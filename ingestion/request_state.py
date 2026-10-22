"""Bounded request-state ring for SZ-CFM correlation.

Concurrency model (lock-based; this structure is *not* lock-free):

* Free slot indices live in a FIFO free list guarded by one ring lock. A slot
  index taken from the free list is owned exclusively by the allocating
  request until it is released, so concurrent requests never share a slot.
  FIFO reuse maximises the time before an index is handed out again.
* Each slot has its own lock guarding its state machine, so transitions on
  different requests do not contend.
* Every allocation increments the slot's ``generation``. A :class:`SlotTicket`
  (slot index, generation, request_id, correlation_id) is the only handle to a
  slot; every operation re-checks all four fields under the slot lock. A late
  AI result carrying an old generation therefore cannot modify a reused slot.
* A full ring is never resolved by evicting an in-flight request. Allocation
  first reclaims slots whose owner exceeded ``stale_after_ns`` (their
  generation is bumped, so the old owner's handle is invalidated), and
  otherwise raises :class:`RingFullError`.

Each slot also owns a fixed region of a :class:`ZeroCopyRingSlotBuffer` that
holds the request's serialized protobuf envelope. Consumers receive a read-only
``memoryview`` of that region (no further copy) that is valid until release;
release zero-fills the region.

State machine::

    FREE -> ALLOCATED -> AI_PENDING -> AI_COMPLETED -> DECIDED -> FREE
                     \\            \\-> AI_UNAVAILABLE -/
                      \\-------------------------------> DECIDED (AI skipped)
    any non-FREE state -> FREE via release (owner) or stale reclamation
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .zero_copy import ZeroCopyRingSlotBuffer


class SlotState(str, Enum):
    FREE = "FREE"
    ALLOCATED = "ALLOCATED"
    AI_PENDING = "AI_PENDING"
    AI_COMPLETED = "AI_COMPLETED"
    AI_UNAVAILABLE = "AI_UNAVAILABLE"
    DECIDED = "DECIDED"


class DeliveryOutcome(str, Enum):
    """Result of matching an asynchronous AI result to a slot."""

    ACCEPTED = "ACCEPTED"
    STALE_GENERATION = "STALE_GENERATION"
    REQUEST_ID_MISMATCH = "REQUEST_ID_MISMATCH"
    CORRELATION_MISMATCH = "CORRELATION_MISMATCH"
    DUPLICATE = "DUPLICATE"
    LATE = "LATE"
    NOT_AWAITING = "NOT_AWAITING"
    UNKNOWN_SLOT = "UNKNOWN_SLOT"


class RingFullError(RuntimeError):
    """No free slot is available and none could be reclaimed."""


class StaleTicketError(RuntimeError):
    """The ticket no longer owns its slot (released or reclaimed)."""


@dataclass(frozen=True)
class SlotTicket:
    slot_index: int
    generation: int
    request_id: str
    correlation_id: str


@dataclass
class _Slot:
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready: threading.Event = field(default_factory=threading.Event)
    generation: int = 0
    state: SlotState = SlotState.FREE
    request_id: str | None = None
    correlation_id: str | None = None
    allocated_ns: int = 0
    ai_result: Any = None
    unavailable_reason: str | None = None
    decision: Any = None
    envelope_length: int = 0


class RequestStateRing:
    def __init__(
        self,
        capacity: int = 1024,
        slot_bytes: int = 65_536,
        stale_after_ns: int = 30_000_000_000,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if stale_after_ns <= 0:
            raise ValueError("stale_after_ns must be positive")
        self.capacity = capacity
        self._stale_after_ns = stale_after_ns
        self._slots = [_Slot() for _ in range(capacity)]
        self._free: deque[int] = deque(range(capacity))
        self._ring_lock = threading.Lock()
        self._memory = ZeroCopyRingSlotBuffer(slot_count=capacity, slot_size=slot_bytes)
        self.slot_bytes = slot_bytes
        self.full_rejections = 0
        self.stale_reclaimed = 0

    # --- allocation / release -------------------------------------------------

    def allocate(self, request_id: str, correlation_id: str) -> SlotTicket:
        """Reserve a slot for one request or raise :class:`RingFullError`."""
        with self._ring_lock:
            if not self._free:
                self._reclaim_stale_locked()
            if not self._free:
                self.full_rejections += 1
                raise RingFullError(f"request-state ring full ({self.capacity} slots in use)")
            index = self._free.popleft()
        slot = self._slots[index]
        with slot.lock:
            slot.generation += 1
            slot.state = SlotState.ALLOCATED
            slot.request_id = request_id
            slot.correlation_id = correlation_id
            slot.allocated_ns = time.monotonic_ns()
            slot.ai_result = None
            slot.unavailable_reason = None
            slot.decision = None
            slot.ready.clear()
            return SlotTicket(index, slot.generation, request_id, correlation_id)

    def release(self, ticket: SlotTicket) -> bool:
        """Wipe and free the slot if the ticket still owns it."""
        slot = self._slots[ticket.slot_index]
        with slot.lock:
            if not self._owns(slot, ticket):
                return False
            self._clear_locked(ticket.slot_index, slot)
        with self._ring_lock:
            self._free.append(ticket.slot_index)
        return True

    def _reclaim_stale_locked(self) -> None:
        """Free slots whose owner exceeded ``stale_after_ns``. Caller holds the ring lock."""
        now = time.monotonic_ns()
        for index, slot in enumerate(self._slots):
            with slot.lock:
                if slot.state is SlotState.FREE or now - slot.allocated_ns < self._stale_after_ns:
                    continue
                # Bump the generation so the stale owner's ticket is invalidated.
                slot.generation += 1
                self._clear_locked(index, slot)
            self._free.append(index)
            self.stale_reclaimed += 1

    def _clear_locked(self, index: int, slot: _Slot) -> None:
        if slot.envelope_length:
            self._memory.wipe_slot(index)
        slot.state = SlotState.FREE
        slot.request_id = slot.correlation_id = None
        slot.ai_result = slot.decision = slot.unavailable_reason = None
        slot.envelope_length = 0
        slot.ready.set()  # wake any waiter; it will observe the ticket is no longer valid

    @staticmethod
    def _owns(slot: _Slot, ticket: SlotTicket) -> bool:
        return (
            slot.state is not SlotState.FREE
            and slot.generation == ticket.generation
            and slot.request_id == ticket.request_id
            and slot.correlation_id == ticket.correlation_id
        )

    def _owned_slot(self, ticket: SlotTicket) -> _Slot:
        slot = self._slots[ticket.slot_index]
        if not self._owns(slot, ticket):
            raise StaleTicketError(f"ticket {ticket} no longer owns slot {ticket.slot_index}")
        return slot

    # --- zero-copy envelope storage ---------------------------------------------

    def store_envelope(self, ticket: SlotTicket, data: bytes) -> memoryview:
        """Copy serialized bytes once into the slot's pre-allocated region."""
        slot = self._slots[ticket.slot_index]
        with slot.lock:
            self._owned_slot(ticket)
            view = self._memory.write_bytes(ticket.slot_index, data)
            slot.envelope_length = len(data)
            return view.toreadonly()

    def envelope_view(self, ticket: SlotTicket) -> memoryview:
        """Read-only view of the stored envelope (no copy); valid until release."""
        slot = self._slots[ticket.slot_index]
        with slot.lock:
            self._owned_slot(ticket)
            return self._memory.get_slot_view(ticket.slot_index).toreadonly()

    # --- state transitions -----------------------------------------------------

    def mark_ai_pending(self, ticket: SlotTicket) -> None:
        slot = self._slots[ticket.slot_index]
        with slot.lock:
            self._owned_slot(ticket)
            if slot.state is not SlotState.ALLOCATED:
                raise RuntimeError(f"cannot await AI from state {slot.state.value}")
            slot.state = SlotState.AI_PENDING

    def deliver_ai_result(
        self,
        slot_index: int,
        generation: int,
        request_id: str,
        correlation_id: str,
        result: Any,
    ) -> DeliveryOutcome:
        """Match an asynchronous AI result to its request; never touches another request's state."""
        if not 0 <= slot_index < self.capacity:
            return DeliveryOutcome.UNKNOWN_SLOT
        slot = self._slots[slot_index]
        with slot.lock:
            if slot.state is SlotState.FREE or slot.generation != generation:
                return DeliveryOutcome.STALE_GENERATION
            if slot.request_id != request_id:
                return DeliveryOutcome.REQUEST_ID_MISMATCH
            if slot.correlation_id != correlation_id:
                return DeliveryOutcome.CORRELATION_MISMATCH
            if slot.state is SlotState.AI_COMPLETED or (
                slot.state is SlotState.DECIDED and slot.ai_result is not None
            ):
                return DeliveryOutcome.DUPLICATE
            if slot.state is SlotState.AI_UNAVAILABLE or (
                slot.state is SlotState.DECIDED and slot.unavailable_reason is not None
            ):
                return DeliveryOutcome.LATE
            if slot.state is not SlotState.AI_PENDING:
                return DeliveryOutcome.NOT_AWAITING
            slot.ai_result = result
            slot.state = SlotState.AI_COMPLETED
            slot.ready.set()
            return DeliveryOutcome.ACCEPTED

    def wait_for_ai(self, ticket: SlotTicket, timeout_s: float) -> tuple[SlotState, Any]:
        """Block the owning request (only) until a result arrives or the deadline passes.

        On timeout the slot moves to ``AI_UNAVAILABLE`` atomically, so a result
        that arrives afterwards is rejected as ``LATE``.
        """
        slot = self._slots[ticket.slot_index]
        slot.ready.wait(timeout_s)
        with slot.lock:
            self._owned_slot(ticket)
            if slot.state is SlotState.AI_PENDING:
                slot.state = SlotState.AI_UNAVAILABLE
                slot.unavailable_reason = "TIMEOUT"
            return slot.state, slot.ai_result

    def mark_ai_unavailable(self, ticket: SlotTicket, reason: str) -> bool:
        """Record that no AI result will be used. Returns False if a result already arrived."""
        slot = self._slots[ticket.slot_index]
        with slot.lock:
            self._owned_slot(ticket)
            if slot.state in (SlotState.ALLOCATED, SlotState.AI_PENDING):
                slot.state = SlotState.AI_UNAVAILABLE
                slot.unavailable_reason = reason
                slot.ready.set()
                return True
            return False

    def commit_decision(self, ticket: SlotTicket, decision: Any) -> bool:
        """Record the final decision exactly once. Returns False on a second attempt."""
        slot = self._slots[ticket.slot_index]
        with slot.lock:
            self._owned_slot(ticket)
            if slot.state is SlotState.DECIDED:
                return False
            if slot.state is SlotState.AI_PENDING:
                raise RuntimeError("cannot decide while the AI result is pending")
            if slot.state is SlotState.ALLOCATED:
                slot.unavailable_reason = slot.unavailable_reason or "SKIPPED"
            slot.decision = decision
            slot.state = SlotState.DECIDED
            return True

    # --- introspection ------------------------------------------------------------

    def state_of(self, ticket: SlotTicket) -> SlotState | None:
        slot = self._slots[ticket.slot_index]
        with slot.lock:
            return slot.state if self._owns(slot, ticket) else None

    def generation_of(self, slot_index: int) -> int:
        slot = self._slots[slot_index]
        with slot.lock:
            return slot.generation

    @property
    def occupancy(self) -> int:
        with self._ring_lock:
            return self.capacity - len(self._free)

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "occupancy": self.occupancy,
            "full_rejections": self.full_rejections,
            "stale_reclaimed": self.stale_reclaimed,
            "slot_bytes": self.slot_bytes,
        }
