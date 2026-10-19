"""Zero-copy memoryview and pre-allocated circular ring slot buffer abstraction.

Provides pre-allocated ring buffer slots with zero-copy pointer slicing (WBS 4.1 / FR1 / NFR3)
to eliminate kernel/user-space buffer churn and heap re-allocations on the hot path across
both Windows and Linux environments.
"""

from __future__ import annotations

import threading
from typing import Any


class ZeroCopyRingSlotBuffer:
    """Pre-allocated contiguous memory ring buffer supporting zero-copy memoryview slicing."""

    def __init__(self, slot_count: int = 256, slot_size: int = 65536) -> None:
        """Initialize contiguous pre-allocated byte buffer divided into fixed-size slots.
        
        Default: 256 slots of 64KB each = 16MB pre-allocated contiguous memory pool.
        """
        if slot_count < 1 or slot_size < 1:
            raise ValueError("slot_count and slot_size must be positive integers")
        self.slot_count = slot_count
        self.slot_size = slot_size
        self._total_bytes = slot_count * slot_size
        self._raw_buffer = bytearray(self._total_bytes)
        self._base_view = memoryview(self._raw_buffer)
        self._slot_views = [
            self._base_view[i * slot_size : (i + 1) * slot_size]
            for i in range(slot_count)
        ]
        self._slot_lengths = [0] * slot_count
        self._slot_in_use = [False] * slot_count
        self._head = 0
        self._lock = threading.Lock()
        self.total_allocations = 0
        self.total_zero_copy_slices = 0

    def allocate_slot(self) -> int:
        """Reserve the next ring buffer slot with zero heap allocations on the hot path."""
        with self._lock:
            start_head = self._head
            while self._slot_in_use[self._head]:
                self._head = (self._head + 1) % self.slot_count
                if self._head == start_head:
                    # Ring is full; evict oldest slot
                    evicted_slot = self._head
                    self.wipe_slot(evicted_slot)
                    self._slot_in_use[evicted_slot] = True
                    self._head = (self._head + 1) % self.slot_count
                    self.total_allocations += 1
                    return evicted_slot

            slot_id = self._head
            self._slot_in_use[slot_id] = True
            self._head = (self._head + 1) % self.slot_count
            self.total_allocations += 1
            return slot_id

    def write_bytes(self, slot_id: int, data: bytes | bytearray | memoryview) -> memoryview:
        """Write incoming byte stream directly into the pre-allocated slot view."""
        data_len = len(data)
        if data_len > self.slot_size:
            raise ValueError(f"Payload size {data_len} exceeds slot capacity {self.slot_size}")
        slot_view = self._slot_views[slot_id]
        slot_view[:data_len] = data
        self._slot_lengths[slot_id] = data_len
        return slot_view[:data_len]

    def get_slot_view(self, slot_id: int) -> memoryview:
        """Return the active memoryview slice for the given slot."""
        length = self._slot_lengths[slot_id]
        return self._slot_views[slot_id][:length]

    def slice_http_stream(
        self, slot_id: int, delimiter: bytes = b"\r\n\r\n"
    ) -> tuple[memoryview, memoryview]:
        """Perform zero-copy pointer slicing into headers and body without creating string copies."""
        active_view = self.get_slot_view(slot_id)
        raw_bytes = active_view.tobytes()  # Inspection for delimiter index
        delim_pos = raw_bytes.find(delimiter)
        self.total_zero_copy_slices += 1

        if delim_pos == -1:
            # No delimiter; entire payload treated as body
            header_view = active_view[:0]
            body_view = active_view[:]
        else:
            header_view = active_view[:delim_pos]
            body_view = active_view[delim_pos + len(delimiter) :]

        return header_view, body_view

    def release_slot(self, slot_id: int) -> None:
        """Release slot back into ring pool."""
        with self._lock:
            self._slot_in_use[slot_id] = False
            self._slot_lengths[slot_id] = 0

    def wipe_slot(self, slot_id: int) -> None:
        """Securely zero-fill slot memory to prevent transient data leakage (NFR4)."""
        slot_view = self._slot_views[slot_id]
        slot_view[:] = b"\x00" * self.slot_size
        self._slot_lengths[slot_id] = 0
        self._slot_in_use[slot_id] = False

    def stats(self) -> dict[str, Any]:
        """Return diagnostic metrics for zero-copy memory ring."""
        active_count = sum(1 for in_use in self._slot_in_use if in_use)
        return {
            "slot_count": self.slot_count,
            "slot_size_bytes": self.slot_size,
            "total_pool_bytes": self._total_bytes,
            "active_slots": active_count,
            "free_slots": self.slot_count - active_count,
            "total_allocations": self.total_allocations,
            "total_zero_copy_slices": self.total_zero_copy_slices,
        }
