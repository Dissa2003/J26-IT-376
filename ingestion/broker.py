"""In-memory event broker for local development and unit tests."""

from __future__ import annotations

from collections.abc import Callable

from shared_contracts.interfaces import EventConsumer, EventProducer
from shared_contracts.models import RawEvent


class InMemoryBroker(EventProducer, EventConsumer):
    """Publish events synchronously to all registered consumers."""

    def __init__(self) -> None:
        """Initialize an empty broker."""
        self._handlers: list[Callable[[RawEvent], None]] = []

    def subscribe(self, handler: Callable[[RawEvent], None]) -> None:
        """Register a consumer callback for future events."""
        self._handlers.append(handler)

    def publish(self, event: RawEvent) -> None:
        """Deliver an event to each registered consumer in registration order."""
        for handler in self._handlers:
            handler(event)
