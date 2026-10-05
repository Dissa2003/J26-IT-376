from __future__ import annotations

from typing import Callable

from shared_contracts.interfaces import EventConsumer, EventProducer
from shared_contracts.models import RawEvent


class InMemoryBroker(EventProducer, EventConsumer):
    """Synchronous stub. Replace with Redis Streams/Kafka implementing the same interfaces."""

    def __init__(self) -> None:
        self._handlers: list[Callable[[RawEvent], None]] = []

    def subscribe(self, handler: Callable[[RawEvent], None]) -> None:
        self._handlers.append(handler)

    def publish(self, event: RawEvent) -> None:
        for h in self._handlers:
            h(event)