"""Event broker implementations for local and Redis Streams deployments."""

from __future__ import annotations

import logging
import socket
import uuid
from collections.abc import Callable
from typing import Any, Protocol

from redis import Redis
from redis.exceptions import RedisError, ResponseError

from shared_contracts.config import Settings, get_settings
from shared_contracts.interfaces import EventConsumer, EventProducer
from shared_contracts.models import RawEvent

log = logging.getLogger(__name__)
EventHandler = Callable[[RawEvent], None]


class RedisStreamClient(Protocol):
    """Subset of Redis client operations used by :class:`RedisStreamBroker`."""

    def ping(self) -> bool: ...

    def xadd(self, name: str, fields: dict[str, str], *, maxlen: int | None = None,
             approximate: bool = True) -> str: ...

    def xgroup_create(self, name: str, groupname: str, *, id: str = "0",
                      mkstream: bool = False) -> bool: ...

    def xreadgroup(
        self, groupname: str, consumername: str, streams: dict[str, str], *,
        count: int | None = None, block: int | None = None,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]: ...

    def xack(self, name: str, groupname: str, *ids: str) -> int: ...


class InMemoryBroker(EventProducer, EventConsumer):
    """Publish events synchronously to all registered consumers."""

    def __init__(self) -> None:
        """Initialize an empty broker."""
        self._handlers: list[EventHandler] = []
        self.dead_letters: list[dict[str, Any]] = []

    def subscribe(self, handler: EventHandler) -> None:
        """Register a consumer callback for future events."""
        self._handlers.append(handler)

    def publish(self, event: RawEvent) -> None:
        """Deliver an event to each registered consumer in registration order."""
        for handler in self._handlers:
            handler(event)

    def publish_dlq(self, payload: dict[str, Any], reason: str) -> None:
        """Store a malformed event for local inspection."""
        self.dead_letters.append({"payload": payload, "reason": reason})


class RedisStreamBroker(EventProducer, EventConsumer):
    """Reliable event broker backed by a Redis Stream consumer group."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        redis_client: RedisStreamClient | None = None,
        stream: str = "events",
        group: str = "ingestion",
        consumer: str | None = None,
        block_ms: int = 1_000,
        batch_size: int = 10,
        max_stream_length: int = 100_000,
    ) -> None:
        """Create a pooled Redis client and ensure the consumer group exists."""
        self._settings = settings or get_settings()
        self._stream = stream
        self._group = group
        self._consumer = consumer or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._block_ms = block_ms
        self._batch_size = batch_size
        self._max_stream_length = max_stream_length
        self._client = redis_client or Redis.from_url(
            self._settings.redis_url,
            decode_responses=True,
            health_check_interval=30,
            socket_connect_timeout=2,
            socket_timeout=max(5, (block_ms // 1_000) + 5),
            retry_on_timeout=True,
            max_connections=20,
        )
        self._ensure_group()

    @property
    def client(self) -> RedisStreamClient:
        """Expose the client for lifecycle management and focused tests."""
        return self._client

    def _ensure_group(self) -> None:
        """Create the stream and group, tolerating an already-created group."""
        try:
            self._client.xgroup_create(
                self._stream, self._group, id="0", mkstream=True
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc).upper():
                log.exception("Unable to create Redis consumer group %s", self._group)
                raise
        except RedisError:
            log.exception("Unable to connect to Redis while creating consumer group")
            raise

    def publish(self, event: RawEvent) -> None:
        """Append a serialized RawEvent to the Redis Stream."""
        try:
            self._client.xadd(
                self._stream,
                {"event": event.model_dump_json()},
                maxlen=self._max_stream_length,
                approximate=True,
            )
        except RedisError:
            log.exception("Failed to publish event %s to Redis Stream", event.event_id)
            raise

    def publish_dlq(self, payload: dict[str, Any], reason: str) -> None:
        """Append a malformed event and its validation reason to the DLQ stream."""
        try:
            self._client.xadd(
                "dlq:events",
                {"payload": _json_dumps(payload), "reason": reason},
                maxlen=self._max_stream_length,
                approximate=True,
            )
        except RedisError:
            log.exception("Failed to publish malformed event to Redis DLQ")
            raise

    def subscribe(self, handler: EventHandler) -> None:
        """Consume, dispatch, and acknowledge events until interrupted.

        Events are acknowledged only after the handler returns successfully. A
        handler exception leaves the message pending for later recovery.
        """
        while True:
            try:
                records = self._client.xreadgroup(
                    self._group,
                    self._consumer,
                    {self._stream: ">"},
                    count=self._batch_size,
                    block=self._block_ms,
                )
                for stream_name, messages in records:
                    for message_id, fields in messages:
                        try:
                            event = RawEvent.model_validate_json(fields["event"])
                            handler(event)
                            self._client.xack(
                                stream_name, self._group, message_id
                            )
                        except (KeyError, TypeError, ValueError):
                            log.exception(
                                "Invalid event %s in Redis Stream %s",
                                message_id,
                                stream_name,
                            )
                        except Exception:
                            log.exception(
                                "Event handler failed for Redis message %s",
                                message_id,
                            )
            except RedisError:
                log.exception("Redis consumer loop failed; retrying")


def create_broker(settings: Settings | None = None) -> EventProducer & EventConsumer:
    """Select Redis in deployed environments and fall back to memory locally."""
    resolved = settings or get_settings()
    if resolved.app_env.lower() == "local":
        log.info("Using in-memory broker for local environment")
        return InMemoryBroker()

    try:
        broker = RedisStreamBroker(resolved)
        # Fail fast during startup rather than switching after accepting traffic.
        broker.client.ping()
        return broker
    except (RedisError, OSError, socket.error):
        log.exception("Redis unavailable; falling back to in-memory broker")
        return InMemoryBroker()


def _json_dumps(payload: dict[str, Any]) -> str:
    """Serialize arbitrary request data for the Redis DLQ."""
    import json

    return json.dumps(payload, default=str, separators=(",", ":"))
