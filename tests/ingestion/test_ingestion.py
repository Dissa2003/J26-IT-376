"""Unit tests for broker delivery and the ingestion API."""

from collections.abc import Callable
import threading
import time
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from ingestion.broker import InMemoryBroker, RedisStreamBroker, RingBufferBroker, create_broker
from ingestion.app import create_app
from ingestion.validator import IngestionValidator
from ingestion.pii_masker import PIIMasker, REDACTED
from ingestion.vector_builder import UnifiedContextVectorBuilder
from shared_contracts.config import Settings
from shared_contracts.interfaces import ProcessingClient
from shared_contracts.models import Label, Prediction, RawEvent


class FakeProcessing(ProcessingClient):
    """Capture-free processing client for app tests."""

    def process(self, event: RawEvent) -> Prediction:
        """Return a deterministic normal prediction."""
        return Prediction(event_id=event.event_id, score=0.1, label=Label.NORMAL, threshold=0.8, model_version="t")


class FlakyProcessing(ProcessingClient):
    """Fail once to model a temporary processing outage."""

    def __init__(self) -> None:
        self.attempts = 0

    def process(self, event: RawEvent) -> Prediction:
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("processing is offline")
        return Prediction(
            event_id=event.event_id,
            score=0.2,
            label=Label.NORMAL,
            threshold=0.8,
            model_version="retry-test",
        )


class CapturingProcessing(FakeProcessing):
    """Processing fake that records the event received downstream."""

    def __init__(self) -> None:
        self.events: list[RawEvent] = []

    def process(self, event: RawEvent) -> Prediction:
        self.events.append(event)
        return super().process(event)


def test_publish_delivers_raw_event_to_subscriber() -> None:
    """A published RawEvent is delivered to each subscribed consumer."""
    broker = InMemoryBroker()
    received: list[RawEvent] = []

    broker.subscribe(received.append)
    event = RawEvent(source="test-source", payload={"value": 1})

    broker.publish(event)

    assert received == [event]


def test_publish_delivers_to_multiple_subscribers() -> None:
    """A published RawEvent is delivered in subscription order."""
    broker = InMemoryBroker()
    received: list[str] = []

    def record(name: str) -> Callable[[RawEvent], None]:
        def handler(event: RawEvent) -> None:
            received.append(f"{name}:{event.event_id}")

        return handler

    broker.subscribe(record("first"))
    broker.subscribe(record("second"))
    event = RawEvent(source="test-source")

    broker.publish(event)

    assert received == [f"first:{event.event_id}", f"second:{event.event_id}"]


def test_ingest_forwards_to_processing() -> None:
    """The HTTP entry point publishes an event to the processing client."""
    c = TestClient(create_app(processing=FakeProcessing()))
    r = c.post("/v1/ingest", json={"source": "s", "payload": {"x": 1}})
    assert r.status_code == 202 and r.json()["label"] == "normal"


def test_ingest_full_path_uses_broker_and_processing_response() -> None:
    """POST /v1/ingest publishes through the broker to the processing client."""
    broker = InMemoryBroker()
    processing = FakeProcessing()
    client = TestClient(create_app(processing=processing, broker=broker))

    response = client.post(
        "/v1/ingest",
        json={"source": "integration-test", "payload": {"value": 4}},
    )

    assert response.status_code == 202
    assert response.json()["label"] == "normal"
    assert response.json()["model_version"] == "t"


def test_processing_outage_retries_without_dropping_event() -> None:
    """A transient processing outage is retried from the in-process queue."""
    processing = FlakyProcessing()
    client = TestClient(
        create_app(
            processing=processing,
            broker=InMemoryBroker(),
            processing_timeout=2.0,
        )
    )

    response = client.post(
        "/v1/ingest",
        json={"source": "retry-test", "payload": {"value": 4}},
    )

    assert response.status_code == 202
    assert response.json()["model_version"] == "retry-test"
    assert processing.attempts == 2


def test_validator_rejects_empty_source() -> None:
    """Events without a meaningful source are invalid."""
    with pytest.raises(ValueError, match="source"):
        IngestionValidator().validate({"source": "  ", "payload": {}})


def test_invalid_ingest_is_written_to_local_dlq() -> None:
    """Invalid requests return 422 and are retained by the local DLQ."""
    broker = InMemoryBroker()
    client = TestClient(create_app(processing=FakeProcessing(), broker=broker))

    response = client.post("/v1/ingest", json={"source": "", "payload": {}})

    assert response.status_code == 422
    assert broker.dead_letters[0]["payload"]["source"] == ""


def test_redis_dlq_uses_dedicated_stream() -> None:
    """Malformed payloads are written to the dedicated Redis DLQ stream."""
    client = MagicMock()
    broker = RedisStreamBroker(Settings(app_env="production"), redis_client=client)
    payload = {"source": "", "payload": {}}

    broker.publish_dlq(payload, "source must be non-empty")

    args, kwargs = client.xadd.call_args
    assert args[0] == "dlq:events"
    assert kwargs["maxlen"] == 100_000


def test_redis_publish_serializes_raw_event() -> None:
    """Redis publish uses XADD with the RawEvent JSON payload."""
    client = MagicMock()
    broker = RedisStreamBroker(
        Settings(app_env="production"),
        redis_client=client,
    )
    event = RawEvent(source="redis-test", payload={"value": 3})

    broker.publish(event)

    client.xadd.assert_called_once()
    args, kwargs = client.xadd.call_args
    assert args[0] == "events"
    assert RawEvent.model_validate_json(args[1]["event"]) == event
    assert kwargs["maxlen"] == 100_000


def test_redis_subscribe_dispatches_and_acknowledges() -> None:
    """Redis records are dispatched and acknowledged after handler success."""
    client = MagicMock()
    event = RawEvent(source="redis-test")
    client.xreadgroup.side_effect = [
        [("events", [("1-0", {"event": event.model_dump_json()})])],
        KeyboardInterrupt,
    ]
    received: list[RawEvent] = []
    broker = RedisStreamBroker(Settings(app_env="production"), redis_client=client)

    with pytest.raises(KeyboardInterrupt):
        broker.subscribe(received.append)

    assert received == [event]
    client.xack.assert_called_once_with("events", "ingestion", "1-0")


def test_create_broker_uses_memory_in_local_environment() -> None:
    """Local development never attempts a Redis connection."""
    broker = create_broker(Settings(app_env="local"))

    assert isinstance(broker, InMemoryBroker)


def test_ring_buffer_drains_events_sequentially() -> None:
    """Buffered events are delivered FIFO by one background worker."""
    broker = RingBufferBroker(maxlen=8)
    received: list[str] = []
    broker.subscribe(lambda event: received.append(event.event_id))
    events = [RawEvent(event_id=str(index), source="test") for index in range(4)]

    for event in events:
        broker.publish(event)

    deadline = time.monotonic() + 1.0
    while len(received) < len(events) and time.monotonic() < deadline:
        time.sleep(0.001)
    broker.stop()

    assert received == ["0", "1", "2", "3"]
    assert broker.evicted_count == 0


def test_ring_buffer_evicts_oldest_event_when_full() -> None:
    """A full circular buffer evicts the oldest event without blocking."""
    broker = RingBufferBroker(maxlen=2)
    started = threading.Event()
    release = threading.Event()

    def slow_handler(event: RawEvent) -> None:
        started.set()
        release.wait(1.0)

    broker.subscribe(slow_handler)
    events = [RawEvent(event_id=str(index), source="test") for index in range(4)]

    broker.publish(events[0])
    assert started.wait(1.0)
    broker.publish(events[1])
    broker.publish(events[2])
    broker.publish(events[3])
    release.set()

    deadline = time.monotonic() + 1.0
    while broker.buffered_count and time.monotonic() < deadline:
        time.sleep(0.001)
    broker.stop()

    assert broker.evicted_count >= 1


def test_create_broker_falls_back_when_redis_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Redis startup failure selects the in-memory fallback."""
    def fail(_: Settings) -> RedisStreamBroker:
        raise RedisConnectionError("connection refused")

    monkeypatch.setattr("ingestion.broker.RedisStreamBroker", fail)

    broker = create_broker(Settings(app_env="production"))

    assert isinstance(broker, InMemoryBroker)


def test_ingest_captures_http_interception_payload() -> None:
    """HTTP interception payloads return an immediate capture acknowledgement."""
    client = TestClient(create_app(processing=FakeProcessing()))

    r = client.post(
        "/v1/ingest",
        json={
            "metadata": {"path": "/orders", "method": "POST"},
            "payload": {"x": 1},
            "threat_score": 0.1,
        },
    )
    assert r.status_code == 202
    body = r.json()
    assert body["event_id"]
    assert body["masked_payload"]["x"] == 1
    assert body["protobuf_bytes_size"] > 0
    assert body["unified_context_vector"] == [0.1, 1.0]
    assert body["pipeline_metrics"]["execution_time_ms"] >= 0


def test_vector_builder_serializes_compact_context_envelope() -> None:
    """The vector starts with the threat score and protobuf output is binary."""
    builder = UnifiedContextVectorBuilder()
    payload = {"temperature": 14.2, "nested": {"pressure": 1450}}

    vector = builder.build(payload, 0.95)
    encoded = builder.serialize("event-1", payload, 0.95, vector)

    assert vector == [0.95, 14.2, 1450.0]
    assert isinstance(encoded, bytes)
    assert encoded


def test_pii_masker_redacts_nested_sensitive_values() -> None:
    """Nested emails, cards, IPs, and credentials are masked in memory."""
    payload = {
        "email": "user@example.com",
        "card": "4111 1111 1111 1111",
        "ip": "192.168.1.10",
        "credentials": {"password": "secret", "user": "alice"},
        "items": [{"token": "abc", "value": 7}],
    }

    masked = PIIMasker().mask(payload)

    assert masked == {
        "email": REDACTED,
        "card": REDACTED,
        "ip": REDACTED,
        "credentials": REDACTED,
        "items": [{"token": REDACTED, "value": 7}],
    }
    assert payload["email"] == "user@example.com"


def test_interception_forwards_masked_payload() -> None:
    """PII is masked before the event reaches downstream processing."""
    processing = CapturingProcessing()
    client = TestClient(create_app(processing=processing))
    response = client.post(
        "/v1/ingest",
        json={
            "metadata": {"path": "/login"},
            "payload": {"email": "user@example.com", "password": "secret"},
            "threat_score": 0.1,
        },
    )

    assert response.status_code == 202
    assert processing.events[0].payload["email"] == REDACTED
    assert processing.events[0].payload["password"] == REDACTED


def test_interception_response_has_context_vector_and_sub_10ms_execution() -> None:
    """The interceptor returns the vector and meets the latency target."""
    client = TestClient(create_app(processing=FakeProcessing()))

    response = client.post(
        "/v1/ingest",
        json={
            "metadata": {"method": "POST", "path": "/payments"},
            "payload": {"amount": 149.95, "retry_count": 2},
            "threat_score": 0.88,
        },
    )

    body = response.json()
    assert response.status_code == 202
    assert body["unified_context_vector"] == [0.88, 149.95, 2.0]
    assert body["pipeline_metrics"]["execution_time_ms"] < 10


def test_invalid_interception_payload_returns_422() -> None:
    """Invalid interceptor envelopes are rejected by the API schema."""
    client = TestClient(create_app(processing=FakeProcessing()))

    response = client.post(
        "/v1/ingest",
        json={
            "metadata": {"path": "/payments"},
            "payload": {"amount": 10},
            "threat_score": 2.0,
        },
    )

    assert response.status_code == 422
