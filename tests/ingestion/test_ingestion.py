"""Unit tests for broker delivery and the ingestion API."""

from collections.abc import Callable
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from ingestion.broker import InMemoryBroker, RedisStreamBroker, create_broker
from ingestion.app import create_app
from ingestion.validator import IngestionValidator
from shared_contracts.config import Settings
from shared_contracts.interfaces import ProcessingClient
from shared_contracts.models import Label, Prediction, RawEvent


class FakeProcessing(ProcessingClient):
    """Capture-free processing client for app tests."""

    def process(self, event: RawEvent) -> Prediction:
        """Return a deterministic normal prediction."""
        return Prediction(event_id=event.event_id, score=0.1, label=Label.NORMAL, threshold=0.8, model_version="t")


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


def test_create_broker_falls_back_when_redis_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Redis startup failure selects the in-memory fallback."""
    def fail(_: Settings) -> RedisStreamBroker:
        raise RedisConnectionError("connection refused")

    monkeypatch.setattr("ingestion.broker.RedisStreamBroker", fail)

    broker = create_broker(Settings(app_env="production"))

    assert isinstance(broker, InMemoryBroker)
