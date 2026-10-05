"""Unit tests for broker delivery and the ingestion API."""

from collections.abc import Callable

from fastapi.testclient import TestClient

from ingestion.broker import InMemoryBroker
from ingestion.app import create_app
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
