from fastapi.testclient import TestClient

from ingestion.app import create_app
from shared_contracts.interfaces import ProcessingClient
from shared_contracts.models import Label, Prediction


class FakeProcessing(ProcessingClient):
    def process(self, event):
        return Prediction(event_id=event.event_id, score=0.1, label=Label.NORMAL, threshold=0.8, model_version="t")


def test_ingest_forwards_to_processing():
    c = TestClient(create_app(processing=FakeProcessing()))
    r = c.post("/v1/ingest", json={"source": "s", "payload": {"x": 1}})
    assert r.status_code == 202 and r.json()["label"] == "normal"