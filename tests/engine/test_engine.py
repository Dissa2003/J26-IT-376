from fastapi.testclient import TestClient

from engine.app import create_app
from engine.service import StubModel, ThresholdPredictionService
from shared_contracts.interfaces import AlertClient


class Capture(AlertClient):
    def __init__(self):
        self.sent = []

    def send_alert(self, alert):
        self.sent.append(alert)


def _client(cap):
    return TestClient(create_app(ThresholdPredictionService(StubModel(), 0.8, "t"), cap))


def test_anomaly_raises_alert():
    cap = Capture()
    r = _client(cap).post("/v1/predict", json={"event_id": "e1", "features": {"x": 0.9}})
    assert r.json()["label"] == "anomaly" and len(cap.sent) == 1


def test_normal_no_alert():
    cap = Capture()
    r = _client(cap).post("/v1/predict", json={"event_id": "e1", "features": {"x": 0.1}})
    assert r.json()["label"] == "normal" and not cap.sent