from fastapi.testclient import TestClient

from api_gateway.app import create_app
from shared_contracts.interfaces import Notifier


class Capture(Notifier):
    def __init__(self):
        self.sent = []

    def notify(self, alert):
        self.sent.append(alert)


def test_auth_required():
    c = TestClient(create_app(Capture()))
    assert c.get("/v1/alerts").status_code == 401
    assert c.get("/v1/alerts", headers={"x-api-key": "dev-key-1"}).json() == []


def test_internal_alert_notifies_and_lists():
    cap = Capture()
    c = TestClient(create_app(cap))
    a = {"event_id": "e", "severity": "high", "score": 0.9, "message": "m"}
    assert c.post("/internal/alerts", json=a).status_code == 201
    assert len(cap.sent) == 1
    assert len(c.get("/v1/alerts", headers={"x-api-key": "dev-key-1"}).json()) == 1