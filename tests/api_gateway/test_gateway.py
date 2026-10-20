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


def test_gateway_cors_headers():
    c = TestClient(create_app(Capture()))
    r = c.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")


def test_gateway_role_escalation_rejected():
    import base64, json
    header_b64 = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(b'{"sub":"user123","role":"user"}').decode().rstrip("=")
    jwt_token = f"{header_b64}.{payload_b64}.fake_sig"

    c = TestClient(create_app(Capture()))
    # Normal user token claiming admin role should be rejected with 403
    r = c.get(
        "/v1/alerts",
        headers={
            "Authorization": f"Bearer {jwt_token}",
            "x-user-role": "admin",
        },
    )
    assert r.status_code == 403


def test_gateway_numeric_validation_returns_400():
    c = TestClient(create_app(Capture()))
    invalid_event = {
        "event_id": "e-bad-num",
        "source": "checkout",
        "payload": {
            "amount": "not_a_number",
            "item": "laptop",
        },
    }
    r = c.post(
        "/v1/events",
        json=invalid_event,
        headers={"x-api-key": "dev-key-1"},
    )
    assert r.status_code == 400
    assert "Invalid numeric format" in r.json()["detail"]