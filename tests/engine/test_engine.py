from fastapi.testclient import TestClient
import z3

from engine.app import create_app
from engine.service import StubModel, ThresholdPredictionService
from engine.verifier import RuntimeLTLVerifier
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


def test_runtime_verifier_allows_valid_context_under_five_ms():
    result = RuntimeLTLVerifier().verify(
        "e-valid",
        {
            "threat_score": 0.2,
            "role": "user",
            "action": "read",
            "rate_limit": 10,
            "request_rate": 2,
            "payload_size": 128,
        },
    )

    assert result.z3_status == "SAT"
    assert result.decision == "ALLOW"
    assert result.violated_rules == []
    assert result.execution_time_ms < 5


def test_runtime_verifier_blocks_policy_violation():
    result = RuntimeLTLVerifier().verify(
        "e-blocked",
        {"threat_score": 1.4, "role": "user", "action": "delete"},
    )

    assert result.z3_status == "UNSAT"
    assert result.decision == "BLOCK"
    assert "threat_score_bounds" in result.violated_rules
    assert "privileged_role_required" in result.violated_rules


def test_runtime_verifier_falls_back_on_unknown(monkeypatch):
    monkeypatch.setattr("z3.Solver.check", lambda self: z3.unknown)

    result = RuntimeLTLVerifier(timeout_ms=1).verify(
        "e-timeout", {"threat_score": 0.2}
    )

    assert result.z3_status == "UNKNOWN"
    assert result.decision == "FALLBACK"