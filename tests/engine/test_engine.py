from fastapi.testclient import TestClient
import z3
import time

from engine.app import create_app
from engine.circuit_breaker import CircuitState, HybridCircuitBreaker
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


def test_hybrid_circuit_breaker_transitions_and_recovers(monkeypatch):
    monkeypatch.setattr("z3.Solver.check", lambda self: z3.unknown)
    breaker = HybridCircuitBreaker(failure_threshold=2, recovery_timeout_ms=0)

    breaker.verify("e1", {"threat_score": 0.1})
    breaker.verify("e2", {"threat_score": 0.1})
    assert breaker.state is CircuitState.OPEN
    assert breaker.timeout_count == 2

    monkeypatch.undo()
    result = breaker.verify("probe", {"threat_score": 0.1})
    assert result.z3_status == "SAT"
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_hybrid_ai_false_positive_is_overridden():
    breaker = HybridCircuitBreaker()

    result = breaker.evaluate_hybrid_decision(
        "e-positive", 0.95, 0.8, {"threat_score": 0.1}
    )

    assert result.label == "normal"
    assert result.decision == "ALLOW"
    assert result.override_applied is True
    assert result.override_reason == "z3_formal_allow_override"
    assert breaker.overrides_applied == 1


def test_hybrid_ai_false_negative_is_overridden():
    breaker = HybridCircuitBreaker()

    result = breaker.evaluate_hybrid_decision(
        "e-negative",
        0.1,
        0.8,
        {"threat_score": 1.4, "role": "user", "action": "delete"},
    )

    assert result.label == "anomaly"
    assert result.decision == "BLOCK"
    assert result.override_applied is True
    assert result.override_reason == "z3_formal_block_override"


def test_hybrid_open_circuit_uses_ai_fast_path(monkeypatch):
    breaker = HybridCircuitBreaker(failure_threshold=1, recovery_timeout_ms=60_000)
    monkeypatch.setattr("z3.Solver.check", lambda self: z3.unknown)
    breaker.verify("trip", {"threat_score": 0.1})

    started = time.perf_counter()
    result = breaker.evaluate_hybrid_decision(
        "fast", 0.9, 0.8, {"threat_score": 0.1}
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert result.label == "anomaly"
    assert result.decision == "BLOCK"
    assert result.override_applied is False
    assert result.override_reason == "circuit_breaker_fast_path"
    assert result.verification.z3_status == "UNKNOWN"
    assert elapsed_ms < 5