"""Tests for runtime_handoff ``runtime-verification-input-1`` (LogiShield -> Runtime Verification), D4-7G3.

The compatibility tests pass the handoff of a genuinely fitted LogiShield prediction into Runtime Verification's real,
unmodified ``HybridVerificationService.evaluate_runtime`` and ``HybridCircuitBreaker`` in process. They need
``z3-solver`` (in the root requirements). They show argument compatibility only: nothing in the running system calls
the handoff, and the threshold used here is Runtime Verification's own parameter, not an agreed value.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

import tokenization_ai.runtime_handoff as rh
from tokenization_ai import adapters
from tokenization_ai.anomaly_signal import AnomalySignal
from tokenization_ai.demo import _DEMO_EVENTS
from tokenization_ai.models import AnomalyPrediction
from tokenization_ai.runtime_handoff import RuntimeVerificationInput, to_runtime_verification_input
from tokenization_ai.service import TokenizationAIService


def _prediction(**over) -> AnomalyPrediction:
    return AnomalyPrediction(**{"event_id": "evt-rv-1", "score": 0.4321, "model_version": "ngram-linear-v1-trained", **over})


def test_the_handoff_carries_the_unchanged_event_id_score_and_score_semantics():
    handoff = to_runtime_verification_input(_prediction())
    assert handoff.handoff_version == "runtime-verification-input-1"
    assert (handoff.signal.event_id, handoff.signal.score, handoff.signal.model_version) == ("evt-rv-1", 0.4321, "ngram-linear-v1-trained")
    assert handoff.signal.score_semantics == "uncalibrated_model_output"
    assert handoff.signal.is_probability is False and handoff.signal.calibrated is False
    assert handoff.evaluate_runtime_kwargs() == {"event_id": "evt-rv-1", "score": 0.4321}


def test_the_adapter_entry_point_returns_the_same_handoff():
    assert adapters.to_runtime_verification_output(_prediction()) == to_runtime_verification_input(_prediction())


def test_no_label_threshold_or_decision_is_added():
    dumped = to_runtime_verification_input(_prediction()).model_dump(mode="json")
    assert set(dumped) == {"handoff_version", "signal"}
    assert set(dumped["signal"]) == {"schema_version", "event_id", "score", "model_version", "score_semantics", "is_probability", "calibrated"}
    assert set(to_runtime_verification_input(_prediction()).evaluate_runtime_kwargs()) == {"event_id", "score"}


def test_the_handoff_is_immutable_strict_and_json_safe():
    handoff = to_runtime_verification_input(_prediction())
    with pytest.raises(ValidationError):
        handoff.handoff_version = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RuntimeVerificationInput(signal=handoff.signal, threshold=0.8)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        RuntimeVerificationInput(handoff_version="runtime-verification-input-2", signal=handoff.signal)  # type: ignore[arg-type]
    assert json.loads(json.dumps(handoff.model_dump(mode="json")))["signal"]["score"] == 0.4321


@pytest.mark.parametrize("bad", [None, {"event_id": "e", "score": 0.5, "model_version": "m"}, "prediction", 0.5])
def test_anything_but_a_prediction_is_rejected(bad):
    with pytest.raises(TypeError):
        to_runtime_verification_input(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("over", [{"event_id": ""}, {"model_version": ""}, {"score": math.nan}, {"score": math.inf}])
def test_an_invalid_prediction_never_becomes_a_handoff(over):
    prediction = AnomalyPrediction.model_construct(**{"event_id": "e", "score": 0.5, "model_version": "m", **over})  # bypasses its own checks
    with pytest.raises(ValidationError):
        to_runtime_verification_input(prediction)


def test_scores_at_the_bounds_are_carried_exactly():
    for score in (0.0, 1.0):
        assert to_runtime_verification_input(_prediction(score=score)).evaluate_runtime_kwargs()["score"] == score


# -- argument compatibility with Runtime Verification's real code ---------------------------------------------------------


def _engine():
    pytest.importorskip("z3")
    from engine.circuit_breaker import HybridCircuitBreaker
    from engine.service import HybridVerificationService, StubModel
    from engine.verifier import RuntimeLTLVerifier
    return HybridVerificationService, HybridCircuitBreaker, RuntimeLTLVerifier, StubModel


def test_a_fitted_logishield_prediction_is_accepted_by_runtime_verifications_evaluate_runtime():
    HybridVerificationService, HybridCircuitBreaker, RuntimeLTLVerifier, StubModel = _engine()
    service = TokenizationAIService.create_default()
    for context, rule_info in _DEMO_EVENTS:
        handoff = adapters.to_runtime_verification_output(service.predict(context, rule_info))
        runtime = HybridVerificationService(StubModel(), threshold=0.8, model_version="engine-under-test",
                                            circuit_breaker=HybridCircuitBreaker(RuntimeLTLVerifier(timeout_ms=50), max_latency_ms=1000.0))
        result = runtime.evaluate_runtime(**handoff.evaluate_runtime_kwargs(),
                                          metadata={"role": "admin", "action": "manage", "rate_limit": 100, "request_rate": 1})
        assert result.decision in ("ALLOW", "BLOCK", "FALLBACK")  # Runtime Verification's own rules decide; nothing is asserted about them
        assert result.verification.execution_time_ms >= 0


def test_the_handoff_matches_the_circuit_breakers_parameters():
    _, HybridCircuitBreaker, RuntimeLTLVerifier, _ = _engine()
    handoff = to_runtime_verification_input(_prediction(score=0.95))
    kwargs = handoff.evaluate_runtime_kwargs()
    decision = HybridCircuitBreaker(RuntimeLTLVerifier(timeout_ms=50), max_latency_ms=1000.0).evaluate_hybrid_decision(
        kwargs["event_id"], kwargs["score"], 0.8, {"role": "admin", "action": "manage", "rate_limit": 100, "request_rate": 1})
    assert decision.decision in ("ALLOW", "BLOCK", "FALLBACK") and decision.label.value in ("anomaly", "normal")


def test_the_module_does_no_io_and_imports_no_other_member_code():
    tree = ast.parse(Path(rh.__file__).read_text(encoding="utf-8"))
    imported = {(("." * n.level) + (n.module or "")) if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert imported <= {"__future__", "typing", "pydantic", ".anomaly_signal", ".models"}, imported
    assert isinstance(to_runtime_verification_input(_prediction()).signal, AnomalySignal)
