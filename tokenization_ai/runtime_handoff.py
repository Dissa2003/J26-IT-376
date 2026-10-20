"""Member 4 -> Runtime Verification handoff ``runtime-verification-input-1``.

Runtime Verification already exposes a score input: ``engine.service.HybridVerificationService
.evaluate_runtime(event_id: str, score: float, ...)`` and ``engine.circuit_breaker.HybridCircuitBreaker
.evaluate_hybrid_decision(event_id, score, anomaly_threshold, ...)``. This module produces exactly the
``event_id`` and ``score`` those calls take, from one :class:`~tokenization_ai.models.AnomalyPrediction`,
and keeps the meaning of the score attached to it. It defines no new Runtime Verification field.

What it is not
--------------
- It performs no I/O and does not import or call Runtime Verification; nothing in the running
  system calls it yet (the gateway does not route requests to this component).
- It sets no threshold and no label. Runtime Verification applies its own ``anomaly_threshold``
  (``score >= threshold`` means anomaly). **The score is an uncalibrated model output, not a
  probability**; a threshold chosen for probabilities is not automatically meaningful for it.
  Agreeing the threshold and failure handling is still open with Runtime Verification.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .anomaly_signal import AnomalySignal, to_anomaly_signal
from .models import AnomalyPrediction

RUNTIME_HANDOFF_VERSION = "runtime-verification-input-1"


class RuntimeVerificationInput(BaseModel):
    """One event's handoff: the strict :class:`AnomalySignal` plus the Runtime Verification call arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    handoff_version: Literal["runtime-verification-input-1"] = RUNTIME_HANDOFF_VERSION
    signal: AnomalySignal

    def evaluate_runtime_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``HybridVerificationService.evaluate_runtime`` (and the circuit breaker's
        ``event_id`` / ``score`` parameters). Only these two values are handed over."""
        return {"event_id": self.signal.event_id, "score": self.signal.score}


def to_runtime_verification_input(prediction: AnomalyPrediction) -> RuntimeVerificationInput:
    """Wrap a prediction; the strict signal validation (finite score in [0, 1], non-empty ids) applies."""
    return RuntimeVerificationInput(signal=to_anomaly_signal(prediction))
