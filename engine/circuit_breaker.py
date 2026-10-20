"""Hybrid runtime verification and deterministic safety fallback."""

from __future__ import annotations

import time
import logging
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shared_contracts.models import Label

from .verifier import RuntimeLTLVerifier, RuntimeVerificationResult

log = logging.getLogger("engine.circuit_breaker")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class HybridDecisionResult(BaseModel):
    """Final decision after combining AI scoring and formal verification."""

    model_config = ConfigDict(frozen=True)

    label: Label
    decision: str = Field(pattern="^(ALLOW|BLOCK|FALLBACK)$")
    override_applied: bool
    override_reason: str
    verification: RuntimeVerificationResult


class HybridCircuitBreaker:
    """Protect the request path from slow or unavailable formal verification."""

    def __init__(
        self,
        verifier: RuntimeLTLVerifier | None = None,
        *,
        failure_threshold: int = 3,
        recovery_timeout_ms: float = 1000.0,
        max_latency_ms: float = 25.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout_ms < 0 or max_latency_ms < 0:
            raise ValueError("recovery_timeout_ms and max_latency_ms must be non-negative")
        self.verifier = verifier or RuntimeLTLVerifier()
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.max_latency_ms = max_latency_ms
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._opened_at_ms: float | None = None

        self.total_evaluations = 0
        self.z3_sat_count = 0
        self.z3_unsat_count = 0
        self.timeout_count = 0
        self.fast_path_count = 0
        self.overrides_applied = 0

    def verify(
        self,
        event_id: str,
        context: Mapping[str, Any] | None = None,
        custom_constraints: Iterable[str] | None = None,
        **context_values: Any,
    ) -> RuntimeVerificationResult:
        """Run formal verification or the circuit-open deterministic fast path."""
        self.total_evaluations += 1
        now_ms = time.perf_counter() * 1000

        if self.state is CircuitState.OPEN:
            if self._opened_at_ms is None or (
                now_ms - self._opened_at_ms < self.recovery_timeout_ms
            ):
                self.fast_path_count += 1
                return self._fast_path_result(event_id)
            self.state = CircuitState.HALF_OPEN

        try:
            result = self.verifier.verify(
                event_id,
                context,
                custom_constraints,
                **context_values,
            )
        except Exception as exc:
            log.exception("Runtime verification failed; using circuit fallback")
            return self._verification_failure(event_id, f"Verifier error: {exc}")

        if result.z3_status == "SAT":
            self.z3_sat_count += 1
        elif result.z3_status == "UNSAT":
            self.z3_unsat_count += 1

        is_timeout = (
            result.z3_status == "UNKNOWN"
            or result.execution_time_ms > self.max_latency_ms
        )
        if is_timeout:
            return self._verification_failure(
                event_id,
                result.proof_summary or "Z3 verification exceeded the latency SLA",
                result if result.z3_status == "UNKNOWN" else None,
            )

        if self.state is CircuitState.HALF_OPEN:
            self._close()
        else:
            self.failure_count = 0
        return result

    def evaluate_hybrid_decision(
        self,
        event_id: str,
        score: float,
        anomaly_threshold: float,
        context: Mapping[str, Any] | None = None,
        custom_constraints: Iterable[str] | None = None,
        **context_values: Any,
    ) -> HybridDecisionResult:
        """Apply formal safety overrides while preserving the AI fast path."""
        ai_label = Label.ANOMALY if score >= anomaly_threshold else Label.NORMAL
        verification = self.verify(
            event_id,
            context,
            custom_constraints,
            **context_values,
        )

        if verification.decision == "ALLOW" and verification.z3_status == "SAT":
            if ai_label is Label.ANOMALY:
                self.overrides_applied += 1
                return HybridDecisionResult(
                    label=Label.NORMAL,
                    decision="ALLOW",
                    override_applied=True,
                    override_reason="z3_formal_allow_override",
                    verification=verification,
                )
            return HybridDecisionResult(
                label=ai_label,
                decision="ALLOW",
                override_applied=False,
                override_reason="none",
                verification=verification,
            )

        if verification.decision == "BLOCK" and verification.z3_status == "UNSAT":
            if ai_label is Label.NORMAL:
                self.overrides_applied += 1
                return HybridDecisionResult(
                    label=Label.ANOMALY,
                    decision="BLOCK",
                    override_applied=True,
                    override_reason="z3_formal_block_override",
                    verification=verification,
                )
            return HybridDecisionResult(
                label=ai_label,
                decision="BLOCK",
                override_applied=False,
                override_reason="none",
                verification=verification,
            )

        self.fast_path_count += 1
        return HybridDecisionResult(
            label=ai_label,
            decision="ALLOW" if ai_label is Label.NORMAL else "BLOCK",
            override_applied=False,
            override_reason="circuit_breaker_fast_path",
            verification=verification,
        )

    def _verification_failure(
        self,
        event_id: str,
        summary: str,
        result: RuntimeVerificationResult | None = None,
    ) -> RuntimeVerificationResult:
        self.timeout_count += 1
        self.failure_count += 1
        if self.state is CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self._open()
        if result is not None and result.z3_status == "UNKNOWN":
            return result
        elapsed = result.execution_time_ms if result else 0.0
        return RuntimeVerificationResult(
            event_id=event_id,
            z3_status="UNKNOWN",
            decision="FALLBACK",
            violated_rules=[],
            proof_summary=summary,
            execution_time_ms=elapsed,
        )

    def _open(self) -> None:
        self.state = CircuitState.OPEN
        self._opened_at_ms = time.perf_counter() * 1000

    def _close(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._opened_at_ms = None

    @staticmethod
    def _fast_path_result(event_id: str) -> RuntimeVerificationResult:
        return RuntimeVerificationResult(
            event_id=event_id,
            z3_status="UNKNOWN",
            decision="FALLBACK",
            violated_rules=[],
            proof_summary="Circuit breaker open; deterministic AI threshold fast path",
            execution_time_ms=0.0,
        )
