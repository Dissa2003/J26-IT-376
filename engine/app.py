from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

# Internal imports for configurations, clients, interfaces, models, and engine services
from shared_contracts.config import get_settings
from shared_contracts.http_clients import HttpAlertClient
from shared_contracts.interfaces import AlertClient, PredictionService
from shared_contracts.models import Alert, FeatureVector, Label, Prediction, Severity

from .circuit_breaker import CircuitState, HybridCircuitBreaker
from .service import HybridVerificationService, StubModel, ThresholdPredictionService

# Initialize the module logger for engine-specific events and errors
log = logging.getLogger("engine")


class RuntimeVerifyRequest(BaseModel):
    """
    Request schema for runtime verification of security events.
    Enforces that at least one form of evaluation context is provided.
    """
    event_id: str
    features: dict[str, float] | None = None
    unified_context_vector: list[float] | None = None
    threat_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    policy_rules: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_context(self) -> "RuntimeVerifyRequest":
        """
        Validate that at least one of features, unified_context_vector, 
        or threat_score is supplied to allow proper evaluation.
        """
        if self.features is None and self.unified_context_vector is None and self.threat_score is None:
            raise ValueError(
                "one of features, unified_context_vector, or threat_score is required"
            )
        return self


class RuntimeVerifyResponse(BaseModel):
    """
    Response schema returning evaluation results, decision state, 
    Z3 verification status, and circuit breaker details.
    """
    event_id: str
    score: float
    label: Label
    decision: str = Field(pattern="^(ALLOW|BLOCK)$")
    z3_status: str
    circuit_state: CircuitState
    override_applied: bool
    override_reason: str
    violated_rules: list[str]
    execution_time_ms: float


def create_app(
    service: PredictionService | None = None,
    alerts: AlertClient | None = None,
    hybrid_service: HybridVerificationService | None = None,
) -> FastAPI:
    """
    Factory function to configure and instantiate the FastAPI application
    with required prediction, verification, and alert delivery services.
    """
    s = get_settings()
    
    # Initialize default services using configuration settings if none are provided
    service = service or ThresholdPredictionService(StubModel(), s.anomaly_threshold, s.model_version)
    hybrid_service = hybrid_service or HybridVerificationService(
        StubModel(), s.anomaly_threshold, s.model_version
    )
    alerts = alerts or HttpAlertClient(s.gateway_url)
    app = FastAPI(title="engine")

    @app.get("/health")
    def health():
        """Health check endpoint to verify engine service availability."""
        return {"status": "ok", "service": "engine"}

    @app.post("/v1/predict")
    def predict(features: FeatureVector) -> Prediction:
        """
        Evaluate input features against the anomaly detection model.
        Triggers an alert if an anomaly label is predicted.
        """
        pred = service.predict(features)
        if pred.label == Label.ANOMALY:
            try:
                # Dispatch critical or high severity alert depending on score threshold
                alerts.send_alert(Alert(
                    event_id=pred.event_id, score=pred.score,
                    severity=Severity.CRITICAL if pred.score >= 0.95 else Severity.HIGH,
                    message=f"Anomaly score {pred.score:.2f} >= {pred.threshold}"))
            except httpx.HTTPError:
                # Log error without failing prediction execution
                log.exception("alert delivery failed")  # prediction must still succeed
        return pred

    @app.post("/v1/verify/runtime", response_model=RuntimeVerifyResponse)
    def verify_runtime(request: RuntimeVerifyRequest) -> RuntimeVerifyResponse:
        """
        Perform runtime logic and formal verification on security events.
        Determines ALLOW/BLOCK decision and dispatches alerts on blocking events.
        """
        context_features = request.features
        
        # Extract features from unified vector if explicitly named features are absent
        if context_features is None and request.unified_context_vector is not None:
            context_features = {
                f"context_{index}": value
                for index, value in enumerate(request.unified_context_vector[1:], 1)
            }
            
        score = request.threat_score
        # Fall back to using index 0 of unified vector if explicit threat_score is absent
        if score is None and request.unified_context_vector:
            score = request.unified_context_vector[0]
            
        # Fall back to maximum feature value if score remains unresolved
        if score is None:
            score = max(context_features.values(), default=0.0) if context_features else 0.0

        # Execute hybrid evaluation and circuit breaker logic
        result = hybrid_service.evaluate_runtime(
            event_id=request.event_id,
            score=score,
            features=context_features,
            unified_context_vector=request.unified_context_vector,
            threat_score=request.threat_score,
            metadata=request.metadata,
            policy_rules=request.policy_rules,
        )
        
        # Map anomaly label to runtime decision
        decision = "BLOCK" if result.label is Label.ANOMALY else "ALLOW"
        if decision == "BLOCK":
            try:
                # Send alert for blocked requests
                alerts.send_alert(Alert(
                    event_id=request.event_id,
                    score=score,
                    severity=Severity.CRITICAL if score >= 0.95 else Severity.HIGH,
                    message=f"Runtime policy blocked event (score {score:.2f})",
                ))
            except httpx.HTTPError:
                log.exception("runtime alert delivery failed")
                
        verification = result.verification
        return RuntimeVerifyResponse(
            event_id=request.event_id,
            score=score,
            label=result.label,
            decision=decision,
            z3_status=verification.z3_status,
            circuit_state=hybrid_service.circuit_breaker.state,
            override_applied=result.override_applied,
            override_reason=result.override_reason,
            violated_rules=verification.violated_rules,
            execution_time_ms=verification.execution_time_ms,
        )

    @app.get("/v1/circuit-breaker/status")
    def circuit_breaker_status() -> dict[str, Any]:
        """
        Retrieve health metrics, evaluation counts, and state 
        information from the Hybrid Circuit Breaker instance.
        """
        breaker = hybrid_service.circuit_breaker
        return {
            "state": breaker.state,
            "failure_count": breaker.failure_count,
            "total_evaluations": breaker.total_evaluations,
            "z3_sat_count": breaker.z3_sat_count,
            "z3_unsat_count": breaker.z3_unsat_count,
            "timeout_count": breaker.timeout_count,
            "fast_path_count": breaker.fast_path_count,
            "overrides_applied": breaker.overrides_applied,
        }

    return app


# Instantiate main FastAPI application instance
app = create_app()