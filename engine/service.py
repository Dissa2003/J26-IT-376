from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from shared_contracts.interfaces import InferenceModel, PredictionService
from shared_contracts.models import FeatureVector, Label, Prediction

from .circuit_breaker import HybridCircuitBreaker, HybridDecisionResult
from .verifier import RuntimeLTLVerifier


class StubModel(InferenceModel):
    """Placeholder: score = max feature value clipped to [0,1]. Swap for a real model."""

    def score(self, features: FeatureVector) -> float:
        return min(1.0, max(0.0, max(features.features.values(), default=0.0)))


class ThresholdPredictionService(PredictionService):
    def __init__(self, model: InferenceModel, threshold: float, model_version: str):
        self._model, self._threshold, self._version = model, threshold, model_version

    def predict(self, features: FeatureVector) -> Prediction:
        s = self._model.score(features)
        return Prediction(
            event_id=features.event_id, score=s,
            label=Label.ANOMALY if s >= self._threshold else Label.NORMAL,
            threshold=self._threshold, model_version=self._version,
        )


class HybridVerificationService(PredictionService):
    """Combine model scoring with runtime formal verification."""

    def __init__(
        self,
        model: InferenceModel,
        threshold: float,
        model_version: str,
        verifier: RuntimeLTLVerifier | None = None,
        circuit_breaker: HybridCircuitBreaker | None = None,
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._version = model_version
        self.circuit_breaker = circuit_breaker or HybridCircuitBreaker(verifier)

    def predict(self, features: FeatureVector) -> Prediction:
        score = self._model.score(features)
        decision = self.evaluate_runtime(
            event_id=features.event_id,
            score=score,
            features=features.features,
        )
        return Prediction(
            event_id=features.event_id,
            score=score,
            label=decision.label,
            threshold=self._threshold,
            model_version=self._version,
        )

    def evaluate_runtime(
        self,
        event_id: str,
        score: float,
        features: Mapping[str, float] | None = None,
        unified_context_vector: list[float] | None = None,
        threat_score: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        policy_rules: Iterable[Any] | None = None,
    ) -> HybridDecisionResult:
        context: dict[str, Any] = dict(features or {})
        vector = unified_context_vector
        if vector is not None:
            if not vector:
                raise ValueError("unified_context_vector must contain threat_score")
            context["threat_score"] = (
                threat_score if threat_score is not None else vector[0]
            )
            context.update(
                {f"context_{index}": value for index, value in enumerate(vector[1:], 1)}
            )
            context["unified_context_vector"] = vector
        elif threat_score is not None:
            context["threat_score"] = threat_score
        if metadata:
            context.update(metadata)

        constraints = self._policy_constraints(context, policy_rules)
        return self.circuit_breaker.evaluate_hybrid_decision(
            event_id,
            score,
            self._threshold,
            context,
            constraints,
        )

    @staticmethod
    def _policy_constraints(
        context: Mapping[str, Any], policy_rules: Iterable[Any] | None
    ) -> list[str]:
        constraints: list[str] = []
        if policy_rules is None:
            return constraints
        role = str(context.get("role", "")).lower()
        action = str(context.get("action", "")).lower()
        resource = str(context.get("resource", "")).lower()
        for rule in policy_rules:
            value = rule.model_dump() if hasattr(rule, "model_dump") else rule
            if not isinstance(value, Mapping):
                raise TypeError("policy_rules must contain mappings or policy models")
            subject = str(value.get("subject", "")).lower()
            rule_action = str(value.get("action", "")).lower()
            rule_resource = str(value.get("resource", "")).lower()
            matches = (
                (not subject or subject in {"everyone", role})
                and (not rule_action or rule_action == action)
                and (not rule_resource or rule_resource == resource)
            )
            effect = str(value.get("effect", "")).lower()
            if matches and effect.endswith("deny"):
                constraints.append("False")
            elif rule_action and subject and subject not in {"everyone", "*"}:
                if effect.endswith("deny"):
                    constraints.append(
                        f"z3.Not(z3.And(role == z3.StringVal('{subject}'), action == z3.StringVal('{rule_action}')))"
                    )
        return constraints