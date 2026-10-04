from __future__ import annotations

from shared_contracts.interfaces import InferenceModel, PredictionService
from shared_contracts.models import FeatureVector, Label, Prediction


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