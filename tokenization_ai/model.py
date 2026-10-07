"""Lightweight AI model that scores prepared inputs for anomalies."""

from __future__ import annotations

from .models import ModelInput


class LightweightAnomalyModel:
    """Compact sequence model producing an anomaly score per input."""

    def predict(self, model_input: ModelInput) -> float:
        """Return an anomaly score in [0, 1]."""
        raise NotImplementedError
