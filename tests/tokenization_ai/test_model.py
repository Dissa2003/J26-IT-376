"""Unit tests for the lightweight anomaly model."""

import pytest

from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import ModelInput


def test_predict_is_not_implemented_yet() -> None:
    """Placeholder until the model is implemented."""
    with pytest.raises(NotImplementedError):
        LightweightAnomalyModel().predict(ModelInput(event_id="e1"))
