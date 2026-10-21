"""Regression tests: no random weights may pass as a trained or calibrated model (D4-7G3).

Background: a dev-branch change added ``LightweightAnomalyModel(initialize_calibrated=True)``, which filled the
weights from ``RandomState(42)`` and set ``is_trained = True``, and ``TokenizationAIService.create_default()``
served that model. Random weights are neither trained nor calibrated, and the flag bypassed the model's own
refusal to predict before ``fit``. These tests pin the restored invariant and the genuinely fitted default.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tokenization_ai.demo import _DEMO_EVENTS, _build_demo_service
from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import ModelInput, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.service import TokenizationAIService


def test_the_model_has_no_random_initialisation_that_claims_training_or_calibration():
    assert not hasattr(LightweightAnomalyModel, "initialize_calibrated_weights")
    assert not hasattr(LightweightAnomalyModel, "create_calibrated")
    assert "initialize_calibrated" not in inspect.signature(LightweightAnomalyModel).parameters
    with pytest.raises(TypeError):
        LightweightAnomalyModel(256, initialize_calibrated=True)  # type: ignore[call-arg]


def test_a_new_model_is_untrained_zero_valued_and_refuses_to_predict():
    model = LightweightAnomalyModel(num_buckets=64)
    assert model.is_trained is False and model.training_loss_history == []
    assert not np.any(model.weights) and model.bias == 0.0
    with pytest.raises(RuntimeError):
        model.predict(ModelInput(event_id="e", input_ids=[1, 2, 3], attention_mask=[1, 1, 1]))


def test_only_fit_marks_a_model_trained():
    model = LightweightAnomalyModel(num_buckets=64)
    inputs = [ModelInput(event_id="a", input_ids=[1, 2, 3], attention_mask=[1, 1, 1]),
              ModelInput(event_id="b", input_ids=[4, 5, 6], attention_mask=[1, 1, 1])]
    model.fit(inputs, [0, 1], epochs=5, learning_rate=0.5)
    assert model.is_trained is True and len(model.training_loss_history) == 5


def test_create_default_serves_a_genuinely_fitted_demo_model_not_random_weights():
    service = TokenizationAIService.create_default()
    model = service.model
    assert model.is_trained is True and len(model.training_loss_history) == 50  # gradient descent actually ran
    assert model.training_loss_history[-1] < model.training_loss_history[0]
    random_draw = np.random.RandomState(42).normal(0.0, 0.05, model.num_buckets)
    assert not np.allclose(model.weights, random_draw) and model.bias != -0.5
    assert model.model_version == "ngram-linear-v1-trained"


def test_create_default_is_the_same_pipeline_as_the_standalone_demo():
    default, demo = TokenizationAIService.create_default(), _build_demo_service()
    for context, rule_info in _DEMO_EVENTS:
        assert default.predict(context, rule_info).score == demo.predict(context, rule_info).score
    assert np.array_equal(default.model.weights, demo.model.weights)


def test_create_default_honours_its_size_arguments_and_still_trains():
    service = TokenizationAIService.create_default(max_length=96, num_buckets=128)
    assert service.tensorizer.max_length == 96 and service.model.num_buckets == 128 and service.model.is_trained


def test_create_default_scores_an_unseen_request_end_to_end_without_claiming_detection():
    service = TokenizationAIService.create_default()
    context = UnifiedApiContext(event_id="evt-smoke", method="POST", path="/api/v1/system/bulk-manage", client_ip="10.0.0.1",
                                headers={"authorization": "Bearer synthetic"}, payload={"action": "manage", "card": "4111-2222-3333-4444"},
                                threat_score=0.95)
    rules = VerifiedRuleInfo(event_id="evt-smoke", rules=[VerifiedRule(rule_id="rule-1", attributes={"effect": "allow"})])
    analysis = service.analyze(context, rules)
    assert 0.0 <= analysis.prediction.score <= 1.0 and len(analysis.model_input.input_ids) > 0
    assert "4111-2222-3333-4444" not in " ".join(analysis.sequence.tokens)  # masking stays active on this path
    assert analysis.masked_paths  # the card number (and the authorization header) were masked by provenance
    doc = " ".join((inspect.getdoc(TokenizationAIService.create_default) or "").split())
    assert "SYNTHETIC" in doc and "not calibrated" in doc and "not a detection model" in doc
