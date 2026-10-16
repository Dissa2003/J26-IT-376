"""Unit tests for the lightweight anomaly model.

Training-correctness tests below use small, explicitly synthetic
fixtures to verify the gradient-descent *mechanism* is implemented
correctly (loss decreases, weights move, training is reproducible).
These fixtures are not real API traffic and their results must never be
read as evidence of real-world anomaly-detection performance -- see
``tokenization_ai/model.py``'s module docstring.
"""

import numpy as np
import pytest

from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import FusedInput, ModelInput, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.tensorizer import Tensorizer
from tokenization_ai.tokenizer import HybridTokenizer
from tokenization_ai.vocabulary import Vocabulary


def _model_input(ids: list[int], mask: list[int] | None = None, event_id: str = "e1") -> ModelInput:
    if mask is None:
        mask = [1] * len(ids)
    return ModelInput(event_id=event_id, input_ids=ids, attention_mask=mask)


def _fused(rules: list[VerifiedRule] | None = None, **context_overrides: object) -> FusedInput:
    fields: dict[str, object] = {"event_id": "e1", "method": "POST", "path": "/orders"}
    fields.update(context_overrides)
    context = UnifiedApiContext(**fields)
    rule_info = VerifiedRuleInfo(event_id="e1", rules=rules or [])
    return FusedInput(context=context, rule_info=rule_info)


# -- Deterministic feature hashing -----------------------------------------


def test_unigram_bucket_is_deterministic_across_calls_and_instances() -> None:
    model_a = LightweightAnomalyModel(num_buckets=64)
    model_b = LightweightAnomalyModel(num_buckets=64)

    assert model_a._unigram_bucket(42) == model_a._unigram_bucket(42)
    assert model_a._unigram_bucket(42) == model_b._unigram_bucket(42)


def test_bigram_bucket_is_deterministic_and_order_dependent() -> None:
    model = LightweightAnomalyModel(num_buckets=64)

    assert model._bigram_bucket(3, 7) == model._bigram_bucket(3, 7)
    # (3, 7) and (7, 3) are different structural events and should not be
    # forced to collide by the hash formula's own symmetry.
    assert model._bigram_bucket(3, 7) != model._bigram_bucket(7, 3)


def test_feature_hashing_never_uses_pythons_randomized_hash(monkeypatch) -> None:
    # If feature extraction ever starts relying on hash(), poisoning the
    # builtin would change the result; it must not.
    model = LightweightAnomalyModel(num_buckets=64)
    before = model._features([5, 9, 5, 9], [1, 1, 1, 1])

    monkeypatch.setattr("builtins.hash", lambda x: 0)
    after = model._features([5, 9, 5, 9], [1, 1, 1, 1])

    assert np.array_equal(before, after)


# -- Unigram/bigram order sensitivity ---------------------------------------


def test_bigram_features_are_sensitive_to_token_order() -> None:
    model = LightweightAnomalyModel(num_buckets=256)

    forward = model._features([5, 9, 13], [1, 1, 1])
    reversed_order = model._features([13, 9, 5], [1, 1, 1])

    assert not np.array_equal(forward, reversed_order)


def test_unigram_counts_are_order_independent_while_bigrams_are_not() -> None:
    model = LightweightAnomalyModel(num_buckets=4096)  # large enough to avoid collisions in this check

    forward = model._features([5, 9, 13], [1, 1, 1])
    reversed_order = model._features([13, 9, 5], [1, 1, 1])

    assert forward.sum() == reversed_order.sum() == 5.0  # 3 unigrams + 2 bigrams, both orders


# -- Attention-mask and padding correctness ---------------------------------


def test_padding_tokens_do_not_contribute_to_features() -> None:
    model = LightweightAnomalyModel(num_buckets=256)

    with_padding = model._features([5, 9, 0, 0], [1, 1, 0, 0])
    without_padding = model._features([5, 9], [1, 1])

    assert np.array_equal(with_padding, without_padding)


def test_no_bigram_forms_across_a_padding_gap() -> None:
    model = LightweightAnomalyModel(num_buckets=4096)

    # Tokens 5 and 9 are separated by padding -- they must not form a bigram.
    gapped = model._features([5, 0, 9], [1, 0, 1])
    adjacent = model._features([5, 9], [1, 1])

    assert np.array_equal(gapped, adjacent)


def test_all_padding_input_produces_a_zero_feature_vector() -> None:
    model = LightweightAnomalyModel(num_buckets=64)

    features = model._features([7, 7, 7], [0, 0, 0])

    assert np.array_equal(features, np.zeros(64))


# -- Invalid input rejection -------------------------------------------------


def test_predict_rejects_mismatched_lengths() -> None:
    model = LightweightAnomalyModel(num_buckets=16)
    model.fit([_model_input([1, 2])], [0], epochs=1)

    with pytest.raises(ValueError):
        model.predict(_model_input([1, 2], mask=[1]))


def test_predict_rejects_empty_input_ids() -> None:
    model = LightweightAnomalyModel(num_buckets=16)
    model.fit([_model_input([1, 2])], [0], epochs=1)

    with pytest.raises(ValueError):
        model.predict(ModelInput(event_id="e1", input_ids=[], attention_mask=[]))


def test_predict_rejects_non_binary_attention_mask() -> None:
    model = LightweightAnomalyModel(num_buckets=16)
    model.fit([_model_input([1, 2])], [0], epochs=1)

    with pytest.raises(ValueError):
        model.predict(_model_input([1, 2], mask=[1, 2]))


def test_fit_rejects_empty_training_data() -> None:
    model = LightweightAnomalyModel(num_buckets=16)

    with pytest.raises(ValueError):
        model.fit([], [], epochs=1)


def test_fit_rejects_mismatched_inputs_and_labels_length() -> None:
    model = LightweightAnomalyModel(num_buckets=16)

    with pytest.raises(ValueError):
        model.fit([_model_input([1])], [0, 1], epochs=1)


def test_fit_rejects_non_binary_labels() -> None:
    model = LightweightAnomalyModel(num_buckets=16)

    with pytest.raises(ValueError):
        model.fit([_model_input([1]), _model_input([2])], [0, 2], epochs=1)


def test_fit_rejects_non_positive_epochs() -> None:
    model = LightweightAnomalyModel(num_buckets=16)

    with pytest.raises(ValueError):
        model.fit([_model_input([1])], [0], epochs=0)


def test_fit_rejects_non_positive_learning_rate() -> None:
    model = LightweightAnomalyModel(num_buckets=16)

    with pytest.raises(ValueError):
        model.fit([_model_input([1])], [0], epochs=5, learning_rate=0.0)


def test_fit_rejects_a_malformed_training_example() -> None:
    model = LightweightAnomalyModel(num_buckets=16)

    with pytest.raises(ValueError):
        model.fit([_model_input([1, 2], mask=[1])], [0], epochs=1)


def test_constructor_rejects_zero_buckets() -> None:
    with pytest.raises(ValueError):
        LightweightAnomalyModel(num_buckets=0)


# -- Untrained prediction rejection ------------------------------------------


def test_predict_raises_runtime_error_before_fit() -> None:
    model = LightweightAnomalyModel(num_buckets=16)

    with pytest.raises(RuntimeError):
        model.predict(_model_input([1, 2, 3]))


def test_untrained_model_has_zero_weights_and_untrained_version_string() -> None:
    model = LightweightAnomalyModel(num_buckets=16)

    assert model.is_trained is False
    assert model.model_version == "ngram-linear-v1-untrained"
    assert np.array_equal(model.weights, np.zeros(16))
    assert model.bias == 0.0


# -- Numerical stability ------------------------------------------------------


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    result = LightweightAnomalyModel._sigmoid(np.array([-1e9, 1e9, 0.0]))

    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(0.0, abs=1e-12)
    assert result[1] == pytest.approx(1.0, abs=1e-12)
    assert result[2] == pytest.approx(0.5)


def test_bce_with_logits_is_stable_for_extreme_logits() -> None:
    logits = np.array([-1e9, 1e9])
    targets = np.array([0.0, 1.0])  # the "easy", correctly-classified case

    loss = LightweightAnomalyModel._bce_with_logits(logits, targets)

    assert np.isfinite(loss)
    assert loss == pytest.approx(0.0, abs=1e-6)


# -- Gradient/training correctness and decreasing loss -----------------------


def _separable_fixture() -> tuple[list[ModelInput], list[int]]:
    """A small, explicitly synthetic, linearly-separable toy fixture.

    Verifies the training *code* is correct; says nothing about real
    attack detection.
    """
    benign = [_model_input([10, 11, 12], event_id=f"benign-{i}") for i in range(4)]
    anomalous = [_model_input([90, 91, 92], event_id=f"anomalous-{i}") for i in range(4)]
    inputs = benign + anomalous
    labels = [0] * len(benign) + [1] * len(anomalous)
    return inputs, labels


def test_training_loss_decreases_on_a_simple_synthetic_fixture() -> None:
    inputs, labels = _separable_fixture()
    model = LightweightAnomalyModel(num_buckets=64)

    history = model.fit(inputs, labels, epochs=200, learning_rate=0.5)

    assert history[-1] < history[0]
    assert history[-1] < 0.2  # converges to a low loss on this trivially separable toy set


def test_fit_moves_weights_away_from_zero() -> None:
    inputs, labels = _separable_fixture()
    model = LightweightAnomalyModel(num_buckets=64)

    model.fit(inputs, labels, epochs=50, learning_rate=0.5)

    assert not np.array_equal(model.weights, np.zeros(64))


def test_fit_sets_is_trained_and_updates_model_version() -> None:
    inputs, labels = _separable_fixture()
    model = LightweightAnomalyModel(num_buckets=64)

    model.fit(inputs, labels, epochs=10, learning_rate=0.1)

    assert model.is_trained is True
    assert model.model_version == "ngram-linear-v1-trained"


# -- Reproducible training ----------------------------------------------------


def test_training_is_reproducible_across_separate_model_instances() -> None:
    inputs, labels = _separable_fixture()

    model_a = LightweightAnomalyModel(num_buckets=64)
    history_a = model_a.fit(inputs, labels, epochs=30, learning_rate=0.3)

    model_b = LightweightAnomalyModel(num_buckets=64)
    history_b = model_b.fit(inputs, labels, epochs=30, learning_rate=0.3)

    assert history_a == history_b
    assert np.array_equal(model_a.weights, model_b.weights)
    assert model_a.bias == model_b.bias


# -- Bounded prediction scores after training ---------------------------------


def test_predictions_are_bounded_after_training() -> None:
    inputs, labels = _separable_fixture()
    model = LightweightAnomalyModel(num_buckets=64)
    model.fit(inputs, labels, epochs=100, learning_rate=0.5)

    for model_input in inputs + [_model_input([1, 2, 3, 4, 5], event_id="unseen")]:
        score = model.predict(model_input)
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)


def test_trained_model_separates_the_synthetic_fixture_it_was_trained_on() -> None:
    # This only demonstrates the training mechanism learned the trivial toy
    # rule it was given -- NOT real-world attack-detection capability.
    inputs, labels = _separable_fixture()
    model = LightweightAnomalyModel(num_buckets=64)
    model.fit(inputs, labels, epochs=300, learning_rate=0.5)

    benign_scores = [model.predict(mi) for mi, label in zip(inputs, labels) if label == 0]
    anomalous_scores = [model.predict(mi) for mi, label in zip(inputs, labels) if label == 1]

    assert max(benign_scores) < min(anomalous_scores)


# -- Input non-mutation --------------------------------------------------------


def test_predict_does_not_mutate_the_model_input() -> None:
    inputs, labels = _separable_fixture()
    model = LightweightAnomalyModel(num_buckets=64)
    model.fit(inputs, labels, epochs=10, learning_rate=0.1)
    probe = _model_input([10, 11, 12])
    before_ids, before_mask = list(probe.input_ids), list(probe.attention_mask)

    model.predict(probe)

    assert probe.input_ids == before_ids
    assert probe.attention_mask == before_mask


def test_fit_does_not_mutate_its_inputs_list() -> None:
    inputs, labels = _separable_fixture()
    snapshot = [(mi.input_ids[:], mi.attention_mask[:]) for mi in inputs]
    model = LightweightAnomalyModel(num_buckets=64)

    model.fit(inputs, labels, epochs=10, learning_rate=0.1)

    for model_input, (ids_before, mask_before) in zip(inputs, snapshot):
        assert model_input.input_ids == ids_before
        assert model_input.attention_mask == mask_before


# -- Model state handling ------------------------------------------------------


def test_predict_after_fit_no_longer_raises() -> None:
    inputs, labels = _separable_fixture()
    model = LightweightAnomalyModel(num_buckets=64)
    model.fit(inputs, labels, epochs=10, learning_rate=0.1)

    score = model.predict(_model_input([10, 11, 12]))

    assert 0.0 <= score <= 1.0


def test_refitting_overwrites_previous_training_state() -> None:
    inputs, labels = _separable_fixture()
    model = LightweightAnomalyModel(num_buckets=64)
    model.fit(inputs, labels, epochs=5, learning_rate=0.1)
    first_weights = model.weights.copy()

    model.fit(inputs, labels, epochs=50, learning_rate=0.5)

    assert model.is_trained is True
    assert not np.array_equal(model.weights, first_weights)


# -- Member-4-only pipeline integration test -----------------------------------


def test_pipeline_integration_tokenizer_vocabulary_tensorizer_model() -> None:
    """HybridTokenizer -> Vocabulary -> Tensorizer -> LightweightAnomalyModel.

    Verifies the four Member 4 stages are wire-compatible end to end,
    using a small, explicitly synthetic training fixture. This is a
    plumbing/compatibility test, not a real-world anomaly-detection
    quality claim.
    """
    benign_fused = _fused(payload={"amount": 10, "currency": "USD"})
    anomalous_fused = _fused(
        payload={"amount": "'; DROP TABLE orders; --", "currency": "XXX"},
        rules=[VerifiedRule(rule_id="R-SUSPICIOUS", attributes={"severity": "high"})],
    )

    tokenizer = HybridTokenizer()
    benign_seq = tokenizer.tokenize(benign_fused)
    anomalous_seq = tokenizer.tokenize(anomalous_fused)

    vocabulary = Vocabulary()
    vocabulary.fit([benign_seq, anomalous_seq])

    max_length = max(len(benign_seq.tokens), len(anomalous_seq.tokens)) + 5
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    training_inputs = [tensorizer.tensorize(benign_seq) for _ in range(4)] + [
        tensorizer.tensorize(anomalous_seq) for _ in range(4)
    ]
    training_labels = [0] * 4 + [1] * 4

    model = LightweightAnomalyModel(num_buckets=128)
    model.fit(training_inputs, training_labels, epochs=200, learning_rate=0.5)

    assert model.is_trained

    benign_score = model.predict(tensorizer.tensorize(benign_seq))
    anomalous_score = model.predict(tensorizer.tensorize(anomalous_seq))

    assert 0.0 <= benign_score <= 1.0
    assert 0.0 <= anomalous_score <= 1.0
