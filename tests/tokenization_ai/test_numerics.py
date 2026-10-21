"""Tests for lambda-power-1 and logit-readout-1 (draft protocol heldout-eval-1.2.0-draft, section 12.11.2).

Hand-made inputs and explicitly assigned model weights only. No model is fitted or scored on any generated dataset, no label is read, and the
confirmatory seeds 7101-7103 are never used. Expected values are derived independently (closed forms, numpy.linalg.eigvalsh, a test-local
re-implementation of the documented hashing).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

import tokenization_ai.heldout_eval.numerics as nm
from tokenization_ai.heldout_eval.arms import ARM_A, ARM_B, ARM_C, ArmInputs
from tokenization_ai.heldout_eval.numerics import (
    EigenvalueConvergenceError,
    EigenvalueError,
    NumericsError,
    estimate_from_arm_inputs,
    estimate_lambda_max,
    learning_rate_table,
    learning_rates,
    raw_logit,
    raw_logits,
    relative_learning_rate,
    verify_logit_compatibility,
)
from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import ModelInput

SOURCE = Path(nm.__file__)
PRIME_U, PRIME_A, PRIME_B = 2654435761, 2654435761, 40503  # documented hashing constants, re-stated here on purpose


def mi(event_id, ids, mask=None):
    return ModelInput(event_id=event_id, input_ids=list(ids), attention_mask=list(mask if mask is not None else [1] * len(ids)))


def ref_features(ids, mask, nb):
    """Independent re-implementation of the documented feature map (unigram + adjacent-bigram hashing, padding excluded)."""
    valid = [i for i, m in zip(ids, mask) if m == 1]
    f = np.zeros(nb)
    for t in valid:
        f[(t * PRIME_U) % nb] += 1
    for a, b in zip(valid, valid[1:]):
        f[(a * PRIME_A + b * PRIME_B) % nb] += 1
    return f


def ref_lambda(inputs, nb):
    X = np.vstack([ref_features(i.input_ids, i.attention_mask, nb) for i in inputs])
    Xa = np.hstack([X, np.ones((len(inputs), 1))])
    return float(np.linalg.eigvalsh(Xa.T @ Xa / len(inputs))[-1])


def random_inputs(n, nb_vocab=40, seed=0):
    rng = random.Random(seed)
    out = []
    for k in range(n):
        length = rng.randint(2, 12)
        out.append(mi(f"r{k}", [rng.randrange(1, nb_vocab) for _ in range(length)]))
    return out


# -- identifiers and constants ----------------------------------------------------------------------------------------


def test_identifiers_and_approved_settings():
    assert nm.LAMBDA_METHOD_ID == "lambda-power-1" and nm.LOGIT_METHOD_ID == "logit-readout-1"
    assert nm.POWER_RELATIVE_TOLERANCE == 1e-10 and nm.POWER_MAX_ITERATIONS == 5000
    assert nm.RELATIVE_STEP_GRID == (0.1, 0.3, 0.8) and nm.NUM_BUCKETS_GRID == (1024, 2048, 4096)


# -- eigenvalue estimator ---------------------------------------------------------------------------------------------


def test_single_bucket_closed_form_with_bias_column():
    # One bucket: every feature lands in it. ids [3, 5] give 2 unigrams + 1 bigram = mass 3, so X = [[3]] and
    # [X|1]^T [X|1] / 1 = [[9, 3], [3, 1]] with largest eigenvalue 9 + 1 = 10.
    est = estimate_lambda_max([mi("a", [3, 5])], split="train", arm=ARM_A, num_buckets=1)
    assert est.lambda_max == pytest.approx(10.0, rel=1e-9)
    assert est.L == est.lambda_max / 4
    assert est.dimension == 2 and est.training_records == 1


def test_two_records_closed_form():
    # Masses 3 and 1 => X = [[3], [1]]; (1/2) [[10, 4], [4, 2]] = [[5, 2], [2, 1]], eigenvalue 3 + 2*sqrt(2).
    est = estimate_lambda_max([mi("a", [7, 9]), mi("b", [1])], split="train", arm=ARM_B, num_buckets=1)
    assert est.lambda_max == pytest.approx(3 + 2 * math.sqrt(2), rel=1e-9)


def test_bias_column_is_included():
    # Without the ones column the same fixture would give 9.0 (a 1 x 1 matrix [[9]]), not 10.
    est = estimate_lambda_max([mi("a", [3, 5])], split="train", arm=ARM_A, num_buckets=1)
    assert est.lambda_max > 9.5


@pytest.mark.parametrize("nb", [4, 16, 64])
def test_agrees_with_eigvalsh_on_small_fixtures(nb):
    inputs = random_inputs(25, seed=nb)
    est = estimate_lambda_max(inputs, split="train", arm=ARM_A, num_buckets=nb)
    assert est.lambda_max == pytest.approx(ref_lambda(inputs, nb), rel=1e-8)
    assert est.converged and est.final_relative_change <= nm.POWER_RELATIVE_TOLERANCE


def test_padding_is_excluded_like_the_model():
    padded = [mi("a", [3, 5, 0, 0], [1, 1, 0, 0]), mi("b", [7, 9, 0], [1, 1, 0])]
    plain = [mi("a", [3, 5]), mi("b", [7, 9])]
    assert estimate_lambda_max(padded, split="train", arm=ARM_A, num_buckets=8).lambda_max == pytest.approx(
        estimate_lambda_max(plain, split="train", arm=ARM_A, num_buckets=8).lambda_max, rel=1e-12)


def test_feature_construction_matches_the_models_fit_path():
    model = LightweightAnomalyModel(num_buckets=32)
    for item in random_inputs(10, seed=3):
        assert np.array_equal(model._features(item.input_ids, item.attention_mask), ref_features(item.input_ids, item.attention_mask, 32))


def test_is_deterministic_and_order_independent():
    inputs = random_inputs(30, seed=5)
    first = estimate_lambda_max(inputs, split="train", arm=ARM_C, num_buckets=16)
    assert estimate_lambda_max(inputs, split="train", arm=ARM_C, num_buckets=16) == first
    shuffled = inputs[:]
    random.Random(9).shuffle(shuffled)
    other = estimate_lambda_max(shuffled, split="train", arm=ARM_C, num_buckets=16)
    assert other.lambda_max == pytest.approx(first.lambda_max, rel=1e-8)
    assert other.record_ids_sha256 == first.record_ids_sha256  # order-free fingerprint of the id set


def test_does_not_mutate_inputs():
    inputs = random_inputs(5, seed=1)
    before = [i.model_dump() for i in inputs]
    estimate_lambda_max(inputs, split="train", arm=ARM_A, num_buckets=8)
    assert [i.model_dump() for i in inputs] == before


def test_non_convergence_fails_without_fallback(monkeypatch):
    monkeypatch.setattr(nm, "POWER_MAX_ITERATIONS", 1)
    with pytest.raises(EigenvalueConvergenceError):
        estimate_lambda_max(random_inputs(10, seed=2), split="train", arm=ARM_A, num_buckets=16)


def test_invalid_estimates_are_refused(monkeypatch):
    for bad in (math.nan, math.inf, 0.0, -1.0):
        monkeypatch.setattr(nm, "_power_iteration", lambda X, bad=bad: (bad, 3, True, 0.0, 0.0))
        with pytest.raises(EigenvalueError):
            estimate_lambda_max([mi("a", [3, 5])], split="train", arm=ARM_A, num_buckets=4)


def test_non_finite_features_are_refused(monkeypatch):
    monkeypatch.setattr(LightweightAnomalyModel, "_features", lambda self, ids, mask: np.full(self.num_buckets, np.nan))
    with pytest.raises(NumericsError):
        estimate_lambda_max([mi("a", [3, 5])], split="train", arm=ARM_A, num_buckets=4)


@pytest.mark.parametrize("nb", [0, -1, 2.5, True, "8", None])
def test_invalid_bucket_counts(nb):
    with pytest.raises(NumericsError):
        estimate_lambda_max([mi("a", [3, 5])], split="train", arm=ARM_A, num_buckets=nb)


def test_invalid_inputs_are_refused():
    ok = mi("a", [3, 5])
    for bad in ([], (), "abc", {"a": ok}):
        with pytest.raises(NumericsError):
            estimate_lambda_max(bad, split="train", arm=ARM_A, num_buckets=4)
    with pytest.raises(NumericsError):
        estimate_lambda_max([ok, {"event_id": "x"}], split="train", arm=ARM_A, num_buckets=4)
    with pytest.raises(NumericsError):
        estimate_lambda_max([ok, mi("b", [1, 2], [1])], split="train", arm=ARM_A, num_buckets=4)  # mask length mismatch
    with pytest.raises(NumericsError):
        estimate_lambda_max([ok, mi("b", [], [])], split="train", arm=ARM_A, num_buckets=4)
    with pytest.raises(NumericsError):
        estimate_lambda_max([ok, mi("b", [1, 2], [1, 2])], split="train", arm=ARM_A, num_buckets=4)  # mask not 0/1
    with pytest.raises(NumericsError):
        estimate_lambda_max([ok, mi("a", [1, 2])], split="train", arm=ARM_A, num_buckets=4)  # duplicate record id


def test_arm_must_be_known():
    with pytest.raises(NumericsError):
        estimate_lambda_max([mi("a", [3, 5])], split="train", arm="D_other", num_buckets=4)


@pytest.mark.parametrize("split", ["validation", "test", "TRAIN", "", " train", None, 1])
def test_only_the_training_split_is_accepted(split):
    with pytest.raises(NumericsError):
        estimate_lambda_max([mi("a", [3, 5])], split=split, arm=ARM_A, num_buckets=4)


def arm_inputs(split):
    ids = ("r1", "r2")
    padded = lambda t: ModelInput(event_id=t[0], input_ids=t[1] + [0] * (6 - len(t[1])), attention_mask=[1] * len(t[1]) + [0] * (6 - len(t[1])))
    rows = [("r1", [3, 5, 7]), ("r2", [9, 11])]
    return ArmInputs(split=split, record_ids=ids, arm_a=tuple(padded(r) for r in rows), arm_b=tuple(padded(r) for r in rows),
                     arm_c=tuple(mi(r[0], r[1]) for r in rows), max_length=6)


def test_arm_inputs_wrapper_enforces_the_declared_split():
    for arm in (ARM_A, ARM_B, ARM_C):
        est = estimate_from_arm_inputs(arm_inputs("train"), arm=arm, num_buckets=16)
        assert est.arm == arm and est.split == "train" and est.training_records == 2
    for split in ("validation", "test"):
        with pytest.raises(NumericsError):
            estimate_from_arm_inputs(arm_inputs(split), arm=ARM_A, num_buckets=16)
    with pytest.raises(NumericsError):
        estimate_from_arm_inputs([mi("a", [1])], arm=ARM_A, num_buckets=16)


def test_estimate_serializes_to_strict_json_with_provenance():
    est = estimate_lambda_max(random_inputs(12, seed=4), split="train", arm=ARM_B, num_buckets=16)
    payload = json.loads(json.dumps(est.to_dict(), allow_nan=False))
    assert payload["method"] == "lambda-power-1" and payload["arm"] == ARM_B and payload["num_buckets"] == 16
    assert payload["split"] == "train" and payload["converged"] is True and payload["random_numbers"] is False
    assert payload["relative_tolerance"] == 1e-10 and payload["max_iterations"] == 5000 and payload["precision"] == "float64"
    assert payload["L"] == payload["lambda_max"] / 4 and payload["fallback_learning_rate"] is None
    assert payload["iterations"] == est.iterations and len(payload["record_ids_sha256"]) == 64
    assert payload["trace"] == pytest.approx(payload["feature_sq_norm_mean"] + 1.0)
    assert payload["lambda_max"] <= payload["trace"] * (1 + 1e-9)  # largest eigenvalue never exceeds the trace


# -- relative learning rates --------------------------------------------------------------------------------------------


def test_relative_learning_rate_arithmetic():
    assert relative_learning_rate(0.1, 2.5) == 0.1 / 2.5
    assert relative_learning_rate(0.8, 4.0) == 0.2


@pytest.mark.parametrize("c", [0, -0.1, 2, 2.5, math.nan, math.inf, True, "0.1", None])
def test_invalid_relative_steps(c):
    with pytest.raises(NumericsError):
        relative_learning_rate(c, 2.0)


@pytest.mark.parametrize("L", [0, -1.0, math.nan, math.inf, True, "2", None])
def test_invalid_L(L):
    with pytest.raises(NumericsError):
        relative_learning_rate(0.1, L)


def test_learning_rates_from_a_known_estimate():
    est = estimate_lambda_max([mi("a", [3, 5])], split="train", arm=ARM_A, num_buckets=1)  # lambda ~ 10, L ~ 2.5
    table = learning_rates(est)
    assert [r["relative_step_c"] for r in table["rates"]] == [0.1, 0.3, 0.8]
    for row in table["rates"]:
        assert row["learning_rate"] == row["relative_step_c"] / est.L
        assert float.fromhex(row["learning_rate_hex"]) == row["learning_rate"]  # exact reproduction
        assert row["rate_times_L"] == pytest.approx(row["relative_step_c"])
    assert table["arm"] == ARM_A and table["num_buckets"] == 1
    assert learning_rates(est) == table  # deterministic
    json.dumps(table, allow_nan=False)


def test_learning_rates_refuse_forged_or_unconverged_estimates():
    est = estimate_lambda_max([mi("a", [3, 5])], split="train", arm=ARM_A, num_buckets=1)
    for bad in (dataclasses.replace(est, converged=False), dataclasses.replace(est, split="validation"),
                dataclasses.replace(est, method="other"), dataclasses.replace(est, L=est.L * 2),
                dataclasses.replace(est, lambda_max=math.nan, L=math.nan), dataclasses.replace(est, lambda_max=0.0, L=0.0)):
        with pytest.raises(NumericsError):
            learning_rates(bad)
    with pytest.raises(NumericsError):
        learning_rates(est, steps=(0.8, 0.1))
    with pytest.raises(NumericsError):
        learning_rates(est, steps=())


def test_learning_rate_table_requires_exactly_the_approved_cells():
    inputs = random_inputs(8, seed=6)
    cells = [estimate_lambda_max(inputs, split="train", arm=a, num_buckets=b) for a in (ARM_A, ARM_B, ARM_C) for b in (1024, 2048, 4096)]
    table = learning_rate_table(cells)
    assert len(table["cells"]) == 9 and table["relative_step_grid"] == [0.1, 0.3, 0.8]
    assert {(c["arm"], c["num_buckets"]) for c in table["cells"]} == {(a, b) for a in (ARM_A, ARM_B, ARM_C) for b in (1024, 2048, 4096)}
    json.dumps(table, allow_nan=False)
    with pytest.raises(NumericsError):
        learning_rate_table(cells[:-1])
    with pytest.raises(NumericsError):
        learning_rate_table(cells + [cells[0]])
    with pytest.raises(NumericsError):
        learning_rate_table([dataclasses.replace(cells[0], num_buckets=999)] + cells[1:])


# -- raw logit ----------------------------------------------------------------------------------------------------------------


def make_model(weights, bias, nb=8):
    model = LightweightAnomalyModel(num_buckets=nb)
    model.weights = np.asarray(weights, dtype=np.float64).copy()
    model.bias = float(bias)
    model.is_trained = True  # explicitly assigned fixture state; no fit
    return model


def test_logit_equals_weights_dot_features_plus_bias_by_hand():
    nb = 8
    weights = np.array([0.5, -1.0, 0.25, 2.0, -0.75, 1.5, 0.0, -0.5])
    model = make_model(weights, 0.3, nb)
    item = mi("a", [1, 2])
    feats = ref_features(item.input_ids, item.attention_mask, nb)
    assert feats.sum() == 3  # two unigrams and one bigram
    expected = float(feats @ weights + 0.3)
    assert raw_logit(model, item) == expected
    assert np.array_equal(model._features(item.input_ids, item.attention_mask), feats)  # exact feature-mapping compatibility


def test_bias_only_and_masked_inputs():
    model = make_model(np.zeros(8), 1.25)
    assert raw_logit(model, mi("a", [4, 6], [0, 0])) == 1.25  # nothing valid: just the bias
    assert raw_logits(model, [mi("a", [1, 2]), mi("b", [3])]) == [1.25, 1.25]


def test_raw_logits_keep_input_order():
    model = make_model(np.arange(8) * 0.1, -0.2)
    items = random_inputs(10, nb_vocab=30, seed=8)
    assert raw_logits(model, items) == [raw_logit(model, i) for i in items]


def test_sigmoid_of_logit_equals_predict_on_non_saturated_examples():
    rng = np.random.default_rng(0)
    model = make_model(rng.normal(0, 0.3, 16), 0.1, 16)
    items = random_inputs(40, seed=11)
    for item in items:
        logit = raw_logit(model, item)
        assert abs(logit) < nm.SIGMOID_COMPARISON_MAX_ABS_LOGIT
        assert model.predict(item) == float(model._sigmoid(logit))  # predict is exactly this sigmoid of this number
        assert model.predict(item) == pytest.approx(nm._stable_sigmoid(logit), abs=1e-12)
    report = verify_logit_compatibility(model, items)
    assert report["checked"] == 40 and report["saturated_not_compared"] == 0 and report["max_abs_difference"] <= 1e-12
    assert report["method"] == "logit-readout-1"


def test_saturated_logits_keep_ranking_information():
    big = make_model(np.full(8, 400.0), 0.0)
    a, b = mi("a", [1, 2]), mi("b", [1, 2, 3, 4])  # mass 3 and 7 => logits 1200 and 2800
    la, lb = raw_logits(big, [a, b])
    assert math.isfinite(la) and math.isfinite(lb) and la < lb
    assert big.predict(a) == 1.0 and big.predict(b) == 1.0  # the sigmoid ties them; the logit does not
    small = make_model(np.full(8, 400.0), -5000.0)
    assert small.predict(a) == 0.0 and small.predict(b) == 0.0
    sa, sb = raw_logits(small, [a, b])
    assert sa < sb < 0
    assert sa == pytest.approx(-5000 + 1200) and sb == pytest.approx(-5000 + 2800)


def test_verify_counts_saturated_and_needs_one_checkable_example():
    model = make_model(np.zeros(8), 0.5)
    ok = mi("a", [1, 2])
    model.weights[:] = 0.0
    report = verify_logit_compatibility(model, [ok])
    assert report["checked"] == 1
    saturated = make_model(np.full(8, 400.0), 0.0)
    with pytest.raises(NumericsError):
        verify_logit_compatibility(saturated, [mi("a", [1, 2])])  # nothing checkable
    mixed = make_model(np.zeros(8), 0.5)
    mixed.weights[(1 * PRIME_U) % 8] = 1000.0
    rep = verify_logit_compatibility(mixed, [mi("sat", [1, 2]), mi("calm", [5, 5, 5, 5, 0], [0, 0, 0, 0, 0])])
    assert rep["saturated_not_compared"] == 1 and rep["checked"] == 1


def test_verify_detects_an_incompatible_predict(monkeypatch):
    model = make_model(np.zeros(8), 0.5)
    monkeypatch.setattr(LightweightAnomalyModel, "predict", lambda self, x: 0.123)
    with pytest.raises(NumericsError):
        verify_logit_compatibility(model, [mi("a", [1, 2])])


def test_unfitted_and_incompatible_models_are_rejected():
    item = mi("a", [1, 2])
    with pytest.raises(NumericsError):
        raw_logit(LightweightAnomalyModel(num_buckets=8), item)  # not fitted
    with pytest.raises(NumericsError):
        raw_logit(object(), item)
    with pytest.raises(NumericsError):
        raw_logits("not a model", [item])
    wrong_shape = make_model(np.zeros(8), 0.0)
    wrong_shape.weights = np.zeros(5)
    with pytest.raises(NumericsError):
        raw_logit(wrong_shape, item)
    two_d = make_model(np.zeros(8), 0.0)
    two_d.weights = np.zeros((8, 1))
    with pytest.raises(NumericsError):
        raw_logit(two_d, item)
    single = make_model(np.zeros(8), 0.0)
    single.weights = np.zeros(8, dtype=np.float32)
    with pytest.raises(NumericsError):
        raw_logit(single, item)
    for bad_weight in (np.nan, np.inf):
        w = np.zeros(8)
        w[3] = bad_weight
        with pytest.raises(NumericsError):
            raw_logit(make_model(w, 0.0), item)
    for bad_bias in (math.nan, math.inf):
        with pytest.raises(NumericsError):
            raw_logit(make_model(np.zeros(8), bad_bias), item)


def test_invalid_inputs_to_the_readout_are_rejected():
    model = make_model(np.zeros(8), 0.0)
    with pytest.raises(NumericsError):
        raw_logit(model, {"event_id": "a"})
    with pytest.raises(NumericsError):
        raw_logit(model, mi("a", [1, 2], [1]))
    with pytest.raises(NumericsError):
        raw_logits(model, [mi("a", [1]), mi("b", [], [])])
    with pytest.raises(NumericsError):
        raw_logits(model, "abc")


def test_overflowing_logit_is_an_explicit_error_not_a_warning():
    model = make_model(np.full(8, 1.7e308), 1.7e308)
    with pytest.raises(NumericsError):
        raw_logit(model, mi("a", [1, 2, 3, 4, 5, 6]))


def test_readout_never_mutates_or_fits_the_model(monkeypatch):
    model = make_model(np.arange(8) * 0.1, 0.4)
    snapshot = (model.weights.copy(), model.bias, model.is_trained, list(model.training_loss_history), model.model_version)

    def boom(*a, **k):
        raise AssertionError("fit must never be called")

    monkeypatch.setattr(LightweightAnomalyModel, "fit", boom)
    items = random_inputs(6, seed=12)
    raw_logits(model, items)
    verify_logit_compatibility(model, items)
    assert np.array_equal(model.weights, snapshot[0]) and model.bias == snapshot[1] and model.is_trained is True
    assert model.training_loss_history == snapshot[3] and model.model_version == snapshot[4]


def test_errors_do_not_leak_values():
    secret = "SECRET-TOKEN-1234"
    bad_weights = np.zeros(8)
    bad_weights[2] = np.nan
    messages = []
    for call in (lambda: raw_logit(make_model(bad_weights, 0.0), mi(secret, [1, 2])),
                 lambda: raw_logit(make_model(np.zeros(8), 0.0), mi(secret, [1, 2], [1])),
                 lambda: raw_logit(make_model(np.full(8, 1.7e308), 0.0), mi(secret, [1, 2, 3, 4, 5, 6])),
                 lambda: estimate_lambda_max([mi(secret, [1, 2], [1])], split="train", arm=ARM_A, num_buckets=4),
                 lambda: estimate_lambda_max([mi(secret, [1, 2])], split="validation", arm=ARM_A, num_buckets=4)):
        with pytest.raises(NumericsError) as caught:
            call()
        messages.append(str(caught.value))
    assert all(secret not in m and "nan" not in m.lower() and "1.7e+308" not in m for m in messages)


# -- provenance and scope -----------------------------------------------------------------------------------------------------


def test_provenance_is_json_safe_and_complete():
    prov = json.loads(json.dumps(nm.provenance(), allow_nan=False))
    assert prov["lambda_method"] == "lambda-power-1" and prov["logit_method"] == "logit-readout-1"
    assert prov["relative_step_grid"] == [0.1, 0.3, 0.8] and prov["num_buckets_grid"] == [1024, 2048, 4096]
    assert prov["relative_tolerance"] == 1e-10 and prov["max_iterations"] == 5000
    assert len(prov["numerics_sha256"]) == 64 and len(prov["model_sha256"]) == 64
    assert set(prov["private_model_interfaces_used"]) == {"_features", "_validate_model_input"}
    assert prov["scores_are_uncalibrated"] is True


def test_source_never_fits_or_uses_randomness_or_confirmatory_seeds():
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "fit" not in called and "random" not in called
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "random" not in imported and "secrets" not in imported
    assert not any(seed in text for seed in ("7101", "7102", "7103"))
