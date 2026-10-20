"""Tests for budget-rule-1 and the post-selection exposure report (draft protocol heldout-eval-1.2.0-draft, section 12.8).

Small hand-made arrays and development-seed (6101) requests only. No model is trained, no detection metric is computed, and the
confirmatory seeds 7101-7103 are never generated.
"""

from __future__ import annotations

import ast
import copy
import json
import random
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.token_budget as tb
from tokenization_ai.fusion import InputFusion
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.policy_sim import simulate_rules_for_record
from tokenization_ai.heldout_eval.request_context import request_to_context
from tokenization_ai.heldout_eval.splits import make_record, split_records
from tokenization_ai.heldout_eval.token_budget import (
    CANDIDATE_GRID,
    RULE_ID,
    BudgetDecision,
    BudgetInputError,
    BudgetNotFeasibleError,
    exposure_report,
    lengths_fingerprint,
    select_shared_max_length,
    sequence_lengths,
    tensorizer_outcome,
    tensorizer_outcomes,
)
from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.models import VerifiedRuleInfo
from tokenization_ai.normalization import Normalizer
from tokenization_ai.tensorizer import Tensorizer
from tokenization_ai.tokenizer import HybridTokenizer
from tokenization_ai.vocabulary import Vocabulary


def _lengths(values, prefix="r"):
    return {f"{prefix}{i}": v for i, v in enumerate(values)}


def _sequence(record, with_rules):
    ctx = request_to_context(record)
    rules = simulate_rules_for_record(record).rule_info if with_rules else VerifiedRuleInfo(event_id=ctx.event_id, rules=[])
    fused = InputFusion().fuse(ctx, rules)
    masked = PrivacyMasker().mask_with_provenance(fused)
    norm = Normalizer().normalize_with_provenance(masked.fused, masked.masked_paths)
    return HybridTokenizer().tokenize(norm.fused, norm.masked_paths)


@pytest.fixture(scope="module")
def dev():
    return generate(GeneratorConfig(seed=6101, n_templates=120))  # development seed only


# -- the rule ----------------------------------------------------------------------------------------------------------


def test_rule_id_and_grid_are_the_documented_ones():
    assert RULE_ID == "budget-rule-1" and CANDIDATE_GRID == (128, 192, 256, 384, 512)


@pytest.mark.parametrize("longest, expected", [
    (1, 128), (127, 128), (128, 128), (129, 192), (192, 192), (193, 256), (256, 256), (257, 384), (384, 384), (385, 512), (512, 512),
])
def test_smallest_candidate_that_fits_with_equality_fitting(longest, expected):
    d = select_shared_max_length(_lengths([1, longest, 1]))
    assert d.status == "selected" and d.selected_max_length == expected == d.require_selected()
    assert d.headroom == expected - longest and d.max == longest


def test_no_fitting_candidate_is_an_explicit_failure_not_a_truncation():
    d = select_shared_max_length(_lengths([100, 513]))
    assert d.status == "no_candidate_fits" and d.selected_max_length is None and d.headroom is None
    assert d.records_over_candidate["512"] == 1 and d.max == 513
    with pytest.raises(BudgetNotFeasibleError):
        d.require_selected()
    assert d.to_dict()["selected_max_length"] is None


def test_statistics_have_the_documented_definitions():
    d = select_shared_max_length(_lengths(range(1, 101)))
    assert (d.min, d.max, d.median, d.p95, d.p99, d.mean) == (1, 100, 50.5, 95, 99, 50.5)
    assert select_shared_max_length(_lengths([3, 1, 2])).median == 2.0
    assert select_shared_max_length(_lengths([4])).p95 == 4 == select_shared_max_length(_lengths([4])).p99
    assert select_shared_max_length(_lengths([1, 2])).mean == 1.5
    d = select_shared_max_length(_lengths([100, 130, 200, 300, 400]))
    assert d.records_over_candidate == {"128": 4, "192": 3, "256": 2, "384": 1, "512": 0}
    assert d.train_records == 5


def test_deterministic_and_independent_of_input_order():
    base = _lengths([random.Random(1).randint(20, 190) for _ in range(300)])
    items = list(base.items())
    a = select_shared_max_length(items)
    random.Random(7).shuffle(items)
    b = select_shared_max_length(items)
    c = select_shared_max_length(dict(reversed(list(base.items()))))
    assert a == b == c and a.lengths_sha256 == lengths_fingerprint(base)


def test_fingerprint_is_canonical_sha256_and_sensitive_to_ids_and_lengths():
    import hashlib

    m = {"b": 3, "a": 2}
    expected = hashlib.sha256(b'[["a",2],["b",3]]').hexdigest()
    assert lengths_fingerprint(m) == lengths_fingerprint({"a": 2, "b": 3}) == expected
    assert lengths_fingerprint({"a": 2, "b": 4}) != expected and lengths_fingerprint({"a": 2, "c": 3}) != expected


@pytest.mark.parametrize("bad", [
    {}, [], {"a": 0}, {"a": -3}, {"a": 1.5}, {"a": True}, {"a": "9"}, {"": 5}, {" a": 5}, {1: 5}, {"a": None},
    "abc", 5, [("a", 1, 2)], [("a", 1), ("a", 2)],
])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(BudgetInputError):
        select_shared_max_length(bad)


@pytest.mark.parametrize("grid", [(), (0, 5), (128, 128), (256, 128), (128.0, 192), (True, 2), [], "128", None])
def test_invalid_grids_are_rejected(grid):
    with pytest.raises(BudgetInputError):
        select_shared_max_length({"a": 5}, grid)


def test_decision_is_json_serializable_and_stable():
    d = select_shared_max_length(_lengths([40, 90, 140]))
    text = json.dumps(d.to_dict(), allow_nan=False, sort_keys=True)
    out = json.loads(text)
    assert out["rule_id"] == RULE_ID and out["selected_max_length"] == 192 and out["candidate_grid"] == [128, 192, 256, 384, 512]
    assert out["length_statistics"]["max"] == 140 and out["basis"] == "arm_b_training_split_token_sequence_lengths"
    assert out["lengths_sha256"] == d.lengths_sha256 and out["headroom"] == 52


def test_decision_cannot_be_influenced_by_validation_or_test_data():
    # The selection function's only data parameter is the training lengths mapping.
    import inspect

    params = list(inspect.signature(select_shared_max_length).parameters)
    assert params == ["train_arm_b_lengths", "grid"]
    train = _lengths([50, 120])
    d1 = select_shared_max_length(train)
    huge_validation, huge_test = _lengths([500, 900], "v"), _lengths([700], "t")
    d2 = select_shared_max_length(train)  # validation/test existing elsewhere changes nothing
    assert d1 == d2 and d1.selected_max_length == 128
    rep = exposure_report(d1, "validation", huge_validation, huge_validation)
    assert rep["budget_changed_by_this_report"] is False and d1.selected_max_length == 128
    assert exposure_report(d1, "test", huge_test, huge_test)["max_length"] == 128


def test_selection_does_not_depend_on_labels_or_metadata(dev):
    recs = list(dev.records)
    flipped = [make_record(r.record_id, method=r.features.method, path=r.features.path, label=1 - r.meta.label,
                           body=r.features.body, headers=dict(r.features.headers), timestamp=7.0, source_id="zz") for r in recs]
    d1 = select_shared_max_length(sequence_lengths({r.record_id: _sequence(r, True) for r in recs}))
    d2 = select_shared_max_length(sequence_lengths({r.record_id: _sequence(r, True) for r in flipped}))
    assert d1 == d2


def test_module_has_no_label_split_or_model_dependencies():
    tree = ast.parse(Path(tb.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"model", "service", "generator", "audit", "policy_sim", "splits", "request_context"}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"label", "meta", "timestamp", "source_id", "predict", "fit"}
    assert "hash(" not in Path(tb.__file__).read_text(encoding="utf-8").replace("sha256", "")


# -- real tokenizer / tensorizer semantics ---------------------------------------------------------------------------------


def test_length_is_the_token_count_including_framing_and_the_rules_segment(dev):
    r = dev.records[0]
    a, b = _sequence(r, False), _sequence(r, True)
    assert a.tokens[0] == "[SEQ_START]" and a.tokens[-1] == "[SEQ_END]"
    assert a.tokens[-3:-1] == ["[RULES_START]", "[RULES_END]"]  # empty rules still add the segment framing
    n_rules = len(simulate_rules_for_record(r).policy_ids)
    assert len(b.tokens) - len(a.tokens) == 20 * n_rules
    lengths = sequence_lengths({"x": a, "y": b})
    assert lengths == {"x": len(a.tokens), "y": len(b.tokens)}
    # What the tensorizer receives is exactly that many ids.
    vocab = Vocabulary()
    vocab.fit([a, b])
    assert len(vocab.encode(b)) == len(b.tokens)


def test_a_fitting_budget_yields_no_truncation_for_any_training_sequence(dev):
    seqs_b = {r.record_id: _sequence(r, True) for r in dev.records}
    seqs_a = {r.record_id: _sequence(r, False) for r in dev.records}
    d = select_shared_max_length(sequence_lengths(seqs_b))
    budget = d.require_selected()
    vocab = Vocabulary()
    vocab.fit(list(seqs_a.values()) + list(seqs_b.values()))
    tensorizer = Tensorizer(vocab, max_length=budget)
    for seqs in (seqs_a, seqs_b):
        for seq in seqs.values():
            model_input = tensorizer.tensorize(seq)
            assert sum(model_input.attention_mask) == len(seq.tokens)  # nothing truncated
    assert set(tensorizer_outcomes(seqs_b, budget).values()) == {"fits"}
    assert set(tensorizer_outcomes(seqs_a, budget).values()) == {"fits"}
    assert d.records_over_candidate[str(budget)] == 0


def _long_record(n_headers):
    headers = {f"x-h{i:03d}": "v" for i in range(n_headers)}
    return make_record("big", method="POST", path="/orders", label=0, body='{"a":1,"b":2}', headers=headers)


def test_outcomes_match_the_real_tensorizer_for_fit_truncate_and_reject(dev):
    # payload-only overflow: truncated
    wide = make_record("w", method="POST", path="/orders", label=0, headers={"user-agent": "T/1"},
                       body=json.dumps({f"field{i:02d}": f"value{i}" for i in range(40)}))
    seq = _sequence(wide, True)
    assert len(seq.tokens) > 128
    assert tensorizer_outcome(seq, 128) == "truncated" and tensorizer_outcome(seq, len(seq.tokens)) == "fits"
    vocab = Vocabulary()
    vocab.fit([seq])
    mi = Tensorizer(vocab, max_length=128).tensorize(seq)
    assert sum(mi.attention_mask) <= 128  # the real tensorizer truncates the payload and does not raise
    # framing alone too long: rejected
    big = _sequence(_long_record(80), True)
    assert tensorizer_outcome(big, 128) == "rejected"
    vocab2 = Vocabulary()
    vocab2.fit([big])
    with pytest.raises(ValueError):
        Tensorizer(vocab2, max_length=128).tensorize(big)
    # the input sequence is not modified by the outcome check
    before = copy.deepcopy(big)
    tensorizer_outcome(big, 128)
    assert big == before


def test_outcome_rejects_invalid_max_length(dev):
    with pytest.raises(BudgetInputError):
        tensorizer_outcome(_sequence(dev.records[0], False), 0)


# -- shared budget and exposure report -----------------------------------------------------------------------------------------


def test_the_same_selected_budget_applies_to_both_arms_and_arm_a_is_never_longer_in_development_data(dev):
    a = sequence_lengths({r.record_id: _sequence(r, False) for r in dev.records})
    b = sequence_lengths({r.record_id: _sequence(r, True) for r in dev.records})
    d = select_shared_max_length(b)
    rep = exposure_report(d, "train", a, b)
    assert rep["max_length"] == d.selected_max_length
    assert rep["arm_a"]["over_budget"] == rep["arm_b"]["over_budget"] == 0 and rep["findings"] == []
    assert rep["paired_exposure"]["records_where_arm_a_longer_than_arm_b"] == 0
    assert rep["paired_exposure"]["mean_length_difference_b_minus_a"] > 0


def test_exposure_counts_and_arm_differences_on_hand_made_lengths():
    d = select_shared_max_length(_lengths([100, 128]))  # budget 128
    a = {"r1": 100, "r2": 130, "r3": 90}
    b = {"r1": 140, "r2": 150, "r3": 120}
    rep = exposure_report(d, "test", a, b)
    assert rep["max_length"] == 128 and rep["split"] == "test"
    assert rep["arm_a"]["over_budget"] == 1 and rep["arm_b"]["over_budget"] == 2
    assert rep["arm_b"]["largest_excess"] == 22 and rep["arm_b"]["truncation_risk"] is True
    p = rep["paired_exposure"]
    assert (p["over_budget_in_either_arm"], p["over_budget_only_in_arm_b"], p["over_budget_only_in_arm_a"],
            p["over_budget_in_both"]) == (2, 1, 0, 1)
    assert p["mean_length_difference_b_minus_a"] == pytest.approx((40 + 20 + 30) / 3, abs=1e-6)
    assert rep["protocol_review_required"] is True
    assert any("test records exceed the frozen budget" in f for f in rep["findings"])
    assert d.selected_max_length == 128  # the frozen decision is untouched
    assert rep["records_excluded_by_this_report"] == 0 and rep["budget_changed_by_this_report"] is False
    json.dumps(rep, allow_nan=False)


def test_exposure_within_budget_has_no_findings():
    d = select_shared_max_length(_lengths([100]))
    rep = exposure_report(d, "validation", {"r": 50}, {"r": 100})
    assert rep["findings"] == [] and rep["protocol_review_required"] is False
    assert rep["arm_b"]["over_budget_share"] == 0.0


def test_missing_and_mismatched_pairs_are_reported_not_hidden():
    d = select_shared_max_length(_lengths([100]))
    rep = exposure_report(d, "validation", {"r1": 50, "r2": 60, "only_a": 70}, {"r1": 90, "r2": 40, "only_b": 95, "only_b2": 95})
    pr = rep["pairing"]
    assert (pr["arm_a_records"], pr["arm_b_records"], pr["paired"]) == (3, 4, 2)
    assert pr["missing_in_arm_a"] == 2 and pr["missing_in_arm_b"] == 1
    assert pr["missing_in_arm_a_ids"] == ["only_b", "only_b2"] and pr["missing_in_arm_b_ids"] == ["only_a"]
    assert rep["paired_exposure"]["records_where_arm_a_longer_than_arm_b"] == 1
    assert any("mismatch" in f for f in rep["findings"]) and any("Arm A is longer" in f for f in rep["findings"])


def test_unexpected_over_budget_training_sequences_are_flagged_explicitly():
    d = select_shared_max_length(_lengths([100]))
    rep = exposure_report(d, "train", {"r": 200}, {"r": 100})
    assert any("Arm A training sequence exceeds" in f for f in rep["findings"]) and rep["protocol_review_required"]
    rep2 = exposure_report(d, "train", {"r": 100}, {"r": 200})
    assert any("Arm B training sequence exceeds" in f for f in rep2["findings"])


def test_tensorizer_outcomes_are_included_and_validated():
    d = select_shared_max_length(_lengths([100]))
    a, b = {"r1": 100, "r2": 100}, {"r1": 100, "r2": 190}
    rep = exposure_report(d, "test", a, b, arm_a_outcomes={"r1": "fits", "r2": "fits"},
                          arm_b_outcomes={"r1": "fits", "r2": "rejected"})
    assert rep["arm_b"]["tensorizer"] == {"fits": 1, "truncated": 0, "rejected": 1} and rep["arm_b"]["rejected_ids"] == ["r2"]
    assert any("would reject" in f for f in rep["findings"])
    with pytest.raises(BudgetInputError):
        exposure_report(d, "test", a, b, arm_b_outcomes={"r1": "fits"})
    with pytest.raises(BudgetInputError):
        exposure_report(d, "test", a, b, arm_b_outcomes={"r1": "fits", "r2": "maybe"})


def test_exposure_requires_a_selected_decision_and_a_known_split():
    failed = select_shared_max_length(_lengths([999]))
    with pytest.raises(BudgetInputError):
        exposure_report(failed, "test", {"r": 1}, {"r": 1})
    ok = select_shared_max_length(_lengths([10]))
    with pytest.raises(BudgetInputError):
        exposure_report(ok, "holdout", {"r": 1}, {"r": 1})
    with pytest.raises(BudgetInputError):
        exposure_report("128", "test", {"r": 1}, {"r": 1})  # type: ignore[arg-type]
    with pytest.raises(BudgetInputError):
        exposure_report(ok, "test", {"r": 0}, {"r": 1})


def test_decision_is_immutable():
    d = select_shared_max_length(_lengths([10]))
    with pytest.raises(Exception):
        d.selected_max_length = 999  # type: ignore[misc]
    assert isinstance(d, BudgetDecision)


def test_training_only_workflow_on_a_development_split(dev):
    res = split_records(dev.records, seed=4401, strict=False)
    by_id = {r.record_id: r for r in dev.records}
    seq_b = {rid: _sequence(by_id[rid], True) for rid in res.members("train")}
    d = select_shared_max_length(sequence_lengths(seq_b))
    assert d.train_records == len(res.members("train")) and d.status == "selected"
    # Later (after the manifest): exposure of other splits against the frozen value, selection untouched.
    for split in ("validation", "test"):
        ids = res.members(split)
        a = sequence_lengths({rid: _sequence(by_id[rid], False) for rid in ids})
        b = sequence_lengths({rid: _sequence(by_id[rid], True) for rid in ids})
        rep = exposure_report(d, split, a, b)
        assert rep["pairing"]["paired"] == len(ids) and rep["max_length"] == d.selected_max_length
