"""Tests for runner-foundation-1 (draft protocol heldout-eval-1.2.0-draft, sections 12.11.2-12.11.4 and 12.12).

The runner fits no model and scores nothing, so these tests do not either. Dataset-level tests use the *development* seed 6101 only, for structure
(partitioning, pairing, preparation, preflight); no label is used to choose anything. Candidate selection is tested on small hand-made result lists.
The confirmatory seeds 7101-7103 are never generated or referenced.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import pickle
import random
import subprocess
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.manifest as mf
import tokenization_ai.heldout_eval.runner as rn
from tokenization_ai.heldout_eval.arms import ARM_A, ARM_B, ARM_C, ARMS, ArmInputs, prepare_split
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.runner import (
    AuthorizationUnavailable,
    RepresentabilityError,
    RunnerError,
    SealedTestSet,
    ValidationCandidateResult,
    aligned_labels,
    approved_candidate_ids,
    canonical_candidate_id,
    partition_records,
    prepare_numerical_configuration,
    prepare_train_configuration,
    preflight_report,
    preflight_split,
    request_test_access,
    request_training_authorization,
    select_candidate,
)
from tokenization_ai.heldout_eval.splits import PRIMARY_SPLIT_SEED, EvalRecord, SplitMetadata, split_records

SOURCE = Path(rn.__file__)


# -- fixtures (development seed only; nothing is fitted or scored) -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    ds = generate(GeneratorConfig(seed=6101, n_templates=250))
    assert ds.config.seed == 6101
    return ds


@pytest.fixture(scope="module")
def split(dataset):
    return split_records(dataset.records, seed=PRIMARY_SPLIT_SEED)


@pytest.fixture(scope="module")
def part(dataset, split):
    return partition_records(dataset.records, split)


@pytest.fixture(scope="module")
def cfg(part):
    return prepare_train_configuration(part.train)


@pytest.fixture(scope="module")
def numerical(cfg):
    return prepare_numerical_configuration(cfg)


# -- A. dataset and split validation ----------------------------------------------------------------------------------------------------


def test_partition_separates_the_three_splits_by_the_split_assignment(dataset, split, part):
    assert set(part.train.record_ids) == set(split.members("train"))
    assert set(part.validation.record_ids) == set(split.members("validation"))
    assert set(part.sealed_test.record_ids) == set(split.members("test"))
    all_ids = [*part.train.record_ids, *part.validation.record_ids, *part.sealed_test.record_ids]
    assert len(all_ids) == len(set(all_ids)) == len(dataset.records)
    assert list(part.train.record_ids) == sorted(part.train.record_ids)  # deterministic order


def test_partition_is_deterministic(dataset, split, part):
    again = partition_records(dataset.records, split)
    assert again.summary == part.summary
    assert again.sealed_test.record_ids == part.sealed_test.record_ids
    assert json.dumps(part.summary, allow_nan=False)


def test_train_and_validation_keep_labels_and_test_does_not(dataset, part):
    true = {r.record_id: r.meta.label for r in dataset.records}
    assert part.train.labels_by_id() == {i: true[i] for i in part.train.record_ids}
    assert part.validation.labels_by_id() == {i: true[i] for i in part.validation.record_ids}
    test_true = [true[i] for i in part.sealed_test.record_ids]
    assert set(test_true) == {0, 1}  # the real test labels do vary...
    assert {r.meta.label for r in part.sealed_test.blinded_records()} == {-1}  # ...but the sealed set holds only the invalid sentinel


def test_sealed_test_set_exposes_no_label_score_or_unsealing_path(part):
    public = {n for n in dir(part.sealed_test) if not n.startswith("_")}
    assert public == {"binding", "blinded_records", "group_by_id", "record_ids"}
    assert not any(isinstance(v, dict) and set(v.values()) <= {0, 1} and v for v in vars(part.sealed_test).values()
                   if set(v) == set(part.sealed_test.record_ids))  # no id -> label table is stored
    assert "label" not in repr(part.sealed_test) and len(part.sealed_test) == len(part.sealed_test.record_ids)
    summary_test = part.summary["test"]
    assert "positives" not in summary_test and summary_test["labels"] == "sealed_not_retained_by_runner"


def test_blinded_labels_do_not_change_what_the_arms_build(dataset, part):
    ids = set(part.sealed_test.record_ids[:30])
    real = [r for r in dataset.records if r.record_id in ids]
    blinded = [r for r in part.sealed_test.blinded_records() if r.record_id in ids]
    a, b = prepare_split(real, "test"), prepare_split(blinded, "test")
    assert [(x.record_id, x.sequence_a.tokens, x.sequence_b.tokens, x.model_input_c.input_ids) for x in a.requests] == \
           [(x.record_id, x.sequence_a.tokens, x.sequence_b.tokens, x.model_input_c.input_ids) for x in b.requests]


def test_blinded_sentinel_can_never_be_mistaken_for_ground_truth(part):
    blinded = part.sealed_test.blinded_records()
    labels = {r.record_id: r.meta.label for r in blinded}
    assert set(labels.values()) == {-1}
    with pytest.raises(RunnerError):
        aligned_labels(list(labels), labels)  # the label-alignment path rejects it
    with pytest.raises(RunnerError):
        rn.TrainData(tuple(blinded))
    with pytest.raises(RunnerError):
        rn.ValidationData(tuple(blinded))
    with pytest.raises(RunnerError):
        rn.ValidationData(tuple(dataclasses.replace(r, meta=SplitMetadata(label=True)) for r in blinded[:3]))  # not a real class either


def test_authentic_test_labels_are_not_held_but_are_recoverable_from_the_source_dataset(dataset, split, part):
    # nothing in the partition holds a test label ...
    test_ids = set(part.sealed_test.record_ids)
    for holder in (part.sealed_test, part.train, part.validation):
        for value in vars(holder).values():
            if isinstance(value, tuple) and value and isinstance(value[0], EvalRecord):
                assert not any(r.record_id in test_ids and r.meta.label in (0, 1) for r in value)
    assert not any(k in test_ids for k in part.train.labels_by_id()) and not any(k in test_ids for k in part.validation.labels_by_id())
    # ... but the caller's source dataset still has them, by record id, and the sealed set binds to exactly that split
    recovered = {r.record_id: r.meta.label for r in dataset.records if r.record_id in test_ids}
    assert set(recovered) == test_ids
    assert aligned_labels(part.sealed_test.record_ids, recovered) == tuple(recovered[i] for i in part.sealed_test.record_ids)
    binding = part.sealed_test.binding()
    assert binding["records"] == len(test_ids) == len(recovered)
    assert binding["ids_sha256"] == rn.ids_fingerprint(sorted(recovered))
    assert binding["split_manifest_sha256"] == split.manifest_sha256 != ""
    assert binding["groups_sha256"] == rn.ids_fingerprint(sorted({split.group_of[i] for i in test_ids}))
    assert part.summary["test"]["ids_sha256"] == binding["ids_sha256"]
    assert set(binding) == {"records", "ids_sha256", "groups_sha256", "split_manifest_sha256"}
    json.dumps(binding, allow_nan=False)


def test_partition_rejects_malformed_inputs(dataset, split):
    recs = list(dataset.records)
    with pytest.raises(RunnerError):
        partition_records([], split)
    with pytest.raises(RunnerError):
        partition_records("abc", split)
    with pytest.raises(RunnerError):
        partition_records([{"a": 1}], split)
    with pytest.raises(RunnerError):
        partition_records(recs, object())
    with pytest.raises(RunnerError):
        partition_records(recs + [recs[0]], split)  # duplicate record id


def test_partition_detects_missing_and_unknown_records(dataset, split):
    recs = list(dataset.records)
    with pytest.raises(RunnerError):
        partition_records(recs[1:], split)  # an assigned record is missing from the data
    extra = dataclasses.replace(recs[0], record_id="unassigned-record")
    with pytest.raises(RunnerError):
        partition_records(recs + [extra], split)  # a record without an assignment


def test_partition_rejects_infeasible_exclusions_and_unknown_splits(dataset, split):
    recs = list(dataset.records)
    bad_report = dataclasses.replace(split.report, problems=("a split is missing a class",))
    with pytest.raises(RunnerError):
        partition_records(recs, dataclasses.replace(split, report=bad_report))
    with pytest.raises(RunnerError):
        partition_records(recs, dataclasses.replace(split, excluded={recs[0].record_id: "conflict"}))
    unknown = dict(split.assignments)
    unknown[recs[0].record_id] = "holdout"
    with pytest.raises(RunnerError):
        partition_records(recs, dataclasses.replace(split, assignments=unknown))


def test_partition_rejects_split_leakage_through_a_shared_group(dataset, split):
    recs = list(dataset.records)
    by_group: dict[str, list[str]] = {}
    for rid, g in split.group_of.items():
        by_group.setdefault(g, []).append(rid)
    group, members = next((g, m) for g, m in sorted(by_group.items()) if len(m) > 1)
    current = split.assignments[members[0]]
    other = next(n for n in ("train", "validation", "test") if n != current)
    leaked = dict(split.assignments)
    leaked[members[1]] = other if split.assignments[members[1]] == current else split.assignments[members[1]]
    if leaked[members[1]] == current:
        leaked[members[1]] = other
    with pytest.raises(RunnerError):
        partition_records(recs, dataclasses.replace(split, assignments=leaked))


def test_partition_requires_both_classes_in_train_and_validation(dataset, split):
    val_ids = set(split.members("validation"))
    only_normal = [dataclasses.replace(r, meta=SplitMetadata(label=0)) if r.record_id in val_ids else r for r in dataset.records]
    with pytest.raises(RunnerError):
        partition_records(only_normal, split)


# -- B. alignment ----------------------------------------------------------------------------------------------------------------------------


def test_labels_are_mapped_by_id_not_by_position():
    ids = ["r3", "r1", "r2"]
    labels = {"r1": 1, "r2": 0, "r3": 1}
    assert aligned_labels(ids, labels) == (1, 1, 0)
    shuffled = dict(sorted(labels.items(), reverse=True))
    assert aligned_labels(ids, shuffled) == (1, 1, 0)
    assert aligned_labels(list(reversed(ids)), labels) == (0, 1, 1)


@pytest.mark.parametrize("labels", [{"r1": 1, "r2": 0}, {"r1": 1, "r2": 0, "r3": 1, "r4": 0}, {"r1": 1, "r2": 0, "r3": 2},
                                    {"r1": 1, "r2": 0, "r3": True}, {"r1": 1, "r2": 0, "r3": None}, {"r1": 1, "r2": 0, "x": 1}])
def test_label_alignment_rejects_missing_extra_and_invalid(labels):
    with pytest.raises(RunnerError):
        aligned_labels(["r1", "r2", "r3"], labels)


def test_label_alignment_rejects_duplicate_ids():
    with pytest.raises(RunnerError):
        aligned_labels(["r1", "r1"], {"r1": 1})


def test_arm_inputs_and_labels_share_one_verified_order(cfg, part):
    assert isinstance(cfg.inputs, ArmInputs) and cfg.inputs.split == "train"
    assert tuple(cfg.inputs.record_ids) == part.train.record_ids
    labels = aligned_labels(cfg.inputs.record_ids, part.train.labels_by_id())
    by_id = part.train.labels_by_id()
    assert all(labels[k] == by_id[rid] for k, rid in enumerate(cfg.inputs.record_ids))


# -- C. train-only configuration ----------------------------------------------------------------------------------------------------------------


def test_train_configuration_comes_from_the_training_split_only(cfg, part):
    d = cfg.to_dict()
    assert d["derived_from"] == "train_split_only" and d["train_records"] == len(part.train.records)
    assert d["vocabulary"]["fitted_on_split"] == "train" and d["vocabulary"]["training_records"] == len(part.train.records)
    assert d["min_frequency"] == 2 and d["max_literal_vocab_size"] is None
    assert d["budget"]["train_records"] == len(part.train.records) and d["budget"]["status"] == "selected"
    assert cfg.decision.selected_max_length in (128, 192, 256, 384, 512)
    assert d["train_ids_sha256"] == rn.ids_fingerprint(part.train.record_ids)
    json.dumps(d, allow_nan=False)


def test_train_configuration_accepts_only_train_data(part):
    with pytest.raises(RunnerError):
        prepare_train_configuration(part.validation)
    with pytest.raises(RunnerError):
        prepare_train_configuration(list(part.train.records))


def test_train_configuration_is_deterministic_and_not_refit_by_later_stages(cfg, part):
    again = prepare_train_configuration(part.train)
    assert again.to_dict() == cfg.to_dict()
    sha_before = cfg.frozen.record["sha256"]
    rn.encode_validation(cfg, part.validation)
    assert cfg.frozen.record["sha256"] == sha_before and cfg.decision.selected_max_length == again.decision.selected_max_length


def test_encode_validation_uses_the_frozen_vocabulary_and_budget(cfg, part):
    val = rn.encode_validation(cfg, part.validation)
    assert val.split == "validation" and val.max_length == cfg.decision.selected_max_length
    assert tuple(val.record_ids) == part.validation.record_ids
    assert all(o == "fits" for o in (*val.outcomes_a, *val.outcomes_b))
    with pytest.raises(RunnerError):
        rn.encode_validation(cfg, part.train)


def test_numerical_configuration_is_train_only_and_has_27_candidates_per_arm(numerical, cfg, part):
    assert len(numerical.estimates) == 9 and all(e.split == "train" for e in numerical.estimates)
    assert {e.record_ids_sha256 for e in numerical.estimates} == {rn.ids_fingerprint(part.train.record_ids)}
    assert len(numerical.candidates) == 81
    for arm in ARMS:
        cands = numerical.candidates_for(arm)
        assert len(cands) == 27 and [c.candidate_id for c in cands] == sorted(approved_candidate_ids(arm))
        assert {c.epochs for c in cands} == {50, 150, 400} and {c.num_buckets for c in cands} == {1024, 2048, 4096}
    cell = {(e.arm, e.num_buckets): e for e in numerical.estimates}
    for c in numerical.candidates:
        assert c.learning_rate == c.relative_step_c / cell[(c.arm, c.num_buckets)].L
    json.dumps(numerical.to_dict(), allow_nan=False)


def test_numerical_configuration_is_deterministic(numerical, cfg):
    assert prepare_numerical_configuration(cfg).to_dict() == numerical.to_dict()
    with pytest.raises(RunnerError):
        prepare_numerical_configuration("not a configuration")


# -- D. representability preflight ------------------------------------------------------------------------------------------------------------------


def test_preflight_passes_under_the_training_budget_for_all_splits(cfg, part):
    report = preflight_report(cfg, part.validation, part.sealed_test)
    assert report["passed"] is True and report["max_length"] == cfg.decision.selected_max_length
    for name, r in report["splits"].items():
        assert r["passed"] and r["labels_accessed"] is False and r["scoring_inputs_encoded"] is False
        assert r["records_excluded"] == 0 and r["budget_changed"] is False and r["arm_c"].startswith("not_budgeted")
        assert r["arm_outcome_counts"][ARM_A]["fits"] == r["records"] == r["arm_outcome_counts"][ARM_B]["fits"]
    assert report["splits"]["test"]["records"] == len(part.sealed_test)
    json.dumps(report, allow_nan=False)


def test_preflight_of_the_test_split_ignores_labels(dataset, part, cfg):
    test_ids = set(part.sealed_test.record_ids)
    real = [r for r in dataset.records if r.record_id in test_ids]
    a = preflight_split(real, "test", cfg.decision).to_dict()
    b = preflight_split(part.sealed_test.blinded_records(), "test", cfg.decision).to_dict()
    assert a == b


def test_preflight_builds_no_scoring_ready_inputs_and_no_arm_c_encoding(monkeypatch, cfg, part):
    import tokenization_ai.heldout_eval.arms as arms_module
    from tokenization_ai.tensorizer import Tensorizer
    from tokenization_ai.vocabulary import Vocabulary

    def forbidden(*args, **kwargs):
        raise AssertionError("the preflight must not build scoring inputs")

    monkeypatch.setattr(arms_module, "encode_request", forbidden)  # Arm C encoder
    monkeypatch.setattr(arms_module, "prepare_split", forbidden)  # the full three-arm preparation
    monkeypatch.setattr(rn, "prepare_split", forbidden)
    monkeypatch.setattr(arms_module, "build_arm_inputs", forbidden)
    monkeypatch.setattr(Tensorizer, "tensorize", forbidden)
    monkeypatch.setattr(Vocabulary, "encode", forbidden)
    monkeypatch.setattr(Vocabulary, "fit", forbidden)
    report = preflight_split(part.sealed_test.blinded_records(), "test", cfg.decision)
    assert report.passed and report.records == len(part.sealed_test)
    assert preflight_split(part.validation.records, "validation", cfg.decision).passed


def test_preflight_outcomes_equal_the_full_preparation_path_without_arm_c(dataset, part, cfg):
    ids = set(part.sealed_test.record_ids[:40])
    blinded = [r for r in part.sealed_test.blinded_records() if r.record_id in ids]
    sequences = rn._budget_sequences(blinded, "test")
    full = prepare_split(blinded, "test")
    assert [(rid, a.tokens, a.roles, b.tokens, b.roles) for rid, a, b in sequences] == \
           [(r.record_id, r.sequence_a.tokens, r.sequence_a.roles, r.sequence_b.tokens, r.sequence_b.roles) for r in full.requests]
    for max_length in (cfg.decision.selected_max_length, 100, 60, 30):
        tight = dataclasses.replace(cfg.decision, selected_max_length=max_length)
        assert preflight_split(blinded, "test", tight).to_dict() == rn.preflight_prepared(full, tight).to_dict() | {"split": "test"}


def test_preflight_retains_nothing_and_returns_only_ids_and_counts(cfg, part):
    report = preflight_split(part.sealed_test.blinded_records(), "test", dataclasses.replace(cfg.decision, selected_max_length=60))
    allowed = {"split", "max_length", "records", "arm_c", "arm_outcome_counts", "failing_totals", "failing_listed", "listed_cap", "passed",
               "action_if_failed", "labels_accessed", "scoring_inputs_encoded", "records_excluded", "budget_changed"}
    assert set(report.to_dict()) == allowed
    assert not any(hasattr(report, a) for a in ("sequences", "model_inputs", "requests", "records_data"))


def test_preflight_rejects_malformed_record_lists(cfg, part):
    recs = list(part.validation.records)
    for bad in ([], ["x"], recs + [recs[0]]):
        with pytest.raises(RunnerError):
            preflight_split(bad, "validation", cfg.decision)


def over_budget(cfg, max_length):
    return dataclasses.replace(cfg.decision, selected_max_length=max_length)


@pytest.mark.parametrize("max_length", [60, 30])
def test_over_budget_requests_fail_closed_with_safe_identifiers_only(cfg, part, max_length):
    tight = over_budget(cfg, max_length)
    report = preflight_split(part.validation.records, "validation", tight)
    assert not report.passed and report.records == len(part.validation.records)
    totals = report.failing_totals
    assert totals["truncated"] + totals["rejected"] > 0
    with pytest.raises(RepresentabilityError) as caught:
        report.require_pass()
    assert caught.value.report is report
    blob = json.dumps(report.to_dict(), allow_nan=False) + str(caught.value)
    by_id = {r.record_id: r for r in part.validation.records}
    for rec in list(by_id.values())[:50]:
        assert rec.features.path not in blob
    assert report.to_dict()["records_excluded"] == 0 and len(report.to_dict()["failing_listed"]["truncated"]) <= 20
    for cat, rows in report.failing.items():
        for rid, arms in rows:
            assert rid in by_id and set(arms) <= {ARM_A, ARM_B}
    # counts: every record falls into exactly one outcome per budgeted arm; nothing was dropped
    for arm in (ARM_A, ARM_B):
        assert sum(report.arm_counts[arm].values()) == report.records


def test_preflight_report_and_encode_validation_stop_when_the_budget_does_not_fit(cfg, part):
    tight = dataclasses.replace(cfg, decision=over_budget(cfg, 60))
    with pytest.raises(RepresentabilityError):
        preflight_report(tight, part.validation, part.sealed_test)
    with pytest.raises(RepresentabilityError):
        rn.encode_validation(tight, part.validation)


def test_a_truncation_is_a_stop_not_a_count_to_tolerate(cfg, part):
    # The arm builder tolerates payload truncation; the runner must not. Pick a budget that truncates but rejects nothing if one exists.
    for max_length in range(cfg.decision.max, 40, -2):
        rep = preflight_split(part.validation.records, "validation", over_budget(cfg, max_length))
        if rep.failing_totals["truncated"] and not rep.failing_totals["rejected"]:
            with pytest.raises(RepresentabilityError):
                rep.require_pass()
            return
    pytest.skip("no purely-truncating budget on this fixture")


def test_preflight_split_rejects_the_wrong_split_and_bad_budget(cfg, part):
    with pytest.raises(RunnerError):
        preflight_split(part.validation.records, "train", cfg.decision)
    with pytest.raises(RunnerError):
        preflight_split(part.validation.records, "validation", "128")
    none_fit = dataclasses.replace(cfg.decision, status="no_candidate_fits", selected_max_length=None)
    with pytest.raises(Exception):
        preflight_split(part.validation.records, "validation", none_fit)


# -- F. validation-only selection -----------------------------------------------------------------------------------------------------------------------


def results_for(arm, ap_of, *, sha="v" * 64, n=100, split="validation"):
    out = []
    for nb in (1024, 2048, 4096):
        for ep in (50, 150, 400):
            for c in (0.1, 0.3, 0.8):
                out.append(ValidationCandidateResult(canonical_candidate_id(arm, nb, ep, c), arm, nb, ep, c, ap_of(nb, ep, c), split, sha, n))
    return out


def test_selection_picks_the_highest_validation_ap():
    res = results_for(ARM_A, lambda nb, ep, c: 0.5)
    res[7] = dataclasses.replace(res[7], validation_ap=0.9)
    decision = select_candidate(res, arm=ARM_A)
    assert decision.selected.candidate_id == res[7].candidate_id and decision.rounded_ap == 0.9
    assert decision.tied_at_rounded_ap == 1 and len(decision.ranking) == 27


def test_ties_at_ten_decimals_break_by_epochs_then_buckets_then_c_then_id():
    base = 0.6
    flat = results_for(ARM_B, lambda nb, ep, c: base)
    assert select_candidate(flat, arm=ARM_B).selected.candidate_id == canonical_candidate_id(ARM_B, 1024, 50, 0.1)
    # only the candidates below tie for the top; every other candidate is clearly worse
    def only(winners):
        return results_for(ARM_B, lambda nb, ep, c: 0.6 if (nb, ep, c) in winners else 0.1)
    assert select_candidate(only({(4096, 400, 0.1), (1024, 150, 0.8), (2048, 50, 0.3)}), arm=ARM_B).selected.epochs == 50
    assert select_candidate(only({(4096, 150, 0.1), (1024, 150, 0.8), (2048, 150, 0.3)}), arm=ARM_B).selected.num_buckets == 1024
    assert select_candidate(only({(2048, 150, 0.8), (2048, 150, 0.3), (2048, 150, 0.1)}), arm=ARM_B).selected.relative_step_c == 0.1
    # tie ranking is by every rule in order, not just the winner
    ranking = select_candidate(only({(4096, 400, 0.1), (1024, 150, 0.8), (2048, 50, 0.3)}), arm=ARM_B).ranking
    assert [r["candidate_id"] for r in ranking[:3]] == [canonical_candidate_id(ARM_B, 2048, 50, 0.3),
                                                          canonical_candidate_id(ARM_B, 1024, 150, 0.8),
                                                          canonical_candidate_id(ARM_B, 4096, 400, 0.1)]


def test_ap_is_compared_after_rounding_to_ten_decimals_but_full_precision_is_kept():
    # 0.7000000000 vs 0.70000000004 round to the same value, so the tie-break (fewer epochs) decides although the second AP is larger
    res = results_for(ARM_C, lambda nb, ep, c: 0.2)
    slow = next(i for i, r in enumerate(res) if (r.num_buckets, r.epochs, r.relative_step_c) == (1024, 400, 0.1))
    fast = next(i for i, r in enumerate(res) if (r.num_buckets, r.epochs, r.relative_step_c) == (1024, 50, 0.1))
    res[slow] = dataclasses.replace(res[slow], validation_ap=0.70000000004)
    res[fast] = dataclasses.replace(res[fast], validation_ap=0.7)
    decision = select_candidate(res, arm=ARM_C)
    assert decision.selected.epochs == 50 and decision.tied_at_rounded_ap == 2
    assert {r["candidate_id"]: r["validation_ap"] for r in decision.ranking}[res[slow].candidate_id] == 0.70000000004  # not rounded away
    # a difference above the tenth decimal is a real difference
    res[slow] = dataclasses.replace(res[slow], validation_ap=0.7000000006)
    assert select_candidate(res, arm=ARM_C).selected.epochs == 400


def test_selection_is_input_order_independent():
    rng = random.Random(3)
    res = results_for(ARM_A, lambda nb, ep, c: round(rng.random(), 3))
    reference = select_candidate(res, arm=ARM_A)
    for _ in range(5):
        shuffled = res[:]
        rng.shuffle(shuffled)
        assert select_candidate(shuffled, arm=ARM_A) == reference


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, 1.5, True, "0.5"])
def test_selection_rejects_undefined_or_invalid_metrics(bad):
    res = results_for(ARM_A, lambda nb, ep, c: 0.5)
    res[3] = dataclasses.replace(res[3], validation_ap=bad)
    with pytest.raises(RunnerError):
        select_candidate(res, arm=ARM_A)


def test_selection_rejects_missing_duplicate_and_unexpected_candidates():
    res = results_for(ARM_A, lambda nb, ep, c: 0.5)
    with pytest.raises(RunnerError):
        select_candidate(res[:-1], arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(res + [res[0]], arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(res[:-1] + [dataclasses.replace(res[0], candidate_id="renamed")], arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(res[:-1] + [dataclasses.replace(res[-1], num_buckets=999)], arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(res[:-1] + [dataclasses.replace(res[-1], epochs=7)], arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(res[:-1] + [dataclasses.replace(res[-1], relative_step_c=0.5)], arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(res + [dataclasses.replace(res[0], candidate_id="extra")], arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(["not a result"] * 27, arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(res, arm="D_other")


@pytest.mark.parametrize("split", ["test", "train", "TEST", ""])
def test_selection_rejects_results_not_labelled_validation(split):
    res = results_for(ARM_A, lambda nb, ep, c: 0.5)
    res[5] = dataclasses.replace(res[5], split=split)
    with pytest.raises(RunnerError):
        select_candidate(res, arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(results_for(ARM_A, lambda nb, ep, c: 0.5, split=split), arm=ARM_A)


def test_selection_rejects_misaligned_results():
    res = results_for(ARM_A, lambda nb, ep, c: 0.5)
    res[0] = dataclasses.replace(res[0], validation_ids_sha256="x" * 64)
    with pytest.raises(RunnerError):
        select_candidate(res, arm=ARM_A)
    res = results_for(ARM_A, lambda nb, ep, c: 0.5)
    res[0] = dataclasses.replace(res[0], validation_records=99)
    with pytest.raises(RunnerError):
        select_candidate(res, arm=ARM_A)
    with pytest.raises(RunnerError):
        select_candidate(results_for(ARM_A, lambda nb, ep, c: 0.5), arm=ARM_A, validation_ids_sha256="z" * 64)
    mixed_arm = results_for(ARM_A, lambda nb, ep, c: 0.5)
    mixed_arm[0] = dataclasses.replace(mixed_arm[0], arm=ARM_B)
    with pytest.raises(RunnerError):
        select_candidate(mixed_arm, arm=ARM_A)
    ok = select_candidate(results_for(ARM_A, lambda nb, ep, c: 0.5, sha="a" * 64), arm=ARM_A, validation_ids_sha256="a" * 64)
    assert ok.selected.validation_ids_sha256 == "a" * 64


def test_selection_decision_serializes_and_states_validation_only_basis():
    payload = json.loads(json.dumps(select_candidate(results_for(ARM_C, lambda nb, ep, c: 0.3 + nb * 1e-6), arm=ARM_C).to_dict(),
                                    allow_nan=False))
    assert payload["basis"] == "validation_split_only" and payload["test_data_used"] is False
    assert payload["ap_round_decimals"] == 10 and payload["candidates"] == 27
    assert payload["selected_candidate_id"].startswith(ARM_C)


def test_selection_interface_cannot_receive_test_data():
    params = list(inspect.signature(select_candidate).parameters)
    assert params == ["results", "arm", "validation_ids_sha256"]
    assert "SealedTestSet" not in str(inspect.signature(select_candidate))
    for name in ("select_candidate", "aligned_labels", "approved_candidate_ids"):
        assert "sealed" not in " ".join(inspect.signature(getattr(rn, name)).parameters).lower()


def test_approved_candidate_ids_cover_the_grid_exactly():
    for arm in ARMS:
        ids = approved_candidate_ids(arm)
        assert len(ids) == len(set(ids)) == 27 and all(i.startswith(arm) for i in ids)
    with pytest.raises(RunnerError):
        approved_candidate_ids("D")
    assert rn.EPOCHS_GRID == (50, 150, 400) and rn.AP_ROUND_DECIMALS == 10 and rn.MIN_FREQUENCY == 2 and rn.MAX_LITERAL_VOCAB_SIZE is None


# -- G. unavailable gates -----------------------------------------------------------------------------------------------------------------------------


def test_training_is_unavailable_without_a_verified_manifest():
    assert rn.TRAINING_AUTHORIZATION_AVAILABLE is False
    with pytest.raises(AuthorizationUnavailable) as caught:
        request_training_authorization()
    assert caught.value.action == "training" and any("manifest" in m for m in caught.value.missing)


def test_test_access_is_unavailable_without_both_records_and_the_attempt_marker():
    assert rn.TEST_SCORING_AVAILABLE is False
    with pytest.raises(AuthorizationUnavailable) as caught:
        request_test_access()
    text = " ".join(caught.value.missing)
    assert "manifest" in text and "selection" in text and "attempt marker" in text


def test_no_caller_supplied_verifier_or_fake_can_unlock_anything(monkeypatch):
    class AlwaysYes:
        def __getattr__(self, name):
            return lambda *a, **k: True

    with pytest.raises(TypeError):
        request_test_access(AlwaysYes())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        request_test_access(manifest_verifier=AlwaysYes())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        request_training_authorization(AlwaysYes())  # type: ignore[call-arg]
    for name in ("verify_manifest", "verify_selection_record", "create_attempt_marker", "manifest_verified"):
        monkeypatch.setattr(rn, name, lambda *a, **k: True, raising=False)
    with pytest.raises(AuthorizationUnavailable):
        request_test_access()
    with pytest.raises(AuthorizationUnavailable):
        request_training_authorization()


def test_sealed_test_set_rejects_unblinded_records(dataset, part):
    unblinded = [r for r in dataset.records if r.record_id in set(part.sealed_test.record_ids)][:5]
    with pytest.raises(RunnerError):
        SealedTestSet([dataclasses.replace(r, meta=SplitMetadata(label=1)) for r in unblinded], {r.record_id: "g" for r in unblinded},
                      split_manifest_sha256="m")
    blinded = list(part.sealed_test.blinded_records())[:5]
    with pytest.raises(RunnerError):
        SealedTestSet(blinded, {blinded[0].record_id: "g"}, split_manifest_sha256="m")
    with pytest.raises(RunnerError):
        SealedTestSet(blinded, {r.record_id: "g" for r in blinded}, split_manifest_sha256=None)  # type: ignore[arg-type]


# -- scope and hygiene -----------------------------------------------------------------------------------------------------------------------------------


def test_runner_has_no_fitting_scoring_unsealing_or_test_encoding_code():
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"fit", "predict", "raw_logit", "raw_logits", "average_precision", "roc_auc", "select_threshold", "paired_group_bootstrap"}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "LightweightAnomalyModel" not in imported and "new_untrained_model" not in imported
    assert not any(seed in text for seed in ("7101", "7102", "7103"))
    assert not any(hasattr(rn, name) for name in ("encode_test", "score_test", "unseal", "unseal_test", "write_manifest", "write_selection_record"))
    assert "random" not in {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}


def test_runner_does_not_mutate_its_inputs(dataset, split):
    snapshot = copy.deepcopy([(r.record_id, r.features, r.meta) for r in dataset.records[:50]])
    partition_records(dataset.records, split)
    assert [(r.record_id, r.features, r.meta) for r in dataset.records[:50]] == snapshot


def test_source_sha256_is_a_stable_digest():
    assert len(rn.source_sha256()) == 64 and rn.source_sha256() == rn.source_sha256()
    assert rn.RUNNER_ID == "runner-foundation-1"


# -- verified training authorization (D4-7C2) -----------------------------------------------------------------------------------------------------
#
# These tests run the REAL manifest verifier on disposable fixtures: a throw-away Git repository with the complete default layout whose protocol header says
# FROZEN, development-seed (6101) structure, and a monkeypatched seed tuple that lets the development seed stand in for the confirmatory seed (so no
# confirmatory seed value appears here). Nothing is fitted or scored, no marker or artifact is created, and no test label is read.

REPO_ROOT = Path(mf.__file__).resolve().parents[2]
FROZEN = "# Protocol" + chr(10) * 2 + "**Protocol version:** `heldout-eval-test-frozen`  **Status: FROZEN**" + chr(10)
DRAFT = "# Protocol" + chr(10) * 2 + "**Protocol version:** `heldout-eval-test-draft`  **Status: DRAFT, not frozen.**" + chr(10)
AUDIT_OK = {"errors": [], "warnings": [], "construction_checks_passed": True}
CREATED = "2026-01-01T00:00:00Z"
LF, CRLF = bytes([10]), bytes([13, 10])


@pytest.fixture(scope="module", autouse=True)
def development_seed_stands_in():
    patch = pytest.MonkeyPatch()
    patch.setattr(mf, "CONFIRMATORY_SEEDS", (6101,))
    yield
    patch.undo()


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          cwd=str(root), capture_output=True, text=True, check=True)
    return done.stdout.strip()


def make_default_repo(root: Path, header: str) -> Path:
    root.mkdir(parents=True)
    for rel in (*mf.DEFAULT_SOURCE_FILES, *mf.DEFAULT_CONFIG_FILES):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel in ("tokenization_ai/heldout_eval/numerics.py", "tokenization_ai/model.py"):
            path.write_bytes((REPO_ROOT / rel).read_bytes().replace(CRLF, LF))  # the manifest cross-checks these two hashes with the live code
        else:
            path.write_text(f"# placeholder for {rel}{LF.decode()}", encoding="utf-8", newline=LF.decode())
    protocol = root / mf.DEFAULT_PROTOCOL_PATH
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(header, encoding="utf-8", newline=LF.decode())
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


class AuthEnv:
    pass


@pytest.fixture(scope="module")
def aenv(tmp_path_factory, dataset, split, part, cfg, numerical):
    e = AuthEnv()
    root = tmp_path_factory.mktemp("auth")
    e.out = root / "out"
    e.out.mkdir()
    e.repo = make_default_repo(root / "repo", FROZEN)
    e.pre = rn.preflight_report(cfg, part.validation, part.sealed_test)
    e.args = dict(dataset=dataset, split_result=split, partition=part, train_config=cfg, numerical=numerical)
    manifest = mf.build_manifest(purpose=mf.PURPOSE_CONFIRMATORY, created_utc=CREATED, repo_root=e.repo, dataset=dataset, split_result=split, partition=part,
                                 audit_report=AUDIT_OK, train_config=cfg, numerical=numerical, preflight=e.pre)
    e.expected = mf.ExpectedRun(purpose=mf.PURPOSE_CONFIRMATORY, branch="main", commit=git(e.repo, "rev-parse", "HEAD"), dataset_seed=6101,
                                generator_fingerprint=dataset.fingerprint, dataset_fingerprint=split.report.dataset_fingerprint,
                                split_seed=PRIMARY_SPLIT_SEED, split_manifest_sha256=split.manifest_sha256,
                                approved_protocol_sha256=manifest["protocol"]["sha256"])
    written = mf.write_manifest(e.out / "manifest.json", manifest, repo_root=e.repo, expected=e.expected)
    e.manifest_bytes, e.manifest_sha = Path(written.path).read_bytes(), written.sha256
    e.manifest = mf.parse_manifest_bytes(e.manifest_bytes)
    e.dataset, e.split, e.part, e.cfg, e.numerical = dataset, split, part, cfg, numerical
    e.files_before = sorted(p.name for p in e.out.iterdir())
    return e


def authorize(e, **over):
    kwargs = dict(manifest_bytes=e.manifest_bytes, expected_manifest_sha256=e.manifest_sha, repo_root=e.repo, expected_run=e.expected, **e.args)
    kwargs.update(over)
    return rn.authorize_training(**kwargs)


def rejected(e, fragment: str | None = None, **over):
    with pytest.raises(rn.TrainingAuthorizationError) as caught:
        authorize(e, **over)
    if fragment:
        assert fragment in str(caught.value)
    return caught.value


def forged(e, fn):
    manifest = copy.deepcopy(e.manifest)
    fn(manifest)
    data = mf.canonical_bytes(manifest)
    return dict(manifest_bytes=data, expected_manifest_sha256=mf.sha256_hex(data))


def variant_manifest_bytes(e, **over):
    manifest = mf.build_manifest(purpose=mf.PURPOSE_CONFIRMATORY, created_utc=CREATED, repo_root=e.repo, dataset=e.dataset, split_result=e.split,
                                 partition=e.part, audit_report=AUDIT_OK, train_config=e.cfg, numerical=e.numerical, preflight=e.pre, **over)
    data = mf.canonical_bytes(manifest)
    return dict(manifest_bytes=data, expected_manifest_sha256=mf.sha256_hex(data))


def test_a_fully_verified_frozen_fixture_is_authorized_and_binds_only_hashes(aenv):
    auth = authorize(aenv)
    assert type(auth) is rn.TrainingAuthorization and rn.require_training_authorization(auth) is auth
    assert auth.purpose == "confirmatory" and auth.manifest_sha256 == aenv.manifest_sha and auth.git_commit == aenv.expected.commit
    assert auth.git_branch == "main" and auth.protocol_sha256 == aenv.expected.approved_protocol_sha256 and auth.protocol_version == "heldout-eval-test-frozen"
    assert auth.dataset_fingerprint == aenv.split.report.dataset_fingerprint and auth.generator_fingerprint == aenv.dataset.fingerprint
    assert auth.split_manifest_sha256 == aenv.split.manifest_sha256 and auth.split_seed == PRIMARY_SPLIT_SEED and auth.dataset_seed == 6101
    assert auth.train_ids_sha256 == rn.ids_fingerprint(aenv.part.train.record_ids) and auth.test_ids_sha256 == aenv.part.sealed_test.binding()["ids_sha256"]
    assert auth.vocabulary_sha256 == aenv.cfg.frozen.record["sha256"] and auth.max_length == aenv.cfg.decision.selected_max_length
    assert auth.model_source_sha256 == aenv.manifest["numerics"]["provenance"]["model_sha256"] and len(auth.sources_sha256) == 64
    assert len(auth.numerics_identity_sha256) == len(auth.candidate_grid_sha256) == 64
    again = authorize(aenv)
    assert again.to_dict() == auth.to_dict()  # deterministic


def test_the_authorization_serializes_without_labels_scores_or_request_content(aenv):
    auth = authorize(aenv)
    payload = auth.to_dict()
    json.dumps(payload, allow_nan=False)
    problems: list[str] = []
    mf._scan_forbidden(payload, "$", problems)
    assert problems == [] and payload["grants_test_access"] is False and payload["fits_models"] is False
    text = json.dumps(payload)
    for rec in list(aenv.part.train.records[:100]) + list(aenv.part.validation.records[:100]):
        assert rec.features.path not in text
    assert not any("label" in key or "score" in key or "prediction" in key for key in payload)


def test_authorization_does_not_imply_test_access_and_creates_nothing(aenv, monkeypatch, tmp_path):
    import tokenization_ai.heldout_eval.attempt_gate as gate
    from tokenization_ai.heldout_eval import numerics
    from tokenization_ai.model import LightweightAnomalyModel

    def forbidden(*args, **kwargs):
        raise AssertionError("authorization must not fit, score, read out or create a marker")

    for target, name in ((LightweightAnomalyModel, "fit"), (LightweightAnomalyModel, "predict"), (numerics, "raw_logits"), (gate, "create_attempt_marker"),
                         (gate, "recover_test_labels"), (gate, "open_test_attempt"), (rn, "select_candidate")):
        monkeypatch.setattr(target, name, forbidden)
    before = {p for p in tmp_path.rglob("*")}
    auth = authorize(aenv)
    assert auth.grants_test_access is False and not hasattr(auth, "labels") and not hasattr(auth, "test_labels")
    assert sorted(p.name for p in aenv.out.iterdir()) == aenv.files_before and {p for p in tmp_path.rglob("*")} == before
    assert not list(aenv.repo.glob("**/test-attempt-*")) and not list(aenv.out.glob("*attempt*")) and not list(aenv.out.glob("*artifact*"))
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_test_access()
    with pytest.raises(TypeError):
        rn.request_test_access(auth)  # type: ignore[call-arg]
    assert rn.TEST_SCORING_AVAILABLE is False
    assert set(aenv.part.sealed_test.__class__.__dict__) .isdisjoint({"labels", "release_labels", "unseal"})


def test_the_argument_free_request_still_never_authorizes():
    assert rn.TRAINING_AUTHORIZATION_AVAILABLE is False and rn.VERIFIED_TRAINING_AUTHORIZATION_PATH == "authorize_training"
    with pytest.raises(rn.AuthorizationUnavailable) as caught:
        rn.request_training_authorization()
    assert "authorize_training" in " ".join(caught.value.missing)


# -- what the caller must supply, and what it cannot change -------------------------------------------------------------------------------------------


def test_the_interface_accepts_no_overrides_or_bypass_flags(aenv):
    parameters = inspect.signature(rn.authorize_training).parameters.values()
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    names = {p.name for p in inspect.signature(rn.authorize_training).parameters.values()}
    assert not names & {"source_files", "config_files", "protocol_path", "allow_environment_mismatch", "verified", "trust", "skip_checks", "callback", "verifier"}
    for override in (dict(source_files=("tokenization_ai/model.py",)), dict(config_files=()), dict(protocol_path="x.md"), dict(allow_environment_mismatch=True),
                     dict(verified=True), dict(require_frozen=False)):
        with pytest.raises(TypeError):
            authorize(aenv, **override)


def test_the_expected_run_must_be_confirmatory_and_describe_the_supplied_run(aenv, dataset, split):
    rejected(aenv, "confirmatory", expected_run=dataclasses.replace(aenv.expected, purpose=mf.PURPOSE_INFRASTRUCTURE, approved_protocol_sha256=None))
    rejected(aenv, "confirmatory", expected_run="trusted")
    rejected(aenv, "does not describe", expected_run=dataclasses.replace(aenv.expected, dataset_fingerprint="1" * 64))
    rejected(aenv, "does not describe", expected_run=dataclasses.replace(aenv.expected, generator_fingerprint="2" * 64))
    rejected(aenv, "does not describe", expected_run=dataclasses.replace(aenv.expected, split_manifest_sha256="3" * 64))
    rejected(aenv, "does not describe", expected_run=dataclasses.replace(aenv.expected, split_seed=1))
    rejected(aenv, "does not describe", expected_run=dataclasses.replace(aenv.expected, dataset_seed=6102))


def test_the_real_draft_protocol_and_the_real_repository_cannot_authorize_training(aenv):
    assert mf.collect_protocol(REPO_ROOT)["status"] == "DRAFT"
    rejected(aenv, "did not verify", repo_root=REPO_ROOT)


def test_a_draft_protocol_or_an_infrastructure_manifest_is_rejected(aenv, tmp_path, monkeypatch, dataset, split, part, cfg, numerical):
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (1,))  # build a genuine infrastructure manifest on a draft protocol
    draft_repo = make_default_repo(tmp_path / "draft", DRAFT)
    manifest = mf.build_manifest(purpose=mf.PURPOSE_INFRASTRUCTURE, created_utc=CREATED, repo_root=draft_repo, dataset=dataset, split_result=split,
                                 partition=part, audit_report=AUDIT_OK, train_config=cfg, numerical=numerical, preflight=aenv.pre)
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (6101,))
    data = mf.canonical_bytes(manifest)
    sha = mf.sha256_hex(data)
    infra = mf.ExpectedRun(purpose=mf.PURPOSE_INFRASTRUCTURE, branch="main", commit=git(draft_repo, "rev-parse", "HEAD"), dataset_seed=6101,
                           generator_fingerprint=dataset.fingerprint, dataset_fingerprint=split.report.dataset_fingerprint, split_seed=PRIMARY_SPLIT_SEED,
                           split_manifest_sha256=split.manifest_sha256)
    rejected(aenv, "confirmatory", manifest_bytes=data, expected_manifest_sha256=sha, repo_root=draft_repo, expected_run=infra)
    claimed = dataclasses.replace(infra, purpose=mf.PURPOSE_CONFIRMATORY, approved_protocol_sha256=manifest["protocol"]["sha256"])
    rejected(aenv, "did not verify", manifest_bytes=data, expected_manifest_sha256=sha, repo_root=draft_repo, expected_run=claimed)


def test_wrong_digests_and_protocol_approval_are_rejected(aenv):
    rejected(aenv, "did not verify", expected_manifest_sha256="0" * 64)
    rejected(aenv, "did not verify", expected_manifest_sha256="short")
    rejected(aenv, "did not verify", expected_run=dataclasses.replace(aenv.expected, approved_protocol_sha256="0" * 64))
    rejected(aenv, "did not verify", expected_run=dataclasses.replace(aenv.expected, approved_protocol_sha256=None))


def test_wrong_branch_or_commit_is_rejected(aenv):
    rejected(aenv, "did not verify", expected_run=dataclasses.replace(aenv.expected, branch="feature/other"))
    rejected(aenv, "did not verify", expected_run=dataclasses.replace(aenv.expected, commit="1" * 40))


def test_a_dirty_working_tree_is_rejected(aenv):
    stray = aenv.repo / "stray.txt"
    stray.write_text("x", encoding="utf-8")
    try:
        rejected(aenv, "did not verify")
    finally:
        stray.unlink()
    assert authorize(aenv)


def test_a_stale_manifest_after_the_repository_moves_on_is_rejected(aenv):
    (aenv.repo / "later.md").write_text("later", encoding="utf-8")
    git(aenv.repo, "add", "-A")
    git(aenv.repo, "commit", "-q", "-m", "later")
    try:
        rejected(aenv, "did not verify")
        rejected(aenv, "did not verify", expected_run=dataclasses.replace(aenv.expected, commit=git(aenv.repo, "rev-parse", "HEAD")))
    finally:
        git(aenv.repo, "reset", "-q", "--hard", aenv.expected.commit)
    assert authorize(aenv)


def test_a_missing_required_source_file_is_rejected(aenv):
    victim = "tokenization_ai/heldout_eval/arm_c.py"
    git(aenv.repo, "rm", "-q", victim)
    git(aenv.repo, "commit", "-q", "-m", "remove")
    try:
        rejected(aenv, "did not verify", expected_run=dataclasses.replace(aenv.expected, commit=git(aenv.repo, "rev-parse", "HEAD")))
    finally:
        git(aenv.repo, "reset", "-q", "--hard", aenv.expected.commit)
    assert (aenv.repo / victim).is_file()


@pytest.mark.parametrize("rel", ["tokenization_ai/heldout_eval/selection_record.py", "tokenization_ai/heldout_eval/attempt_gate.py",
                                 "tokenization_ai/heldout_eval/model_artifacts.py", "tokenization_ai/heldout_eval/runner.py"])
def test_an_omitted_or_substituted_source_entry_is_rejected(aenv, rel):
    rejected(aenv, "did not verify", **forged(aenv, lambda m: m["sources"].pop(rel)))
    rejected(aenv, "did not verify", **forged(aenv, lambda m: m["sources"].update({rel: "f" * 64})))


def test_a_partial_source_list_with_valid_hashes_is_rejected(aenv):
    partial = tuple(rel for rel in mf.REQUIRED_SOURCE_FILES if rel != "tokenization_ai/heldout_eval/attempt_gate.py")
    rejected(aenv, "did not verify", **variant_manifest_bytes(aenv, source_files=partial))


def test_the_runners_own_coverage_check_rejects_a_partial_manifest_if_the_verifier_were_fooled(aenv, monkeypatch):
    partial = tuple(rel for rel in mf.REQUIRED_SOURCE_FILES if rel not in ("tokenization_ai/heldout_eval/attempt_gate.py",))
    data = variant_manifest_bytes(aenv, source_files=partial)
    lying = mf.VerificationResult(ok=True, manifest_sha256="0" * 64, purpose=mf.PURPOSE_CONFIRMATORY, protocol_status="FROZEN", environment_differences=(),
                                  checks=(), source_list_is_default=True, missing_required_sources=())
    monkeypatch.setattr(mf, "verify_manifest_bytes", lambda *a, **k: lying)
    rejected(aenv, "source coverage", **data)


def test_an_environment_mismatch_is_rejected(aenv):
    rejected(aenv, "EnvironmentMismatchError", **forged(aenv, lambda m: m["environment"]["python"].update(version="0.0.1")))
    rejected(aenv, "EnvironmentMismatchError", **forged(aenv, lambda m: m["environment"]["numpy"].update(version="0.0.1")))


@pytest.mark.parametrize("raw", [b"", b"[]", b"not json", b'{"a":1,"a":2}' + LF, b'{"a": 1}' + LF])
def test_malformed_or_noncanonical_manifests_are_rejected_even_with_a_matching_digest(aenv, raw):
    rejected(aenv, "did not verify", manifest_bytes=raw, expected_manifest_sha256=mf.sha256_hex(raw))


def test_pretty_printed_truncated_and_tampered_manifests_are_rejected(aenv):
    pretty = json.dumps(aenv.manifest, indent=2).encode()
    rejected(aenv, "did not verify", manifest_bytes=pretty, expected_manifest_sha256=mf.sha256_hex(pretty))
    half = aenv.manifest_bytes[: len(aenv.manifest_bytes) // 2]
    rejected(aenv, "did not verify", manifest_bytes=half, expected_manifest_sha256=mf.sha256_hex(half))
    flipped = bytearray(aenv.manifest_bytes)
    flipped[len(flipped) // 2] ^= 1
    rejected(aenv, "did not verify", manifest_bytes=bytes(flipped))
    rejected(aenv, "exact bytes", manifest_bytes=json.loads(aenv.manifest_bytes))


@pytest.mark.parametrize("mutation", [lambda m: m["numerics"]["estimates"][0].update(lambda_max=-1.0, L=-0.25),
                                      lambda m: m["numerics"]["candidates"].pop(),
                                      lambda m: m["numerics"]["learning_rates"]["cells"][0]["rates"][0].update(learning_rate=0.5),
                                      lambda m: m["arms"]["vocabulary"].update(min_frequency=1),
                                      lambda m: m["arms"]["preflight"].update(passed=False),
                                      lambda m: m.update(source_coverage="complete")])
def test_a_manifest_with_invalid_numerics_vocabulary_or_forged_metadata_is_rejected(aenv, mutation):
    rejected(aenv, "did not verify", **forged(aenv, mutation))


# -- binding the verified manifest to the prepared run ----------------------------------------------------------------------------------------------------


def test_a_different_split_or_partition_does_not_match_the_manifest(aenv, dataset):
    other_split = split_records(dataset.records, seed=PRIMARY_SPLIT_SEED + 1)
    other_part = partition_records(dataset.records, other_split)
    rejected(aenv, "does not describe", split_result=other_split)
    rejected(aenv, "does not match the prepared run", partition=other_part)


def test_wrong_vocabulary_budget_or_training_ids_do_not_match_the_manifest(aenv, cfg):
    bad_vocab = dataclasses.replace(cfg, frozen=dataclasses.replace(cfg.frozen, record={**cfg.frozen.record, "sha256": "0" * 64}))
    rejected(aenv, "train-only vocabulary", train_config=bad_vocab)
    bad_budget = dataclasses.replace(cfg, decision=dataclasses.replace(cfg.decision, selected_max_length=256))
    rejected(aenv, "shared token budget", train_config=bad_budget)


def test_a_wrong_numerical_configuration_or_candidate_grid_does_not_match_the_manifest(aenv, numerical):
    rejected(aenv, "numerical configuration", numerical=dataclasses.replace(numerical, estimates=numerical.estimates[:-1]))
    changed = dataclasses.replace(numerical.candidates[0], learning_rate=numerical.candidates[0].learning_rate * 2)
    rejected(aenv, "candidate grid", numerical=dataclasses.replace(numerical, candidates=(changed, *numerical.candidates[1:])))
    rejected(aenv, "candidate grid", numerical=dataclasses.replace(numerical, candidates=numerical.candidates[:-1]))
    extra = dataclasses.replace(numerical.candidates[0], candidate_id="extra")
    rejected(aenv, "candidate grid", numerical=dataclasses.replace(numerical, candidates=(*numerical.candidates, extra)))


def test_the_preflight_is_recomputed_and_must_match_the_manifest(aenv, monkeypatch):
    monkeypatch.setattr(rn, "preflight_report", lambda *a, **k: {"passed": True})
    rejected(aenv, "representability preflight")

    def failing(*args, **kwargs):
        report = rn.preflight_split(aenv.part.validation.records, "validation", dataclasses.replace(aenv.cfg.decision, selected_max_length=60))
        report.require_pass()

    monkeypatch.setattr(rn, "preflight_report", failing)
    rejected(aenv, "representability preflight")


def test_objects_of_the_wrong_type_are_rejected(aenv):
    for over in (dict(dataset=object()), dict(split_result="split"), dict(partition=None), dict(train_config={}), dict(numerical=[])):
        rejected(aenv, "runner's own objects", **over)


# -- the context cannot be forged, modified or reused as something else ---------------------------------------------------------------------------------------


def valid_fields(auth):
    return {name: getattr(auth, name) for name in rn._AUTH_FIELDS}


def test_the_authorization_cannot_be_constructed_without_verification(aenv):
    auth = authorize(aenv)
    with pytest.raises(rn.TrainingAuthorizationError):
        rn.TrainingAuthorization(object(), **valid_fields(auth))
    with pytest.raises(rn.TrainingAuthorizationError):
        rn.TrainingAuthorization(None, **valid_fields(auth))
    with pytest.raises(rn.TrainingAuthorizationError):
        rn.TrainingAuthorization(rn._AUTH_KEY, **{**valid_fields(auth), "extra": 1})
    with pytest.raises(rn.TrainingAuthorizationError):
        rn.TrainingAuthorization(rn._AUTH_KEY, manifest_sha256="a" * 64)


def test_the_authorization_is_immutable_and_tampering_is_detected(aenv):
    auth = authorize(aenv)
    with pytest.raises(AttributeError):
        auth.git_commit = "0" * 40  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del auth.manifest_sha256
    with pytest.raises(AttributeError):
        auth.extra = 1  # type: ignore[attr-defined]
    object.__setattr__(auth, "git_commit", "0" * 40)  # deliberate bypass of the guard, as malicious in-process code could
    with pytest.raises(rn.TrainingAuthorizationError, match="modified"):
        rn.require_training_authorization(auth)


def test_only_a_genuine_authorization_satisfies_the_training_stage_check(aenv):
    auth = authorize(aenv)

    class Imposter(rn.TrainingAuthorization):
        pass

    class Duck:
        def __getattr__(self, name):
            return "x"

    for bad in (None, "authorized", {"verified": True}, Duck(), object(), 1, aenv.manifest):
        with pytest.raises(rn.TrainingAuthorizationError):
            rn.require_training_authorization(bad)
    with pytest.raises(rn.TrainingAuthorizationError):
        rn.require_training_authorization(Imposter(rn._AUTH_KEY, **valid_fields(auth)))  # a subclass instance is not an authorization
    assert rn.require_training_authorization(auth) is auth


def test_the_authorization_is_not_picklable_or_copyable(aenv):
    auth = authorize(aenv)
    for operation in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            operation(auth)
    assert "..." in repr(auth) and auth.git_commit not in repr(auth)


def test_errors_do_not_echo_manifest_values_or_secrets(aenv):
    secret = "SECRET-VALUE-424242"
    errors = [rejected(aenv, expected_manifest_sha256=secret), rejected(aenv, **forged(aenv, lambda m: m["git"].update(branch=secret))),
              rejected(aenv, **forged(aenv, lambda m: m["data"].update(dataset_fingerprint=secret))),
              rejected(aenv, manifest_bytes=secret.encode(), expected_manifest_sha256=mf.sha256_hex(secret.encode()))]
    assert all(secret not in str(e) for e in errors)


def test_the_authorization_code_never_fits_scores_or_unseals():
    tree = ast.parse(Path(rn.__file__).read_text(encoding="utf-8"))
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "authorize_training")
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(func) if isinstance(n, ast.Call)}
    assert not called & {"fit", "predict", "raw_logit", "raw_logits", "average_precision", "select_threshold", "select_candidate", "create_attempt_marker",
                         "recover_test_labels", "open_test_attempt", "write_model_artifact", "write_selection_record", "request_test_access"}
    assert "7101" not in Path(rn.__file__).read_text(encoding="utf-8")
