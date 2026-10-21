"""Tests for training-stage-1 (draft protocol heldout-eval-1.2.0-draft, section 12.12.9).

DEVELOPMENT-ONLY. A small development-seed (6101) dataset, a throw-away FROZEN fixture repository and a monkeypatched confirmatory-seed tuple that lets the
development seed stand in for the confirmatory seed. The 81 candidate fits here exercise the orchestration only; no metric is reported or asserted beyond
internal consistency, and no confirmatory seed, dataset, manifest, record or marker is created.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import os
import pickle
import stat
from pathlib import Path

import numpy as np
import pytest

import _heldout_chain as hc
import tokenization_ai.heldout_eval.manifest as mf
import tokenization_ai.heldout_eval.model_artifacts as ma
import tokenization_ai.heldout_eval.numerics as numerics
import tokenization_ai.heldout_eval.runner as rn
import tokenization_ai.heldout_eval.selection_record as sr
import tokenization_ai.heldout_eval.training as tr
from tokenization_ai.heldout_eval.arms import ARM_A, ARM_B, ARM_C, ARMS
from tokenization_ai.heldout_eval.splits import PRIMARY_SPLIT_SEED, EvalRecord, SplitMetadata, split_records
from tokenization_ai.model import LightweightAnomalyModel

GRID = [(nb, ep, c) for nb in (1024, 2048, 4096) for ep in (50, 150, 400) for c in (0.1, 0.3, 0.8)]


@pytest.fixture(scope="module", autouse=True)
def development_seed_stands_in():
    patch = pytest.MonkeyPatch()
    patch.setattr(mf, "CONFIRMATORY_SEEDS", (hc.DEV_SEED,))
    yield
    patch.undo()


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    c = hc.build_chain(tmp_path_factory.mktemp("training"))
    c.plan = tr.build_training_plan(c.auth, **hc.plan_args(c))
    c.artifact_dir = c.root / "artifacts"
    c.artifact_dir.mkdir()
    c.trained = tr.train_candidates(c.plan, artifact_dir=c.artifact_dir, repo_root=c.repo)
    c.records_dir = c.root / "records"
    c.records_dir.mkdir()
    c.selection = tr.select_on_validation(c.plan, c.trained, validation=c.part.validation, manifest_bytes=c.manifest_bytes,
                                          expected_manifest_sha256=c.manifest_sha, repo_root=c.repo, expected_run=c.expected,
                                          record_path=c.records_dir / "selection.json", created_utc=hc.CREATED)
    return c


_UNSET = object()


def plan_error(c, fragment=None, auth=_UNSET, **over):
    with pytest.raises(tr.TrainingStageError) as caught:
        tr.build_training_plan(c.auth if auth is _UNSET else auth, **hc.plan_args(c, **over))
    if fragment:
        assert fragment in str(caught.value), str(caught.value)
    return caught.value


# -- D4-7D1: the plan ---------------------------------------------------------------------------------------------------------------------------


def test_the_plan_holds_exactly_the_81_approved_candidates_in_a_deterministic_order(chain):
    plan = chain.plan
    assert len(plan.candidates) == 81 and [c.index for c in plan.candidates] == list(range(81))
    assert [c.candidate_id for c in plan.candidates] == sorted(c.candidate_id for c in plan.candidates)
    for arm in ARMS:
        cells = {(c.num_buckets, c.epochs, c.relative_step_c) for c in plan.candidates if c.arm == arm}
        assert cells == set(GRID)
    assert {c.candidate_id for c in plan.candidates} == {cid for arm in ARMS for cid in rn.approved_candidate_ids(arm)}


def test_the_plan_uses_the_authorized_train_only_learning_rates_and_epochs(chain):
    manifest_rates = {x["candidate_id"]: x for x in chain.manifest["numerics"]["candidates"]}
    estimates = {(e["arm"], e["num_buckets"]): e for e in chain.manifest["numerics"]["estimates"]}
    for c in chain.plan.candidates:
        est = estimates[(c.arm, c.num_buckets)]
        assert c.learning_rate == manifest_rates[c.candidate_id]["learning_rate"] == numerics.relative_learning_rate(c.relative_step_c, est["L"])
        assert c.L == est["L"] == c.lambda_max / 4 and c.lambda_method == "lambda-power-1" and c.epochs in (50, 150, 400)


def test_the_plan_is_deterministic_and_bound_to_the_authorization(chain):
    again = tr.build_training_plan(chain.auth, **hc.plan_args(chain))
    assert again.plan_sha256 == chain.plan.plan_sha256 and again.to_dict() == chain.plan.to_dict()
    assert chain.plan.authorization is chain.auth and chain.plan.to_dict()["manifest_sha256"] == chain.manifest_sha
    json.dumps(chain.plan.to_dict(), allow_nan=False)


def test_the_plan_holds_training_labels_only_aligned_by_record_id(chain):
    plan = chain.plan
    by_id = chain.part.train.labels_by_id()
    assert plan.train_record_ids == tuple(sorted(by_id)) and plan.train_labels == tuple(by_id[i] for i in plan.train_record_ids)
    assert set(plan.train_record_ids) == set(chain.split.members("train"))
    assert not set(plan.train_record_ids) & set(plan.validation_ids) and not set(plan.train_record_ids) & set(plan.test_ids)
    assert "validation_labels" not in plan.__slots__ and "test_labels" not in plan.__slots__ and len(plan.validation_labels_sha256) == 64
    assert "label" not in repr(plan)


@pytest.mark.parametrize("bad", [None, "authorized", {"verified": True}, object()])
def test_a_missing_or_forged_authorization_is_rejected(chain, bad):
    plan_error(chain, "genuine", auth=bad)


def test_a_modified_authorization_is_rejected(chain):
    auth = hc.authorize(chain)
    object.__setattr__(auth, "max_length", 999)  # deliberate in-process tampering
    plan_error(chain, "genuine", auth=auth)


def test_an_authorization_for_another_split_is_rejected(chain):
    other = split_records(chain.dataset.records, seed=PRIMARY_SPLIT_SEED + 1)
    other_part = rn.partition_records(chain.dataset.records, other)
    plan_error(chain, "does not match", split_result=other)
    plan_error(chain, "does not match", partition=other_part)


def test_substituted_request_content_is_rejected(chain):
    train = list(chain.part.train.records)
    swapped = [EvalRecord(train[0].record_id, train[1].features, train[0].meta), *train[1:]]
    part = dataclasses.replace(chain.part, train=rn.TrainData(tuple(swapped)))
    plan_error(chain, "request content", partition=part)


def test_a_changed_training_or_validation_label_is_rejected(chain):
    train = list(chain.part.train.records)
    flipped = [EvalRecord(train[0].record_id, train[0].features, SplitMetadata(label=1 - train[0].meta.label)), *train[1:]]
    plan_error(chain, "labels differ", partition=dataclasses.replace(chain.part, train=rn.TrainData(tuple(flipped))))
    val = list(chain.part.validation.records)
    vflipped = [EvalRecord(val[0].record_id, val[0].features, SplitMetadata(label=1 - val[0].meta.label)), *val[1:]]
    plan_error(chain, "labels differ", partition=dataclasses.replace(chain.part, validation=rn.ValidationData(tuple(vflipped))))


def test_overlapping_or_missing_split_records_are_rejected(chain):
    leaked = rn.ValidationData(tuple(sorted([*chain.part.validation.records, chain.part.train.records[0]], key=lambda r: r.record_id)))
    plan_error(chain, "more than one split", partition=dataclasses.replace(chain.part, validation=leaked))
    fewer = rn.TrainData(chain.part.train.records[1:])
    plan_error(chain, "membership", partition=dataclasses.replace(chain.part, train=fewer))


def test_a_tampered_split_result_or_group_leakage_is_rejected(chain):
    rid = chain.split.members("train")[0]
    moved = dict(chain.split.assignments)
    moved[rid] = "validation"
    plan_error(chain, "split result", split_result=dataclasses.replace(chain.split, assignments=moved))
    val_group = chain.split.group_of[chain.split.members("validation")[0]]
    regrouped = dict(chain.split.group_of)
    regrouped[rid] = val_group
    plan_error(chain, "split", split_result=dataclasses.replace(chain.split, group_of=regrouped))


def test_a_different_dataset_is_rejected(chain):
    plan_error(chain, "generator output", dataset=dataclasses.replace(chain.dataset, fingerprint="0" * 64))


def test_tampered_training_inputs_vocabulary_or_budget_are_rejected(chain):
    cfg = chain.cfg
    inputs = cfg.inputs
    reversed_inputs = rn.ArmInputs(split="train", record_ids=tuple(reversed(inputs.record_ids)), arm_a=tuple(reversed(inputs.arm_a)),
                                   arm_b=tuple(reversed(inputs.arm_b)), arm_c=tuple(reversed(inputs.arm_c)), max_length=inputs.max_length,
                                   outcomes_a=tuple(reversed(inputs.outcomes_a)), outcomes_b=tuple(reversed(inputs.outcomes_b)))
    plan_error(chain, "training configuration", train_config=dataclasses.replace(cfg, inputs=reversed_inputs))
    plan_error(chain, "training configuration", train_config=dataclasses.replace(cfg, decision=dataclasses.replace(cfg.decision, selected_max_length=256)))
    plan_error(chain, "training configuration",
               train_config=dataclasses.replace(cfg, frozen=dataclasses.replace(cfg.frozen, record={**cfg.frozen.record, "sha256": "0" * 64})))
    altered = list(inputs.arm_b)
    altered[0] = altered[0].model_copy(update={"input_ids": [5, *altered[0].input_ids[1:]]})
    plan_error(chain, "training configuration", train_config=dataclasses.replace(cfg, inputs=dataclasses.replace(inputs, arm_b=tuple(altered))))


def test_a_tampered_numerical_configuration_or_grid_is_rejected(chain):
    num = chain.numerical
    plan_error(chain, "numerical", numerical=dataclasses.replace(num, estimates=num.estimates[:-1]))
    changed = dataclasses.replace(num.candidates[0], learning_rate=num.candidates[0].learning_rate * 2)
    plan_error(chain, "numerical", numerical=dataclasses.replace(num, candidates=(changed, *num.candidates[1:])))
    plan_error(chain, "candidate grid", numerical=dataclasses.replace(num, candidates=num.candidates[:-1]))


def test_objects_of_the_wrong_type_are_rejected(chain):
    for over in (dict(dataset=object()), dict(split_result=None), dict(partition="p"), dict(train_config={}), dict(numerical=[])):
        plan_error(chain, "runner's own objects", **over)


def test_the_plan_cannot_be_forged_modified_or_serialized(chain):
    with pytest.raises(tr.TrainingStageError):
        tr.TrainingPlan(object())
    with pytest.raises(AttributeError):
        chain.plan.train_labels = ()  # type: ignore[misc]
    for op in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            op(chain.plan)


# -- D4-7D2: training ---------------------------------------------------------------------------------------------------------------------------


def test_training_persists_81_distinct_verified_artifacts_and_a_ledger(chain):
    trained = chain.trained
    assert len(trained.digests) == 81 and len(set(trained.digests.values())) == 81 and set(trained.digests) == {c.candidate_id for c in chain.plan.candidates}
    names = ma.expected_artifact_names(chain.manifest_sha)
    for cid, digest in trained.digests.items():
        path = Path(trained.artifact_dir) / names[cid]
        assert path.is_file() and mf.sha256_hex(path.read_bytes()) == digest and not (path.stat().st_mode & stat.S_IWRITE)
    ledger_path = Path(trained.ledger_path)
    ledger = json.loads(ledger_path.read_bytes())
    assert mf.sha256_hex(ledger_path.read_bytes()) == trained.ledger_sha256 and not (ledger_path.stat().st_mode & stat.S_IWRITE)
    assert ledger["ledger_version"] == "training-ledger-1" and ledger["plan_sha256"] == chain.plan.plan_sha256 and ledger["labels_recorded"] is False
    assert [row["candidate_id"] for row in ledger["candidates"]] == [c.candidate_id for c in chain.plan.candidates]
    assert {row["candidate_id"]: row["artifact_sha256"] for row in ledger["candidates"]} == dict(trained.digests)
    problems: list[str] = []
    mf._scan_forbidden(ledger, "$", problems)
    assert problems == []


def test_every_artifact_reloads_into_the_trained_parameters_and_readout(chain):
    loaded = ma.load_candidate_set(chain.trained.artifact_dir, expected_manifest_sha256=chain.manifest_sha, expected_digests=dict(chain.trained.digests),
                                   expected_learning_rates={c.candidate_id: c.learning_rate for c in chain.plan.candidates},
                                   expected_model_sha256=chain.auth.model_source_sha256)
    assert len(loaded) == 81
    for c in chain.plan.candidates[:6]:
        art = loaded[c.candidate_id]
        assert art.metadata["training"]["train_ids_sha256"] == chain.auth.train_ids_sha256 and art.model.num_buckets == c.num_buckets
        assert art.fingerprint["weights_sha256"] == chain.trained.weights_sha256[c.candidate_id]


def test_training_uses_only_training_records_training_labels_and_the_approved_settings(chain, tmp_path, monkeypatch):
    calls = []
    real_fit = LightweightAnomalyModel.fit

    def spy(self, inputs, labels, *, epochs, learning_rate=0.1):
        calls.append(([m.event_id for m in inputs], list(labels), epochs, learning_rate, self.num_buckets, self.is_trained))
        return real_fit(self, inputs, labels, epochs=epochs, learning_rate=learning_rate)

    monkeypatch.setattr(LightweightAnomalyModel, "fit", spy)
    directory = tmp_path / "again"
    directory.mkdir()
    again = tr.train_candidates(chain.plan, artifact_dir=directory, repo_root=chain.repo)
    assert len(calls) == 81
    train_ids = list(chain.plan.train_record_ids)
    for (ids, labels, epochs, lr, nb, already_trained), cand in zip(calls, chain.plan.candidates):
        assert ids == train_ids and labels == list(chain.plan.train_labels)  # training records and training labels only, in plan order
        assert (epochs, lr, nb) == (cand.epochs, cand.learning_rate, cand.num_buckets) and already_trained is False  # a fresh model each time
    assert dict(again.digests) == dict(chain.trained.digests)  # training is deterministic: identical artifacts, byte for byte


def test_training_never_overwrites_or_silently_retries(chain):
    before = {p.name: p.read_bytes() for p in Path(chain.trained.artifact_dir).iterdir()}
    with pytest.raises(tr.TrainingStageError, match="already exist"):
        tr.train_candidates(chain.plan, artifact_dir=chain.trained.artifact_dir, repo_root=chain.repo)
    assert {p.name: p.read_bytes() for p in Path(chain.trained.artifact_dir).iterdir()} == before


def test_a_failure_part_way_stops_the_stage_and_leaves_partial_artifacts_for_review(chain, tmp_path, monkeypatch):
    real_fit = LightweightAnomalyModel.fit
    count = {"n": 0}

    def flaky(self, inputs, labels, *, epochs, learning_rate=0.1):
        count["n"] += 1
        if count["n"] == 5:
            raise ValueError("simulated failure")
        return real_fit(self, inputs, labels, epochs=epochs, learning_rate=learning_rate)

    monkeypatch.setattr(LightweightAnomalyModel, "fit", flaky)
    directory = tmp_path / "partial"
    directory.mkdir()
    with pytest.raises(tr.TrainingStageError, match="candidate 4"):
        tr.train_candidates(chain.plan, artifact_dir=directory, repo_root=chain.repo)
    assert count["n"] == 5 and len(list(directory.iterdir())) == 4 and not (directory / tr.ledger_filename(chain.manifest_sha)).exists()
    monkeypatch.undo()
    with pytest.raises(tr.TrainingStageError, match="already exist"):  # the partial run blocks a silent retry
        tr.train_candidates(chain.plan, artifact_dir=directory, repo_root=chain.repo)


def test_non_finite_training_results_stop_the_stage(chain, tmp_path, monkeypatch):
    real_fit = LightweightAnomalyModel.fit

    def poisoned(self, inputs, labels, *, epochs, learning_rate=0.1):
        out = real_fit(self, inputs, labels, epochs=epochs, learning_rate=learning_rate)
        self.weights[0] = np.nan
        return out

    monkeypatch.setattr(LightweightAnomalyModel, "fit", poisoned)
    directory = tmp_path / "nan"
    directory.mkdir()
    with pytest.raises(tr.TrainingStageError, match="non-finite"):
        tr.train_candidates(chain.plan, artifact_dir=directory, repo_root=chain.repo)
    assert list(directory.iterdir()) == []


def test_artifacts_cannot_be_written_inside_the_repository_or_a_missing_directory(chain, tmp_path):
    with pytest.raises(tr.TrainingStageError):
        tr.train_candidates(chain.plan, artifact_dir=chain.repo, repo_root=chain.repo)
    with pytest.raises(tr.TrainingStageError):
        tr.train_candidates(chain.plan, artifact_dir=tmp_path / "missing", repo_root=chain.repo)
    with pytest.raises(tr.TrainingStageError):
        tr.train_candidates("plan", artifact_dir=tmp_path, repo_root=chain.repo)


def test_training_refuses_a_dirty_moved_or_changed_repository(chain, tmp_path):
    directory = tmp_path / "artifacts"
    directory.mkdir()
    stray = chain.repo / "stray.txt"
    stray.write_text("x", encoding="utf-8")
    try:
        with pytest.raises(tr.TrainingStageError, match="authorized clean commit"):
            tr.train_candidates(chain.plan, artifact_dir=directory, repo_root=chain.repo)
    finally:
        stray.unlink()
    original = hc.git(chain.repo, "rev-parse", "HEAD")
    source = chain.repo / "tokenization_ai/heldout_eval/arm_c.py"
    source.write_text("# changed after authorization" + chr(10), encoding="utf-8", newline=chr(10))
    hc.git(chain.repo, "commit", "-q", "-am", "change a source after authorization")
    try:
        with pytest.raises(tr.TrainingStageError, match="authorized clean commit"):
            tr.train_candidates(chain.plan, artifact_dir=directory, repo_root=chain.repo)
    finally:
        hc.git(chain.repo, "reset", "-q", "--hard", original)
    assert list(directory.iterdir()) == []


def test_trained_candidates_cannot_be_forged(chain):
    with pytest.raises(tr.TrainingStageError):
        tr.TrainedCandidates(object())
    with pytest.raises(AttributeError):
        chain.trained.digests = {}  # type: ignore[misc]


# -- D4-7D2: validation scoring and the selection record ------------------------------------------------------------------------------------------


def test_the_selection_record_is_written_and_verified_with_all_81_candidates(chain):
    sel = chain.selection
    assert sel.candidates_scored == 81 and sel.verification.ok and sel.verification.authorizes_test_scoring is False
    record = mf.parse_manifest_bytes(Path(sel.record_path).read_bytes())
    assert mf.sha256_hex(Path(sel.record_path).read_bytes()) == sel.record_sha256 and record["record_version"] == "validation-selection-record-2"
    assert record["artifacts"]["candidates"] == 81 and record["test_status"]["test_scored"] is False
    for arm in ARMS:
        block = record["arms"][arm]
        assert block["candidate_verification"]["candidates_verified"] == 27 and block["selected"]["candidate_id"] == sel.selected_candidates[arm]
        assert block["threshold"]["scale"] == "raw_logit" and block["selected"]["artifact_sha256"] == chain.trained.digests[sel.selected_candidates[arm]]
        assert all(row["artifact_sha256"] == chain.trained.digests[row["candidate_id"]] for row in block["candidates"])


def test_the_record_verifies_again_independently_with_the_real_verifier(chain):
    val_inputs = rn.encode_validation(chain.cfg, chain.part.validation)
    evidence = sr.ValidationEvidence(arm_inputs=val_inputs, labels_by_id=chain.part.validation.labels_by_id(), frozen=chain.cfg.frozen,
                                     split_result=chain.split, artifact_dir=chain.trained.artifact_dir, artifact_digests=dict(chain.trained.digests))
    result = sr.verify_selection_record(chain.selection.record_path, expected_sha256=chain.selection.record_sha256, manifest_bytes=chain.manifest_bytes,
                                        expected_manifest_sha256=chain.manifest_sha, repo_root=chain.repo, expected_run=chain.expected, evidence=evidence)
    assert result.ok and dict(result.selected_candidates) == dict(chain.selection.selected_candidates)


def test_the_selected_candidates_follow_the_approved_rule_on_recomputed_validation_ap(chain):
    val_inputs = rn.encode_validation(chain.cfg, chain.part.validation)
    labels = list(rn.aligned_labels(val_inputs.record_ids, chain.part.validation.labels_by_id()))
    loaded = ma.load_candidate_set(chain.trained.artifact_dir, expected_manifest_sha256=chain.manifest_sha, expected_digests=dict(chain.trained.digests),
                                   expected_learning_rates={c.candidate_id: c.learning_rate for c in chain.plan.candidates})
    for arm in ARMS:
        attr = {ARM_A: "arm_a", ARM_B: "arm_b", ARM_C: "arm_c"}[arm]
        results = []
        for c in (c for c in chain.plan.candidates if c.arm == arm):
            ap = numerics.raw_logits(loaded[c.candidate_id].model, list(getattr(val_inputs, attr)))
            from tokenization_ai.heldout_eval.detection_metrics import average_precision
            results.append(rn.ValidationCandidateResult(c.candidate_id, arm, c.num_buckets, c.epochs, c.relative_step_c, average_precision(ap, labels),
                                                        "validation", rn.ids_fingerprint(list(val_inputs.record_ids)), len(labels)))
        assert rn.select_candidate(results, arm=arm).selected.candidate_id == chain.selection.selected_candidates[arm]


def test_validation_requires_the_same_plan_trained_candidates_and_validation_data(chain, tmp_path):
    def call(**over):
        kwargs = dict(validation=chain.part.validation, manifest_bytes=chain.manifest_bytes, expected_manifest_sha256=chain.manifest_sha,
                      repo_root=chain.repo, expected_run=chain.expected, record_path=tmp_path / "r.json", created_utc=hc.CREATED)
        plan = over.pop("plan", chain.plan)
        trained = over.pop("trained", chain.trained)
        kwargs.update(over)
        return tr.select_on_validation(plan, trained, **kwargs)

    with pytest.raises(tr.TrainingStageError):
        call(trained="trained")
    with pytest.raises(tr.TrainingStageError):
        call(plan=object())
    with pytest.raises(tr.TrainingStageError):
        call(expected_manifest_sha256="0" * 64)
    with pytest.raises(tr.TrainingStageError):
        call(manifest_bytes=b"{}")
    val = list(chain.part.validation.records)
    flipped = rn.ValidationData(tuple([EvalRecord(val[0].record_id, val[0].features, SplitMetadata(label=1 - val[0].meta.label)), *val[1:]]))
    with pytest.raises(tr.TrainingStageError):
        call(validation=flipped)
    with pytest.raises(tr.TrainingStageError):
        call(validation=chain.part.train)
    assert not (tmp_path / "r.json").exists()


def test_a_tampered_artifact_after_training_is_detected_before_any_record_is_written(chain, tmp_path):
    victim = Path(chain.trained.artifact_dir) / ma.expected_artifact_names(chain.manifest_sha)[chain.plan.candidates[10].candidate_id]
    original = victim.read_bytes()
    os.chmod(victim, stat.S_IWRITE | stat.S_IREAD)
    try:
        data = bytearray(original)
        data[200] ^= 1
        victim.write_bytes(bytes(data))
        with pytest.raises(tr.TrainingStageError, match="did not verify"):
            tr.select_on_validation(chain.plan, chain.trained, validation=chain.part.validation, manifest_bytes=chain.manifest_bytes,
                                    expected_manifest_sha256=chain.manifest_sha, repo_root=chain.repo, expected_run=chain.expected,
                                    record_path=tmp_path / "r.json", created_utc=hc.CREATED)
        assert not (tmp_path / "r.json").exists()
    finally:
        victim.write_bytes(original)
        os.chmod(victim, stat.S_IREAD)


def test_a_changed_training_ledger_is_detected(chain, tmp_path):
    ledger = Path(chain.trained.ledger_path)
    original = ledger.read_bytes()
    os.chmod(ledger, stat.S_IWRITE | stat.S_IREAD)
    try:
        ledger.write_bytes(original + b" ")
        with pytest.raises(tr.TrainingStageError, match="ledger"):
            tr.select_on_validation(chain.plan, chain.trained, validation=chain.part.validation, manifest_bytes=chain.manifest_bytes,
                                    expected_manifest_sha256=chain.manifest_sha, repo_root=chain.repo, expected_run=chain.expected,
                                    record_path=tmp_path / "r.json", created_utc=hc.CREATED)
    finally:
        ledger.write_bytes(original)
        os.chmod(ledger, stat.S_IREAD)


def test_the_selection_record_is_write_once(chain):
    with pytest.raises(tr.TrainingStageError):
        tr.select_on_validation(chain.plan, chain.trained, validation=chain.part.validation, manifest_bytes=chain.manifest_bytes,
                                expected_manifest_sha256=chain.manifest_sha, repo_root=chain.repo, expected_run=chain.expected,
                                record_path=chain.selection.record_path, created_utc=hc.CREATED)


# -- isolation and hygiene -------------------------------------------------------------------------------------------------------------------------


def test_no_stage_can_receive_test_labels_or_test_scoring_inputs():
    for function in (tr.build_training_plan, tr.train_candidates, tr.select_on_validation):
        names = set(inspect.signature(function).parameters)
        assert not names & {"sealed_test", "test", "test_labels", "label_provider", "attempt_root"}
    assert set(inspect.signature(tr.train_candidates).parameters) == {"plan", "artifact_dir", "repo_root"}  # no validation data either


def test_the_training_module_never_scores_tests_or_opens_the_gate():
    text = Path(tr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"open_test_attempt", "recover_test_labels", "create_attempt_marker", "request_test_access", "paired_group_bootstrap", "roc_auc",
                         "build_arm_inputs", "labels_in_order"}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "attempt_gate" not in imported and "confirmatory_scoring" not in imported
    assert not any(seed in text for seed in ("7101", "7102", "7103"))


def test_the_training_module_is_protected_by_the_manifest():
    assert "tokenization_ai/heldout_eval/training.py" in mf.REQUIRED_SOURCE_FILES
