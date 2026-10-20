"""Tests for validation-selection-record-2 (draft protocol heldout-eval-1.2.0-draft, sections 12.11.4, 12.12.3 and 12.12.6).

No research model is trained or scored. The 81 candidate models are *mock artifacts*: ``LightweightAnomalyModel`` objects with explicitly assigned random
weights (no ``fit``), persisted with ``model_artifacts`` and verified by the real loader, so the APs and thresholds the verifier recomputes are properties of the
fixture, not detection results. Dataset objects come from the development seed 6101 (structure only); Git state lives in a throw-away repository under pytest's
temporary directory. No confirmatory seed is used. To keep the suite fast, ``numerics.raw_logits`` is wrapped by a cache keyed on the exact weights, bias and
inputs; every cache miss calls the real function.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
import random
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

import tokenization_ai.heldout_eval.manifest as mf
import tokenization_ai.heldout_eval.model_artifacts as ma
import tokenization_ai.heldout_eval.numerics as numerics
import tokenization_ai.heldout_eval.runner as rn
import tokenization_ai.heldout_eval.selection_record as sr
from tokenization_ai.heldout_eval.arms import ARM_A, ARM_B, ARM_C, ARMS
from tokenization_ai.heldout_eval.detection_metrics import ThresholdSelection, average_precision
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.runner import ValidationCandidateResult
from tokenization_ai.heldout_eval.selection_record import (
    SelectionRecordError,
    SelectionRecordSchemaError,
    SelectionRecordVerificationError,
    SelectionRecordWriteError,
    ValidationEvidence,
    build_selection_record,
    verify_selection_record,
    verify_selection_record_bytes,
    write_selection_record,
)
from tokenization_ai.heldout_eval.splits import PRIMARY_SPLIT_SEED, split_records
from tokenization_ai.model import LightweightAnomalyModel

REPO_ROOT = Path(sr.__file__).resolve().parents[2]
SRC = ("src/a.py", "src/b.py")
CFG = ("requirements.txt",)
PROTO = "docs/PROTOCOL.md"
HEADER = "# Protocol\n\n**Protocol version:** `heldout-eval-test-draft`  **Status: DRAFT, not frozen.**\n"
AUDIT_OK = {"errors": [], "warnings": [], "construction_checks_passed": True}
CREATED = "2026-01-01T00:00:00Z"
GRID = [(nb, ep, c) for nb in (1024, 2048, 4096) for ep in (50, 150, 400) for c in (0.1, 0.3, 0.8)]
CANDIDATES = [(arm, nb, ep, c) for arm in ARMS for (nb, ep, c) in GRID]
INPUT_ATTR = {ARM_A: "arm_a", ARM_B: "arm_b", ARM_C: "arm_c"}


def cid_of(arm, nb, ep, c) -> str:
    return rn.canonical_candidate_id(arm, nb, ep, c)


# -- caching wrapper (test speed only) ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def cached_logits():
    real = numerics.raw_logits
    cache: dict = {}

    def wrapper(model, inputs):
        try:
            key = (hashlib.sha256(np.ascontiguousarray(model.weights, dtype="<f8").tobytes()).hexdigest(), float(model.bias), model.num_buckets,
                   model.is_trained, len(inputs), inputs[0].event_id if len(inputs) else None, inputs[-1].event_id if len(inputs) else None,
                   hash(tuple(tuple(m.input_ids[:6]) for m in inputs[:20])))
        except Exception:  # noqa: BLE001 - anything unusual goes straight to the real function
            return real(model, inputs)
        if key not in cache:
            cache[key] = real(model, inputs)
        return list(cache[key])

    numerics.raw_logits = wrapper
    yield
    numerics.raw_logits = real


# -- fixtures ------------------------------------------------------------------------------------------------------------------------------


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          cwd=str(root), capture_output=True, text=True, check=True)
    return done.stdout.strip()


def make_repo(root: Path) -> Path:
    root.mkdir()
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "src/a.py").write_text("A = 1\n", encoding="utf-8", newline="\n")
    (root / "src/b.py").write_text("B = 2\n", encoding="utf-8", newline="\n")
    (root / "requirements.txt").write_text("numpy>=1.26\n", encoding="utf-8", newline="\n")
    (root / PROTO).write_text(HEADER, encoding="utf-8", newline="\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def mock_model(arm: str, nb: int, k: int = 0, scale: float = 0.05, bias: float = 0.125) -> LightweightAnomalyModel:
    rng = np.random.default_rng({ARM_A: 11, ARM_B: 12, ARM_C: 13}[arm] + nb + 1000 * k)
    model = LightweightAnomalyModel(num_buckets=nb)
    model.weights = rng.normal(0.0, scale, nb)
    model.bias = bias
    model.is_trained = True  # explicit fixture state: no fit
    return model


def zero_model(nb: int, bias: float = 0.125) -> LightweightAnomalyModel:
    model = LightweightAnomalyModel(num_buckets=nb)
    model.bias = bias
    model.is_trained = True  # all logits equal: every AP equals the prevalence, so any set of such candidates ties exactly
    return model


class Env:
    pass


def training_for(manifest: dict, arm: str, nb: int) -> dict:
    est = next(x for x in manifest["numerics"]["estimates"] if x["arm"] == arm and x["num_buckets"] == nb)
    is_c = arm == ARM_C
    return {"train_ids_sha256": manifest["arms"]["alignment"]["train_ids_sha256"],
            "vocabulary_sha256": None if is_c else manifest["arms"]["vocabulary"]["sha256"],
            "max_length": None if is_c else manifest["arms"]["budget"]["selected_max_length"],
            "lambda_method": est["method"], "lambda_max": est["lambda_max"], "L": est["L"]}


def write_all(e, directory: Path, models: dict) -> dict:
    """Persist all 81 candidate models with model_artifacts; returns the independently established digests (candidate id -> SHA-256)."""
    directory.mkdir(parents=True, exist_ok=True)
    rates = ma.learning_rates_from_manifest(e.manifest)
    digests = {}
    for arm, nb, ep, c in CANDIDATES:
        cid = cid_of(arm, nb, ep, c)
        written = ma.write_model_artifact(directory, models[cid], repo_root=e.repo, arm=arm, num_buckets=nb, epochs=ep, relative_step_c=c,
                                          learning_rate=rates[cid], manifest_sha256=e.manifest_sha, training=training_for(e.manifest, arm, nb))
        digests[cid] = written.sha256
    return digests


def ap_of(e, arm: str, model) -> float:
    inputs = list(getattr(e.val_inputs, INPUT_ATTR[arm]))
    return average_precision(numerics.raw_logits(model, inputs), e.labels_list)


def results_from(e, aps: dict) -> dict:
    return {arm: [ValidationCandidateResult(cid_of(arm, nb, ep, c), arm, nb, ep, c, aps[cid_of(arm, nb, ep, c)], "validation", e.ids_fp,
                                            len(e.val_inputs.record_ids)) for (nb, ep, c) in GRID] for arm in ARMS}


def build_record(e, results, evidence):
    return build_selection_record(manifest_bytes=e.manifest_bytes, expected_manifest_sha256=e.manifest_sha, results_by_arm=results, evidence=evidence,
                                  created_utc=CREATED)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env()
    root = tmp_path_factory.mktemp("sel")
    e.root = root
    e.repo = make_repo(root / "repo")
    e.out = root / "out"
    e.out.mkdir()
    ds = generate(GeneratorConfig(seed=6101, n_templates=250))
    assert ds.config.seed == 6101
    split = split_records(ds.records, seed=PRIMARY_SPLIT_SEED)
    part = rn.partition_records(ds.records, split)
    cfg = rn.prepare_train_configuration(part.train)
    numerical = rn.prepare_numerical_configuration(cfg)
    pre = rn.preflight_report(cfg, part.validation, part.sealed_test)
    e.kw = {"source_files": SRC, "config_files": CFG, "protocol_path": PROTO}
    manifest = mf.build_manifest(purpose=mf.PURPOSE_INFRASTRUCTURE, created_utc=CREATED, repo_root=e.repo, dataset=ds, split_result=split,
                                 partition=part, audit_report=AUDIT_OK, train_config=cfg, numerical=numerical, preflight=pre, **e.kw)
    e.expected_run = mf.ExpectedRun(purpose=mf.PURPOSE_INFRASTRUCTURE, branch="main", commit=git(e.repo, "rev-parse", "HEAD"), dataset_seed=6101,
                                    generator_fingerprint=ds.fingerprint, dataset_fingerprint=split.report.dataset_fingerprint,
                                    split_seed=PRIMARY_SPLIT_SEED, split_manifest_sha256=split.manifest_sha256)
    written = mf.write_manifest(e.out / "manifest.json", manifest, repo_root=e.repo, expected=e.expected_run, **e.kw)
    e.manifest_bytes, e.manifest_sha = Path(written.path).read_bytes(), written.sha256
    e.manifest = mf.parse_manifest_bytes(e.manifest_bytes)
    e.val_inputs = rn.encode_validation(cfg, part.validation)
    e.labels = part.validation.labels_by_id()
    e.labels_list = list(rn.aligned_labels(e.val_inputs.record_ids, e.labels))
    e.split, e.cfg, e.part, e.ds = split, cfg, part, ds
    e.ids_fp = rn.ids_fingerprint(list(e.val_inputs.record_ids))
    e.models = {cid_of(arm, nb, ep, c): mock_model(arm, nb, k) for k, (arm, nb, ep, c) in enumerate(CANDIDATES)}
    e.artifact_dir = root / "artifacts"
    e.digests = write_all(e, e.artifact_dir, e.models)
    e.evidence = ValidationEvidence(arm_inputs=e.val_inputs, labels_by_id=e.labels, frozen=cfg.frozen, split_result=split, artifact_dir=e.artifact_dir,
                                    artifact_digests=e.digests)
    e.aps = {cid_of(arm, nb, ep, c): ap_of(e, arm, e.models[cid_of(arm, nb, ep, c)]) for (arm, nb, ep, c) in CANDIDATES}
    e.results = results_from(e, e.aps)
    e.winners = {arm: rn.select_candidate(e.results[arm], arm=arm).selected for arm in ARMS}
    e.record = build_record(e, e.results, e.evidence)
    return e


def fresh_artifacts(e, tmp_path) -> Path:
    """A writable copy of the 81 artifact files, for tests that tamper with them."""
    target = tmp_path / "artifacts"
    shutil.copytree(e.artifact_dir, target)
    for p in target.iterdir():
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    return target


def artifact_path(e, directory: Path, cid: str) -> Path:
    return directory / ma.artifact_filename(e.manifest_sha, cid)


def evidence_with(e, **over) -> ValidationEvidence:
    return dataclasses.replace(e.evidence, **over)


def scenario(e, tmp_path, overrides: dict):
    """A full, internally consistent record/evidence pair in which some candidates carry different models; ``overrides`` maps candidate id -> model."""
    models = {**e.models, **overrides}
    directory = tmp_path / "scenario"
    digests = write_all(e, directory, models)
    aps = {cid: (ap_of(e, next(a for a in ARMS if cid.startswith(a)), models[cid]) if cid in overrides else e.aps[cid]) for cid in models}
    results = results_from(e, aps)
    evidence = evidence_with(e, artifact_dir=directory, artifact_digests=digests)
    return build_record(e, results, evidence), evidence, results, aps


def verify(e, record, *, evidence=None, manifest_bytes=None, manifest_sha=None, expected_run=None, digest=None, data=None, **kw):
    data = data if data is not None else mf.canonical_bytes(record)
    return verify_selection_record_bytes(data, expected_sha256=digest or mf.sha256_hex(data), manifest_bytes=manifest_bytes or e.manifest_bytes,
                                         expected_manifest_sha256=manifest_sha or e.manifest_sha, repo_root=e.repo,
                                         expected_run=expected_run or e.expected_run, evidence=evidence or e.evidence, **e.kw, **kw)


def mutated(record, fn):
    out = copy.deepcopy(record)
    fn(out)
    return out


def write(e, record=None, name="rec.json", evidence=None, **kw):
    return write_selection_record(e.out / name, e.record if record is None else record, manifest_bytes=e.manifest_bytes,
                                  expected_manifest_sha256=e.manifest_sha, repo_root=e.repo, expected_run=e.expected_run,
                                  evidence=evidence or e.evidence, **e.kw, **kw)


# -- schema, determinism ---------------------------------------------------------------------------------------------------------------


def test_record_has_the_required_structure_and_is_strict_json(env):
    r = env.record
    assert r["record_version"] == "validation-selection-record-2" and set(r) == sr._TOP_KEYS and set(r["arms"]) == set(ARMS)
    assert r["manifest"]["sha256"] == env.manifest_sha and r["manifest"]["git_commit"] == env.expected_run.commit
    assert r["manifest"]["split_manifest_sha256"] == env.split.manifest_sha256 and r["manifest"]["protocol_sha256"] == env.manifest["protocol"]["sha256"]
    assert r["validation"] == {"split": "validation", "records": len(env.val_inputs.record_ids), "ids_sha256": env.ids_fp}
    assert r["test_status"] == {"test_scored": False, "test_labels_accessed": False, "test_scores_accessed": False, "test_data_in_record": False}
    assert r["artifacts"]["candidates"] == 81 and r["artifacts"]["format"] == "model-artifact-1" and r["artifacts"]["manifest_sha256"] == env.manifest_sha
    assert r["vocabulary"]["sha256"] == env.manifest["arms"]["vocabulary"]["sha256"] and r["budget"]["rule_id"] == "budget-rule-1"
    assert r["statistics"] == env.manifest["statistics"] and r["rules"] == env.manifest["selection"]
    assert r["code"]["selection_record_sha256"] == sr.source_sha256()
    json.loads(mf.canonical_bytes(r))
    sr.validate_record_structure(r)


def test_per_arm_blocks_hold_all_27_verified_candidates_and_the_recomputed_winner(env):
    for arm in ARMS:
        block = env.record["arms"][arm]
        win = env.winners[arm]
        sel = block["selected"]
        assert (sel["num_buckets"], sel["epochs"], sel["relative_step_c"]) == (win.num_buckets, win.epochs, win.relative_step_c)
        assert sel["candidate_id"] == win.candidate_id and sel["lambda_method"] == "lambda-power-1" and sel["L"] == sel["lambda_max"] / 4
        assert sel["learning_rate"] == sel["relative_step_c"] / sel["L"] and sel["artifact_sha256"] == env.digests[win.candidate_id]
        assert sel["validation_ap"] == env.aps[win.candidate_id] and sel["rounded_ap"] == round(env.aps[win.candidate_id], 10)
        rows = block["candidates"]
        assert len(rows) == 27 and [r["candidate_id"] for r in rows] == sorted(r["candidate_id"] for r in rows)
        manifest_lr = {x["candidate_id"]: x["learning_rate"] for x in env.manifest["numerics"]["candidates"]}
        for r in rows:
            assert r["learning_rate"] == manifest_lr[r["candidate_id"]] and r["artifact_sha256"] == env.digests[r["candidate_id"]]
            assert r["validation_ap"] == env.aps[r["candidate_id"]] and r["recomputed_validation_ap"] == pytest.approx(r["validation_ap"], abs=1e-12)
            assert r["rounded_ap"] == round(r["validation_ap"], 10)
            assert r["weights_sha256"] == sr.model_artifact_fingerprint(env.models[r["candidate_id"]])["weights_sha256"]
        ver = block["candidate_verification"]
        assert ver["candidates_verified"] == 27 and ver["all_validation_aps_recomputed"] is True and ver["artifact_format"] == "model-artifact-1"
        th = block["threshold"]
        assert th["scale"] == "raw_logit" and th["probability_threshold"] is False and th["basis"] == "validation_scores_and_labels_only"
        assert th["value"] is not None and sum(th["counts"].values()) == len(env.val_inputs.record_ids)
        assert block["model"]["weights_sha256"] == sr.model_artifact_fingerprint(env.models[win.candidate_id])["weights_sha256"]
        assert block["model"]["model_sha256"] == env.manifest["numerics"]["provenance"]["model_sha256"] and block["model"]["logit_method"] == "logit-readout-1"
        assert block["tie_diagnostics"]["validation_records"] == len(env.val_inputs.record_ids)
        assert block["tie_break_trace"][0]["rule"] == "highest_rounded_validation_ap"


def test_building_is_deterministic_and_independent_of_candidate_order(env):
    again = build_record(env, env.results, env.evidence)
    assert mf.canonical_bytes(again) == mf.canonical_bytes(env.record)
    shuffled = {arm: random.Random(5).sample(rows, len(rows)) for arm, rows in env.results.items()}
    assert mf.canonical_bytes(build_record(env, shuffled, env.evidence)) == mf.canonical_bytes(env.record)


def test_model_artifact_fingerprint_is_exact_and_sensitive():
    m = mock_model(ARM_A, 1024)
    fp = sr.model_artifact_fingerprint(m)
    assert fp["weights_sha256"] == hashlib.sha256(np.ascontiguousarray(m.weights, dtype="<f8").tobytes()).hexdigest()
    assert fp["bias_hex"] == float(m.bias).hex() and fp["num_buckets"] == 1024 and fp["is_trained"] is True
    assert sr.model_artifact_fingerprint(mock_model(ARM_A, 1024)) == fp
    changed = mock_model(ARM_A, 1024)
    changed.weights[7] += 1e-9
    assert sr.model_artifact_fingerprint(changed)["weights_sha256"] != fp["weights_sha256"]
    other_bias = mock_model(ARM_A, 1024)
    other_bias.bias += 1e-12
    assert sr.model_artifact_fingerprint(other_bias)["bias_hex"] != fp["bias_hex"]
    for bad in (LightweightAnomalyModel(num_buckets=8), object()):
        with pytest.raises(SelectionRecordVerificationError):
            sr.model_artifact_fingerprint(bad)  # type: ignore[arg-type]
    nan_model = mock_model(ARM_A, 1024)
    nan_model.weights[0] = np.nan
    with pytest.raises(SelectionRecordVerificationError):
        sr.model_artifact_fingerprint(nan_model)


# -- all-candidate recomputation ----------------------------------------------------------------------------------------------------------


def test_all_81_candidates_are_loaded_and_every_validation_ap_is_recomputed(env, monkeypatch):
    seen: set[str] = set()
    real = numerics.raw_logits

    def spy(model, inputs):
        if len(inputs) == len(env.val_inputs.record_ids):
            seen.add(hashlib.sha256(np.ascontiguousarray(model.weights, dtype="<f8").tobytes()).hexdigest())
        return real(model, inputs)

    monkeypatch.setattr(numerics, "raw_logits", spy)
    result = verify(env, env.record)
    assert result.ok and len(seen) == 81  # one logit readout of the full validation set per distinct candidate model
    assert seen == {sr.model_artifact_fingerprint(m)["weights_sha256"] for m in env.models.values()}


def test_loaded_artifacts_are_used_not_in_memory_models(env, tmp_path):
    """The verifier reads the persisted files: replacing a file (and its trusted digest) with another valid artifact changes the outcome."""
    directory = fresh_artifacts(env, tmp_path)
    cid = cid_of(ARM_A, 1024, 50, 0.1)
    other = mock_model(ARM_A, 1024, k=999)
    path = artifact_path(env, directory, cid)
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    path.unlink()
    rates = ma.learning_rates_from_manifest(env.manifest)
    new = ma.write_model_artifact(directory, other, repo_root=env.repo, arm=ARM_A, num_buckets=1024, epochs=50, relative_step_c=0.1,
                                  learning_rate=rates[cid], manifest_sha256=env.manifest_sha, training=training_for(env.manifest, ARM_A, 1024))
    evidence = evidence_with(env, artifact_dir=directory, artifact_digests={**env.digests, cid: new.sha256})
    with pytest.raises(SelectionRecordVerificationError):  # a validly persisted but different model no longer reproduces the recorded AP and weights hash
        verify(env, env.record, evidence=evidence)


def test_recomputed_ap_values_agree_with_an_independent_computation(env):
    rng = random.Random(0)
    for arm in ARMS:
        for row in rng.sample(env.record["arms"][arm]["candidates"], 4):
            model = env.models[row["candidate_id"]]
            inputs = list(getattr(env.val_inputs, INPUT_ATTR[arm]))
            logits = [float(np.dot(model._features(i.input_ids, i.attention_mask), model.weights) + model.bias) for i in inputs]
            assert average_precision(logits, env.labels_list) == pytest.approx(row["recomputed_validation_ap"], abs=1e-12)


def test_a_false_non_selected_ap_claim_is_rejected(env):
    for arm in ARMS:
        win = env.winners[arm].candidate_id
        other = next(r for r in env.results[arm] if r.candidate_id != win)
        for delta in (1e-6, -1e-6, 0.3, -other.validation_ap):
            results = {a: list(rows) for a, rows in env.results.items()}
            results[arm] = [dataclasses.replace(r, validation_ap=max(0.0, min(1.0, r.validation_ap + delta))) if r.candidate_id == other.candidate_id else r
                            for r in results[arm]]
            with pytest.raises(SelectionRecordVerificationError) as caught:
                build_record(env, results, env.evidence)
            assert other.candidate_id in str(caught.value) and "persisted model" in str(caught.value)


def test_a_claim_within_floating_point_tolerance_is_accepted_but_is_not_a_license_to_change_the_winner(env):
    results = {a: list(rows) for a, rows in env.results.items()}
    arm = ARM_B
    other = next(r for r in results[arm] if r.candidate_id != env.winners[arm].candidate_id)
    results[arm] = [dataclasses.replace(r, validation_ap=r.validation_ap + 1e-14) if r is other else r for r in results[arm]]
    record = build_record(env, results, env.evidence)
    assert verify(env, record).ok and record["arms"][arm]["selected"]["candidate_id"] == env.winners[arm].candidate_id


def test_a_falsely_preferred_winner_is_rejected(env):
    arm = ARM_A
    win = env.winners[arm]
    runner_up = sorted((r for r in env.results[arm] if r.candidate_id != win.candidate_id), key=lambda r: -r.validation_ap)[0]
    # (1) the winner's claimed AP is lowered so that the runner-up appears to win: its claim no longer matches its persisted model
    results = {a: list(rows) for a, rows in env.results.items()}
    results[arm] = [dataclasses.replace(r, validation_ap=runner_up.validation_ap - 1e-3) if r.candidate_id == win.candidate_id else r for r in results[arm]]
    with pytest.raises(SelectionRecordVerificationError):
        build_record(env, results, env.evidence)
    # (2) the record claims the runner-up as the selected candidate while all APs are honest
    forged = mutated(env.record, lambda r: r["arms"][arm]["selected"].update(candidate_id=runner_up.candidate_id, num_buckets=runner_up.num_buckets,
                                                                              epochs=runner_up.epochs, relative_step_c=runner_up.relative_step_c))
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, forged)


# -- write-once -----------------------------------------------------------------------------------------------------------------------------


def test_write_creates_canonical_bytes_returns_their_digest_and_verifies(env):
    written = write(env, name="ok.json")
    data = Path(written.path).read_bytes()
    assert data == mf.canonical_bytes(env.record) and written.sha256 == mf.sha256_hex(data) and written.size == len(data)
    assert not (Path(written.path).stat().st_mode & stat.S_IWRITE)
    result = verify_selection_record(written.path, expected_sha256=written.sha256, manifest_bytes=env.manifest_bytes,
                                     expected_manifest_sha256=env.manifest_sha, repo_root=env.repo, expected_run=env.expected_run,
                                     evidence=env.evidence, **env.kw)
    assert result.ok and result.manifest_sha256 == env.manifest_sha
    assert result.selected_candidates == {arm: env.winners[arm].candidate_id for arm in ARMS}


def test_an_existing_record_is_never_overwritten(env):
    first = write(env, name="once.json")
    before = Path(first.path).read_bytes()
    with pytest.raises(SelectionRecordWriteError):
        write(env, name="once.json")
    assert Path(first.path).read_bytes() == before


def test_write_refuses_repository_paths_and_missing_directories(env):
    with pytest.raises(SelectionRecordWriteError):
        write_selection_record(env.repo / "rec.json", env.record, manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha,
                               repo_root=env.repo, expected_run=env.expected_run, evidence=env.evidence, **env.kw)
    assert not (env.repo / "rec.json").exists()
    with pytest.raises(SelectionRecordWriteError):
        write(env, name="nope/rec.json")


def test_a_failed_write_is_left_for_review_and_blocks_a_silent_retry(env, monkeypatch):
    def broken(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(SelectionRecordWriteError):
        write(env, name="failed.json")
    monkeypatch.undo()
    assert (env.out / "failed.json").exists()
    with pytest.raises(SelectionRecordWriteError):
        write(env, name="failed.json")


def test_invalid_records_are_never_written(env):
    bad = mutated(env.record, lambda r: r["arms"][ARM_A]["selected"].update(candidate_id=cid_of(ARM_A, 4096, 400, 0.8)))
    for name, record in (("bad1.json", bad), ("bad2.json", {}), ("bad3.json", {"record_version": sr.RECORD_VERSION})):
        with pytest.raises(SelectionRecordError):
            write(env, record, name=name)
        assert not (env.out / name).exists()


def test_the_writer_requires_the_same_artifact_evidence_as_the_verifier(env, tmp_path):
    for name, evidence in (("a.json", evidence_with(env, artifact_digests={k: v for k, v in env.digests.items() if k != cid_of(ARM_A, 1024, 50, 0.1)})),
                           ("b.json", evidence_with(env, artifact_dir=tmp_path / "missing"))):
        with pytest.raises(SelectionRecordVerificationError):
            write(env, name=name, evidence=evidence)
        assert not (env.out / name).exists()


# -- digest and bytes -------------------------------------------------------------------------------------------------------------------------


def test_verification_succeeds_on_the_untouched_record(env):
    result = verify(env, env.record)
    assert result.ok and result.authorizes_test_scoring is False and result.test_scored is False


def test_digest_and_byte_integrity(env):
    data = mf.canonical_bytes(env.record)
    flipped = bytearray(data)
    flipped[len(flipped) // 3] ^= 1
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, None, data=bytes(flipped), digest=mf.sha256_hex(data))
    for bad in (None, "", "abc", mf.sha256_hex(data).upper(), 5):
        with pytest.raises(SelectionRecordVerificationError):
            verify_selection_record_bytes(data, expected_sha256=bad, manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha,
                                          repo_root=env.repo, expected_run=env.expected_run, evidence=env.evidence, **env.kw)  # type: ignore[arg-type]
    with pytest.raises(SelectionRecordVerificationError):
        verify_selection_record(env.out / "missing.json", expected_sha256=mf.sha256_hex(data), manifest_bytes=env.manifest_bytes,
                                expected_manifest_sha256=env.manifest_sha, repo_root=env.repo, expected_run=env.expected_run, evidence=env.evidence,
                                **env.kw)


@pytest.mark.parametrize("raw", [b"", b"[]", b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n', b'{"a":1e999}\n', b'{"a": 1}\n', b'{"a":1}',
                                 b'{"a":1}\n{"b":2}\n', b'{"record_version":"validation-selection-record-2",', b"\xff\xfe"])
def test_malformed_partial_or_non_canonical_bytes_are_rejected_even_with_a_matching_digest(env, raw):
    with pytest.raises(SelectionRecordError):
        verify_selection_record_bytes(raw, expected_sha256=mf.sha256_hex(raw), manifest_bytes=env.manifest_bytes,
                                      expected_manifest_sha256=env.manifest_sha, repo_root=env.repo, expected_run=env.expected_run,
                                      evidence=env.evidence, **env.kw)


def test_truncated_real_record_is_rejected(env):
    data = mf.canonical_bytes(env.record)
    for cut in (len(data) // 2, len(data) - 1):
        with pytest.raises(SelectionRecordError):
            verify_selection_record_bytes(data[:cut], expected_sha256=mf.sha256_hex(data[:cut]), manifest_bytes=env.manifest_bytes,
                                          expected_manifest_sha256=env.manifest_sha, repo_root=env.repo, expected_run=env.expected_run,
                                          evidence=env.evidence, **env.kw)


# -- schema versions --------------------------------------------------------------------------------------------------------------------------


def test_the_superseded_version_1_record_is_refused_explicitly(env):
    assert sr.SUPERSEDED_VERSIONS == ("validation-selection-record-1",)
    old = mutated(env.record, lambda r: r.update(record_version="validation-selection-record-1"))
    with pytest.raises(SelectionRecordSchemaError) as caught:
        verify(env, old)
    assert "superseded" in str(caught.value)
    with pytest.raises(SelectionRecordSchemaError):
        sr.validate_record_structure({"record_version": "validation-selection-record-1", "arms": {}})  # even a skeleton of the old format
    v1_shaped = {k: v for k, v in env.record.items() if k not in ("artifacts",)}
    v1_shaped["record_version"] = "validation-selection-record-1"
    with pytest.raises(SelectionRecordSchemaError):
        verify(env, v1_shaped)


@pytest.mark.parametrize("version", ["validation-selection-record-3", "validation-selection-record-0", "", None, 2])
def test_unknown_record_versions_fail_closed(env, version):
    with pytest.raises(SelectionRecordSchemaError):
        verify(env, mutated(env.record, lambda r: r.update(record_version=version)))


def test_a_version_2_record_missing_the_new_evidence_blocks_is_rejected(env):
    for mutation in (lambda r: r.pop("artifacts"), lambda r: r["arms"][ARM_A].pop("candidate_verification"),
                     lambda r: r["arms"][ARM_B]["candidates"][0].pop("artifact_sha256"), lambda r: r["arms"][ARM_C]["candidates"][0].pop("recomputed_validation_ap"),
                     lambda r: r["arms"][ARM_C]["candidates"][0].pop("weights_sha256")):
        with pytest.raises(SelectionRecordError):
            verify(env, mutated(env.record, mutation))


def test_the_evidence_type_no_longer_accepts_selected_only_models():
    fields = {f.name for f in dataclasses.fields(ValidationEvidence)}
    assert fields == {"arm_inputs", "labels_by_id", "frozen", "split_result", "artifact_dir", "artifact_digests"} and "models" not in fields
    with pytest.raises(TypeError):
        ValidationEvidence(arm_inputs=None, labels_by_id={}, frozen=None, split_result=None, models={})  # type: ignore[call-arg]


# -- manifest binding -------------------------------------------------------------------------------------------------------------------------


def test_the_record_binds_the_exact_manifest_digest(env):
    other = mf.build_manifest(purpose=mf.PURPOSE_INFRASTRUCTURE, created_utc="2027-01-01T00:00:00Z", repo_root=env.repo, dataset=env.ds,
                              split_result=env.split, partition=env.part, audit_report=AUDIT_OK, train_config=env.cfg,
                              numerical=rn.prepare_numerical_configuration(env.cfg),
                              preflight=rn.preflight_report(env.cfg, env.part.validation, env.part.sealed_test), **env.kw)
    other_bytes = mf.canonical_bytes(other)
    other_sha = mf.sha256_hex(other_bytes)
    assert other_sha != env.manifest_sha
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, manifest_bytes=other_bytes, manifest_sha=other_sha)


def test_wrong_manifest_digest_or_bytes_are_rejected(env):
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, manifest_sha="0" * 64)
    flipped = bytearray(env.manifest_bytes)
    flipped[10] ^= 1
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, manifest_bytes=bytes(flipped))
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, manifest_sha="short")


def test_the_manifest_must_pass_its_own_independent_verifier(env):
    wrong_commit = dataclasses.replace(env.expected_run, commit="1" * 40)
    with pytest.raises(SelectionRecordVerificationError) as caught:
        verify(env, env.record, expected_run=wrong_commit)
    assert "manifest did not verify" in str(caught.value)
    (env.repo / "stray.txt").write_text("x", encoding="utf-8")
    try:
        with pytest.raises(SelectionRecordVerificationError):
            verify(env, env.record)
    finally:
        (env.repo / "stray.txt").unlink()
    assert verify(env, env.record).ok


@pytest.mark.parametrize("field,value", [("sha256", "a" * 64), ("git_commit", "b" * 40), ("git_branch", "other"), ("dataset_seed", 6102),
                                         ("dataset_fingerprint", "c" * 64), ("generator_fingerprint", "d" * 64), ("split_seed", 1),
                                         ("split_manifest_sha256", "e" * 64), ("protocol_sha256", "f" * 64), ("version", "pre-run-manifest-2"),
                                         ("purpose", "confirmatory"), ("protocol_version", "x")])
def test_manifest_identity_fields_in_the_record_must_equal_the_manifest(env, field, value):
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, lambda r: r["manifest"].update({field: value})))


def test_a_stale_record_after_the_repository_moves_on_is_rejected(env):
    (env.repo / "docs/extra.md").write_text("later\n", encoding="utf-8")
    git(env.repo, "add", "-A")
    git(env.repo, "commit", "-q", "-m", "later")
    try:
        with pytest.raises(SelectionRecordVerificationError):
            verify(env, env.record)
    finally:
        git(env.repo, "reset", "-q", "--hard", env.expected_run.commit)
    assert verify(env, env.record).ok


# -- candidate table and winner ---------------------------------------------------------------------------------------------------------------------------


def candidate_mutations():
    return [
        ("missing", lambda r: r["arms"][ARM_A]["candidates"].pop()),
        ("duplicate", lambda r: r["arms"][ARM_A]["candidates"].append(copy.deepcopy(r["arms"][ARM_A]["candidates"][0]))),
        ("extra", lambda r: r["arms"][ARM_A]["candidates"].append({**r["arms"][ARM_A]["candidates"][0], "candidate_id": "extra"})),
        ("id_mismatch", lambda r: r["arms"][ARM_A]["candidates"][3].update(candidate_id="A_request_only|nb=00001|ep=0001|c=0.10")),
        ("id_param_mismatch", lambda r: r["arms"][ARM_A]["candidates"][3].update(epochs=400 if r["arms"][ARM_A]["candidates"][3]["epochs"] != 400 else 50)),
        ("unexpected_buckets", lambda r: r["arms"][ARM_A]["candidates"][3].update(num_buckets=999)),
        ("unexpected_c", lambda r: r["arms"][ARM_A]["candidates"][3].update(relative_step_c=0.5)),
        ("ap_none", lambda r: r["arms"][ARM_A]["candidates"][3].update(validation_ap=None)),
        ("ap_negative", lambda r: r["arms"][ARM_A]["candidates"][3].update(validation_ap=-0.1)),
        ("ap_above_one", lambda r: r["arms"][ARM_A]["candidates"][3].update(validation_ap=1.5)),
        ("ap_string", lambda r: r["arms"][ARM_A]["candidates"][3].update(validation_ap="0.5")),
        ("ap_claim_changed", lambda r: r["arms"][ARM_A]["candidates"][3].update(validation_ap=r["arms"][ARM_A]["candidates"][3]["validation_ap"] + 1e-6)),
        ("recomputed_ap_changed", lambda r: r["arms"][ARM_B]["candidates"][4].update(recomputed_validation_ap=0.99)),
        ("rounded_wrong", lambda r: r["arms"][ARM_A]["candidates"][3].update(rounded_ap=0.123)),
        ("lr_wrong", lambda r: r["arms"][ARM_A]["candidates"][3].update(learning_rate=0.5)),
        ("lr_hex_wrong", lambda r: r["arms"][ARM_A]["candidates"][3].update(learning_rate_hex="0x1.0p+0")),
        ("artifact_digest_changed", lambda r: r["arms"][ARM_C]["candidates"][7].update(artifact_sha256="1" * 64)),
        ("artifact_digest_swapped", lambda r: r["arms"][ARM_C]["candidates"][7].update(artifact_sha256=r["arms"][ARM_C]["candidates"][8]["artifact_sha256"])),
        ("weights_hash_changed", lambda r: r["arms"][ARM_C]["candidates"][7].update(weights_sha256="2" * 64)),
        ("winner_claim_changed", lambda r: r["arms"][ARM_A]["selected"].update(candidate_id=cid_of(ARM_A, 4096, 400, 0.8), num_buckets=4096, epochs=400,
                                                                                relative_step_c=0.8)),
        ("selected_ap_changed", lambda r: r["arms"][ARM_A]["selected"].update(validation_ap=r["arms"][ARM_A]["selected"]["validation_ap"] + 0.01)),
        ("selected_artifact_changed", lambda r: r["arms"][ARM_A]["selected"].update(artifact_sha256="3" * 64)),
        ("selected_lr_changed", lambda r: r["arms"][ARM_A]["selected"].update(learning_rate=0.01)),
        ("selected_L_changed", lambda r: r["arms"][ARM_A]["selected"].update(L=1.0)),
        ("lambda_method_changed", lambda r: r["arms"][ARM_A]["selected"].update(lambda_method="lambda-power-0")),
        ("verification_flag_changed", lambda r: r["arms"][ARM_A]["candidate_verification"].update(all_validation_aps_recomputed=False)),
        ("verification_count_changed", lambda r: r["arms"][ARM_A]["candidate_verification"].update(candidates_verified=26)),
        ("artifacts_count_changed", lambda r: r["artifacts"].update(candidates=80)),
        ("artifacts_manifest_changed", lambda r: r["artifacts"].update(manifest_sha256="4" * 64)),
        ("trace_changed", lambda r: r["arms"][ARM_A]["tie_break_trace"][0].update(survivors=5)),
        ("extra_arm", lambda r: r["arms"].update({"D_other": copy.deepcopy(r["arms"][ARM_A])})),
        ("missing_arm", lambda r: r["arms"].pop(ARM_C)),
    ]


@pytest.mark.parametrize("name,mutation", candidate_mutations(), ids=[n for n, _ in candidate_mutations()])
def test_candidate_table_violations_are_rejected(env, name, mutation):
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, mutation))


def test_results_with_a_wrong_split_label_or_fingerprint_cannot_be_built(env):
    for change in (dict(split="test"), dict(split="train"), dict(validation_ids_sha256="0" * 64), dict(validation_records=3)):
        results = {arm: list(rows) for arm, rows in env.results.items()}
        results[ARM_A][4] = dataclasses.replace(results[ARM_A][4], **change)
        with pytest.raises(SelectionRecordError):
            build_record(env, results, env.evidence)


def test_missing_or_duplicate_candidates_cannot_be_built(env):
    for broken in (lambda rows: rows[:-1], lambda rows: rows + [rows[0]]):
        results = {arm: list(rows) for arm, rows in env.results.items()}
        results[ARM_C] = broken(results[ARM_C])
        with pytest.raises(SelectionRecordError):
            build_record(env, results, env.evidence)
    with pytest.raises(SelectionRecordError):
        build_record(env, {ARM_A: env.results[ARM_A]}, env.evidence)


# -- rounding and every tie-break level -----------------------------------------------------------------------------------------------------------------------


def test_a_27_way_exact_tie_is_resolved_by_fewer_epochs_then_smaller_buckets_then_smaller_c(env, tmp_path):
    overrides = {cid_of(ARM_A, nb, ep, c): zero_model(nb) for (nb, ep, c) in GRID}  # every logit equal: all 27 APs equal the prevalence exactly
    record, evidence, results, aps = scenario(env, tmp_path, overrides)
    assert len({aps[cid_of(ARM_A, nb, ep, c)] for (nb, ep, c) in GRID}) == 1
    block = record["arms"][ARM_A]
    assert block["selected"]["candidate_id"] == cid_of(ARM_A, 1024, 50, 0.1) and block["selected"]["tied_at_rounded_ap"] == 27
    assert [s["rule"] for s in block["tie_break_trace"]] == ["highest_rounded_validation_ap", "fewer_epochs", "smaller_num_buckets",
                                                              "smaller_relative_step_c", "candidate_id_lexicographic"]
    assert [s["survivors"] for s in block["tie_break_trace"]] == [27, 9, 3, 1, 1]
    assert verify(env, record, evidence=evidence).ok
    other_arms = {arm: record["arms"][arm]["selected"]["candidate_id"] for arm in (ARM_B, ARM_C)}
    assert other_arms == {arm: env.winners[arm].candidate_id for arm in (ARM_B, ARM_C)}  # untouched arms keep their honest winners


def test_ties_between_identical_models_at_the_top_follow_the_epoch_then_c_rules(env, tmp_path):
    arm = ARM_B
    best = env.winners[arm]
    nb = best.num_buckets
    twin = lambda ep, c: (cid_of(arm, nb, ep, c), env.models[best.candidate_id])  # the best model's exact parameters under other hyperparameters
    group = dict([twin(400, 0.1), twin(150, 0.8), twin(50, 0.8), twin(50, 0.3)])
    overrides = {}
    for cid, model in group.items():
        copy_model = LightweightAnomalyModel(num_buckets=nb)
        copy_model.weights, copy_model.bias, copy_model.is_trained = model.weights.copy(), model.bias, True
        overrides[cid] = copy_model
    record, evidence, results, aps = scenario(env, tmp_path, overrides)
    tied = [cid for cid in {best.candidate_id, *group} if aps[cid] == aps[best.candidate_id]]
    assert len(tied) == len({best.candidate_id, *group})  # all of them tie exactly at the top
    parse = lambda cid: next((nb_, ep, c) for (nb_, ep, c) in GRID if cid_of(arm, nb_, ep, c) == cid)
    expected = min(tied, key=lambda cid: (parse(cid)[1], parse(cid)[0], parse(cid)[2]))
    block = record["arms"][arm]
    assert block["selected"]["candidate_id"] == expected and block["selected"]["tied_at_rounded_ap"] == len(tied)
    assert block["tie_break_trace"][0]["survivors"] == len(tied) and block["tie_break_trace"][-1]["survivors"] == 1
    assert verify(env, record, evidence=evidence).ok


def test_ap_rounding_to_ten_decimals_is_part_of_the_recomputed_selection(env):
    for arm in ARMS:
        win = env.winners[arm]
        rows = {r["candidate_id"]: r for r in env.record["arms"][arm]["candidates"]}
        assert rows[win.candidate_id]["rounded_ap"] == round(rows[win.candidate_id]["recomputed_validation_ap"], 10)
        assert max(r["rounded_ap"] for r in rows.values()) == env.record["arms"][arm]["selected"]["rounded_ap"]


# -- learning rates, numerical bindings ------------------------------------------------------------------------------------------------------------------------


def test_learning_rates_must_equal_the_manifests_train_only_rates(env):
    for arm in ARMS:
        for index in (0, 13, 26):
            with pytest.raises(SelectionRecordError):
                verify(env, mutated(env.record, lambda r, a=arm, i=index: r["arms"][a]["candidates"][i].update(learning_rate=1e-9)))
    manifest = copy.deepcopy(env.manifest)
    manifest["numerics"]["candidates"][0]["learning_rate"] = 0.5
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, manifest_bytes=mf.canonical_bytes(manifest))


# -- candidate artifacts ------------------------------------------------------------------------------------------------------------------------------------


def test_a_tampered_candidate_artifact_is_rejected(env, tmp_path):
    directory = fresh_artifacts(env, tmp_path)
    cid = cid_of(ARM_C, 2048, 150, 0.3)
    path = artifact_path(env, directory, cid)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(SelectionRecordVerificationError) as caught:
        verify(env, env.record, evidence=evidence_with(env, artifact_dir=directory))
    assert "artifacts did not verify" in str(caught.value)


def test_a_substituted_artifact_is_rejected_even_when_the_trusted_digests_are_swapped_too(env, tmp_path):
    directory = fresh_artifacts(env, tmp_path)
    a, b = cid_of(ARM_A, 1024, 50, 0.1), cid_of(ARM_A, 1024, 50, 0.3)
    pa, pb = artifact_path(env, directory, a), artifact_path(env, directory, b)
    da, db = pa.read_bytes(), pb.read_bytes()
    pa.write_bytes(db)
    pb.write_bytes(da)
    with pytest.raises(SelectionRecordVerificationError):  # the digests no longer match the files
        verify(env, env.record, evidence=evidence_with(env, artifact_dir=directory))
    swapped = {**env.digests, a: env.digests[b], b: env.digests[a]}
    with pytest.raises(SelectionRecordVerificationError):  # digests follow the files, but each file now claims the other candidate's identity
        verify(env, env.record, evidence=evidence_with(env, artifact_dir=directory, artifact_digests=swapped))


def test_a_missing_artifact_file_is_rejected(env, tmp_path):
    directory = fresh_artifacts(env, tmp_path)
    artifact_path(env, directory, cid_of(ARM_B, 4096, 400, 0.8)).unlink()
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, artifact_dir=directory))


def test_missing_extra_duplicate_and_malformed_trusted_digests_are_rejected(env):
    cid = cid_of(ARM_A, 1024, 50, 0.1)
    other = cid_of(ARM_A, 1024, 50, 0.3)
    for digests in ({k: v for k, v in env.digests.items() if k != cid}, {**env.digests, "extra": "0" * 64}, {**env.digests, other: env.digests[cid]},
                    {**env.digests, cid: "short"}, {**env.digests, cid: None}, {}, None, list(env.digests)):
        with pytest.raises(SelectionRecordVerificationError):
            verify(env, env.record, evidence=evidence_with(env, artifact_digests=digests))  # type: ignore[arg-type]


def test_a_wrong_trusted_digest_for_one_candidate_is_rejected(env):
    cid = cid_of(ARM_C, 1024, 400, 0.8)
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, artifact_digests={**env.digests, cid: "0" * 64}))


def test_digests_copied_from_the_record_are_not_a_trust_authority(env, tmp_path):
    """An attacker edits an artifact and updates the digest *in the record*; the verifier's independently supplied digest still catches it."""
    directory = fresh_artifacts(env, tmp_path)
    cid = cid_of(ARM_B, 2048, 50, 0.3)
    path = artifact_path(env, directory, cid)
    forged = json.loads(path.read_bytes())
    forged["model"]["bias_hex"] = (0.5).hex()
    forged["model"]["fingerprint"]["bias_hex"] = (0.5).hex()
    new_bytes = mf.canonical_bytes(forged)
    path.write_bytes(new_bytes)
    new_digest = mf.sha256_hex(new_bytes)
    record = mutated(env.record, lambda r: [row.update(artifact_sha256=new_digest) for row in r["arms"][ARM_B]["candidates"] if row["candidate_id"] == cid])
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, record, evidence=evidence_with(env, artifact_dir=directory))  # trusted digests unchanged
    # the record's own copy of the digest is only ever compared with the trusted one, never used to locate or authenticate a file
    assert record["arms"][ARM_B]["candidates"] != env.record["arms"][ARM_B]["candidates"]


def forged_artifact(env, tmp_path, cid, **over):
    """Replace one candidate's artifact by a validly persisted artifact with some wrong binding; returns evidence that trusts the new file's digest."""
    directory = fresh_artifacts(env, tmp_path)
    arm = next(a for a in ARMS if cid.startswith(a))
    nb, ep, c = next(g for g in GRID if cid_of(arm, *g) == cid)
    path = artifact_path(env, directory, cid)
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    path.unlink()
    kwargs = dict(repo_root=env.repo, arm=arm, num_buckets=nb, epochs=ep, relative_step_c=c, learning_rate=ma.learning_rates_from_manifest(env.manifest)[cid],
                  manifest_sha256=env.manifest_sha, training=training_for(env.manifest, arm, nb))
    kwargs.update(over)
    model = kwargs.pop("model", env.models[cid])
    written = ma.write_model_artifact(directory, model, **kwargs)
    return evidence_with(env, artifact_dir=directory, artifact_digests={**env.digests, cid: written.sha256})


def test_an_artifact_bound_to_another_manifest_is_rejected(env, tmp_path):
    cid = cid_of(ARM_A, 2048, 150, 0.8)
    directory = fresh_artifacts(env, tmp_path)
    other_manifest = "d" * 64
    rates = ma.learning_rates_from_manifest(env.manifest)
    artifact_path(env, directory, cid).unlink()
    written = ma.write_model_artifact(directory, env.models[cid], repo_root=env.repo, arm=ARM_A, num_buckets=2048, epochs=150, relative_step_c=0.8,
                                      learning_rate=rates[cid], manifest_sha256=other_manifest, training=training_for(env.manifest, ARM_A, 2048))
    placed = directory / ma.artifact_filename(env.manifest_sha, cid)
    placed.write_bytes(Path(written.path).read_bytes())  # placed where the verified manifest's lookup expects it
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, artifact_dir=directory, artifact_digests={**env.digests, cid: written.sha256}))


def test_an_artifact_with_a_wrong_learning_rate_is_rejected(env, tmp_path):
    cid = cid_of(ARM_A, 1024, 150, 0.3)
    est = training_for(env.manifest, ARM_A, 1024)
    other = {**est, "lambda_max": est["lambda_max"] * 2, "L": est["L"] * 2}
    other["L"] = other["lambda_max"] / 4
    evidence = forged_artifact(env, tmp_path, cid, training=other, learning_rate=ma.numerics.relative_learning_rate(0.3, other["L"]))
    with pytest.raises(SelectionRecordVerificationError):  # internally consistent artifact, but not the manifest's train-only learning rate
        verify(env, env.record, evidence=evidence)


def test_an_artifact_with_different_training_provenance_is_rejected(env, tmp_path):
    cid = cid_of(ARM_B, 1024, 50, 0.1)
    for over in (dict(train_ids_sha256="9" * 64), dict(vocabulary_sha256="8" * 64), dict(max_length=77)):
        sub = tmp_path / f"p{len(over)}{list(over)[0]}"
        sub.mkdir()
        evidence = forged_artifact(env, sub, cid, training={**training_for(env.manifest, ARM_B, 1024), **over})
        with pytest.raises(SelectionRecordVerificationError) as caught:
            verify(env, env.record, evidence=evidence)
        assert "training provenance" in str(caught.value)


def test_an_artifact_for_a_stale_model_source_is_rejected(env, tmp_path):
    directory = fresh_artifacts(env, tmp_path)
    cid = cid_of(ARM_C, 4096, 50, 0.1)
    path = artifact_path(env, directory, cid)
    forged = json.loads(path.read_bytes())
    forged["model_source_sha256"] = "7" * 64
    new_bytes = mf.canonical_bytes(forged)
    path.write_bytes(new_bytes)
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, artifact_dir=directory, artifact_digests={**env.digests, cid: mf.sha256_hex(new_bytes)}))


def test_the_artifact_directory_must_exist_and_be_a_directory(env, tmp_path):
    some_file = tmp_path / "file"
    some_file.write_text("x", encoding="utf-8")
    for directory in (tmp_path / "missing", some_file, None, 5):
        with pytest.raises(SelectionRecordVerificationError):
            verify(env, env.record, evidence=evidence_with(env, artifact_dir=directory))


# -- evidence ------------------------------------------------------------------------------------------------------------------------------------


def test_evidence_is_mandatory_and_validated(env):
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=object())  # type: ignore[arg-type]
    with pytest.raises(SelectionRecordVerificationError):
        verify_selection_record_bytes(mf.canonical_bytes(env.record), expected_sha256=mf.sha256_hex(mf.canonical_bytes(env.record)),
                                      manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha, repo_root=env.repo,
                                      expected_run=env.expected_run, evidence=None, **env.kw)  # type: ignore[arg-type]
    import inspect
    params = inspect.signature(verify_selection_record_bytes).parameters
    assert {"expected_sha256", "manifest_bytes", "expected_manifest_sha256", "evidence", "expected_run"} <= set(params)
    assert not any(name.startswith(("trust", "skip", "mock", "allow_unverified", "verified")) for name in params)


def test_vocabulary_and_budget_provenance_must_match_the_manifest(env):
    for mutation in (lambda r: r["vocabulary"].update(sha256="6" * 64), lambda r: r["vocabulary"].update(min_frequency=1),
                     lambda r: r["vocabulary"].update(fitted_on_split="validation"), lambda r: r["budget"].update(selected_max_length=512),
                     lambda r: r["budget"].update(rule_id="budget-rule-2")):
        with pytest.raises(SelectionRecordError):
            verify(env, mutated(env.record, mutation))


def test_the_supplied_frozen_vocabulary_must_match_the_manifest(env):
    class Other:
        record = {"sha256": "5" * 64}

    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, frozen=Other()))


def test_validation_split_binding(env):
    for mutation in (lambda r: r["validation"].update(ids_sha256="4" * 64), lambda r: r["validation"].update(records=5),
                     lambda r: r["validation"].update(split="test")):
        with pytest.raises(SelectionRecordError):
            verify(env, mutated(env.record, mutation))
    other_split = split_records(env.ds.records, seed=PRIMARY_SPLIT_SEED + 1)
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, split_result=other_split))
    shifted = {k: v for k, v in list(env.labels.items())[:-1]}
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, labels_by_id=shifted))


def test_wrong_or_misaligned_validation_labels_change_the_recomputed_results_and_are_rejected(env):
    flipped = {k: 1 - v for k, v in env.labels.items()}
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, labels_by_id=flipped))
    ids = list(env.labels)
    swapped = dict(env.labels)
    a = next(i for i in ids if env.labels[i] == 0)
    b = next(i for i in ids if env.labels[i] == 1)
    swapped[a], swapped[b] = swapped[b], swapped[a]  # two records exchange labels: ids stay aligned, one pairing is wrong
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, labels_by_id=swapped))
    renamed = {f"{k}-x": v for k, v in env.labels.items()}
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, labels_by_id=renamed))


def test_test_split_inputs_cannot_serve_as_validation_evidence(env):
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, arm_inputs=dataclasses.replace(env.val_inputs, split="test")))
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=evidence_with(env, arm_inputs=env.cfg.inputs))


# -- threshold -------------------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,mutation", [
    ("value_changed", lambda r: r["arms"][ARM_A]["threshold"].update(value=r["arms"][ARM_A]["threshold"]["value"] + 0.5)),
    ("claim_all_negative", lambda r: r["arms"][ARM_A]["threshold"].update(value=None, predicts_all_negative=True)),
    ("all_negative_flag_only", lambda r: r["arms"][ARM_A]["threshold"].update(predicts_all_negative=True)),
    ("probability_scale", lambda r: r["arms"][ARM_A]["threshold"].update(scale="probability")),
    ("probability_flag", lambda r: r["arms"][ARM_A]["threshold"].update(probability_threshold=True)),
    ("basis_changed", lambda r: r["arms"][ARM_A]["threshold"].update(basis="test_scores")),
    ("rule_changed", lambda r: r["arms"][ARM_A]["threshold"].update(rule="max_test_f1")),
    ("counts_changed", lambda r: r["arms"][ARM_A]["threshold"]["counts"].update(tp=r["arms"][ARM_A]["threshold"]["counts"]["tp"] + 1)),
    ("f1_changed", lambda r: r["arms"][ARM_A]["threshold"].update(validation_f1=0.999)),
    ("metrics_changed", lambda r: r["arms"][ARM_A]["threshold"]["metrics"].update(precision=1.0)),
    ("status_changed", lambda r: r["arms"][ARM_A]["threshold"]["metric_status"].update(precision="undefined")),
    ("candidates_considered", lambda r: r["arms"][ARM_A]["threshold"].update(candidates_considered=3)),
    ("tie_diag_changed", lambda r: r["arms"][ARM_C]["tie_diagnostics"].update(distinct_values=1)),
    ("compat_changed", lambda r: r["arms"][ARM_B]["model"]["readout_compatibility"].update(checked=1)),
])
def test_unverifiable_or_invalid_threshold_claims_are_rejected(env, name, mutation):
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, mutation))


def test_the_threshold_is_recomputed_from_the_selected_candidates_own_artifact(env, tmp_path):
    """Swap the selected candidate's artifact for a different valid one that keeps the recorded AP (a twin with a perturbed weight): the threshold changes."""
    arm = ARM_A
    win = env.winners[arm]
    twin = LightweightAnomalyModel(num_buckets=win.num_buckets)
    twin.weights, twin.bias, twin.is_trained = env.models[win.candidate_id].weights.copy(), env.models[win.candidate_id].bias + 1e-3, True
    evidence = forged_artifact(env, tmp_path, win.candidate_id, model=twin)
    with pytest.raises(SelectionRecordVerificationError):  # a different bias changes the recorded model fingerprint and threshold
        verify(env, env.record, evidence=evidence)


def test_threshold_none_is_a_valid_all_negative_representation():
    selection = ThresholdSelection(rule="max_validation_f1_ties_to_highest_threshold", threshold=None, f1=0.0,
                                   validation_counts=None, candidates_considered=4)  # type: ignore[arg-type]
    block = sr.threshold_block(selection, [0.3, -1.0, 2.0, 0.1], [1, 0, 1, 0])
    assert block["value"] is None and block["predicts_all_negative"] is True and block["scale"] == "raw_logit"
    assert block["counts"] == {"tp": 0, "fp": 0, "tn": 2, "fn": 2}
    assert block["metrics"]["precision"] is None and block["metrics"]["recall"] == 0.0 and block["metrics"]["f1"] == 0.0
    assert block["metric_status"]["precision"] == "undefined" and block["metric_status"]["recall"] == "defined"
    json.dumps(block, allow_nan=False)
    problems: list[str] = []
    sr._scan({"threshold": block}, "$", problems)
    assert problems == []


def test_threshold_block_for_a_logit_cut_is_inclusive_and_labelled_raw_logit():
    selection = ThresholdSelection(rule="max_validation_f1_ties_to_highest_threshold", threshold=0.5, f1=1.0,
                                   validation_counts=None, candidates_considered=3)  # type: ignore[arg-type]
    block = sr.threshold_block(selection, [0.5, 0.4, 2.0], [1, 0, 1])
    assert block["counts"] == {"tp": 2, "fp": 0, "tn": 1, "fn": 0} and block["value"] == 0.5 and block["probability_threshold"] is False


# -- leakage and forbidden content --------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(test_labels=[0, 1]),
    lambda r: r.update(test_scores=[0.1]),
    lambda r: r.update(test_predictions=[1]),
    lambda r: r["arms"][ARM_A].update(test_scores=[0.1]),
    lambda r: r["arms"][ARM_A]["threshold"].update(test_f1=0.5),
    lambda r: r["arms"][ARM_A]["selected"].update(test_ap=0.5),
    lambda r: r["validation"].update(test_records=5),
    lambda r: r["validation"].update(labels=[0, 1]),
    lambda r: r["arms"][ARM_A]["threshold"].update(scores=[0.1, 0.2]),
    lambda r: r["arms"][ARM_A].update(logits=[1.0]),
    lambda r: r["arms"][ARM_A].update(body="GET /secret"),
    lambda r: r["arms"][ARM_A]["model"].update(predictions=[1]),
    lambda r: r["arms"][ARM_A]["candidates"][0].update(test_validation_ap=0.5),
    lambda r: r["artifacts"].update(test_artifacts=[1]),
    lambda r: r["manifest"].update(request_bodies=["x"]),
    lambda r: r["test_status"].update(test_scored=True),
    lambda r: r["test_status"].update(test_labels_accessed=True),
    lambda r: r["test_status"].update(test_scores_accessed=True),
    lambda r: r["test_status"].update(test_data_in_record=True),
    lambda r: r.pop("test_status"),
    lambda r: r.update(extra_field="x"),
    lambda r: r.pop("manifest"),
    lambda r: r["limitations"].pop(),
    lambda r: r.update(created_utc="soon"),
])
def test_test_derived_or_forbidden_content_is_rejected(env, mutation):
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, mutation))


def test_the_record_contains_no_request_content_labels_or_scores(env):
    text = mf.canonical_bytes(env.record).decode("ascii")
    for rec in env.ds.records[:200]:
        assert rec.features.path not in text
    assert text.count('"validation_ap"') >= 3 * 27 and text.count('"artifact_sha256"') >= 3 * 27
    problems: list[str] = []
    sr._scan(env.record, "$", problems)
    assert problems == []
    assert not any(k in env.record for k in ("labels", "scores", "predictions", "test_labels", "test_scores"))


def test_errors_do_not_echo_record_values(env):
    secret = "SECRET-VALUE-424242"
    for mutation in (lambda r: r["manifest"].update(git_branch=secret), lambda r: r["arms"][ARM_A]["selected"].update(candidate_id=secret),
                     lambda r: r["arms"][ARM_B]["model"].update(weights_sha256=secret), lambda r: r["vocabulary"].update(sha256=secret),
                     lambda r: r["arms"][ARM_C]["candidates"][0].update(validation_ap=secret), lambda r: r["arms"][ARM_C]["candidates"][1].update(artifact_sha256=secret),
                     lambda r: r.update(**{secret: 1})):
        with pytest.raises(SelectionRecordError) as caught:
            verify(env, mutated(env.record, mutation))
        assert secret not in str(caught.value)


def test_artifact_failures_do_not_echo_digests_or_parameters(env, tmp_path):
    directory = fresh_artifacts(env, tmp_path)
    cid = cid_of(ARM_C, 2048, 150, 0.3)
    path = artifact_path(env, directory, cid)
    data = bytearray(path.read_bytes())
    data[100] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(SelectionRecordVerificationError) as caught:
        verify(env, env.record, evidence=evidence_with(env, artifact_dir=directory))
    message = str(caught.value)
    assert env.digests[cid] not in message and mf.sha256_hex(bytes(data)) not in message and str(env.aps[cid]) not in message


# -- no scoring authorization ---------------------------------------------------------------------------------------------------------------------------------------


def test_a_verified_record_never_authorizes_test_scoring(env):
    result = verify(env, env.record)
    assert result.authorizes_test_scoring is False and result.test_scored is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.authorizes_test_scoring = True  # type: ignore[misc]
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_test_access()
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_training_authorization()
    assert rn.TEST_SCORING_AVAILABLE is False and rn.TRAINING_AUTHORIZATION_AVAILABLE is False


def test_module_has_no_attempt_marker_unsealing_training_or_scoring_code():
    text = Path(sr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for name in ("create_attempt_marker", "unseal", "unseal_test", "score_test", "fit_candidate", "authorize_test_scoring", "request_test_access"):
        assert not hasattr(sr, name)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not called & {"fit", "predict", "paired_group_bootstrap", "request_test_access", "request_training_authorization"}
    assert "meta" not in attrs and "blinded_records" not in attrs
    assert not any(seed in text for seed in ("7101", "7102", "7103"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "SealedTestSet" not in imported


def test_the_selection_record_module_is_covered_by_the_manifest_source_list():
    assert "tokenization_ai/heldout_eval/selection_record.py" in mf.DEFAULT_SOURCE_FILES
