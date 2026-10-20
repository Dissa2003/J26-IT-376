"""Tests for validation-selection-record-1 (draft protocol heldout-eval-1.2.0-draft, sections 12.11.4 and 12.12.3).

No research model is trained or scored. Fitted models are *mock artifacts*: untrained ``LightweightAnomalyModel`` objects with explicitly assigned
random weights, so the AP and threshold the verifier recomputes are properties of the fixture, not detection results. Dataset objects come from the
development seed 6101 (structure only); Git state lives in a throw-away repository under pytest's temporary directory. No confirmatory seed is used.
To keep the suite fast, ``numerics.raw_logits`` is wrapped by a cache keyed on the exact weights, bias and inputs; every cache miss calls the real function.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
import random
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

import tokenization_ai.heldout_eval.manifest as mf
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

SRC = ("src/a.py", "src/b.py")
CFG = ("requirements.txt",)
PROTO = "docs/PROTOCOL.md"
HEADER = "# Protocol\n\n**Protocol version:** `heldout-eval-test-draft`  **Status: DRAFT, not frozen.**\n"
AUDIT_OK = {"errors": [], "warnings": [], "construction_checks_passed": True}
CREATED = "2026-01-01T00:00:00Z"
WINNERS = {ARM_A: (1024, 150, 0.3), ARM_B: (2048, 50, 0.8), ARM_C: (4096, 400, 0.1)}
GRID = [(nb, ep, c) for nb in (1024, 2048, 4096) for ep in (50, 150, 400) for c in (0.1, 0.3, 0.8)]


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


def mock_model(arm: str, nb: int) -> LightweightAnomalyModel:
    rng = np.random.default_rng({"A_request_only": 11, "B_simulated_rules": 12, "C_char3": 13}[arm] + nb)
    model = LightweightAnomalyModel(num_buckets=nb)
    model.weights = rng.normal(0.0, 0.02, nb)
    model.bias = 0.05
    model.is_trained = True  # explicit mock artifact: no fit
    return model


class Env:
    pass


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env()
    root = tmp_path_factory.mktemp("sel")
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
    e.manifest_bytes = Path(written.path).read_bytes()
    e.manifest_sha = written.sha256
    e.manifest = mf.parse_manifest_bytes(e.manifest_bytes)
    e.val_inputs = rn.encode_validation(cfg, part.validation)
    e.labels = part.validation.labels_by_id()
    e.split, e.cfg, e.part, e.ds = split, cfg, part, ds
    e.models = {arm: mock_model(arm, WINNERS[arm][0]) for arm in ARMS}
    e.evidence = ValidationEvidence(arm_inputs=e.val_inputs, labels_by_id=e.labels, frozen=cfg.frozen, split_result=split, models=e.models)
    e.ids_fp = rn.ids_fingerprint(list(e.val_inputs.record_ids))
    e.actual_ap = {arm: actual_ap(e, arm, e.models[arm]) for arm in ARMS}
    e.results = make_results(e, WINNERS)
    e.record = build_selection_record(manifest_bytes=e.manifest_bytes, expected_manifest_sha256=e.manifest_sha, results_by_arm=e.results,
                                      evidence=e.evidence, created_utc=CREATED)
    return e


def actual_ap(e, arm, model) -> float:
    inputs = list(getattr(e.val_inputs, {ARM_A: "arm_a", ARM_B: "arm_b", ARM_C: "arm_c"}[arm]))
    labels = rn.aligned_labels(e.val_inputs.record_ids, e.labels)
    return average_precision(numerics.raw_logits(model, inputs), list(labels))


def make_results(e, winners, tied=None, ap_override=None):
    """27 results per arm: the ``winners`` candidate carries the real AP of its mock model, every other candidate is clearly lower (``tied`` ones equal)."""
    out = {}
    for arm in ARMS:
        ap = e.actual_ap[arm] if ap_override is None or arm not in ap_override else ap_override[arm]
        rows = []
        for k, (nb, ep, c) in enumerate(GRID):
            if (nb, ep, c) == winners[arm] or (tied and (nb, ep, c) in tied.get(arm, ())):
                value = ap
            else:
                value = ap * (0.30 + 0.001 * k)
            rows.append(ValidationCandidateResult(rn.canonical_candidate_id(arm, nb, ep, c), arm, nb, ep, c, value, "validation", e.ids_fp,
                                                  len(e.val_inputs.record_ids)))
        out[arm] = rows
    return out


def scenario(e, tied):
    """Tie scenario: for each arm the set ``tied`` all carry the real AP of the model of the *rule's* winner; returns (record, evidence)."""
    winners = {}
    for arm, group in tied.items():
        winners[arm] = min(group, key=lambda t: (t[1], t[0], t[2]))  # epochs, num_buckets, c
    models = {arm: mock_model(arm, winners[arm][0]) for arm in ARMS}
    aps = {arm: actual_ap(e, arm, models[arm]) for arm in ARMS}
    results = make_results(e, winners, tied=tied, ap_override=aps)
    evidence = dataclasses.replace(e.evidence, models=models)
    record = build_selection_record(manifest_bytes=e.manifest_bytes, expected_manifest_sha256=e.manifest_sha, results_by_arm=results,
                                    evidence=evidence, created_utc=CREATED)
    return record, evidence, results


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
    assert r["record_version"] == "validation-selection-record-1" and set(r) == sr._TOP_KEYS and set(r["arms"]) == set(ARMS)
    assert r["manifest"]["sha256"] == env.manifest_sha and r["manifest"]["git_commit"] == env.expected_run.commit
    assert r["manifest"]["split_manifest_sha256"] == env.split.manifest_sha256 and r["manifest"]["protocol_sha256"] == env.manifest["protocol"]["sha256"]
    assert r["validation"] == {"split": "validation", "records": len(env.val_inputs.record_ids), "ids_sha256": env.ids_fp}
    assert r["test_status"] == {"test_scored": False, "test_labels_accessed": False, "test_scores_accessed": False, "test_data_in_record": False}
    assert r["vocabulary"]["sha256"] == env.manifest["arms"]["vocabulary"]["sha256"] and r["budget"]["rule_id"] == "budget-rule-1"
    assert r["statistics"] == env.manifest["statistics"] and r["rules"] == env.manifest["selection"]
    assert r["code"]["selection_record_sha256"] == sr.source_sha256()
    json.loads(mf.canonical_bytes(r))
    sr.validate_record_structure(r)


def test_per_arm_blocks_hold_the_approved_candidates_selection_and_artifacts(env):
    for arm in ARMS:
        block = env.record["arms"][arm]
        nb, ep, c = WINNERS[arm]
        sel = block["selected"]
        assert (sel["num_buckets"], sel["epochs"], sel["relative_step_c"]) == (nb, ep, c) and sel["candidate_id"] == rn.canonical_candidate_id(arm, nb, ep, c)
        assert sel["lambda_method"] == "lambda-power-1" and sel["learning_rate"] == c / sel["L"] and sel["L"] == sel["lambda_max"] / 4
        assert sel["validation_ap"] == env.actual_ap[arm] and sel["rounded_ap"] == round(env.actual_ap[arm], 10)
        rows = block["candidates"]
        assert len(rows) == 27 and [r["candidate_id"] for r in rows] == sorted(r["candidate_id"] for r in rows)
        assert all(r["rounded_ap"] == round(r["validation_ap"], 10) for r in rows)
        manifest_lr = {x["candidate_id"]: x["learning_rate"] for x in env.manifest["numerics"]["candidates"]}
        assert all(r["learning_rate"] == manifest_lr[r["candidate_id"]] for r in rows)
        th = block["threshold"]
        assert th["scale"] == "raw_logit" and th["probability_threshold"] is False and th["basis"] == "validation_scores_and_labels_only"
        assert th["value"] is not None and th["predicts_all_negative"] is False and set(th["counts"]) == {"tp", "fp", "tn", "fn"}
        assert sum(th["counts"].values()) == len(env.val_inputs.record_ids)
        assert block["model"]["weights_sha256"] == sr.model_artifact_fingerprint(env.models[arm])["weights_sha256"]
        assert block["model"]["model_sha256"] == env.manifest["numerics"]["provenance"]["model_sha256"] and block["model"]["logit_method"] == "logit-readout-1"
        assert block["tie_diagnostics"]["validation_records"] == len(env.val_inputs.record_ids)
        assert block["tie_break_trace"][0]["rule"] == "highest_rounded_validation_ap" and block["tie_break_trace"][0]["survivors"] == 1


def test_building_is_deterministic_and_independent_of_candidate_order(env):
    again = build_selection_record(manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha, results_by_arm=env.results,
                                   evidence=env.evidence, created_utc=CREATED)
    assert mf.canonical_bytes(again) == mf.canonical_bytes(env.record)
    shuffled = {arm: random.Random(5).sample(rows, len(rows)) for arm, rows in env.results.items()}
    third = build_selection_record(manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha, results_by_arm=shuffled,
                                   evidence=env.evidence, created_utc=CREATED)
    assert mf.canonical_bytes(third) == mf.canonical_bytes(env.record)


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
    assert result.selected_candidates == {arm: rn.canonical_candidate_id(arm, *WINNERS[arm]) for arm in ARMS}


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
    bad = mutated(env.record, lambda r: r["arms"][ARM_A]["selected"].update(candidate_id=rn.canonical_candidate_id(ARM_A, 4096, 400, 0.8)))
    for name, record in (("bad1.json", bad), ("bad2.json", {}), ("bad3.json", {"record_version": sr.RECORD_VERSION})):
        with pytest.raises(SelectionRecordError):
            write(env, record, name=name)
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
                                 b'{"a":1}\n{"b":2}\n', b'{"record_version":"validation-selection-record-1",', b"\xff\xfe"])
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


# -- manifest binding -------------------------------------------------------------------------------------------------------------------------


def test_the_record_binds_the_exact_manifest_digest(env, tmp_path):
    other = mf.build_manifest(purpose=mf.PURPOSE_INFRASTRUCTURE, created_utc="2027-01-01T00:00:00Z", repo_root=env.repo, dataset=env.ds,
                              split_result=env.split, partition=env.part, audit_report=AUDIT_OK, train_config=env.cfg,
                              numerical=rn.prepare_numerical_configuration(env.cfg), preflight=rn.preflight_report(env.cfg, env.part.validation, env.part.sealed_test),
                              **env.kw)
    other_bytes = mf.canonical_bytes(other)
    other_sha = mf.sha256_hex(other_bytes)
    assert other_sha != env.manifest_sha
    with pytest.raises(SelectionRecordVerificationError):  # a genuine, verifiable manifest - but not the one the record names
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
            verify(env, env.record)  # dirty tree: the manifest no longer verifies, so neither does the record
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
            verify(env, env.record)  # HEAD is no longer the manifest's commit
    finally:
        git(env.repo, "reset", "-q", "--hard", env.expected_run.commit)
    assert verify(env, env.record).ok


# -- candidate table and winner ---------------------------------------------------------------------------------------------------------------------------


def candidate_mutations():
    cid = lambda nb, ep, c: rn.canonical_candidate_id(ARM_A, nb, ep, c)
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
        ("rounded_wrong", lambda r: r["arms"][ARM_A]["candidates"][3].update(rounded_ap=0.123)),
        ("lr_wrong", lambda r: r["arms"][ARM_A]["candidates"][3].update(learning_rate=0.5)),
        ("lr_hex_wrong", lambda r: r["arms"][ARM_A]["candidates"][3].update(learning_rate_hex="0x1.0p+0")),
        ("winner_claim_changed", lambda r: r["arms"][ARM_A]["selected"].update(candidate_id=cid(4096, 400, 0.8), num_buckets=4096, epochs=400, relative_step_c=0.8)),
        ("selected_ap_changed", lambda r: r["arms"][ARM_A]["selected"].update(validation_ap=r["arms"][ARM_A]["selected"]["validation_ap"] + 0.01)),
        ("selected_lr_changed", lambda r: r["arms"][ARM_A]["selected"].update(learning_rate=0.01)),
        ("selected_L_changed", lambda r: r["arms"][ARM_A]["selected"].update(L=1.0)),
        ("lambda_method_changed", lambda r: r["arms"][ARM_A]["selected"].update(lambda_method="lambda-power-0")),
        ("trace_changed", lambda r: r["arms"][ARM_A]["tie_break_trace"][0].update(survivors=5)),
        ("extra_arm", lambda r: r["arms"].update({"D_other": copy.deepcopy(r["arms"][ARM_A])})),
        ("missing_arm", lambda r: r["arms"].pop(ARM_C)),
    ]


@pytest.mark.parametrize("name,mutation", candidate_mutations(), ids=[n for n, _ in candidate_mutations()])
def test_candidate_table_violations_are_rejected(env, name, mutation):
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, mutation))


def test_a_claimed_winner_that_is_not_the_highest_ap_is_rejected(env):
    # raise another candidate's AP above the winner's, but leave the winner claim untouched
    def bump(r):
        rows = r["arms"][ARM_B]["candidates"]
        other = next(x for x in rows if x["candidate_id"] != r["arms"][ARM_B]["selected"]["candidate_id"])
        other["validation_ap"] = min(1.0, other["validation_ap"] + 0.5)
        other["rounded_ap"] = round(other["validation_ap"], 10)
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, mutated(env.record, bump))


def test_results_with_a_wrong_split_label_or_fingerprint_cannot_be_built(env):
    for change in (dict(split="test"), dict(split="train"), dict(validation_ids_sha256="0" * 64), dict(validation_records=3)):
        results = {arm: list(rows) for arm, rows in env.results.items()}
        results[ARM_A][4] = dataclasses.replace(results[ARM_A][4], **change)
        with pytest.raises(SelectionRecordError):
            build_selection_record(manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha, results_by_arm=results,
                                   evidence=env.evidence, created_utc=CREATED)


def test_missing_or_duplicate_candidates_cannot_be_built(env):
    for broken in (lambda rows: rows[:-1], lambda rows: rows + [rows[0]]):
        results = {arm: list(rows) for arm, rows in env.results.items()}
        results[ARM_C] = broken(results[ARM_C])
        with pytest.raises(SelectionRecordError):
            build_selection_record(manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha, results_by_arm=results,
                                   evidence=env.evidence, created_utc=CREATED)
    with pytest.raises(SelectionRecordError):
        build_selection_record(manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha,
                               results_by_arm={ARM_A: env.results[ARM_A]}, evidence=env.evidence, created_utc=CREATED)


# -- rounding and every tie-break level -----------------------------------------------------------------------------------------------------------------------


def verified_scenario(env, tied):
    record, evidence, _ = scenario(env, tied)
    assert verify(env, record, evidence=evidence).ok
    return record


def test_ties_at_ten_decimals_are_broken_by_fewer_epochs_first(env):
    tied = {arm: {(4096, 400, 0.1), (1024, 150, 0.8), (2048, 50, 0.3)} for arm in ARMS}
    record = verified_scenario(env, tied)
    block = record["arms"][ARM_A]
    assert block["selected"]["epochs"] == 50 and block["selected"]["num_buckets"] == 2048 and block["selected"]["tied_at_rounded_ap"] == 3
    assert [s["rule"] for s in block["tie_break_trace"]] == ["highest_rounded_validation_ap", "fewer_epochs", "smaller_num_buckets",
                                                              "smaller_relative_step_c", "candidate_id_lexicographic"]
    assert [s["survivors"] for s in block["tie_break_trace"]] == [3, 1, 1, 1, 1]


def test_ties_with_equal_epochs_are_broken_by_smaller_num_buckets(env):
    tied = {arm: {(4096, 150, 0.1), (1024, 150, 0.8), (2048, 150, 0.3)} for arm in ARMS}
    block = verified_scenario(env, tied)["arms"][ARM_B]
    assert block["selected"]["num_buckets"] == 1024 and [s["survivors"] for s in block["tie_break_trace"]] == [3, 3, 1, 1, 1]


def test_ties_with_equal_epochs_and_buckets_are_broken_by_smaller_c(env):
    tied = {arm: {(2048, 150, 0.8), (2048, 150, 0.3), (2048, 150, 0.1)} for arm in ARMS}
    block = verified_scenario(env, tied)["arms"][ARM_C]
    assert block["selected"]["relative_step_c"] == 0.1 and [s["survivors"] for s in block["tie_break_trace"]] == [3, 3, 3, 1, 1]


def test_rounding_to_ten_decimals_defines_a_tie_but_full_precision_is_kept(env):
    arm = ARM_A
    base = env.actual_ap[arm]
    results = make_results(env, WINNERS)
    slow = rn.canonical_candidate_id(arm, 1024, 400, 0.1)  # same num_buckets as the winner, so the same mock artifact fits either
    # the slower candidate carries the AP rounded to ten decimals: it differs from the winner's full-precision AP (by at most 5e-11) but rounds to the
    # same value, so it ties and fewer epochs (the winner, 150) decides
    tied_ap = round(base, 10)
    results[arm] = [dataclasses.replace(r, validation_ap=tied_ap) if r.candidate_id == slow else r for r in results[arm]]
    record = build_selection_record(manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha, results_by_arm=results,
                                    evidence=env.evidence, created_utc=CREATED)
    block = record["arms"][arm]
    assert block["selected"]["epochs"] == 150 and block["selected"]["tied_at_rounded_ap"] == 2
    assert next(r for r in block["candidates"] if r["candidate_id"] == slow)["validation_ap"] == tied_ap  # stored at full precision as supplied
    assert verify(env, record).ok


def test_a_higher_ap_beyond_ten_decimals_wins_regardless_of_epochs(env):
    arm = ARM_B  # winner nb=2048, ep=50, c=0.8; give a 400-epoch candidate with the same nb a clearly higher AP and a matching model
    nb, _, c = WINNERS[arm]
    model = mock_model(arm, nb)
    ap = env.actual_ap[arm]
    results = make_results(env, WINNERS)
    rival = rn.canonical_candidate_id(arm, nb, 400, 0.3)
    results[arm] = [dataclasses.replace(r, validation_ap=ap + 1e-8) if r.candidate_id == rival else r for r in results[arm]]
    # the model's real AP is `ap`, so claiming ap + 1e-8 for the selected rival is detected as not matching the model
    with pytest.raises(SelectionRecordVerificationError):
        build_selection_record(manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha, results_by_arm=results,
                               evidence=dataclasses.replace(env.evidence, models={**env.models, arm: model}), created_utc=CREATED)


# -- learning rates, numerical bindings ------------------------------------------------------------------------------------------------------------------------


def test_learning_rates_must_equal_the_manifests_train_only_rates(env):
    for arm in ARMS:
        for index in (0, 13, 26):
            with pytest.raises(SelectionRecordError):
                verify(env, mutated(env.record, lambda r, a=arm, i=index: r["arms"][a]["candidates"][i].update(learning_rate=1e-9)))
    manifest = copy.deepcopy(env.manifest)
    manifest["numerics"]["candidates"][0]["learning_rate"] = 0.5  # a doctored manifest cannot even be parsed as the real one: digest mismatch
    forged = mf.canonical_bytes(manifest)
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, manifest_bytes=forged)


# -- model artifacts ------------------------------------------------------------------------------------------------------------------------------------------


def with_models(env, **changes):
    return dataclasses.replace(env.evidence, models={**env.models, **changes})


def test_a_changed_model_artifact_is_rejected(env):
    tampered = mock_model(ARM_A, WINNERS[ARM_A][0])
    tampered.weights[3] += 1e-6
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=with_models(env, **{ARM_A: tampered}))
    biased = mock_model(ARM_B, WINNERS[ARM_B][0])
    biased.bias += 0.5
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=with_models(env, **{ARM_B: biased}))


def test_substituted_stale_or_missing_artifacts_are_rejected(env):
    with pytest.raises(SelectionRecordVerificationError):  # another arm's artifact under this arm's name
        verify(env, env.record, evidence=with_models(env, **{ARM_A: env.models[ARM_C]}))
    with pytest.raises(SelectionRecordVerificationError):  # wrong bucket count
        verify(env, env.record, evidence=with_models(env, **{ARM_A: mock_model(ARM_A, 2048)}))
    with pytest.raises(SelectionRecordVerificationError):  # unfitted
        verify(env, env.record, evidence=with_models(env, **{ARM_A: LightweightAnomalyModel(num_buckets=1024)}))
    models = dict(env.models)
    models.pop(ARM_C)
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, models=models))
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, models={**env.models, "D_other": env.models[ARM_A]}))
    broken = mock_model(ARM_A, 1024)
    broken.weights = np.zeros(7)
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=with_models(env, **{ARM_A: broken}))


@pytest.mark.parametrize("field,value", [("weights_sha256", "0" * 64), ("bias_hex", (0.0).hex()), ("num_buckets", 512), ("is_trained", False),
                                         ("model_sha256", "9" * 64), ("logit_method", "logit-readout-0"), ("weights_dtype", "float32")])
def test_model_fields_in_the_record_must_equal_the_recomputed_ones(env, field, value):
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, lambda r: r["arms"][ARM_A]["model"].update({field: value})))


def test_the_model_source_hash_must_equal_the_manifests(env):
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, lambda r: r["code"].update(model_sha256="8" * 64)))
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, lambda r: r["code"].update(selection_record_sha256="7" * 64)))


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
    assert not any(name.startswith(("trust", "skip", "mock", "allow_unverified")) for name in params)


# -- vocabulary, budget, validation split ------------------------------------------------------------------------------------------------------------------------


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
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, frozen=Other()))


def test_validation_split_binding(env):
    for mutation in (lambda r: r["validation"].update(ids_sha256="4" * 64), lambda r: r["validation"].update(records=5),
                     lambda r: r["validation"].update(split="test")):
        with pytest.raises(SelectionRecordError):
            verify(env, mutated(env.record, mutation))
    other_split = split_records(env.ds.records, seed=PRIMARY_SPLIT_SEED + 1)
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, split_result=other_split))
    shifted = {k: v for k, v in list(env.labels.items())[:-1]}
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, labels_by_id=shifted))


def test_wrong_validation_labels_change_the_recomputed_results_and_are_rejected(env):
    flipped = {k: 1 - v for k, v in env.labels.items()}
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, labels_by_id=flipped))


def test_test_split_inputs_cannot_serve_as_validation_evidence(env):
    bad_inputs = dataclasses.replace(env.val_inputs, split="test")
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, arm_inputs=bad_inputs))
    train_inputs = env.cfg.inputs
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, arm_inputs=train_inputs))


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


def test_a_threshold_not_recomputable_from_the_supplied_validation_evidence_is_rejected(env):
    # evidence whose labels disagree with the claim (threshold computed from other data) cannot reproduce the recorded threshold
    rng = random.Random(1)
    shuffled_labels = dict(zip(env.labels, rng.sample(list(env.labels.values()), len(env.labels))))
    with pytest.raises(SelectionRecordVerificationError):
        verify(env, env.record, evidence=dataclasses.replace(env.evidence, labels_by_id=shuffled_labels))


def test_threshold_none_is_a_valid_all_negative_representation():
    selection = ThresholdSelection(rule="max_validation_f1_ties_to_highest_threshold", threshold=None, f1=0.0,
                                   validation_counts=None, candidates_considered=4)  # type: ignore[arg-type]
    block = sr.threshold_block(selection, [0.3, -1.0, 2.0, 0.1], [1, 0, 1, 0])
    assert block["value"] is None and block["predicts_all_negative"] is True and block["scale"] == "raw_logit"
    assert block["counts"] == {"tp": 0, "fp": 0, "tn": 2, "fn": 2}
    assert block["metrics"]["precision"] is None and block["metrics"]["recall"] == 0.0 and block["metrics"]["f1"] == 0.0
    assert block["metric_status"]["precision"] == "undefined" and block["metric_status"]["recall"] == "defined"
    json.dumps(block, allow_nan=False)
    record_like = {"threshold": block}
    problems: list[str] = []
    sr._scan(record_like, "$", problems)
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
    lambda r: r["manifest"].update(request_bodies=["x"]),
    lambda r: r["test_status"].update(test_scored=True),
    lambda r: r["test_status"].update(test_labels_accessed=True),
    lambda r: r["test_status"].update(test_scores_accessed=True),
    lambda r: r["test_status"].update(test_data_in_record=True),
    lambda r: r.pop("test_status"),
    lambda r: r.update(extra_field="x"),
    lambda r: r.pop("manifest"),
    lambda r: r["limitations"].pop(),
    lambda r: r.update(record_version="validation-selection-record-2"),
    lambda r: r.update(created_utc="soon"),
])
def test_test_derived_or_forbidden_content_is_rejected(env, mutation):
    with pytest.raises(SelectionRecordError):
        verify(env, mutated(env.record, mutation))


def test_the_record_contains_no_request_content_labels_or_scores(env):
    text = mf.canonical_bytes(env.record).decode("ascii")
    for rec in env.ds.records[:200]:
        assert rec.features.path not in text
    assert text.count('"validation_ap"') >= 3 * 27
    problems: list[str] = []
    sr._scan(env.record, "$", problems)
    assert problems == []
    assert not any(k in env.record for k in ("labels", "scores", "predictions", "test_labels", "test_scores"))


def test_errors_do_not_echo_record_values(env):
    secret = "SECRET-VALUE-424242"
    for mutation in (lambda r: r["manifest"].update(git_branch=secret), lambda r: r["arms"][ARM_A]["selected"].update(candidate_id=secret),
                     lambda r: r["arms"][ARM_B]["model"].update(weights_sha256=secret), lambda r: r["vocabulary"].update(sha256=secret),
                     lambda r: r["arms"][ARM_C]["candidates"][0].update(validation_ap=secret), lambda r: r.update(**{secret: 1})):
        with pytest.raises(SelectionRecordError) as caught:
            verify(env, mutated(env.record, mutation))
        assert secret not in str(caught.value)


# -- no scoring authorization --------------------------------------------------------------------------------------------------------------------------------


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
