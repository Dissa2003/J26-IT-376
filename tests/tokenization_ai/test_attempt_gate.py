"""Tests for test-attempt-gate-1 and test-attempt-marker-1 (draft protocol heldout-eval-1.2.0-draft, sections 12.11.4 and 12.12.4).

Nothing here scores a test set or fits a model, and no confirmatory data exists. The full-path tests run the REAL manifest and validation-selection
verifiers on disposable fixtures: a throw-away Git repository whose protocol header says FROZEN, development-seed (6101) structure, **81 mock fitted candidate artifacts** (27 per arm, persisted with ``model_artifacts`` and verified against
independently supplied digests, as validation-selection-record-2 requires), and a
monkeypatched seed tuple that lets the development seed stand in for the confirmatory seed (so no confirmatory seed value appears here). The gate's production
code has no flag or callback that could do the same. Direct construction of ``VerifiedAttempt`` with the module's private key is used only to test the marker
and recovery primitives quickly; it is a test fixture, not a production path. ``numerics.raw_logits`` is cached for speed (every miss calls the real function).
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import pickle
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

import tokenization_ai.heldout_eval.manifest as mf
import tokenization_ai.heldout_eval.model_artifacts as ma
import tokenization_ai.heldout_eval.numerics as numerics
import tokenization_ai.heldout_eval.runner as rn
import tokenization_ai.heldout_eval.selection_record as sr
import tokenization_ai.heldout_eval.attempt_gate as gate
from tokenization_ai.heldout_eval.arms import ARM_A, ARM_B, ARM_C, ARMS
from tokenization_ai.heldout_eval.detection_metrics import average_precision
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.splits import PRIMARY_SPLIT_SEED, EvalRecord, SplitMetadata, split_records
from tokenization_ai.model import LightweightAnomalyModel

REPO_ROOT = Path(mf.__file__).resolve().parents[2]
LF, CRLF = bytes([10]), bytes([13, 10])
FROZEN = "# Protocol\n\n**Protocol version:** `heldout-eval-test-frozen`  **Status: FROZEN**\n"
DRAFT = "# Protocol\n\n**Protocol version:** `heldout-eval-test-draft`  **Status: DRAFT, not frozen.**\n"
AUDIT_OK = {"errors": [], "warnings": [], "construction_checks_passed": True}
CREATED = "2026-01-01T00:00:00Z"
WINNERS = {ARM_A: (1024, 150, 0.3), ARM_B: (2048, 50, 0.8), ARM_C: (4096, 400, 0.1)}
GRID = [(nb, ep, c) for nb in (1024, 2048, 4096) for ep in (50, 150, 400) for c in (0.1, 0.3, 0.8)]


@pytest.fixture(scope="module", autouse=True)
def cached_logits():
    real = numerics.raw_logits
    cache: dict = {}

    def wrapper(model, inputs):
        try:
            key = (hashlib.sha256(np.ascontiguousarray(model.weights, dtype="<f8").tobytes()).hexdigest(), float(model.bias), model.num_buckets,
                   model.is_trained, len(inputs), inputs[0].event_id if len(inputs) else None, inputs[-1].event_id if len(inputs) else None,
                   hash(tuple(tuple(m.input_ids[:6]) for m in inputs[:20])))
        except Exception:  # noqa: BLE001
            return real(model, inputs)
        if key not in cache:
            cache[key] = real(model, inputs)
        return list(cache[key])

    numerics.raw_logits = wrapper
    yield
    numerics.raw_logits = real


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


def make_repo(root: Path, header: str) -> Path:
    """A throw-away repository with a small file at every path of the code-defined default lists (the gate accepts no other layout)."""
    root.mkdir()
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


def mock_model(arm: str, nb: int, k: int = 0) -> LightweightAnomalyModel:
    rng = np.random.default_rng({ARM_A: 11, ARM_B: 12, ARM_C: 13}[arm] + nb + 1000 * k)
    model = LightweightAnomalyModel(num_buckets=nb)
    model.weights = rng.normal(0.0, 0.02, nb)
    model.bias = 0.05
    model.is_trained = True
    return model


class Env:
    pass


def build_env(root: Path, header: str, purpose: str) -> Env:
    e = Env()
    e.root = root
    e.repo = make_repo(root / "repo", header)
    e.out = root / "out"
    e.out.mkdir()
    ds = generate(GeneratorConfig(seed=6101, n_templates=250))
    split = split_records(ds.records, seed=PRIMARY_SPLIT_SEED)
    part = rn.partition_records(ds.records, split)
    cfg = rn.prepare_train_configuration(part.train)
    numerical = rn.prepare_numerical_configuration(cfg)
    pre = rn.preflight_report(cfg, part.validation, part.sealed_test)
    e.kw = {}  # the gate and the verifiers use only the code-defined lists; nothing is overridden
    e.cfg, e.numerical, e.pre = cfg, numerical, pre
    manifest = mf.build_manifest(purpose=purpose, created_utc=CREATED, repo_root=e.repo, dataset=ds, split_result=split, partition=part,
                                 audit_report=AUDIT_OK, train_config=cfg, numerical=numerical, preflight=pre, **e.kw)
    approved = manifest["protocol"]["sha256"] if purpose == mf.PURPOSE_CONFIRMATORY else None
    e.expected = mf.ExpectedRun(purpose=purpose, branch="main", commit=git(e.repo, "rev-parse", "HEAD"), dataset_seed=6101,
                                generator_fingerprint=ds.fingerprint, dataset_fingerprint=split.report.dataset_fingerprint,
                                split_seed=PRIMARY_SPLIT_SEED, split_manifest_sha256=split.manifest_sha256, approved_protocol_sha256=approved)
    written = mf.write_manifest(e.out / "manifest.json", manifest, repo_root=e.repo, expected=e.expected, **e.kw)
    e.manifest_bytes, e.manifest_sha = Path(written.path).read_bytes(), written.sha256
    e.manifest = mf.parse_manifest_bytes(e.manifest_bytes)
    val = rn.encode_validation(cfg, part.validation)
    candidates = [(arm, nb, ep, c) for arm in ARMS for (nb, ep, c) in GRID]  # exactly 27 per arm, 81 in total
    rates = ma.learning_rates_from_manifest(e.manifest)
    e.artifact_dir = root / "artifacts"
    e.artifact_dir.mkdir()
    digests, models = {}, {}
    for k, (arm, nb, ep, c) in enumerate(candidates):
        cid = rn.canonical_candidate_id(arm, nb, ep, c)
        models[cid] = mock_model(arm, nb, k)
        est = next(x for x in e.manifest["numerics"]["estimates"] if x["arm"] == arm and x["num_buckets"] == nb)
        training = {"train_ids_sha256": e.manifest["arms"]["alignment"]["train_ids_sha256"],
                    "vocabulary_sha256": None if arm == ARM_C else e.manifest["arms"]["vocabulary"]["sha256"],
                    "max_length": None if arm == ARM_C else e.manifest["arms"]["budget"]["selected_max_length"],
                    "lambda_method": est["method"], "lambda_max": est["lambda_max"], "L": est["L"]}
        digests[cid] = ma.write_model_artifact(e.artifact_dir, models[cid], repo_root=e.repo, arm=arm, num_buckets=nb, epochs=ep, relative_step_c=c,
                                               learning_rate=rates[cid], manifest_sha256=e.manifest_sha, training=training).sha256
    assert len(models) == 81 and len(digests) == 81
    e.models, e.digests = models, digests
    e.evidence = sr.ValidationEvidence(arm_inputs=val, labels_by_id=part.validation.labels_by_id(), frozen=cfg.frozen, split_result=split,
                                       artifact_dir=e.artifact_dir, artifact_digests=digests)
    ids_fp = rn.ids_fingerprint(list(val.record_ids))
    labels = list(rn.aligned_labels(val.record_ids, e.evidence.labels_by_id))
    results = {}
    for arm in ARMS:
        inputs = list(getattr(val, {ARM_A: "arm_a", ARM_B: "arm_b", ARM_C: "arm_c"}[arm]))
        results[arm] = [rn.ValidationCandidateResult(rn.canonical_candidate_id(arm, nb, ep, c), arm, nb, ep, c,
                                                     average_precision(numerics.raw_logits(models[rn.canonical_candidate_id(arm, nb, ep, c)], inputs), labels),
                                                     "validation", ids_fp, len(labels)) for (nb, ep, c) in GRID]
    record = sr.build_selection_record(manifest_bytes=e.manifest_bytes, expected_manifest_sha256=e.manifest_sha, results_by_arm=results,
                                       evidence=e.evidence, created_utc=CREATED)
    rec_written = sr.write_selection_record(e.out / "record.json", record, manifest_bytes=e.manifest_bytes, expected_manifest_sha256=e.manifest_sha,
                                            repo_root=e.repo, expected_run=e.expected, evidence=e.evidence, **e.kw)
    e.record_bytes, e.record_sha = Path(rec_written.path).read_bytes(), rec_written.sha256
    e.ds, e.split, e.part = ds, split, part
    e.sealed = part.sealed_test
    test_ids = set(e.sealed.record_ids)
    e.source = [r for r in ds.records if r.record_id in test_ids]
    return e


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return build_env(tmp_path_factory.mktemp("gate"), FROZEN, mf.PURPOSE_CONFIRMATORY)


class Provider:
    def __init__(self, records, root=None, experiment_id=None):
        self.records, self.calls, self.marker_seen = records, 0, None
        self.root, self.experiment_id = root, experiment_id

    def __call__(self):
        self.calls += 1
        if self.root is not None:
            self.marker_seen = gate.inspect_attempt(self.root, self.experiment_id).state
        return list(self.records)


def attempt(env, root, provider=None, **over):
    kwargs = dict(attempt_root=root, manifest_bytes=env.manifest_bytes, expected_manifest_sha256=env.manifest_sha, record_bytes=env.record_bytes,
                  expected_record_sha256=env.record_sha, repo_root=env.repo, expected_run=env.expected, evidence=env.evidence, sealed_test=env.sealed,
                  label_provider=provider or Provider(env.source), created_utc=CREATED, **env.kw)
    kwargs.update(over)
    return gate.open_test_attempt(**kwargs)


@pytest.fixture()
def root(tmp_path):
    path = tmp_path / "attempts"
    path.mkdir()
    return path


def experiment_id(env) -> str:
    return gate.experiment_identity(env.manifest)


def fake_verified(env, **over) -> gate.VerifiedAttempt:
    fields = dict(experiment_id=experiment_id(env), manifest_sha256=env.manifest_sha, selection_record_sha256=env.record_sha, dataset_seed=6101,
                  dataset_fingerprint=env.split.report.dataset_fingerprint, generator_fingerprint=env.ds.fingerprint,
                  split_manifest_sha256=env.split.manifest_sha256, test_ids_sha256=env.manifest["arms"]["alignment"]["test_ids_sha256"],
                  test_records=len(env.sealed))
    fields.update(over)
    return gate.VerifiedAttempt(gate._KEY, **fields)


def marker_files(root):
    return sorted(p.name for p in Path(root).iterdir() if ".failure-" not in p.name)


def failure_files(root):
    return sorted(p.name for p in Path(root).iterdir() if ".failure-" in p.name)


# -- happy path with the real verifiers ----------------------------------------------------------------------------------------------------------


def test_the_real_verifiers_open_the_gate_for_a_disposable_confirmatory_fixture(env, root):
    provider = Provider(env.source, root, experiment_id(env))
    outcome = attempt(env, root, provider)
    assert outcome.test_scored is False and outcome.experiment_id == experiment_id(env) and provider.calls == 1
    assert provider.marker_seen == "consumed"  # the marker already existed when the provider ran
    truth = {r.record_id: r.meta.label for r in env.source}
    assert outcome.labels.record_ids == tuple(sorted(truth)) and outcome.labels.labels_in_order() == tuple(truth[i] for i in sorted(truth))
    assert marker_files(root) == [f"test-attempt-{experiment_id(env)}.json"] and failure_files(root) == []


def test_marker_content_is_versioned_bounded_and_free_of_sensitive_content(env, root):
    outcome = attempt(env, root)
    data = Path(outcome.marker_path).read_bytes()
    marker = gate.parse_marker_bytes(data)
    assert len(data) < gate.MAX_MARKER_BYTES and data == mf.canonical_bytes(marker) and mf.sha256_hex(data) == outcome.marker_sha256
    assert marker["marker_version"] == "test-attempt-marker-1" and marker["lifecycle_state"] == "attempt_started" and marker["created_utc"] == CREATED
    assert marker["manifest_sha256"] == env.manifest_sha and marker["selection_record_sha256"] == env.record_sha
    assert marker["experiment_id"] == experiment_id(env) and marker["split_manifest_sha256"] == env.split.manifest_sha256
    assert marker["dataset_fingerprint"] == env.split.report.dataset_fingerprint and marker["test_records"] == len(env.sealed)
    text = data.decode("ascii").lower()
    for forbidden in ("label", "score", "prediction", "logit", "body", "path"):
        assert forbidden not in text
    assert Path(outcome.marker_path).parent == root.resolve() or Path(outcome.marker_path).parent == root
    assert not (Path(outcome.marker_path).stat().st_mode & 0o200)


def test_experiment_identity_depends_on_the_dataset_split_not_on_the_manifest_digest(env):
    ident = experiment_id(env)
    assert ident == gate.experiment_identity(env.manifest) and len(ident) == 64
    other = copy.deepcopy(env.manifest)
    other["created_utc"] = "2030-01-01T00:00:00Z"
    other["protocol"]["sha256"] = "9" * 64
    assert gate.experiment_identity(other) == ident  # a re-written manifest or record cannot buy a second attempt
    for change in (lambda m: m["data"]["split"].update(manifest_sha256="1" * 64), lambda m: m["data"].update(dataset_fingerprint="2" * 64),
                   lambda m: m["data"].update(dataset_seed=1), lambda m: m["data"]["generator"].update(fingerprint="3" * 64),
                   lambda m: m["data"]["split"].update(seed=1)):
        changed = copy.deepcopy(env.manifest)
        change(changed)
        assert gate.experiment_identity(changed) != ident


# -- verification gates (nothing is created on failure) -------------------------------------------------------------------------------------------


def assert_blocked(env, root, exc=gate.GateVerificationError, **over):
    provider = Provider(env.source)
    with pytest.raises(exc):
        attempt(env, root, provider, **over)
    assert provider.calls == 0 and marker_files(root) == [] and failure_files(root) == []


def test_wrong_digests_are_rejected_without_a_marker(env, root):
    assert_blocked(env, root, expected_manifest_sha256="0" * 64)
    assert_blocked(env, root, expected_record_sha256="0" * 64)
    assert_blocked(env, root, expected_record_sha256=env.manifest_sha)  # the record's digest field cannot be swapped for the manifest's
    assert_blocked(env, root, expected_manifest_sha256="short")


def test_a_tampered_manifest_or_record_is_rejected(env, root):
    flipped = bytearray(env.manifest_bytes)
    flipped[12] ^= 1
    assert_blocked(env, root, manifest_bytes=bytes(flipped))
    flipped = bytearray(env.record_bytes)
    flipped[12] ^= 1
    assert_blocked(env, root, record_bytes=bytes(flipped))


def test_unvalidated_dictionaries_and_verified_flags_are_not_accepted(env, root):
    assert_blocked(env, root, record_bytes=json.loads(env.record_bytes))
    assert_blocked(env, root, manifest_bytes=json.loads(env.manifest_bytes))
    assert_blocked(env, root, record_bytes=env.record_bytes.decode("ascii"))
    params = set(inspect.signature(gate.open_test_attempt).parameters) | set(inspect.signature(gate.verify_attempt_inputs).parameters)
    assert not any(name.startswith(("verified", "trust", "skip", "mock", "verifier", "callback", "allow_unverified")) for name in params)
    provider = Provider(env.source)
    with pytest.raises(TypeError):
        attempt(env, root, provider, verified=True)  # unknown keywords are not a way in
    assert provider.calls == 0 and marker_files(root) == []


def test_a_selection_record_for_another_manifest_is_rejected(env, root):
    other = mf.build_manifest(purpose=mf.PURPOSE_CONFIRMATORY, created_utc="2027-01-01T00:00:00Z", repo_root=env.repo, dataset=env.ds,
                              split_result=env.split, partition=env.part, audit_report=AUDIT_OK, train_config=rn.prepare_train_configuration(env.part.train),
                              numerical=rn.prepare_numerical_configuration(rn.prepare_train_configuration(env.part.train)),
                              preflight=rn.preflight_report(rn.prepare_train_configuration(env.part.train), env.part.validation, env.part.sealed_test),
                              **env.kw)
    other_bytes = mf.canonical_bytes(other)
    assert_blocked(env, root, manifest_bytes=other_bytes, expected_manifest_sha256=mf.sha256_hex(other_bytes))


def test_a_stale_commit_or_dirty_repository_is_rejected(env, root):
    (env.repo / "stray.txt").write_text("x", encoding="utf-8")
    try:
        assert_blocked(env, root)
    finally:
        (env.repo / "stray.txt").unlink()
    (env.repo / "later.md").write_text("later\n", encoding="utf-8")
    git(env.repo, "add", "-A")
    git(env.repo, "commit", "-q", "-m", "later")
    try:
        assert_blocked(env, root)
    finally:
        git(env.repo, "reset", "-q", "--hard", env.expected.commit)
    assert marker_files(root) == []


def test_a_changed_source_file_is_rejected(env, root):
    changed = "tokenization_ai/heldout_eval/arm_c.py"
    (env.repo / changed).write_text("A = 2" + LF.decode(), encoding="utf-8", newline=LF.decode())
    try:
        assert_blocked(env, root)  # dirty
    finally:
        git(env.repo, "checkout", "--", changed)


def test_a_draft_protocol_or_an_infrastructure_run_cannot_open_the_gate(env, root, tmp_path, monkeypatch):
    assert_blocked(env, root, expected_run=dataclasses.replace(env.expected, purpose=mf.PURPOSE_INFRASTRUCTURE))
    assert_blocked(env, root, expected_run=dataclasses.replace(env.expected, approved_protocol_sha256=None))
    assert_blocked(env, root, expected_run=dataclasses.replace(env.expected, approved_protocol_sha256="0" * 64))
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (1,))  # build a genuine infrastructure fixture on a draft protocol
    (tmp_path / "draft").mkdir()
    draft = build_env(tmp_path / "draft", DRAFT, mf.PURPOSE_INFRASTRUCTURE)
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (6101,))
    draft_root = tmp_path / "draft_attempts"
    draft_root.mkdir()
    with pytest.raises(gate.GateVerificationError):
        attempt(draft, draft_root)
    confirmatory_claim = dataclasses.replace(draft.expected, purpose=mf.PURPOSE_CONFIRMATORY, approved_protocol_sha256=draft.manifest["protocol"]["sha256"])
    with pytest.raises(gate.GateVerificationError):
        attempt(draft, draft_root, expected_run=confirmatory_claim)
    assert marker_files(draft_root) == []


def test_the_real_draft_protocol_means_no_real_run_can_open_the_gate(env, root):
    assert mf.collect_protocol(REPO_ROOT)["status"] == "DRAFT"
    assert_blocked(env, root, repo_root=REPO_ROOT)  # the real repository's protocol, commit and sources do not match any confirmatory manifest


def writable_copy_of_artifacts(env, root) -> Path:
    copy_dir = root.parent / "artifacts_copy"
    shutil.copytree(env.artifact_dir, copy_dir)
    for p in copy_dir.iterdir():
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    return copy_dir


def test_wrong_validation_evidence_is_rejected_by_the_real_selection_verifier(env, root):
    # (1) a tampered candidate artifact: a flipped byte no longer matches its independently supplied digest
    tampered = writable_copy_of_artifacts(env, root)
    victim = sorted(tampered.iterdir())[0]
    data = bytearray(victim.read_bytes())
    data[50] ^= 1
    victim.write_bytes(bytes(data))
    assert_blocked(env, root, evidence=dataclasses.replace(env.evidence, artifact_dir=tampered))
    # (2) the original meaning: a *wrong model* for a candidate. Here a validly persisted artifact with different weights, whose digest the caller trusts:
    # it no longer reproduces the recorded validation AP of that candidate.
    swapped = root.parent / "artifacts_swapped"
    shutil.copytree(env.artifact_dir, swapped)
    cid = rn.canonical_candidate_id(ARM_A, 1024, 150, 0.3)
    old_file = swapped / ma.artifact_filename(env.manifest_sha, cid)
    os.chmod(old_file, stat.S_IWRITE | stat.S_IREAD)
    old_file.unlink()
    wrong = mock_model(ARM_A, 1024, k=999)
    wrong.weights[0] += 1.0
    est = next(x for x in env.manifest["numerics"]["estimates"] if x["arm"] == ARM_A and x["num_buckets"] == 1024)
    rewritten = ma.write_model_artifact(swapped, wrong, repo_root=env.repo, arm=ARM_A, num_buckets=1024, epochs=150, relative_step_c=0.3,
                                        learning_rate=ma.learning_rates_from_manifest(env.manifest)[cid], manifest_sha256=env.manifest_sha,
                                        training={"train_ids_sha256": env.manifest["arms"]["alignment"]["train_ids_sha256"],
                                                  "vocabulary_sha256": env.manifest["arms"]["vocabulary"]["sha256"],
                                                  "max_length": env.manifest["arms"]["budget"]["selected_max_length"],
                                                  "lambda_method": est["method"], "lambda_max": est["lambda_max"], "L": est["L"]})
    assert_blocked(env, root, evidence=dataclasses.replace(env.evidence, artifact_dir=swapped, artifact_digests={**env.digests, cid: rewritten.sha256}))
    flipped = {k: 1 - v for k, v in env.evidence.labels_by_id.items()}
    assert_blocked(env, root, evidence=dataclasses.replace(env.evidence, labels_by_id=flipped))


def test_the_gate_requires_the_full_81_candidate_artifact_evidence(env, root):
    cid = rn.canonical_candidate_id(ARM_C, 4096, 400, 0.1)
    assert len(env.models) == 81 and len(env.digests) == 81 and sum(1 for k in env.digests if k.startswith(ARM_A)) == 27
    assert sum(1 for k in env.digests if k.startswith(ARM_B)) == 27 and sum(1 for k in env.digests if k.startswith(ARM_C)) == 27
    assert_blocked(env, root, evidence=dataclasses.replace(env.evidence, artifact_digests={k: v for k, v in env.digests.items() if k != cid}))
    assert_blocked(env, root, evidence=dataclasses.replace(env.evidence, artifact_digests={**env.digests, cid: "0" * 64}))
    assert_blocked(env, root, evidence=dataclasses.replace(env.evidence, artifact_dir=root.parent / "no_such_directory"))
    with pytest.raises(TypeError):
        dataclasses.replace(env.evidence, models={})  # the obsolete selected-only evidence no longer exists


# -- manifest source coverage (D4-7C1) -------------------------------------------------------------------------------------------------------------


NEW_MODULES = ("tokenization_ai/heldout_eval/selection_record.py", "tokenization_ai/heldout_eval/attempt_gate.py",
               "tokenization_ai/heldout_eval/model_artifacts.py", "tokenization_ai/heldout_eval/training.py",
               "tokenization_ai/heldout_eval/confirmatory_scoring.py")


def variant_manifest(env, **over):
    """A genuine, valid manifest of the fixture repository built with some deviation (for example a shorter source list)."""
    return mf.build_manifest(purpose=mf.PURPOSE_CONFIRMATORY, created_utc=CREATED, repo_root=env.repo, dataset=env.ds, split_result=env.split,
                            partition=env.part, audit_report=AUDIT_OK, train_config=env.cfg, numerical=env.numerical, preflight=env.pre, **over)


def blocked_by_manifest(env, root, manifest: dict, fragment: str | None = None):
    data = mf.canonical_bytes(manifest)
    provider = Provider(env.source)
    with pytest.raises(gate.GateVerificationError) as caught:
        attempt(env, root, provider, manifest_bytes=data, expected_manifest_sha256=mf.sha256_hex(data))
    assert provider.calls == 0 and marker_files(root) == [] and failure_files(root) == []
    if fragment:
        assert fragment in str(caught.value)


def test_complete_default_source_coverage_is_accepted_in_an_isolated_authorized_fixture(env, root):
    required = tuple(mf.REQUIRED_SOURCE_FILES)
    assert len(required) == 27 and set(NEW_MODULES) <= set(required) and set(env.manifest["sources"]) == set(required)
    assert set(env.manifest["config_files"]) == set(mf.DEFAULT_CONFIG_FILES)
    result = mf.verify_manifest_bytes(env.manifest_bytes, expected_sha256=env.manifest_sha, repo_root=env.repo, expected=env.expected,
                                      source_files=required, config_files=mf.DEFAULT_CONFIG_FILES, protocol_path=mf.DEFAULT_PROTOCOL_PATH)
    assert result.source_list_is_default is True and result.missing_required_sources == ()
    gate._check_source_coverage(env.manifest, result, env.repo)  # the gate's own check agrees
    assert attempt(env, root).test_scored is False


def test_the_gate_critical_modules_are_part_of_the_trusted_required_list():
    assert set(gate.GATE_CRITICAL_SOURCES) <= set(mf.REQUIRED_SOURCE_FILES)
    assert all((REPO_ROOT / rel).is_file() for rel in gate.GATE_CRITICAL_SOURCES)


@pytest.mark.parametrize("rel", [*NEW_MODULES, "tokenization_ai/heldout_eval/runner.py", "tokenization_ai/model.py"])
def test_an_omitted_source_is_rejected_even_with_a_recomputed_digest(env, root, rel):
    blocked_by_manifest(env, root, mutated_manifest(env, lambda m: m["sources"].pop(rel)))


@pytest.mark.parametrize("rel", NEW_MODULES)
def test_a_substituted_source_hash_is_rejected_by_the_real_verifier(env, root, rel):
    blocked_by_manifest(env, root, mutated_manifest(env, lambda m: m["sources"].update({rel: "f" * 64})))


def mutated_manifest(env, fn) -> dict:
    manifest = copy.deepcopy(env.manifest)
    fn(manifest)
    return manifest


def test_a_partial_source_list_is_rejected_although_every_listed_hash_is_valid(env, root):
    partial = tuple(rel for rel in mf.REQUIRED_SOURCE_FILES if rel != "tokenization_ai/heldout_eval/attempt_gate.py")
    manifest = variant_manifest(env, source_files=partial)
    assert set(manifest["sources"]) == set(partial)
    assert manifest["sources"] == {rel: v for rel, v in env.manifest["sources"].items() if rel in partial}  # every listed hash is the genuine one
    blocked_by_manifest(env, root, manifest)


def test_the_gates_own_coverage_check_rejects_a_partial_manifest_even_if_the_verifier_were_fooled(env, root, monkeypatch):
    partial = tuple(rel for rel in mf.REQUIRED_SOURCE_FILES if rel not in NEW_MODULES)
    manifest = variant_manifest(env, source_files=partial)
    lying = mf.VerificationResult(ok=True, manifest_sha256="0" * 64, purpose=mf.PURPOSE_CONFIRMATORY, protocol_status="FROZEN",
                                  environment_differences=(), checks=(), source_list_is_default=True, missing_required_sources=())
    monkeypatch.setattr(mf, "verify_manifest_bytes", lambda *a, **k: lying)  # a verifier that wrongly reports complete, default coverage
    blocked_by_manifest(env, root, manifest, "source coverage")


@pytest.mark.parametrize("reported", [dict(source_list_is_default=False, missing_required_sources=()), dict(source_list_is_default=True,
                                                                                                           missing_required_sources=("x",)),
                                      dict(source_list_is_default=None, missing_required_sources=())])
def test_reported_incomplete_coverage_is_rejected_even_for_a_complete_manifest(env, root, monkeypatch, reported):
    result = mf.VerificationResult(ok=True, manifest_sha256=env.manifest_sha, purpose=mf.PURPOSE_CONFIRMATORY, protocol_status="FROZEN",
                                   environment_differences=(), checks=(), **reported)
    monkeypatch.setattr(mf, "verify_manifest_bytes", lambda *a, **k: result)
    blocked_by_manifest(env, root, env.manifest, "source coverage")


def test_a_verifier_result_without_coverage_fields_is_not_accepted(env, root, monkeypatch):
    monkeypatch.setattr(mf, "verify_manifest_bytes", lambda *a, **k: object())
    blocked_by_manifest(env, root, env.manifest, "source coverage")


def test_forged_coverage_metadata_in_the_manifest_is_rejected(env, root):
    blocked_by_manifest(env, root, mutated_manifest(env, lambda m: m.update(source_coverage="complete")))
    blocked_by_manifest(env, root, mutated_manifest(env, lambda m: m["protocol"].update(source_coverage="complete")))
    blocked_by_manifest(env, root, mutated_manifest(env, lambda m: m["config_files"].pop(next(iter(m["config_files"])))))
    blocked_by_manifest(env, root, mutated_manifest(env, lambda m: m["sources"].update({"tokenization_ai/extra.py": "a" * 64})))


def test_a_missing_required_executable_source_is_rejected(env, root):
    victim = "tokenization_ai/heldout_eval/arm_c.py"
    original = env.expected.commit
    git(env.repo, "rm", "-q", victim)
    git(env.repo, "commit", "-q", "-m", "remove a required source")
    try:
        assert_blocked(env, root, expected_run=dataclasses.replace(env.expected, commit=git(env.repo, "rev-parse", "HEAD")))
    finally:
        git(env.repo, "reset", "-q", "--hard", original)
    assert (env.repo / victim).is_file()


def test_the_gate_accepts_no_override_of_the_trusted_lists_or_the_environment_check(env, root):
    import inspect
    for function in (gate.open_test_attempt, gate.verify_attempt_inputs):
        parameters = inspect.signature(function).parameters.values()
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
        assert not {"source_files", "config_files", "protocol_path", "allow_environment_mismatch"} & {p.name for p in parameters}
    for override in (dict(source_files=("tokenization_ai/model.py",)), dict(config_files=()), dict(protocol_path="docs/PROTOCOL.md"),
                     dict(allow_environment_mismatch=True), dict(require_complete_sources=False)):
        provider = Provider(env.source)
        with pytest.raises(TypeError):
            attempt(env, root, provider, **override)
        assert provider.calls == 0 and marker_files(root) == [] and failure_files(root) == []


def test_a_shrunken_trusted_required_list_cannot_remove_a_module_the_gate_depends_on(env, root, monkeypatch):
    shrunk = tuple(rel for rel in mf.REQUIRED_SOURCE_FILES if rel != "tokenization_ai/heldout_eval/selection_record.py")
    manifest = variant_manifest(env, source_files=shrunk)  # a manifest that is internally consistent with the shrunken list
    monkeypatch.setattr(mf, "REQUIRED_SOURCE_FILES", shrunk)
    blocked_by_manifest(env, root, manifest, "omits a module the gate depends on")


def test_all_81_candidate_artifacts_are_still_verified_through_the_gate(env, root, monkeypatch):
    seen: set[str] = set()
    real = numerics.raw_logits
    size = len(env.evidence.arm_inputs.record_ids)

    def spy(model, inputs):
        if len(inputs) == size:
            seen.add(hashlib.sha256(np.ascontiguousarray(model.weights, dtype="<f8").tobytes()).hexdigest())
        return real(model, inputs)

    monkeypatch.setattr(numerics, "raw_logits", spy)
    assert attempt(env, root).test_scored is False
    assert len(seen) == 81 and len(env.digests) == 81


def test_the_real_repository_and_its_draft_protocol_cannot_authorize_test_access(env, root):
    assert mf.collect_protocol(REPO_ROOT)["status"] == "DRAFT"
    provider = Provider(env.source)
    with pytest.raises(gate.GateVerificationError):
        attempt(env, root, provider, repo_root=REPO_ROOT)
    assert provider.calls == 0 and marker_files(root) == []
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_test_access()
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_training_authorization()


def test_a_sealed_set_that_does_not_match_the_manifest_is_rejected(env, root):
    other_split = split_records(env.ds.records, seed=PRIMARY_SPLIT_SEED + 1)
    other_part = rn.partition_records(env.ds.records, other_split)
    assert_blocked(env, root, sealed_test=other_part.sealed_test)
    smaller = rn.SealedTestSet(list(env.sealed.blinded_records())[:-1], {i: g for i, g in env.sealed.group_by_id.items() if i != env.sealed.record_ids[-1]},
                               split_manifest_sha256=env.split.manifest_sha256)
    assert_blocked(env, root, sealed_test=smaller)
    wrong_hash = rn.SealedTestSet(list(env.sealed.blinded_records()), env.sealed.group_by_id, split_manifest_sha256="0" * 64)
    assert_blocked(env, root, sealed_test=wrong_hash)
    regrouped = dict(env.sealed.group_by_id)
    regrouped[env.sealed.record_ids[0]] = "other-group"
    assert_blocked(env, root, sealed_test=rn.SealedTestSet(list(env.sealed.blinded_records()), regrouped, split_manifest_sha256=env.split.manifest_sha256))
    assert_blocked(env, root, sealed_test="sealed")


# -- one-time marker -------------------------------------------------------------------------------------------------------------------------


def test_an_existing_marker_blocks_every_repeat_attempt_and_is_never_touched(env, root):
    first = attempt(env, root)
    before = Path(first.marker_path).read_bytes()
    provider = Provider(env.source)
    with pytest.raises(gate.AttemptAlreadyConsumed):
        attempt(env, root, provider)
    with pytest.raises(gate.AttemptAlreadyConsumed):
        attempt(env, root, provider, created_utc="2031-01-01T00:00:00Z")
    assert provider.calls == 0 and Path(first.marker_path).read_bytes() == before and marker_files(root) == [Path(first.marker_path).name]


def test_the_attempt_is_per_experiment_not_per_record_pair(env, root):
    first = gate.create_attempt_marker(root, fake_verified(env))
    with pytest.raises(gate.AttemptAlreadyConsumed):  # a different selection record for the same dataset and split cannot reopen the test set
        gate.create_attempt_marker(root, fake_verified(env, selection_record_sha256="7" * 64, manifest_sha256="8" * 64))
    assert Path(first.path).read_bytes() == first.marker_bytes
    other_experiment = gate.create_attempt_marker(root, fake_verified(env, experiment_id="a" * 64))
    assert other_experiment.path != first.path


def test_the_marker_path_is_derived_only_from_the_experiment_identity(env, root):
    receipt = gate.create_attempt_marker(root, fake_verified(env))
    assert Path(receipt.path).name == f"test-attempt-{experiment_id(env)}.json"
    for bad in ("../escape", "x" * 64, "A" * 64, "", None, "a" * 63):
        with pytest.raises(gate.AttemptGateError):
            gate.marker_path(root, bad)  # type: ignore[arg-type]
    assert marker_files(root) == [Path(receipt.path).name]


def test_the_attempt_directory_must_be_a_trusted_directory_outside_the_repository(env, tmp_path):
    verified = fake_verified(env)
    with pytest.raises(gate.MarkerCreationError):
        gate.create_attempt_marker(env.repo / "docs", verified, repo_root=env.repo)
    with pytest.raises(gate.MarkerCreationError):
        gate.create_attempt_marker(env.repo, verified, repo_root=env.repo)
    with pytest.raises(gate.MarkerCreationError):
        gate.create_attempt_marker(tmp_path / "missing", verified)
    some_file = tmp_path / "file.txt"
    some_file.write_text("x", encoding="utf-8")
    with pytest.raises(gate.MarkerCreationError):
        gate.create_attempt_marker(some_file, verified)
    with pytest.raises(gate.MarkerCreationError):
        gate.create_attempt_marker(tmp_path, verified, created_utc="today")
    with pytest.raises(gate.GateVerificationError):
        gate.create_attempt_marker(tmp_path, object())  # type: ignore[arg-type]
    assert not list(env.repo.glob("**/test-attempt-*"))


def test_a_verified_attempt_or_receipt_cannot_be_forged(env):
    with pytest.raises(gate.GateVerificationError):
        gate.VerifiedAttempt(object(), experiment_id="a" * 64)
    with pytest.raises(gate.AttemptGateError):
        gate.MarkerReceipt(object(), fake_verified(env), "p", b"x")
    with pytest.raises(gate.AttemptGateError):
        gate.RecoveredTestLabels(object(), ("a",), (1,))


@pytest.mark.parametrize("content", [b"", b"{", b'{"marker_version":"test-attempt-marker-1"}\n', b"not json", b"\xff\xfe", b"x" * 5000, b'{"a":1,"a":2}\n'])
def test_a_partial_or_malformed_marker_is_a_consumed_attempt_that_blocks(env, root, content):
    path = gate.marker_path(root, experiment_id(env))
    path.write_bytes(content)
    status = gate.inspect_attempt(root, experiment_id(env))
    assert status.state == "invalid_marker" and status.blocks_new_attempt
    with pytest.raises(gate.AttemptAlreadyConsumed):
        gate.create_attempt_marker(root, fake_verified(env))
    provider = Provider(env.source)
    with pytest.raises(gate.AttemptAlreadyConsumed):
        attempt(env, root, provider)
    assert provider.calls == 0 and path.read_bytes() == content  # untouched, not repaired


def test_a_substituted_marker_for_another_experiment_is_invalid_and_still_blocks(env, root):
    receipt = gate.create_attempt_marker(root, fake_verified(env, experiment_id="b" * 64))
    path = gate.marker_path(root, experiment_id(env))
    path.write_bytes(receipt.marker_bytes)  # a valid marker, but for a different experiment, sitting at this experiment's path
    assert gate.inspect_attempt(root, experiment_id(env)).state == "invalid_marker"
    with pytest.raises(gate.AttemptAlreadyConsumed):
        gate.create_attempt_marker(root, fake_verified(env))


def test_marker_parsing_is_strict():
    base = {"marker_version": "test-attempt-marker-1", "gate_version": gate.GATE_VERSION, "lifecycle_state": "attempt_started", "experiment_id": "a" * 64,
            "manifest_sha256": "b" * 64, "selection_record_sha256": "c" * 64, "dataset_seed": 1, "dataset_fingerprint": "d" * 64,
            "generator_fingerprint": "e" * 64, "split_manifest_sha256": "f" * 64, "test_ids_sha256": "0" * 64, "test_records": 3,
            "created_utc": CREATED}
    assert gate.parse_marker_bytes(mf.canonical_bytes(base)) == base
    for change in ({"marker_version": "x"}, {"lifecycle_state": "done"}, {"experiment_id": "short"}, {"created_utc": "soon"}, {"test_records": "3"},
                   {"extra": 1}, {"dataset_seed": 1.5}):
        with pytest.raises(gate.AttemptGateError):
            gate.parse_marker_bytes(mf.canonical_bytes({**base, **change}))
    missing = dict(base)
    missing.pop("gate_version")
    with pytest.raises(gate.AttemptGateError):
        gate.parse_marker_bytes(mf.canonical_bytes(missing))
    with pytest.raises(gate.AttemptGateError):
        gate.parse_marker_bytes(json.dumps(base, indent=2).encode())  # not canonical


def test_creation_failure_leaves_nothing_and_blocks_nothing(env, root, monkeypatch):
    def refuse(path, flags, mode=0o777):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "open", refuse)
    with pytest.raises(gate.MarkerCreationError):
        gate.create_attempt_marker(root, fake_verified(env))
    monkeypatch.undo()
    assert marker_files(root) == [] and gate.inspect_attempt(root, experiment_id(env)).state == "absent"


def test_persistence_failures_leave_the_marker_and_count_as_a_spent_attempt(env, root, monkeypatch):
    real_fsync = os.fsync

    def broken(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(gate.MarkerPersistenceError):
        gate.create_attempt_marker(root, fake_verified(env))
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert marker_files(root) == [f"test-attempt-{experiment_id(env)}.json"]
    assert gate.inspect_attempt(root, experiment_id(env)).blocks_new_attempt
    with pytest.raises(gate.AttemptAlreadyConsumed):
        gate.create_attempt_marker(root, fake_verified(env))


def test_a_read_back_mismatch_is_a_persistence_failure_and_reports_it(env, root, monkeypatch):
    monkeypatch.setattr(gate, "_read_back", lambda path: b"different bytes")
    provider = Provider(env.source)
    with pytest.raises(gate.MarkerPersistenceError):
        attempt(env, root, provider)
    monkeypatch.undo()
    assert provider.calls == 0  # no label was recovered
    assert marker_files(root) == [f"test-attempt-{experiment_id(env)}.json"]
    reports = failure_files(root)
    assert len(reports) == 1
    report = json.loads((root / reports[0]).read_bytes())
    assert report["stage"] == "marker_persistence" and report["category"] == "marker_persistence_failed" and report["retry_authorized"] is False
    with pytest.raises(gate.AttemptAlreadyConsumed):
        attempt(env, root, provider)


def test_exactly_one_of_many_concurrent_attempts_wins(env, root):
    verified = fake_verified(env)
    barrier = threading.Barrier(16)
    results: list = []

    def worker():
        barrier.wait()
        try:
            results.append(gate.create_attempt_marker(root, verified))
        except gate.AttemptAlreadyConsumed as exc:
            results.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [r for r in results if isinstance(r, gate.MarkerReceipt)]
    losers = [r for r in results if isinstance(r, gate.AttemptAlreadyConsumed)]
    assert len(winners) == 1 and len(losers) == 15 and len(results) == 16
    assert gate.inspect_attempt(root, experiment_id(env)).state == "consumed"
    assert Path(winners[0].path).read_bytes() == winners[0].marker_bytes


def test_the_marker_survives_a_restart(env, root):
    gate.create_attempt_marker(root, fake_verified(env))
    code = ("import sys\nfrom tokenization_ai.heldout_eval import attempt_gate as g\n"
            "print(g.inspect_attempt(sys.argv[1], sys.argv[2]).state)\n"
            "try:\n    g._write_exclusive(g.marker_path(sys.argv[1], sys.argv[2]), b'x')\nexcept g.AttemptAlreadyConsumed:\n    print('blocked')\n")
    done = subprocess.run([sys.executable, "-c", code, str(root), experiment_id(env)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["consumed", "blocked"]


# -- ordering of the label provider -----------------------------------------------------------------------------------------------------------------


def test_the_label_provider_is_never_called_when_verification_or_the_marker_fails(env, root):
    provider = Provider(env.source)
    with pytest.raises(gate.GateVerificationError):
        attempt(env, root, provider, expected_record_sha256="0" * 64)
    gate.create_attempt_marker(root, fake_verified(env))
    with pytest.raises(gate.AttemptAlreadyConsumed):
        attempt(env, root, provider)
    assert provider.calls == 0


def test_recovery_requires_a_marker_receipt_and_an_intact_marker(env, root):
    receipt = gate.create_attempt_marker(root, fake_verified(env))
    provider = Provider(env.source)
    with pytest.raises(gate.LabelRecoveryError) as caught:
        gate.recover_test_labels(object(), env.sealed, env.split, provider)  # type: ignore[arg-type]
    assert caught.value.category == "marker_unverifiable"
    Path(receipt.path).chmod(0o600)
    Path(receipt.path).write_bytes(b"tampered")
    with pytest.raises(gate.LabelRecoveryError) as caught:
        gate.recover_test_labels(receipt, env.sealed, env.split, provider)
    assert caught.value.category == "marker_unverifiable" and provider.calls == 0
    Path(receipt.path).unlink()
    with pytest.raises(gate.LabelRecoveryError):
        gate.recover_test_labels(receipt, env.sealed, env.split, provider)
    assert provider.calls == 0


# -- authentic label recovery -----------------------------------------------------------------------------------------------------------------------


def recover(env, root, rows, **over):
    receipt = gate.create_attempt_marker(Path(tempfile.mkdtemp(dir=root)), fake_verified(env))  # a fresh attempt directory per call
    args = dict(sealed_test=env.sealed, split_result=env.split)
    args.update(over)
    return gate.recover_test_labels(receipt, args["sealed_test"], args["split_result"], lambda: rows)


def expect_category(env, root, rows, category, **over):
    with pytest.raises(gate.LabelRecoveryError) as caught:
        recover(env, root, rows, **over)
    assert caught.value.category == category
    return caught.value


def test_authentic_labels_are_recovered_in_the_sealed_id_order(env, root):
    shuffled = list(env.source)
    np.random.default_rng(0).shuffle(shuffled)
    labels = recover(env, root, shuffled)
    truth = {r.record_id: r.meta.label for r in env.source}
    assert labels.record_ids == env.sealed.record_ids and labels.labels_in_order() == tuple(truth[i] for i in env.sealed.record_ids)
    assert set(labels.labels_in_order()) == {0, 1}


def test_the_blinded_placeholder_is_never_accepted_as_ground_truth(env, root):
    expect_category(env, root, list(env.sealed.blinded_records()), "placeholder_label")


@pytest.mark.parametrize("label", [2, -2, True, None, 0.5, "1"])
def test_invalid_recovered_labels_are_rejected(env, root, label):
    rows = list(env.source)
    rows[3] = EvalRecord(rows[3].record_id, rows[3].features, SplitMetadata(label=label))  # type: ignore[arg-type]
    expect_category(env, root, rows, "invalid_label")


def test_missing_extra_and_duplicate_source_records_are_rejected(env, root):
    expect_category(env, root, env.source[:-1], "records_misaligned")
    expect_category(env, root, env.source + [env.source[0]], "records_misaligned")
    stranger = EvalRecord("not-a-test-record", env.source[0].features, SplitMetadata(label=1))
    expect_category(env, root, env.source + [stranger], "records_misaligned")
    expect_category(env, root, env.source[:-1] + [stranger], "records_misaligned")
    expect_category(env, root, [r for r in env.source] + ["not a record"], "records_misaligned")
    expect_category(env, root, [r for r in env.part.train.records][:5], "records_misaligned")  # training records are not test records


def test_a_source_whose_request_content_differs_is_rejected(env, root):
    rows = list(env.source)
    features = dataclasses.replace(rows[0].features, path=rows[0].features.path + "-altered")
    rows[0] = EvalRecord(rows[0].record_id, features, rows[0].meta)
    expect_category(env, root, rows, "content_mismatch")


def test_split_group_and_fingerprint_mismatches_are_rejected(env, root):
    other_split = split_records(env.ds.records, seed=PRIMARY_SPLIT_SEED + 1)
    expect_category(env, root, env.source, "binding_mismatch", split_result=other_split)
    regrouped = dataclasses.replace(env.split, group_of={**env.split.group_of, env.sealed.record_ids[0]: "elsewhere"})
    expect_category(env, root, env.source, "group_mismatch", split_result=regrouped)
    wrong_hash = rn.SealedTestSet(list(env.sealed.blinded_records()), env.sealed.group_by_id, split_manifest_sha256="0" * 64)
    expect_category(env, root, env.source, "binding_mismatch", sealed_test=wrong_hash)
    reduced = rn.SealedTestSet(list(env.sealed.blinded_records())[:-1], {i: g for i, g in env.sealed.group_by_id.items() if i != env.sealed.record_ids[-1]},
                               split_manifest_sha256=env.split.manifest_sha256)
    expect_category(env, root, env.source, "binding_mismatch", sealed_test=reduced)
    expect_category(env, root, env.source, "binding_mismatch", sealed_test="sealed")


def test_a_failing_provider_is_reported_by_class_name_only(env, root):
    receipt = gate.create_attempt_marker(root, fake_verified(env))

    def boom():
        raise RuntimeError("SECRET-LABELS-1010")

    with pytest.raises(gate.LabelRecoveryError) as caught:
        gate.recover_test_labels(receipt, env.sealed, env.split, boom)
    assert caught.value.category == "provider_failed" and "SECRET" not in str(caught.value) and "RuntimeError" in str(caught.value)


def test_recovered_labels_are_redacted_and_not_serializable(env, root):
    labels = recover(env, root, env.source)
    expected = f"RecoveredTestLabels(records={len(env.sealed)}, labels=<withheld>)"
    assert repr(labels) == expected and str(labels) == expected
    for operation in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            operation(labels)
    public = {n for n in dir(labels) if not n.startswith("_")}
    assert public == {"labels_in_order", "record_ids"}
    assert not hasattr(labels, "to_dict") and not hasattr(labels, "__iter__") and not hasattr(labels, "__dict__")


# -- failures after the marker ------------------------------------------------------------------------------------------------------------------------


def test_a_crash_after_the_marker_leaves_it_intact_and_writes_a_redacted_report(env, root):
    def crash():
        raise RuntimeError("SECRET-LABELS-1010 label=1 score=0.9")

    with pytest.raises(gate.PostMarkerFailure) as caught:
        attempt(env, root, crash)
    failure = caught.value
    assert failure.stage == "label_recovery" and failure.category == "provider_failed" and failure.report_path
    assert "SECRET" not in str(failure) and "manual methodological review" in str(failure)
    marker = root / f"test-attempt-{experiment_id(env)}.json"
    assert marker.exists() and gate.inspect_attempt(root, experiment_id(env)).state == "consumed"
    report_bytes = Path(failure.report_path).read_bytes()
    report = json.loads(report_bytes)
    assert report["report_version"] == "test-attempt-failure-report-1" and report["marker_present"] is True
    assert report["retry_authorized"] is False and report["automatic_retry"] is False and report["manifest_sha256"] == env.manifest_sha
    assert report["selection_record_sha256"] == env.record_sha and report["experiment_id"] == experiment_id(env)
    assert set(report) == {"report_version", "gate_version", "experiment_id", "manifest_sha256", "selection_record_sha256", "stage", "category",
                           "marker_present", "retry_authorized", "automatic_retry", "required_next_step", "created_utc"}
    assert b"SECRET" not in report_bytes and b"label=" not in report_bytes and b"0.9" not in report_bytes


def test_there_is_no_automatic_retry_and_a_failure_report_does_not_unlock_anything(env, root):
    def crash():
        raise OSError("boom")

    with pytest.raises(gate.PostMarkerFailure):
        attempt(env, root, crash)
    marker_before = (root / f"test-attempt-{experiment_id(env)}.json").read_bytes()
    provider = Provider(env.source)
    with pytest.raises(gate.AttemptAlreadyConsumed):
        attempt(env, root, provider)  # same inputs, a second try: still spent
    with pytest.raises(gate.AttemptAlreadyConsumed):
        attempt(env, root, provider, label_provider=Provider(env.source))
    assert provider.calls == 0 and (root / f"test-attempt-{experiment_id(env)}.json").read_bytes() == marker_before
    assert len(failure_files(root)) == 1 and gate.inspect_attempt(root, experiment_id(env)).blocks_new_attempt
    (root / failure_files(root)[0]).unlink()  # even removing the report (a manual act) does not remove the marker
    assert gate.inspect_attempt(root, experiment_id(env)).blocks_new_attempt


@pytest.mark.parametrize("rows_fn,category", [
    (lambda e: list(e.sealed.blinded_records()), "placeholder_label"),
    (lambda e: e.source[:-1], "records_misaligned"),
    (lambda e: [EvalRecord(r.record_id, r.features, SplitMetadata(label=3)) for r in e.source], "invalid_label"),
])
def test_label_validation_failures_after_the_marker_are_reported_by_category(env, root, rows_fn, category):
    with pytest.raises(gate.PostMarkerFailure) as caught:
        attempt(env, root, lambda: rows_fn(env))
    assert caught.value.category == category and caught.value.stage == "label_recovery"
    assert len(failure_files(root)) == 1 and len(marker_files(root)) == 1
    report = json.loads((root / failure_files(root)[0]).read_bytes())
    assert report["category"] == category and report["retry_authorized"] is False


def test_an_unexpected_error_inside_recovery_is_still_reported_safely(env, root, monkeypatch):
    def broken(*args, **kwargs):
        raise KeyError("SECRET-LABELS-1010")

    monkeypatch.setattr(gate, "recover_test_labels", broken)
    with pytest.raises(gate.PostMarkerFailure) as caught:
        attempt(env, root)
    assert caught.value.category == "unexpected_error" and "SECRET" not in str(caught.value)
    assert marker_files(root) and b"SECRET" not in (root / failure_files(root)[0]).read_bytes()


def test_a_failure_report_that_cannot_be_written_is_reported_as_such(env, root, monkeypatch):
    monkeypatch.setattr(gate, "write_failure_report", lambda *a, **k: None)

    def crash():
        raise RuntimeError("x")

    with pytest.raises(gate.PostMarkerFailure) as caught:
        attempt(env, root, crash)
    assert caught.value.report_path is None and len(marker_files(root)) == 1


def test_failure_reports_are_numbered_exclusively_and_never_overwritten(env, root):
    verified = fake_verified(env)
    paths = [gate.write_failure_report(root, gate.build_failure_report(verified=verified, stage="label_recovery", category="provider_failed"))
             for _ in range(3)]
    assert len(set(paths)) == 3 and all(Path(p).exists() for p in paths)
    first = Path(paths[0]).read_bytes()
    gate.write_failure_report(root, gate.build_failure_report(verified=verified, stage="label_recovery", category="invalid_label"))
    assert Path(paths[0]).read_bytes() == first
    for bad in (dict(stage="nope", category="provider_failed"), dict(stage="label_recovery", category="nope")):
        with pytest.raises(gate.AttemptGateError):
            gate.build_failure_report(verified=verified, **bad)


# -- no scoring, no real data, runner untouched -------------------------------------------------------------------------------------------------------------


def test_the_runner_gates_stay_blocked_and_are_not_wired_to_the_gate(env, root):
    attempt(env, root)  # even after a successful gate run
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_test_access()
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_training_authorization()
    assert rn.TEST_SCORING_AVAILABLE is False and rn.TRAINING_AUTHORIZATION_AVAILABLE is False


def test_the_gate_module_never_scores_trains_or_touches_confirmatory_data():
    text = Path(gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"fit", "predict", "raw_logit", "raw_logits", "average_precision", "roc_auc", "select_threshold", "paired_group_bootstrap",
                         "request_test_access", "request_training_authorization", "generate", "split_records"}
    assert not any(seed in text for seed in ("7101", "7102", "7103"))
    assert not any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (n.name.startswith("test_") or n.name.startswith("Test")) for n in ast.walk(tree))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "LightweightAnomalyModel" not in imported and "generate" not in imported
    assert not hasattr(gate, "score_test") and not hasattr(gate, "unseal") and not hasattr(gate, "delete_marker") and not hasattr(gate, "reset_attempt")


def test_nothing_in_the_gate_deletes_or_rewrites_markers():
    text = Path(gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not called & {"unlink", "remove", "rmdir", "rmtree", "rename", "replace", "truncate", "write_text", "write_bytes"}
    assert 'open(path, "xb")' in text or "O_EXCL" in text


def test_no_confirmatory_attempt_marker_was_created_in_this_phase():
    assert not list((REPO_ROOT / "tokenization_ai").glob("**/test-attempt-*")) and not list((REPO_ROOT / "tests").glob("**/test-attempt-*"))
    assert gate.source_sha256() == mf.file_sha256(Path(gate.__file__))
