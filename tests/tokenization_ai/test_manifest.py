"""Tests for pre-run-manifest-2 (draft protocol heldout-eval-1.2.0-draft, sections 12.12 and 12.15).

No model is fitted or scored and no confirmatory dataset exists. Dataset objects come from the *development* seed 6101 (structure only); Git
behavior is tested in throw-away repositories under pytest's temporary directory, never in the user's repository (which is only read, for hashes).
Confirmatory seed gating is tested by monkeypatching the seed tuple, so no confirmatory seed value appears in these tests.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.manifest as mf
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.manifest import (
    EnvironmentMismatchError,
    ExpectedRun,
    ManifestError,
    ManifestSchemaError,
    ManifestVerificationError,
    ManifestWriteError,
    ProtocolNotFrozenError,
    build_manifest,
    canonical_bytes,
    parse_manifest_bytes,
    sha256_hex,
    verify_manifest,
    verify_manifest_bytes,
    write_manifest,
)
from tokenization_ai.heldout_eval.runner import (
    partition_records,
    prepare_numerical_configuration,
    prepare_train_configuration,
    preflight_report,
)
from tokenization_ai.heldout_eval.splits import PRIMARY_SPLIT_SEED, split_records

REPO_ROOT = Path(mf.__file__).resolve().parents[2]
SRC = ("src/a.py", "src/b.py")
CFG = ("requirements.txt",)
PROTO = "docs/PROTOCOL.md"
DRAFT_HEADER = "# Protocol\n\n**Protocol version:** `heldout-eval-test-draft`  **Status: DRAFT, not frozen.**\n"
FROZEN_HEADER = "# Protocol\n\n**Protocol version:** `heldout-eval-test-frozen`  **Status: FROZEN**\n"
AUDIT_OK = {"errors": [], "warnings": [], "construction_checks_passed": True}
CREATED = "2026-01-01T00:00:00Z"


# -- fixtures -------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def objs():
    ds = generate(GeneratorConfig(seed=6101, n_templates=250))
    assert ds.config.seed == 6101
    split = split_records(ds.records, seed=PRIMARY_SPLIT_SEED)
    part = partition_records(ds.records, split)
    cfg = prepare_train_configuration(part.train)
    numerical = prepare_numerical_configuration(cfg)
    preflight = preflight_report(cfg, part.validation, part.sealed_test)
    return {"ds": ds, "split": split, "part": part, "cfg": cfg, "num": numerical, "pre": preflight}


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          cwd=str(root), capture_output=True, text=True, check=True)
    return done.stdout.strip()


def make_repo(root: Path, header: str = DRAFT_HEADER) -> Path:
    root.mkdir()
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "src/a.py").write_text("A = 1\n", encoding="utf-8", newline="\n")
    (root / "src/b.py").write_text("B = 2\n", encoding="utf-8", newline="\n")
    (root / "requirements.txt").write_text("numpy>=1.26\n", encoding="utf-8", newline="\n")
    (root / PROTO).write_text(header, encoding="utf-8", newline="\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


class Ctx:
    def __init__(self, repo: Path, tmp: Path, objs: dict, header_status_frozen: bool = False):
        self.repo, self.tmp, self.objs = repo, tmp, objs
        self.kw = {"source_files": SRC, "config_files": CFG, "protocol_path": PROTO}
        self.out = tmp / "out"
        self.out.mkdir(exist_ok=True)
        self.ops = {"attempt_root": tmp / "ops" / "attempts", "results_root": tmp / "ops" / "results"}
        for d in self.ops.values():
            d.mkdir(parents=True, exist_ok=True)

    def build(self, purpose: str = mf.PURPOSE_INFRASTRUCTURE, **over) -> dict:
        """Confirmatory manifests bind this context's canonical directories unless ``operational_roots`` is given explicitly."""
        o = self.objs
        kw = {"operational_roots": self.ops if purpose == mf.PURPOSE_CONFIRMATORY else None, **self.kw, **over}
        return build_manifest(purpose=purpose, created_utc=CREATED, repo_root=self.repo, dataset=o["ds"], split_result=o["split"],
                              partition=o["part"], audit_report=AUDIT_OK, train_config=o["cfg"], numerical=o["num"], preflight=o["pre"], **kw)

    def expected(self, **over) -> ExpectedRun:
        o = self.objs
        base = dict(purpose=mf.PURPOSE_INFRASTRUCTURE, branch="main", commit=git(self.repo, "rev-parse", "HEAD"), dataset_seed=6101,
                    generator_fingerprint=o["ds"].fingerprint, dataset_fingerprint=o["split"].report.dataset_fingerprint,
                    split_seed=PRIMARY_SPLIT_SEED, split_manifest_sha256=o["split"].manifest_sha256)
        base.update(over)
        return ExpectedRun(**base)

    def verify_dict(self, manifest: dict, expected: ExpectedRun | None = None, **kw):
        data = canonical_bytes(manifest)
        return verify_manifest_bytes(data, expected_sha256=sha256_hex(data), repo_root=self.repo, expected=expected or self.expected(),
                                     **self.kw, **kw)

    def write(self, manifest: dict | None = None, name: str = "m.json", expected: ExpectedRun | None = None, **kw):
        return write_manifest(self.out / name, self.build() if manifest is None else manifest, repo_root=self.repo, expected=expected or self.expected(), **self.kw, **kw)


@pytest.fixture()
def ctx(tmp_path, objs):
    return Ctx(make_repo(tmp_path / "repo"), tmp_path, objs)


def mutate(manifest: dict, fn) -> dict:
    out = copy.deepcopy(manifest)
    fn(out)
    return out


# -- canonical JSON ---------------------------------------------------------------------------------------------------------------------


def test_canonical_bytes_are_deterministic_and_key_order_independent():
    a = {"b": [1, 2.5, None, True], "a": {"y": "é", "x": (1, 2)}}
    b = {"a": {"x": [1, 2], "y": "é"}, "b": [1, 2.5, None, True]}
    assert canonical_bytes(a) == canonical_bytes(b)
    data = canonical_bytes(a)
    assert data.endswith(b"\n") and data.isascii() and b" " not in data
    assert json.loads(data) == {"a": {"x": [1, 2], "y": "é"}, "b": [1, 2.5, None, True]}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), {1: "x"}, {"k": object()}, {"k": {1, 2}}, {"k": "x" * 5000}])
def test_canonical_bytes_reject_non_json_values(bad):
    with pytest.raises(ManifestSchemaError):
        canonical_bytes(bad)


@pytest.mark.parametrize("raw", [b"", b"[]", b"3", b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n', b'{"a":-Infinity}\n', b'{"a":1e999}\n',
                                 b'{"a": 1}\n', b'{"a":1}', b'{"a":1}\n\n', b'{"a":1}\n{"b":2}\n', b'{"a":', b"\xff\xfe", b'{"b":1,"a":2}\n'])
def test_parse_rejects_malformed_ambiguous_and_non_canonical_bytes(raw):
    with pytest.raises(ManifestSchemaError):
        parse_manifest_bytes(raw)


def test_parse_accepts_canonical_bytes_and_round_trips():
    value = {"a": [1, {"b": 0.1}], "c": "x"}
    assert parse_manifest_bytes(canonical_bytes(value)) == value
    with pytest.raises(ManifestSchemaError):
        parse_manifest_bytes(None)  # type: ignore[arg-type]


def test_nested_duplicate_keys_are_rejected():
    with pytest.raises(ManifestSchemaError):
        parse_manifest_bytes(b'{"a":{"x":1,"x":2}}\n')


# -- building and structure -----------------------------------------------------------------------------------------------------------------


def test_built_manifest_is_valid_complete_and_json_serializable(ctx):
    m = ctx.build()
    assert m["manifest_version"] == "pre-run-manifest-2" and m["purpose"] == "infrastructure_test" and m["confirmatory"] is False
    assert m["operations"] is None  # an infrastructure manifest may leave the operational directories unbound
    assert set(m) == mf._TOP_KEYS
    assert m["protocol"]["status"] == "DRAFT" and m["protocol"]["approved_for_confirmatory"] is False
    assert m["git"]["branch"] == "main" and m["git"]["commit"] == git(ctx.repo, "rev-parse", "HEAD") and m["git"]["clean"] is True
    assert set(m["sources"]) == set(SRC) and set(m["config_files"]) == set(CFG)
    assert m["data"]["generator"]["version"] == "synthgen-3" and m["data"]["simulated_policy"]["catalog_version"] == "simpol-1"
    assert m["arms"]["arms_version"] == "arms-1" and m["arms"]["request_to_context"] == "request_to_context-v1"
    assert m["arms"]["arm_c_encoder"]["encoder_id"] == "armc-char3-v1" and m["arms"]["budget"]["rule_id"] == "budget-rule-1"
    assert m["numerics"]["provenance"]["lambda_method"] == "lambda-power-1" and m["numerics"]["provenance"]["logit_method"] == "logit-readout-1"
    assert len(m["numerics"]["estimates"]) == 9 and len(m["numerics"]["candidates"]) == 81
    stats = m["statistics"]
    assert stats["bootstrap"]["replicates"] == 2000 and stats["bootstrap"]["seed"] == 8201 and stats["improvement_rule"]["margin"] == 0.0
    assert m["selection"]["epochs"] == [50, 150, 400] and m["selection"]["relative_step_c"] == [0.1, 0.3, 0.8]
    assert m["environment"]["python"]["version"] and m["environment"]["numpy"]["version"]
    json.loads(canonical_bytes(m))
    mf.validate_manifest(m)


def test_build_is_deterministic(ctx):
    assert canonical_bytes(ctx.build()) == canonical_bytes(ctx.build())


def test_train_only_numerics_and_alignment_are_bound(ctx):
    m = ctx.build()
    train_ids = m["arms"]["alignment"]["train_ids_sha256"]
    assert {e["record_ids_sha256"] for e in m["numerics"]["estimates"]} == {train_ids}
    assert {e["split"] for e in m["numerics"]["estimates"]} == {"train"}
    assert m["arms"]["vocabulary"]["fitted_on_split"] == "train" and m["arms"]["preflight"]["passed"] is True
    assert m["data"]["counts"]["train"]["records"] == m["arms"]["budget"]["train_records"]


def test_no_test_labels_scores_or_predictions_are_recorded(ctx, objs):
    m = ctx.build()
    assert set(m["data"]["counts"]["test"]) == {"records", "groups", "ids_sha256"}
    problems: list[str] = []
    mf._scan_forbidden(m, "$", problems)
    assert problems == []
    text = canonical_bytes(m).decode("ascii")
    for rec in objs["ds"].records[:200]:
        assert rec.features.path not in text  # no request content
    assert '"positives"' in text  # train and validation counts exist ...
    assert text.count('"positives"') == 2  # ... and nothing else carries a label statistic


def test_manifest_module_source_has_no_label_or_model_access():
    text = Path(mf.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "meta" not in attrs and not called & {"fit", "predict", "raw_logit", "raw_logits", "average_precision", "paired_group_bootstrap"}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "LightweightAnomalyModel" not in imported and "SealedTestSet" not in imported


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(extra="x"),
    lambda m: m.pop("data"),
    lambda m: m["data"].pop("audit"),
    lambda m: m["data"].update(sha256="x" * 64),
    lambda m: m["git"].update(clean=False),
    lambda m: m["git"].update(commit="abc"),
    lambda m: m["protocol"].update(sha256="short"),
    lambda m: m["protocol"].update(status="DRAFTY"),
    lambda m: m.update(created_utc="yesterday"),
    lambda m: m.update(manifest_version="pre-run-manifest-1"),  # the superseded version (no operational-directory binding) is no longer accepted
    lambda m: m.update(manifest_version="pre-run-manifest-3"),
    lambda m: m.pop("operations"),
    lambda m: m.update(operations={"policy": "operations-binding-1"}),
    lambda m: m.update(purpose="other"),
    lambda m: m.update(confirmatory=True),
    lambda m: m.update(sha256="a" * 64),  # a self-reported digest is not an accepted field
    lambda m: m["sources"].update({"src/a.py": "nothex"}),
    lambda m: m.update(sources={}),
    lambda m: m["limitations"].pop(),
    lambda m: m["statistics"]["bootstrap"].update(replicates=1000),
    lambda m: m["statistics"]["improvement_rule"].update(margin=0.01),
    lambda m: m["statistics"].update(primary_contrast="AP(C) - AP(A)"),
    lambda m: m["selection"].update(ap_round_decimals=6),
    lambda m: m["selection"]["epochs"].append(800),
    lambda m: m["arms"].update(arms_version="arms-2"),
    lambda m: m["arms"]["arm_c_encoder"].update(ngram=4),
    lambda m: m["arms"]["simulated_rules"].update(verification_status="verified"),
    lambda m: m["arms"]["vocabulary"].update(min_frequency=1),
    lambda m: m["arms"]["vocabulary"].update(fitted_on_split="validation"),
    lambda m: m["arms"]["budget"].update(selected_max_length=77),
    lambda m: m["arms"]["preflight"].update(passed=False),
    lambda m: m["arms"]["preflight"]["splits"]["test"].update(labels_accessed=True),
    lambda m: m["arms"]["preflight"]["splits"]["test"].update(scoring_inputs_encoded=True),
    lambda m: m["arms"]["alignment"].update(train_records=1),
    lambda m: m["data"]["counts"]["test"].update(positives=5),
    lambda m: m["data"].update(splitter_excluded=3),
    lambda m: m["data"]["audit"].update(errors=2),
    lambda m: m["data"]["audit"].update(construction_checks_passed=False),
    lambda m: m["data"]["split"].update(requirements_met=False),
    lambda m: m["data"]["generator"].update(version="synthgen-9"),
    lambda m: m["arms"]["alignment"].update(scores=[0.1]),  # forbidden field names
    lambda m: m["selection"].update(predictions=[1]),
])
def test_structure_and_consistency_violations_are_rejected(ctx, mutation):
    with pytest.raises(ManifestSchemaError):
        mf.validate_manifest(mutate(ctx.build(), mutation))


@pytest.mark.parametrize("mutation", [
    lambda m: m["numerics"]["estimates"][0].update(lambda_max=-1.0, L=-0.25),
    lambda m: m["numerics"]["estimates"][0].update(lambda_max=0.0, L=0.0),
    lambda m: m["numerics"]["estimates"][0].update(L=1.0),
    lambda m: m["numerics"]["estimates"][0].update(converged=False),
    lambda m: m["numerics"]["estimates"][0].update(iterations=0),
    lambda m: m["numerics"]["estimates"][0].update(final_relative_change=1e-3),
    lambda m: m["numerics"]["estimates"][0].update(relative_tolerance=1e-6),
    lambda m: m["numerics"]["estimates"][0].update(max_iterations=10),
    lambda m: m["numerics"]["estimates"][0].update(split="validation"),
    lambda m: m["numerics"]["estimates"][0].update(record_ids_sha256="0" * 64),
    lambda m: m["numerics"]["estimates"][0].update(method="lambda-power-0"),
    lambda m: m["numerics"]["estimates"][0].update(num_buckets=999),
    lambda m: m["numerics"]["estimates"].pop(),
    lambda m: m["numerics"]["estimates"].append(copy.deepcopy(m["numerics"]["estimates"][0])),
    lambda m: m["numerics"]["learning_rates"]["cells"][0]["rates"][0].update(learning_rate=0.5),
    lambda m: m["numerics"]["learning_rates"]["cells"][0]["rates"][0].update(learning_rate_hex="0x1.0p+0"),
    lambda m: m["numerics"]["learning_rates"]["cells"][0]["rates"][0].update(relative_step_c=0.5),
    lambda m: m["numerics"]["learning_rates"]["cells"][0].update(L=3.0),
    lambda m: m["numerics"]["learning_rates"].update(relative_step_grid=[0.1, 0.3]),
    lambda m: m["numerics"]["candidates"].pop(),
    lambda m: m["numerics"]["candidates"][0].update(learning_rate=1.0),
    lambda m: m["numerics"]["candidates"][0].update(epochs=7),
    lambda m: m["numerics"]["candidates"][0].update(candidate_id="x"),
    lambda m: m["numerics"]["grids"].update(epochs=[50]),
    lambda m: m["numerics"].update(candidates=m["numerics"]["candidates"] + [m["numerics"]["candidates"][0]]),
])
def test_invalid_numerical_configuration_is_rejected(ctx, mutation):
    with pytest.raises(ManifestSchemaError):
        mf.validate_manifest(mutate(ctx.build(), mutation))


# -- write-once ------------------------------------------------------------------------------------------------------------------------


def test_write_creates_the_canonical_file_and_returns_the_digest_of_its_bytes(ctx):
    manifest = ctx.build()
    written = ctx.write(manifest)
    path = Path(written.path)
    data = path.read_bytes()
    assert data == canonical_bytes(manifest) and written.sha256 == sha256_hex(data) and written.size == len(data) and written.purpose == "infrastructure_test"
    assert not (path.stat().st_mode & stat.S_IWRITE)  # read-only after finalization
    result = verify_manifest(path, expected_sha256=written.sha256, repo_root=ctx.repo, expected=ctx.expected(), **ctx.kw)
    assert result.ok and result.protocol_status == "DRAFT" and result.environment_differences == () and "expected_digest" in result.checks


def test_an_existing_manifest_is_never_overwritten(ctx):
    first = ctx.write()
    before = Path(first.path).read_bytes()
    with pytest.raises(ManifestWriteError):
        ctx.write()
    with pytest.raises(ManifestWriteError):
        ctx.write(mutate(ctx.build(), lambda m: m.update(created_utc="2027-01-01T00:00:00Z")))
    assert Path(first.path).read_bytes() == before


def test_a_file_that_appears_between_the_check_and_the_write_is_not_replaced(ctx, monkeypatch):
    target = ctx.out / "race.json"
    real_lexists = os.path.lexists

    def late(path):
        if Path(path) == target:
            target.write_bytes(b"someone else's file")
            return False  # pretend the path was free when checked
        return real_lexists(path)

    monkeypatch.setattr(os.path, "lexists", late)
    with pytest.raises(ManifestWriteError):
        write_manifest(target, ctx.build(), repo_root=ctx.repo, expected=ctx.expected(), **ctx.kw)
    assert target.read_bytes() == b"someone else's file"


def test_write_refuses_paths_inside_the_repository_and_missing_directories(ctx):
    with pytest.raises(ManifestWriteError):
        write_manifest(ctx.repo / "manifest.json", ctx.build(), repo_root=ctx.repo, expected=ctx.expected(), **ctx.kw)
    assert not (ctx.repo / "manifest.json").exists()
    with pytest.raises(ManifestWriteError):
        write_manifest(ctx.out / "nope" / "m.json", ctx.build(), repo_root=ctx.repo, expected=ctx.expected(), **ctx.kw)


def test_a_failed_write_is_left_in_place_and_blocks_a_silent_retry(ctx, monkeypatch):
    def broken(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(ManifestWriteError):
        ctx.write(name="failed.json")
    monkeypatch.undo()
    assert (ctx.out / "failed.json").exists()  # nothing was repaired or deleted behind the caller's back
    with pytest.raises(ManifestWriteError):
        ctx.write(name="failed.json")


def test_invalid_manifests_are_not_written(ctx):
    for broken in (mutate(ctx.build(), lambda m: m["git"].update(clean=False)), {"manifest_version": "pre-run-manifest-1"}, {}):
        with pytest.raises(ManifestError):
            ctx.write(broken, name="bad.json")
        assert not (ctx.out / "bad.json").exists()


def test_writer_refuses_a_dirty_or_wrong_repository_state_without_writing(ctx):
    manifest = ctx.build()
    (ctx.repo / "src/a.py").write_text("A = 99\n", encoding="utf-8", newline="\n")
    with pytest.raises(ManifestVerificationError):
        ctx.write(manifest, name="dirty.json")
    assert not (ctx.out / "dirty.json").exists()


# -- verification and tamper detection ----------------------------------------------------------------------------------------------------------


def test_verify_accepts_an_untouched_manifest(ctx):
    result = ctx.verify_dict(ctx.build())
    assert result.ok and result.manifest_sha256 and result.purpose == "infrastructure_test"


def test_a_single_flipped_byte_fails_the_external_digest(ctx):
    written = ctx.write()
    data = bytearray(Path(written.path).read_bytes())
    data[len(data) // 2] ^= 0x01
    with pytest.raises(ManifestVerificationError):
        verify_manifest_bytes(bytes(data), expected_sha256=written.sha256, repo_root=ctx.repo, expected=ctx.expected(), **ctx.kw)


def test_a_missing_or_malformed_external_digest_is_refused(ctx):
    data = canonical_bytes(ctx.build())
    for bad in (None, "", "abc", sha256_hex(data).upper(), 123):
        with pytest.raises(ManifestVerificationError):
            verify_manifest_bytes(data, expected_sha256=bad, repo_root=ctx.repo, expected=ctx.expected(), **ctx.kw)  # type: ignore[arg-type]
    with pytest.raises(ManifestVerificationError):
        verify_manifest_bytes(data, expected_sha256=sha256_hex(data), repo_root=ctx.repo, expected="trusted?", **ctx.kw)  # type: ignore[arg-type]
    with pytest.raises(ManifestVerificationError):
        verify_manifest(ctx.out / "missing.json", expected_sha256=sha256_hex(data), repo_root=ctx.repo, expected=ctx.expected(), **ctx.kw)


def test_a_self_consistent_forgery_with_its_own_digest_still_fails_independent_checks(ctx):
    # The attacker recomputes the digest of the edited file. The digest then matches, but the content no longer matches the repository.
    forged = mutate(ctx.build(), lambda m: m["sources"].update({"src/a.py": "f" * 64}))
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(forged)
    forged = mutate(ctx.build(), lambda m: m["config_files"].update({"requirements.txt": "e" * 64}))
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(forged)
    forged = mutate(ctx.build(), lambda m: m["protocol"].update(sha256="d" * 64))
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(forged)
    forged = mutate(ctx.build(), lambda m: m["git"].update(commit="c" * 40))
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(forged)
    forged = mutate(ctx.build(), lambda m: m["git"].update(branch="other"))
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(forged)


def test_the_clean_flag_inside_the_manifest_is_not_trusted(ctx):
    manifest = ctx.build()  # records clean = True
    (ctx.repo / "untracked.txt").write_text("stray\n", encoding="utf-8")
    with pytest.raises(ManifestVerificationError) as caught:
        ctx.verify_dict(manifest)
    assert "not clean" in str(caught.value)


def test_a_modified_tracked_source_file_is_detected(ctx):
    manifest = ctx.build()
    (ctx.repo / "src/b.py").write_text("B = 3\n", encoding="utf-8", newline="\n")
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(manifest)  # dirty
    git(ctx.repo, "add", "-A")
    git(ctx.repo, "commit", "-q", "-m", "change")
    new_commit = git(ctx.repo, "rev-parse", "HEAD")
    with pytest.raises(ManifestVerificationError) as caught:  # clean again, but the manifest is stale
        ctx.verify_dict(manifest, ctx.expected(commit=new_commit))
    assert "sources differ" in str(caught.value) or "git.commit" in str(caught.value)
    with pytest.raises(ManifestVerificationError):  # the old expected commit no longer matches HEAD either
        ctx.verify_dict(manifest)


def test_a_manifest_for_another_commit_is_stale(ctx):
    manifest = ctx.build()
    (ctx.repo / "docs/extra.md").write_text("unrelated\n", encoding="utf-8")
    git(ctx.repo, "add", "-A")
    git(ctx.repo, "commit", "-q", "-m", "docs")
    new_commit = git(ctx.repo, "rev-parse", "HEAD")
    for expected in (ctx.expected(), ctx.expected(commit=new_commit)):
        with pytest.raises(ManifestVerificationError):
            ctx.verify_dict(manifest, expected)


def test_wrong_expected_branch_or_commit_is_refused(ctx):
    manifest = ctx.build()
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(manifest, ctx.expected(branch="feature/other"))
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(manifest, ctx.expected(commit="1" * 40))
    git(ctx.repo, "checkout", "-q", "-b", "side")
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(manifest)


@pytest.mark.parametrize("field,value", [("generator_fingerprint", "1" * 64), ("dataset_fingerprint", "2" * 64), ("split_manifest_sha256", "3" * 64),
                                         ("dataset_seed", 6102), ("split_seed", 1)])
def test_wrong_dataset_or_split_provenance_is_refused(ctx, field, value):
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(ctx.build(), ctx.expected(**{field: value}))


def test_the_manifest_cannot_override_the_expected_dataset_values(ctx):
    forged = mutate(ctx.build(), lambda m: m["data"].update(dataset_fingerprint="9" * 64))
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(forged)  # the caller's trusted value, not the manifest's, decides


def test_purpose_must_match_the_expected_run(ctx):
    with pytest.raises(ManifestVerificationError):
        ctx.verify_dict(ctx.build(), ctx.expected(purpose=mf.PURPOSE_CONFIRMATORY))


def test_invalid_expected_values_are_rejected():
    good = dict(purpose="infrastructure_test", branch="main", commit="a" * 40, dataset_seed=1, generator_fingerprint="b" * 64,
                dataset_fingerprint="c" * 64, split_seed=2, split_manifest_sha256="d" * 64)
    ExpectedRun(**good)
    for field, value in [("purpose", "x"), ("branch", ""), ("commit", "abc"), ("dataset_seed", True), ("split_seed", 1.5), ("generator_fingerprint", "zz"),
                         ("dataset_fingerprint", "A" * 64), ("split_manifest_sha256", None), ("approved_protocol_sha256", "short")]:
        with pytest.raises(ManifestSchemaError):
            ExpectedRun(**{**good, field: value})


def test_verification_errors_do_not_echo_manifest_values(ctx):
    secret = "SECRET-VALUE-12345"
    for mutation in (lambda m: m["git"].update(branch=secret), lambda m: m["data"].update(dataset_fingerprint=secret),
                     lambda m: m["protocol"].update(version=secret), lambda m: m["environment"]["python"].update(version=secret),
                     lambda m: m["arms"]["budget"].update(rule_id=secret)):
        with pytest.raises(ManifestError) as caught:
            ctx.verify_dict(mutate(ctx.build(), mutation))
        assert secret not in str(caught.value)


# -- protocol authorization ---------------------------------------------------------------------------------------------------------------------


def test_confirmatory_purpose_cannot_be_built_from_a_draft_protocol(ctx, monkeypatch):
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (6101,))  # make the development seed play the confirmatory seed for this gate test only
    with pytest.raises(ManifestSchemaError):
        ctx.build(purpose=mf.PURPOSE_CONFIRMATORY)


def test_infrastructure_manifest_cannot_use_a_confirmatory_seed(ctx, monkeypatch):
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (6101,))
    with pytest.raises(ManifestSchemaError):
        ctx.build()


def test_frozen_protocol_authorization_is_required_for_a_confirmatory_manifest(tmp_path, objs, monkeypatch):
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (6101,))
    frozen = Ctx(make_repo(tmp_path / "repo", FROZEN_HEADER), tmp_path, objs)
    manifest = frozen.build(purpose=mf.PURPOSE_CONFIRMATORY)
    assert manifest["protocol"]["status"] == "FROZEN" and manifest["protocol"]["approved_for_confirmatory"] is True
    protocol_hash = manifest["protocol"]["sha256"]
    confirmatory = dict(purpose=mf.PURPOSE_CONFIRMATORY)
    with pytest.raises(ProtocolNotFrozenError):  # no external approval at all
        frozen.verify_dict(manifest, frozen.expected(**confirmatory))
    with pytest.raises(ProtocolNotFrozenError):  # approval of a different protocol file
        frozen.verify_dict(manifest, frozen.expected(approved_protocol_sha256="0" * 64, **confirmatory))
    with pytest.raises(ProtocolNotFrozenError):
        frozen.write(manifest, expected=frozen.expected(**confirmatory))
    assert not (frozen.out / "m.json").exists()
    assert frozen.verify_dict(manifest, frozen.expected(approved_protocol_sha256=protocol_hash, **confirmatory)).ok


def frozen_ctx(tmp_path, objs, monkeypatch) -> Ctx:
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (6101,))
    return Ctx(make_repo(tmp_path / "repo", FROZEN_HEADER), tmp_path, objs)


def test_a_confirmatory_manifest_binds_distinct_canonical_operational_directories(tmp_path, objs, monkeypatch):
    frozen = frozen_ctx(tmp_path, objs, monkeypatch)
    manifest = frozen.build(purpose=mf.PURPOSE_CONFIRMATORY)
    ops = manifest["operations"]
    assert ops["policy"] == "operations-binding-1" and set(ops) == {"policy", "scope", "attempt_root", "results_root"}
    for name, path in frozen.ops.items():
        assert ops[name] == mf.describe_directory(path) and ops[name]["inode"] != 0
        assert ops[name]["path"] == os.path.normcase(os.path.realpath(path))
    approved = frozen.expected(purpose=mf.PURPOSE_CONFIRMATORY, approved_protocol_sha256=manifest["protocol"]["sha256"])
    assert frozen.verify_dict(manifest, approved).ok and "operational_directories" in frozen.verify_dict(manifest, approved).checks
    with pytest.raises(ManifestSchemaError):  # a confirmatory manifest without the binding is invalid
        frozen.build(purpose=mf.PURPOSE_CONFIRMATORY, operational_roots=None)
    with pytest.raises(ManifestSchemaError):
        frozen.build(purpose=mf.PURPOSE_CONFIRMATORY, operational_roots={"attempt_root": frozen.ops["attempt_root"]})
    with pytest.raises(ManifestSchemaError):  # the same directory twice
        frozen.build(purpose=mf.PURPOSE_CONFIRMATORY, operational_roots={"attempt_root": frozen.ops["attempt_root"],
                                                                         "results_root": frozen.ops["attempt_root"]})
    for bad in (tmp_path / "missing", frozen.repo / "docs", frozen.repo):
        with pytest.raises(ManifestVerificationError):
            frozen.build(purpose=mf.PURPOSE_CONFIRMATORY, operational_roots={**frozen.ops, "attempt_root": bad})
    for mutation in (lambda m: m["operations"]["attempt_root"].update(inode=0), lambda m: m["operations"].update(policy="other"),
                     lambda m: m["operations"]["results_root"].pop("device"), lambda m: m["operations"].update(scope="global"),
                     lambda m: m["operations"].update(results_root=m["operations"]["attempt_root"]), lambda m: m.update(operations=None)):
        with pytest.raises(ManifestSchemaError):
            mf.validate_manifest(mutate(manifest, mutation))


def test_a_moved_or_recreated_operational_directory_fails_verification(tmp_path, objs, monkeypatch):
    frozen = frozen_ctx(tmp_path, objs, monkeypatch)
    manifest = frozen.build(purpose=mf.PURPOSE_CONFIRMATORY)
    approved = frozen.expected(purpose=mf.PURPOSE_CONFIRMATORY, approved_protocol_sha256=manifest["protocol"]["sha256"])
    results = frozen.ops["results_root"]
    results.rename(results.with_name("parked"))
    with pytest.raises(ManifestVerificationError) as caught:
        frozen.verify_dict(manifest, approved)
    assert "operations.results_root" in str(caught.value)
    results.mkdir()  # the same path, a new directory
    with pytest.raises(ManifestVerificationError):
        frozen.verify_dict(manifest, approved)
    results.rmdir()
    results.with_name("parked").rename(results)
    assert frozen.verify_dict(manifest, approved).ok


def test_directory_identity_resolves_aliases_and_refuses_other_directories(tmp_path):
    target, other = tmp_path / "target", tmp_path / "other"
    target.mkdir()
    other.mkdir()
    ident = mf.describe_directory(target)
    assert mf.describe_directory(str(target) + os.sep) == ident and mf.describe_directory(other) != ident
    if os.name == "nt":
        assert mf.describe_directory(str(target).upper()) == ident
    alias = tmp_path / "alias"
    try:
        os.symlink(target, alias, target_is_directory=True)
    except (OSError, NotImplementedError):
        alias = None
        if os.name == "nt":
            import _winapi
            alias = tmp_path / "junction"
            _winapi.CreateJunction(str(target), str(alias))
    if alias is not None:
        assert mf.describe_directory(alias) == ident
    ops = {"policy": mf.OPERATIONS_POLICY, "attempt_root": ident}
    assert mf.check_bound_directory(ops, "attempt_root", target) == Path(os.path.realpath(target))
    for wrong in (other, tmp_path / "missing"):
        with pytest.raises(ManifestVerificationError):
            mf.check_bound_directory(ops, "attempt_root", wrong)
    for bad_ops in (None, {}, {"policy": "other", "attempt_root": ident}):
        with pytest.raises(ManifestVerificationError):
            mf.check_bound_directory(bad_ops, "attempt_root", target)
    with pytest.raises(ManifestVerificationError):
        mf.check_bound_directory(ops, "artifact_root", target)  # only the two bound kinds exist


def test_a_draft_protocol_file_cannot_pass_as_frozen_even_with_a_forged_manifest(ctx, monkeypatch):
    built = ctx.build()
    monkeypatch.setattr(mf, "CONFIRMATORY_SEEDS", (6101,))
    forged = mutate(built, lambda m: (m.update(purpose="confirmatory", confirmatory=True),
                                            m["protocol"].update(status="FROZEN", approved_for_confirmatory=True)))
    with pytest.raises(ManifestError):
        ctx.verify_dict(forged, ctx.expected(purpose="confirmatory", approved_protocol_sha256=forged["protocol"]["sha256"]))


def test_protocol_header_parser():
    assert mf.parse_protocol_header(DRAFT_HEADER) == ("heldout-eval-test-draft", "DRAFT")
    assert mf.parse_protocol_header(FROZEN_HEADER) == ("heldout-eval-test-frozen", "FROZEN")
    for bad in ("", "no header", "**Protocol version:** `x`", "**Protocol version:** `x` **Status: MAYBE**"):
        with pytest.raises(ManifestVerificationError):
            mf.parse_protocol_header(bad)


# -- environment ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mutation,label", [
    (lambda m: m["environment"]["python"].update(version="0.0.1"), "python.version"),
    (lambda m: m["environment"]["numpy"].update(version="0.0.1"), "numpy.version"),
    (lambda m: m["environment"]["platform"].update(system="Plan9"), "platform.system"),
    (lambda m: m["environment"]["threads"].update(OMP_NUM_THREADS="7777"), "threads"),
    (lambda m: m["environment"]["numpy"].update(blas={"name": "other", "version": "0", "configuration": None}), "numpy.blas"),
])
def test_environment_mismatch_is_explicit(ctx, mutation, label):
    manifest = mutate(ctx.build(), mutation)
    with pytest.raises(EnvironmentMismatchError) as caught:
        ctx.verify_dict(manifest)
    assert label in str(caught.value)
    result = ctx.verify_dict(manifest, allow_environment_mismatch=True)
    assert label in result.environment_differences and result.ok


def test_current_thread_configuration_is_part_of_the_comparison(ctx, monkeypatch):
    manifest = ctx.build()
    monkeypatch.setenv("OMP_NUM_THREADS", "3" if os.environ.get("OMP_NUM_THREADS") != "3" else "4")
    with pytest.raises(EnvironmentMismatchError):
        ctx.verify_dict(manifest)


# -- real repository (read-only) ------------------------------------------------------------------------------------------------------------------


def test_default_source_and_config_files_exist_and_hash_exactly():
    hashes = mf.hash_files(REPO_ROOT, mf.DEFAULT_SOURCE_FILES)
    assert set(hashes) == set(mf.DEFAULT_SOURCE_FILES) and "tokenization_ai/heldout_eval/manifest.py" in hashes
    for rel, digest in hashes.items():
        assert digest == sha256_hex((REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    assert set(mf.hash_files(REPO_ROOT, mf.DEFAULT_CONFIG_FILES)) == set(mf.DEFAULT_CONFIG_FILES)
    assert mf.source_sha256() == hashes["tokenization_ai/heldout_eval/manifest.py"]


def test_missing_or_escaping_paths_are_refused(tmp_path):
    with pytest.raises(ManifestVerificationError):
        mf.hash_files(REPO_ROOT, ["tokenization_ai/does_not_exist.py"])
    with pytest.raises(ManifestVerificationError):
        mf.hash_files(REPO_ROOT / "tokenization_ai", ["../requirements.txt"])


def test_the_real_protocol_is_still_a_draft_so_no_confirmatory_manifest_can_exist_yet():
    info = mf.collect_protocol(REPO_ROOT)
    assert info["status"] == "DRAFT" and info["version"].startswith("heldout-eval-") and len(info["sha256"]) == 64


def test_non_repository_roots_are_refused(tmp_path):
    with pytest.raises(ManifestVerificationError):
        mf.collect_git_state(tmp_path)
    nested = make_repo(tmp_path / "outer")
    (nested / "inner").mkdir()
    with pytest.raises(ManifestVerificationError):
        mf.collect_git_state(nested / "inner")  # a sub-directory is not the repository top level


def test_environment_snapshot_is_json_safe_and_complete():
    env = mf.collect_environment()
    json.dumps(env, allow_nan=False)
    assert env["python"]["version"] and env["numpy"]["version"] and set(env["threads"]) == set(mf.THREAD_ENV_VARS)
    assert mf.environment_differences(env, env) == []


# -- source coverage (D4-7B) --------------------------------------------------------------------------------------------------------------------------


LF, CRLF = bytes([10]), bytes([13, 10])
NEW_MODULES = ("tokenization_ai/heldout_eval/selection_record.py", "tokenization_ai/heldout_eval/attempt_gate.py", "tokenization_ai/heldout_eval/model_artifacts.py",
               "tokenization_ai/heldout_eval/training.py", "tokenization_ai/heldout_eval/confirmatory_scoring.py")


def make_default_repo(root: Path, header: str = DRAFT_HEADER) -> Path:
    """A throw-away repository holding a tiny file at every path of the code-defined default lists."""
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
    protocol.write_text(header, encoding="utf-8", newline="\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture()
def dctx(tmp_path, objs):
    ctx = Ctx(make_default_repo(tmp_path / "repo"), tmp_path, objs)
    ctx.kw = {}  # the code-defined default lists, no overrides
    return ctx


def test_the_default_source_list_protects_the_three_new_modules():
    assert len(mf.DEFAULT_SOURCE_FILES) == 27 and len(set(mf.DEFAULT_SOURCE_FILES)) == 27
    for rel in NEW_MODULES:
        assert rel in mf.DEFAULT_SOURCE_FILES and rel in mf.REQUIRED_SOURCE_FILES
    assert mf.REQUIRED_SOURCE_FILES == mf.DEFAULT_SOURCE_FILES and mf.missing_required_sources(mf.DEFAULT_SOURCE_FILES) == ()
    assert set(mf.missing_required_sources(["tokenization_ai/model.py"])) == set(mf.REQUIRED_SOURCE_FILES) - {"tokenization_ai/model.py"}
    assert mf.hash_files(REPO_ROOT, NEW_MODULES).keys() == set(NEW_MODULES)


def test_every_held_out_evaluation_source_and_core_dependency_is_protected():
    """No executable evaluation module, and no core pipeline module that the evaluation imports, may be missing from the source list."""
    pkg = REPO_ROOT / "tokenization_ai"
    listed = set(mf.DEFAULT_SOURCE_FILES)
    on_disk = {f"tokenization_ai/heldout_eval/{p.name}" for p in (pkg / "heldout_eval").glob("*.py")}
    assert on_disk <= listed, sorted(on_disk - listed)
    imported_core = set()
    for path in (pkg / "heldout_eval").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module:
                imported_core.add(f"tokenization_ai/{node.module.split('.')[0]}.py")
            elif isinstance(node, ast.ImportFrom) and node.level == 2 and not node.module:
                imported_core |= {f"tokenization_ai/{a.name}.py" for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 1:
                names = [node.module.split(".")[0]] if node.module else [a.name for a in node.names]
                for name in names:
                    if (pkg / "heldout_eval" / f"{name}.py").exists():
                        imported_core.add(f"tokenization_ai/heldout_eval/{name}.py")
    imported_core = {rel for rel in imported_core if (REPO_ROOT / rel).exists()}
    assert imported_core <= listed, sorted(imported_core - listed)


def test_hashes_are_normalized_content_hashes_not_raw_byte_hashes(tmp_path):
    lf, crlf, lone_cr = tmp_path / "lf.txt", tmp_path / "crlf.txt", tmp_path / "cr.txt"
    lf.write_bytes(b"a\nb\n")
    crlf.write_bytes(b"a\r\nb\r\n")
    lone_cr.write_bytes(b"a\rb\r")
    assert mf.file_sha256(lf) == mf.file_sha256(crlf) == sha256_hex(b"a\nb\n")
    assert sha256_hex(crlf.read_bytes()) != mf.file_sha256(crlf)  # a raw-byte hash would differ
    assert mf.file_sha256(lone_cr) == sha256_hex(b"a\rb\r")  # a lone CR is not normalized


def test_a_manifest_built_with_the_default_list_records_and_verifies_every_required_source(dctx):
    manifest = dctx.build()
    assert set(manifest["sources"]) == set(mf.REQUIRED_SOURCE_FILES) and all(rel in manifest["sources"] for rel in NEW_MODULES)
    result = dctx.verify_dict(manifest)
    assert result.ok and result.source_list_is_default is True and result.missing_required_sources == ()


@pytest.mark.parametrize("rel", NEW_MODULES)
def test_omitting_a_required_source_from_the_manifest_fails_verification(dctx, rel):
    forged = mutate(dctx.build(), lambda m: m["sources"].pop(rel))
    with pytest.raises(ManifestVerificationError):
        dctx.verify_dict(forged)  # the verifier derives the required list from code, not from the manifest


@pytest.mark.parametrize("rel", NEW_MODULES)
def test_a_substituted_source_hash_fails_verification(dctx, rel):
    forged = mutate(dctx.build(), lambda m: m["sources"].update({rel: "f" * 64}))
    with pytest.raises(ManifestVerificationError):
        dctx.verify_dict(forged)


@pytest.mark.parametrize("rel", NEW_MODULES)
def test_a_changed_required_source_is_detected_dirty_and_after_commit(dctx, rel):
    manifest = dctx.build()
    (dctx.repo / rel).write_text("# changed\n", encoding="utf-8", newline="\n")
    with pytest.raises(ManifestVerificationError):
        dctx.verify_dict(manifest)  # dirty tree
    git(dctx.repo, "add", "-A")
    git(dctx.repo, "commit", "-q", "-m", "change")
    new_commit = git(dctx.repo, "rev-parse", "HEAD")
    with pytest.raises(ManifestVerificationError) as caught:
        dctx.verify_dict(manifest, dctx.expected(commit=new_commit))
    assert "sources differ" in str(caught.value) or "git.commit" in str(caught.value)


@pytest.mark.parametrize("rel", NEW_MODULES)
def test_a_missing_required_source_file_is_rejected_at_build_and_at_verification(dctx, rel):
    manifest = dctx.build()
    git(dctx.repo, "rm", "-q", rel)
    git(dctx.repo, "commit", "-q", "-m", "remove")
    with pytest.raises(ManifestVerificationError):
        dctx.verify_dict(manifest, dctx.expected(commit=git(dctx.repo, "rev-parse", "HEAD")))
    with pytest.raises(ManifestVerificationError):
        dctx.build()


def test_a_custom_source_list_is_reported_as_incomplete(ctx):
    result = ctx.verify_dict(ctx.build())  # the fixture repo uses a two-file list passed by the trusted caller
    assert result.ok and result.source_list_is_default is False
    assert set(NEW_MODULES) <= set(result.missing_required_sources) and len(result.missing_required_sources) == len(mf.REQUIRED_SOURCE_FILES)


def test_the_real_repository_hashes_all_required_sources():
    hashes = mf.hash_files(REPO_ROOT, mf.REQUIRED_SOURCE_FILES)
    assert set(hashes) == set(mf.REQUIRED_SOURCE_FILES) and all(len(v) == 64 for v in hashes.values())
