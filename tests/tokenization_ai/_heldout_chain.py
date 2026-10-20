"""Shared, disposable fixture builder for the training-stage and confirmatory-scoring tests (not a test module).

Everything here is DEVELOPMENT-ONLY: the development seed 6101 with a small template count, a throw-away Git repository with the complete default source
layout whose protocol header says FROZEN, and a monkeypatched confirmatory-seed tuple (set by the calling test module) that lets the development seed stand in
for the confirmatory seed. No confirmatory seed value appears here, nothing is reported, and every file lives under pytest's temporary directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import tokenization_ai.heldout_eval.manifest as mf
import tokenization_ai.heldout_eval.runner as rn
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.splits import PRIMARY_SPLIT_SEED, split_records

REPO_ROOT = Path(mf.__file__).resolve().parents[2]
DEV_SEED = 6101
DEV_TEMPLATES = 80
FROZEN = "# Protocol" + chr(10) * 2 + "**Protocol version:** `heldout-eval-test-frozen`  **Status: FROZEN**" + chr(10)
DRAFT = "# Protocol" + chr(10) * 2 + "**Protocol version:** `heldout-eval-test-draft`  **Status: DRAFT, not frozen.**" + chr(10)
AUDIT_OK = {"errors": [], "warnings": [], "construction_checks_passed": True}
CREATED = "2026-01-01T00:00:00Z"
LF, CRLF = bytes([10]), bytes([13, 10])


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          cwd=str(root), capture_output=True, text=True, check=True)
    return done.stdout.strip()


def make_default_repo(root: Path, header: str = FROZEN) -> Path:
    """A throw-away repository with a small file at every path of the code-defined default lists."""
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


class Chain:
    """Holds every object of one disposable development chain."""


def build_chain(root: Path, n_templates: int = DEV_TEMPLATES) -> Chain:
    """Dataset, split, partition, train and numerical configuration, preflight, FROZEN fixture repository, confirmatory manifest and authorization."""
    c = Chain()
    c.root = root
    c.out = root / "out"
    c.out.mkdir(parents=True)
    c.dataset = generate(GeneratorConfig(seed=DEV_SEED, n_templates=n_templates))
    assert c.dataset.config.seed == DEV_SEED
    c.split = split_records(c.dataset.records, seed=PRIMARY_SPLIT_SEED)
    c.part = rn.partition_records(c.dataset.records, c.split)
    c.cfg = rn.prepare_train_configuration(c.part.train)
    c.numerical = rn.prepare_numerical_configuration(c.cfg)
    c.pre = rn.preflight_report(c.cfg, c.part.validation, c.part.sealed_test)
    c.repo = make_default_repo(root / "repo", FROZEN)
    manifest = mf.build_manifest(purpose=mf.PURPOSE_CONFIRMATORY, created_utc=CREATED, repo_root=c.repo, dataset=c.dataset, split_result=c.split,
                                 partition=c.part, audit_report=AUDIT_OK, train_config=c.cfg, numerical=c.numerical, preflight=c.pre)
    c.expected = mf.ExpectedRun(purpose=mf.PURPOSE_CONFIRMATORY, branch="main", commit=git(c.repo, "rev-parse", "HEAD"), dataset_seed=DEV_SEED,
                                generator_fingerprint=c.dataset.fingerprint, dataset_fingerprint=c.split.report.dataset_fingerprint,
                                split_seed=PRIMARY_SPLIT_SEED, split_manifest_sha256=c.split.manifest_sha256,
                                approved_protocol_sha256=manifest["protocol"]["sha256"])
    written = mf.write_manifest(c.out / "manifest.json", manifest, repo_root=c.repo, expected=c.expected)
    c.manifest_bytes, c.manifest_sha = Path(written.path).read_bytes(), written.sha256
    c.manifest = mf.parse_manifest_bytes(c.manifest_bytes)
    c.auth = authorize(c)
    return c


def authorize(c: Chain, **over) -> rn.TrainingAuthorization:
    kwargs = dict(manifest_bytes=c.manifest_bytes, expected_manifest_sha256=c.manifest_sha, repo_root=c.repo, expected_run=c.expected, dataset=c.dataset,
                  split_result=c.split, partition=c.part, train_config=c.cfg, numerical=c.numerical)
    kwargs.update(over)
    return rn.authorize_training(**kwargs)


def plan_args(c: Chain, **over) -> dict:
    kwargs = dict(dataset=c.dataset, split_result=c.split, partition=c.part, train_config=c.cfg, numerical=c.numerical)
    kwargs.update(over)
    return kwargs
