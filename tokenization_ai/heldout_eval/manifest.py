"""``pre-run-manifest-1``: the write-once pre-run manifest and its independent verifier (draft protocol heldout-eval-1.2.0-draft, section 12.12).

Infrastructure only. This module trains nothing, scores nothing, generates no dataset and **reads no test label, score or prediction**; it records
hashes, versions, counts and the train-only numerical configuration, and it never stores request bodies. A manifest produced here is authorization
*evidence*; wiring it into the runner's blocked training gate is a later milestone (``runner.request_training_authorization`` is unchanged).

Trust model
-----------
A digest stored inside the manifest proves nothing, so verification never believes the manifest about itself:

* the **expected SHA-256 of the exact file bytes** is supplied from outside (``verify_manifest(expected_sha256=...)``);
* Git branch, commit and cleanliness are re-read from the repository (``git`` is run, nothing copied from the manifest), and the caller states the
  branch and commit it expects (:class:`ExpectedRun`);
* every source and configuration file is re-hashed from its exact content (LF-normalized), and the file *lists* are code constants or trusted caller
  parameters, never taken from the manifest;
* versions, grids, learning-rate arithmetic, the statistical protocol, the selection rules and the limitations are re-derived from the code and compared
  by equality; the environment is compared with the running one;
* dataset seed, fingerprints and split hash are compared with the caller's trusted values.

Any mismatch stops verification with :class:`ManifestVerificationError` (messages name fields and categories only, never values).

Write-once
----------
The file is created with exclusive creation (``"xb"``), written once as canonical JSON (sorted keys, compact separators, ASCII, ``allow_nan=False``,
one trailing newline), flushed and fsynced, then read back and re-verified byte for byte. An existing path is never overwritten. If a write fails after the
file exists, it is **left in place** and :class:`ManifestWriteError` is raised: nothing is repaired or replaced silently, and a leftover file blocks a retry
until it is reviewed. The manifest path must be outside the repository, so writing it cannot dirty the working tree it certifies.

Confirmatory versus infrastructure manifests
--------------------------------------------
``purpose = "confirmatory"`` requires a protocol file whose header says ``Status: FROZEN``, an externally approved SHA-256 of that file, and a dataset seed
in the confirmatory plan. ``purpose = "infrastructure_test"`` allows a draft protocol but forbids the confirmatory seeds, and is rejected by a verifier
that expects a confirmatory run. The protocol is currently a DRAFT, so no confirmatory manifest can be created yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import platform
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import arm_c, detection_metrics as dm, numerics, policy_sim, request_context, runner
from . import synthetic_catalog
from .arms import ARM_A, ARM_B, ARM_C, ARMS, ARMS_VERSION
from .generator import GENERATOR_VERSION, SyntheticDataset
from .runner import NumericalConfiguration, Partition, TrainConfiguration
from .splits import PROTOCOL_VERSION as SPLIT_PROTOCOL_VERSION
from .splits import TEMPLATE_VERSION, SplitResult

MANIFEST_VERSION = "pre-run-manifest-1"
PURPOSE_CONFIRMATORY = "confirmatory"
PURPOSE_INFRASTRUCTURE = "infrastructure_test"
PURPOSES = (PURPOSE_CONFIRMATORY, PURPOSE_INFRASTRUCTURE)
CONFIRMATORY_SEEDS = (7101, 7102, 7103)
DEFAULT_PROTOCOL_PATH = "tokenization_ai/heldout_eval/PROTOCOL.md"
DEFAULT_SOURCE_FILES: tuple[str, ...] = (
    "tokenization_ai/models.py", "tokenization_ai/fusion.py", "tokenization_ai/masking.py", "tokenization_ai/normalization.py",
    "tokenization_ai/tokenizer.py", "tokenization_ai/vocabulary.py", "tokenization_ai/tensorizer.py", "tokenization_ai/model.py",
    "tokenization_ai/heldout_eval/splits.py", "tokenization_ai/heldout_eval/synthetic_catalog.py", "tokenization_ai/heldout_eval/generator.py",
    "tokenization_ai/heldout_eval/audit.py", "tokenization_ai/heldout_eval/policy_sim.py", "tokenization_ai/heldout_eval/policy_diagnostics.py",
    "tokenization_ai/heldout_eval/request_context.py", "tokenization_ai/heldout_eval/arm_c.py", "tokenization_ai/heldout_eval/token_budget.py",
    "tokenization_ai/heldout_eval/arms.py", "tokenization_ai/heldout_eval/detection_metrics.py", "tokenization_ai/heldout_eval/numerics.py",
    "tokenization_ai/heldout_eval/runner.py", "tokenization_ai/heldout_eval/manifest.py",
)
DEFAULT_CONFIG_FILES: tuple[str, ...] = ("requirements.txt", "tokenization_ai/requirements-ml.txt")
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_STRING_LENGTH = 4096
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_FORBIDDEN_KEY_TOKENS = frozenset({"label", "labels", "score", "scores", "prediction", "predictions", "logit", "logits", "body", "bodies"})

LIMITATIONS: tuple[str, ...] = (
    "synthetic held-out evaluation only; no real-world detection claim",
    "labels are defined by same-author scenario construction",
    "Arm B uses simulated simpol-1 rules, not Member 2 verified rules",
    "Arm C sees query and value characters that Arms A and B reduce to atomic tokens or the unknown token",
    "scores are uncalibrated logits, not probabilities",
    "bootstrap intervals describe synthetic-generator sampling uncertainty over template groups only",
    "a passing audit does not show absence of leakage or shortcuts",
)


class ManifestError(ValueError):
    """Base class. Messages hold field paths, categories and counts only, never values, request content, labels or scores."""


class ManifestSchemaError(ManifestError):
    """The manifest is malformed, ambiguous, incomplete or internally inconsistent."""


class ManifestWriteError(ManifestError):
    """The manifest could not be written (existing path, I/O failure). Nothing was repaired or replaced."""


class ManifestVerificationError(ManifestError):
    """A check against the independently derived truth failed."""


class ProtocolNotFrozenError(ManifestVerificationError):
    """A confirmatory manifest needs a frozen, externally approved protocol."""


class EnvironmentMismatchError(ManifestVerificationError):
    """The running environment differs from the recorded one."""


# -- canonical JSON ------------------------------------------------------------------------------------------------------------


def _plain(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            raise ManifestSchemaError(f"{path}: string value is too long")
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ManifestSchemaError(f"{path}: non-finite number")
        return number
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestSchemaError(f"{path}: non-string key")
            out[key] = _plain(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise ManifestSchemaError(f"{path}: unsupported value type")


def canonical_bytes(value: Any) -> bytes:
    """Deterministic JSON: sorted keys, compact separators, ASCII, no NaN or infinity, one trailing newline."""
    return (json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ManifestSchemaError("duplicate key in the JSON object")
        out[key] = value
    return out


def _reject_constant(_: str) -> Any:
    raise ManifestSchemaError("NaN or infinity is not allowed")


def _parse_float(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ManifestSchemaError("a number is not finite")
    return number


def parse_manifest_bytes(data: bytes) -> dict[str, Any]:
    """Strictly parse manifest bytes: UTF-8 JSON object, no duplicate keys, no NaN/infinity, and **canonical** (re-serializes to the same bytes)."""
    if not isinstance(data, (bytes, bytearray)) or not data or len(data) > MAX_MANIFEST_BYTES:
        raise ManifestSchemaError("manifest bytes are missing or too large")
    try:
        parsed = json.loads(bytes(data).decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_reject_constant,
                            parse_float=_parse_float)
    except ManifestSchemaError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ManifestSchemaError("the manifest is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise ManifestSchemaError("the manifest must be a JSON object")
    if canonical_bytes(parsed) != bytes(data):
        raise ManifestSchemaError("the manifest is not in canonical serialization (partial, reformatted or altered)")
    return parsed


# -- trusted expectations ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedRun:
    """Values the caller states from outside the manifest. Verification compares the manifest with these and with the live repository."""

    purpose: str
    branch: str
    commit: str
    dataset_seed: int
    generator_fingerprint: str
    dataset_fingerprint: str
    split_seed: int
    split_manifest_sha256: str
    approved_protocol_sha256: str | None = None  # required for a confirmatory manifest: the approved hash of the frozen protocol file

    def __post_init__(self) -> None:
        problems = []
        if self.purpose not in PURPOSES:
            problems.append("purpose")
        if not isinstance(self.branch, str) or not self.branch:
            problems.append("branch")
        if not isinstance(self.commit, str) or not _HEX40.fullmatch(self.commit):
            problems.append("commit")
        if isinstance(self.dataset_seed, bool) or not isinstance(self.dataset_seed, int):
            problems.append("dataset_seed")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            problems.append("split_seed")
        for name in ("generator_fingerprint", "dataset_fingerprint", "split_manifest_sha256"):
            if not isinstance(getattr(self, name), str) or not _HEX64.fullmatch(getattr(self, name)):
                problems.append(name)
        if self.approved_protocol_sha256 is not None and (not isinstance(self.approved_protocol_sha256, str)
                                                          or not _HEX64.fullmatch(self.approved_protocol_sha256)):
            problems.append("approved_protocol_sha256")
        if problems:
            raise ManifestSchemaError(f"invalid expectation field(s): {', '.join(problems)}")


# -- live repository, files and environment ---------------------------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        raise ManifestVerificationError("git could not be run") from None
    if done.returncode != 0:
        raise ManifestVerificationError("git reported an error for this repository")
    return done.stdout


def collect_git_state(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Branch, full commit and cleanliness read from the repository itself (untracked files count as dirty; ignored files do not)."""
    root = Path(repo_root)
    top = _git(root, "rev-parse", "--show-toplevel").strip()
    if not top or not os.path.samefile(top, root):
        raise ManifestVerificationError("repo_root is not the top level of a Git repository")
    commit = _git(root, "rev-parse", "HEAD").strip()
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    return {"branch": branch, "commit": commit, "clean": status.strip() == "", "dirty_entries": len([ln for ln in status.splitlines() if ln.strip()])}


def file_sha256(path: Path) -> str:
    """SHA-256 of the exact file content with CRLF normalized to LF (stable across checkouts)."""
    return sha256_hex(path.read_bytes().replace(b"\r\n", b"\n"))


def hash_files(repo_root: str | os.PathLike[str], relative_paths: Sequence[str]) -> dict[str, str]:
    root = Path(repo_root).resolve()
    out = {}
    for rel in relative_paths:
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file():
            raise ManifestVerificationError(f"required file is missing or outside the repository: {rel}")
        out[rel] = file_sha256(path)
    return dict(sorted(out.items()))


def parse_protocol_header(text: str) -> tuple[str, str]:
    """``(version, status)`` from the protocol's first lines; status is ``DRAFT`` or ``FROZEN``."""
    head = text[:3000]
    version = re.search(r"\*\*Protocol version:\*\*\s*`([^`]+)`", head)
    status = re.search(r"\*\*Status:\s*([A-Za-z]+)", head)
    if not version or not status or status.group(1).upper() not in ("DRAFT", "FROZEN"):
        raise ManifestVerificationError("the protocol header does not state a version and a DRAFT or FROZEN status")
    return version.group(1), status.group(1).upper()


def collect_protocol(repo_root: str | os.PathLike[str], protocol_path: str = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    path = Path(repo_root) / protocol_path
    if not path.is_file():
        raise ManifestVerificationError("the protocol file is missing")
    version, status = parse_protocol_header(path.read_text(encoding="utf-8"))
    return {"version": version, "status": status, "sha256": file_sha256(path)}


def collect_environment() -> dict[str, Any]:
    """Python, numpy, BLAS, platform and thread configuration of the running process."""
    blas: dict[str, Any] = {"name": None, "version": None, "configuration": None}
    simd: list[str] = []
    try:
        cfg = np.show_config(mode="dicts")
        found = cfg.get("Build Dependencies", {}).get("blas", {})
        blas = {"name": found.get("name"), "version": found.get("version"), "configuration": found.get("openblas configuration")}
        simd = sorted(cfg.get("SIMD Extensions", {}).get("baseline", []))
    except Exception:  # noqa: BLE001 - older numpy: record that BLAS details are unavailable instead of guessing
        pass
    return {
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "numpy": {"version": np.__version__, "blas": blas, "simd_baseline": simd},
        "platform": {"system": platform.system(), "machine": platform.machine(), "release": platform.release()},
        "threads": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "cpu_count": os.cpu_count(),
    }


_ENVIRONMENT_COMPARED = (("python", "version"), ("python", "implementation"), ("numpy", "version"), ("numpy", "blas"), ("numpy", "simd_baseline"),
                         ("platform", "system"), ("platform", "machine"), ("threads", None))


def environment_differences(recorded: Mapping[str, Any], current: Mapping[str, Any] | None = None) -> list[str]:
    current = collect_environment() if current is None else current
    diffs = []
    for section, key in _ENVIRONMENT_COMPARED:
        a = recorded.get(section) if key is None else (recorded.get(section) or {}).get(key)
        b = current.get(section) if key is None else (current.get(section) or {}).get(key)
        if a != b:
            diffs.append(section if key is None else f"{section}.{key}")
    return diffs


# -- constant sections (re-derived from code at verification) ----------------------------------------------------------------------------------


def statistics_section() -> dict[str, Any]:
    return {
        "detection_metrics_id": dm.METRICS_ID,
        "primary_contrast": f"AP({ARM_B}) - AP({ARM_A})",
        "primary_metric": "average_precision_stepwise_no_interpolation",
        "improvement_rule": {
            "margin": 0.0, "evidence_of_improvement": "status ok and the whole 95% interval strictly above zero",
            "possible_worse": "the whole interval strictly below zero", "otherwise": "no improvement claimed",
            "not_estimable": "inconclusive", "scope": "study-specific evidence rule, not proof of real-world improvement",
        },
        "bootstrap": {
            "unit": "normalized_template_group", "replicates": dm.BOOTSTRAP_REPLICATES, "seed": dm.BOOTSTRAP_SEED,
            "confidence": dm.BOOTSTRAP_CONFIDENCE, "interval": "percentile_linear_interpolation_valid_replicates_only",
            "min_valid_percent": dm.MIN_VALID_PERCENT, "min_valid_convention": "prespecified convention, not a statistical theorem",
            "undefined_replicates_replaced": False, "mixed_label_groups_intact": True, "same_draw_for_both_arms": True,
        },
        "secondary": ["roc_auc", "precision_recall_f1_and_confusion_at_validation_chosen_threshold", "arm_c_contrasts_descriptive_only"],
        "ranking_representation": f"raw_logit ({numerics.LOGIT_METHOD_ID}); uncalibrated, thresholds are logit-scale",
        "truncation_exclusion_policy": "fail_closed: no exclusion, no truncation, no re-budgeting; stop for protocol review",
        "interpretation": "synthetic held-out performance and feasibility only",
    }


def selection_section() -> dict[str, Any]:
    return {
        "model_selection": "highest validation average precision", "ap_round_decimals": runner.AP_ROUND_DECIMALS,
        "tie_break_order": ["fewer_epochs", "smaller_num_buckets", "smaller_relative_step_c", "candidate_id_lexicographic"],
        "threshold_rule": dm.THRESHOLD_RULE, "threshold_basis": "validation scores and labels only, logit scale",
        "num_buckets": list(numerics.NUM_BUCKETS_GRID), "epochs": list(runner.EPOCHS_GRID), "relative_step_c": list(numerics.RELATIVE_STEP_GRID),
        "candidates_per_arm": len(numerics.NUM_BUCKETS_GRID) * len(runner.EPOCHS_GRID) * len(numerics.RELATIVE_STEP_GRID),
        "min_frequency": runner.MIN_FREQUENCY, "max_literal_vocab_size": runner.MAX_LITERAL_VOCAB_SIZE,
        "fresh_model_per_candidate": True, "training_split_only": True, "validation_only_selection": True,
    }


def component_versions() -> dict[str, Any]:
    return {
        "arms_version": ARMS_VERSION, "arms": {"A": ARM_A, "B": ARM_B, "C": ARM_C}, "request_to_context": request_context.MAPPING_ID,
        "arm_c_encoder": dict(arm_c.encoder_parameters()), "runner": runner.RUNNER_ID,
        "tokenizer": "HybridTokenizer (see sources)", "budget_rule": "budget-rule-1",
        "simulated_rules": {"catalog_version": policy_sim.CATALOG_VERSION, "catalog_sha256": policy_sim.CATALOG_SHA256,
                            "provenance": policy_sim.SIMULATED_PROVENANCE, "verification_status": policy_sim.NOT_VERIFIED_STATUS},
    }


# -- building -----------------------------------------------------------------------------------------------------------------------------


def _audit_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    try:
        errors, warnings, passed = report["errors"], report["warnings"], report["construction_checks_passed"]
    except (KeyError, TypeError):
        raise ManifestSchemaError("the audit report lacks errors, warnings or construction_checks_passed") from None
    if not isinstance(errors, (list, tuple)) or not isinstance(warnings, (list, tuple)) or not isinstance(passed, bool):
        raise ManifestSchemaError("the audit report fields are malformed")
    return {"errors": len(errors), "warnings": len(warnings), "construction_checks_passed": passed}


def build_manifest(
    *, purpose: str, created_utc: str, repo_root: str | os.PathLike[str], dataset: SyntheticDataset, split_result: SplitResult,
    partition: Partition, audit_report: Mapping[str, Any], train_config: TrainConfiguration, numerical: NumericalConfiguration,
    preflight: Mapping[str, Any], reserve_seed_log: Sequence[Mapping[str, Any]] = (),
    source_files: Sequence[str] = DEFAULT_SOURCE_FILES, config_files: Sequence[str] = DEFAULT_CONFIG_FILES,
    protocol_path: str = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Assemble a manifest from the live repository and the real runner objects (nothing is written). Raises if the result is not valid."""
    if purpose not in PURPOSES:
        raise ManifestSchemaError("purpose must be 'confirmatory' or 'infrastructure_test'")
    if not (isinstance(dataset, SyntheticDataset) and isinstance(split_result, SplitResult) and isinstance(partition, Partition)
            and isinstance(train_config, TrainConfiguration) and isinstance(numerical, NumericalConfiguration)):
        raise ManifestSchemaError("dataset, split, partition, train configuration and numerical configuration must be the runner's own objects")
    git = collect_git_state(repo_root)
    protocol = collect_protocol(repo_root, protocol_path)
    summary = partition.summary
    report = split_result.report
    cfg = train_config.to_dict()
    manifest = {
        "manifest_version": MANIFEST_VERSION, "purpose": purpose, "confirmatory": purpose == PURPOSE_CONFIRMATORY, "created_utc": created_utc,
        "protocol": {**protocol, "approved_for_confirmatory": protocol["status"] == "FROZEN",
                     "note": "a DRAFT protocol cannot authorize a confirmatory manifest"},
        "git": {"branch": git["branch"], "commit": git["commit"], "clean": git["clean"], "untracked_files_considered": True},
        "sources": hash_files(repo_root, source_files), "config_files": hash_files(repo_root, config_files),
        "data": {
            "generator": {"version": dataset.generator_version, "catalog_version": synthetic_catalog.CATALOG_VERSION, "template_version": TEMPLATE_VERSION,
                          "config": dataset.config.to_dict(), "fingerprint": dataset.fingerprint},
            "dataset_seed": dataset.config.seed, "dataset_fingerprint": report.dataset_fingerprint,
            "split": {"method": report.method, "seed": report.seed, "ratios": list(report.ratios), "parameters": dict(report.parameters),
                      "manifest_sha256": split_result.manifest_sha256, "protocol_version": SPLIT_PROTOCOL_VERSION,
                      "template_version": report.template_version, "requirements_met": report.meets_requirements},
            "counts": {
                "train": {k: summary["train"][k] for k in ("records", "groups", "positives")},
                "validation": {k: summary["validation"][k] for k in ("records", "groups", "positives")},
                "test": {k: summary["test"][k] for k in ("records", "groups", "ids_sha256")},
            },
            "splitter_excluded": len(split_result.excluded), "audit": _audit_summary(audit_report), "isolation_checked": True,
            "simulated_policy": {"catalog_version": policy_sim.CATALOG_VERSION, "catalog_sha256": policy_sim.CATALOG_SHA256},
            "reserve_seed_log": [dict(entry) for entry in reserve_seed_log],
        },
        "arms": {
            **component_versions(),
            "vocabulary": {**cfg["vocabulary"], "min_frequency": cfg["min_frequency"], "max_literal_vocab_size": cfg["max_literal_vocab_size"]},
            "budget": cfg["budget"],
            "alignment": {"train_records": cfg["train_records"], "train_ids_sha256": cfg["train_ids_sha256"],
                          "validation_records": summary["validation"]["records"], "test_records": summary["test"]["records"],
                          "test_ids_sha256": summary["test"]["ids_sha256"], "pairing": "ArmInputs enforces identical record ids and order in all arms"},
            "preflight": dict(preflight),
        },
        "numerics": {
            "provenance": numerics.provenance(),
            "estimates": [e.to_dict() for e in numerical.estimates], "learning_rates": dict(numerical.learning_rates),
            "candidates": [c.to_dict() for c in numerical.candidates],
            "grids": {"num_buckets": list(numerics.NUM_BUCKETS_GRID), "epochs": list(runner.EPOCHS_GRID),
                      "relative_step_c": list(numerics.RELATIVE_STEP_GRID)},
        },
        "selection": selection_section(), "statistics": statistics_section(),
        "environment": collect_environment(), "limitations": list(LIMITATIONS),
    }
    manifest = _plain(manifest)
    validate_manifest(manifest)
    return manifest


# -- structure and internal consistency -------------------------------------------------------------------------------------------------------


def _is_hex64(v: Any) -> bool:
    return isinstance(v, str) and bool(_HEX64.fullmatch(v))


def _scan_forbidden(value: Any, path: str, problems: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            tokens = set(re.split(r"[^a-z0-9]+", key.lower()))
            carries_data = isinstance(item, (dict, list)) or (isinstance(item, (int, float)) and not isinstance(item, bool))
            if tokens & _FORBIDDEN_KEY_TOKENS and carries_data:  # descriptors and flags (strings, booleans) are fine; data-bearing values are not
                problems.append(f"{path}.{key}: a label, score, prediction or body field is not allowed in a manifest")
            _scan_forbidden(item, f"{path}.{key}", problems)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _scan_forbidden(item, f"{path}[{i}]", problems)


_TOP_KEYS = {"manifest_version", "purpose", "confirmatory", "created_utc", "protocol", "git", "sources", "config_files", "data", "arms", "numerics",
             "selection", "statistics", "environment", "limitations"}
_SECTION_KEYS = {
    "protocol": {"version", "status", "sha256", "approved_for_confirmatory", "note"},
    "git": {"branch", "commit", "clean", "untracked_files_considered"},
    "data": {"generator", "dataset_seed", "dataset_fingerprint", "split", "counts", "splitter_excluded", "audit", "isolation_checked",
             "simulated_policy", "reserve_seed_log"},
    "arms": {"arms_version", "arms", "request_to_context", "arm_c_encoder", "runner", "tokenizer", "budget_rule", "simulated_rules", "vocabulary",
             "budget", "alignment", "preflight"},
    "numerics": {"provenance", "estimates", "learning_rates", "candidates", "grids"},
}


def _check_numerics(m: Mapping[str, Any], problems: list[str]) -> None:
    num = m["numerics"]
    train_ids = m["arms"]["alignment"]["train_ids_sha256"]
    estimates = num["estimates"]
    cells = {}
    for est in estimates:
        key = (est["arm"], est["num_buckets"])
        ok = (est["method"] == numerics.LAMBDA_METHOD_ID and est["split"] == "train" and est["converged"] is True
              and est["relative_tolerance"] == numerics.POWER_RELATIVE_TOLERANCE and est["max_iterations"] == numerics.POWER_MAX_ITERATIONS
              and isinstance(est["lambda_max"], float) and math.isfinite(est["lambda_max"]) and est["lambda_max"] > 0
              and est["L"] == est["lambda_max"] / 4 and 1 <= est["iterations"] <= numerics.POWER_MAX_ITERATIONS
              and est["final_relative_change"] <= numerics.POWER_RELATIVE_TOLERANCE and est["record_ids_sha256"] == train_ids
              and est["training_records"] == m["arms"]["alignment"]["train_records"] and est["arm"] in ARMS
              and est["num_buckets"] in numerics.NUM_BUCKETS_GRID and est.get("fallback_learning_rate", None) is None)
        if not ok or key in cells:
            problems.append(f"numerics.estimates[{est.get('arm')}, {est.get('num_buckets')}]: invalid, unconverged, not train-only or duplicated")
        cells[key] = est
    if set(cells) != {(a, b) for a in ARMS for b in numerics.NUM_BUCKETS_GRID}:
        problems.append("numerics.estimates: must cover exactly the 9 arm x num_buckets cells")
        return
    table = num["learning_rates"]
    if table.get("method") != numerics.LAMBDA_METHOD_ID or table.get("relative_step_grid") != list(numerics.RELATIVE_STEP_GRID):
        problems.append("numerics.learning_rates: wrong method or relative-step grid")
    rate_of: dict[tuple[str, int, float], float] = {}
    for cell in table.get("cells", []):
        est = cells.get((cell["arm"], cell["num_buckets"]))
        if est is None or cell["L"] != est["L"] or cell["lambda_max"] != est["lambda_max"]:
            problems.append("numerics.learning_rates: a cell does not match its eigenvalue estimate")
            continue
        for row in cell["rates"]:
            expected = numerics.relative_learning_rate(row["relative_step_c"], est["L"])
            if row["learning_rate"] != expected or row["learning_rate_hex"] != expected.hex() or row["relative_step_c"] not in numerics.RELATIVE_STEP_GRID:
                problems.append("numerics.learning_rates: a learning rate is not c / L of its estimate")
            rate_of[(cell["arm"], cell["num_buckets"], row["relative_step_c"])] = row["learning_rate"]
    if len(rate_of) != len(ARMS) * len(numerics.NUM_BUCKETS_GRID) * len(numerics.RELATIVE_STEP_GRID):
        problems.append("numerics.learning_rates: incomplete")
    seen = set()
    for cand in num["candidates"]:
        ident = runner.canonical_candidate_id(cand["arm"], cand["num_buckets"], cand["epochs"], cand["relative_step_c"])
        if (cand["candidate_id"] != ident or cand["epochs"] not in runner.EPOCHS_GRID or ident in seen
                or rate_of.get((cand["arm"], cand["num_buckets"], cand["relative_step_c"])) != cand["learning_rate"]):
            problems.append("numerics.candidates: an inconsistent, unapproved or duplicate candidate")
        seen.add(ident)
    if len(seen) != len(ARMS) * 27:
        problems.append("numerics.candidates: must hold exactly 27 candidates per arm")
    if num["grids"] != {"num_buckets": list(numerics.NUM_BUCKETS_GRID), "epochs": list(runner.EPOCHS_GRID),
                        "relative_step_c": list(numerics.RELATIVE_STEP_GRID)}:
        problems.append("numerics.grids: differ from the approved grids")


def _check_arms(m: Mapping[str, Any], problems: list[str]) -> None:
    arms, data = m["arms"], m["data"]
    counts = data["counts"]
    vocab, budget, align, pre = arms["vocabulary"], arms["budget"], arms["alignment"], arms["preflight"]
    if (vocab.get("fitted_on_split") != "train" or vocab.get("training_records") != counts["train"]["records"] or vocab.get("min_frequency") != 2
            or vocab.get("max_literal_vocab_size") is not None or not _is_hex64(vocab.get("sha256"))):
        problems.append("arms.vocabulary: must be fitted on the training split only with the approved settings")
    if (budget.get("status") != "selected" or budget.get("rule_id") != "budget-rule-1" or budget.get("train_records") != counts["train"]["records"]
            or budget.get("selected_max_length") not in (budget.get("candidate_grid") or [])):
        problems.append("arms.budget: must be a selected training-split budget-rule-1 decision")
    if align["train_records"] != counts["train"]["records"] or align["validation_records"] != counts["validation"]["records"] \
            or align["test_records"] != counts["test"]["records"] or align["test_ids_sha256"] != counts["test"]["ids_sha256"] \
            or not _is_hex64(align["train_ids_sha256"]):
        problems.append("arms.alignment: does not match the split counts")
    splits = pre.get("splits", {})
    ok = pre.get("passed") is True and set(splits) == {"train", "validation", "test"} and pre.get("max_length") == budget.get("selected_max_length")
    for name in ("train", "validation", "test"):
        part = splits.get(name, {})
        ok = ok and part.get("passed") is True and part.get("labels_accessed") is False and part.get("scoring_inputs_encoded") is False \
            and part.get("records_excluded") == 0 and part.get("budget_changed") is False \
            and part.get("records") == counts[name]["records"] and part.get("max_length") == budget.get("selected_max_length")
    if not ok:
        problems.append("arms.preflight: every split must have passed the label-free representability preflight under the frozen budget")
    if arms["simulated_rules"].get("verification_status") != policy_sim.NOT_VERIFIED_STATUS:
        problems.append("arms.simulated_rules: must state the not-verified simulated status")


def _check_data(m: Mapping[str, Any], problems: list[str]) -> None:
    data = m["data"]
    purpose, seed = m["purpose"], data["dataset_seed"]
    if (purpose == PURPOSE_CONFIRMATORY) != (seed in CONFIRMATORY_SEEDS):
        problems.append("data.dataset_seed: a confirmatory manifest needs a confirmatory seed and an infrastructure manifest must not use one")
    if seed != data["generator"]["config"].get("seed"):
        problems.append("data.dataset_seed: differs from the generator configuration")
    if seed in (7102, 7103) and not data["reserve_seed_log"]:
        problems.append("data.reserve_seed_log: a reserve seed needs a logged reason (protocol 12.5)")
    if not data["split"]["requirements_met"] or data["splitter_excluded"] != 0 or data["isolation_checked"] is not True:
        problems.append("data.split: must meet its requirements, with no exclusions and a passed isolation check")
    audit = data["audit"]
    if audit["errors"] != 0 or audit["construction_checks_passed"] is not True:
        problems.append("data.audit: construction checks must have passed with no errors")
    if set(data["counts"]["test"]) != {"records", "groups", "ids_sha256"}:
        problems.append("data.counts.test: may hold only records, groups and the id hash")
    if data["generator"]["version"] != GENERATOR_VERSION or data["generator"]["catalog_version"] != synthetic_catalog.CATALOG_VERSION:
        problems.append("data.generator: version differs from the code")


def validate_manifest(m: Any) -> None:
    """Structure and internal consistency of a manifest dictionary (no repository access). Raises :class:`ManifestSchemaError` listing field paths."""
    problems: list[str] = []
    if not isinstance(m, dict) or set(m) != _TOP_KEYS:
        raise ManifestSchemaError("the manifest does not have exactly the required top-level fields")
    for name, keys in _SECTION_KEYS.items():
        if not isinstance(m[name], dict) or set(m[name]) != keys:
            problems.append(f"{name}: missing or unexpected fields")
    if problems:
        raise ManifestSchemaError("; ".join(problems))
    if m["manifest_version"] != MANIFEST_VERSION:
        problems.append("manifest_version")
    if m["purpose"] not in PURPOSES or m["confirmatory"] is not (m["purpose"] == PURPOSE_CONFIRMATORY):
        problems.append("purpose")
    if not isinstance(m["created_utc"], str) or not _UTC.fullmatch(m["created_utc"]):
        problems.append("created_utc")
    p = m["protocol"]
    if p["status"] not in ("DRAFT", "FROZEN") or not _is_hex64(p["sha256"]) or p["approved_for_confirmatory"] is not (p["status"] == "FROZEN") \
            or not isinstance(p["version"], str):
        problems.append("protocol")
    if m["purpose"] == PURPOSE_CONFIRMATORY and p["status"] != "FROZEN":
        problems.append("protocol: a confirmatory manifest needs a FROZEN protocol")
    g = m["git"]
    if not isinstance(g["branch"], str) or not g["branch"] or not isinstance(g["commit"], str) or not _HEX40.fullmatch(g["commit"]) \
            or g["clean"] is not True or g["untracked_files_considered"] is not True:
        problems.append("git: needs a branch, a full commit and a clean working tree")
    for name in ("sources", "config_files"):
        files = m[name]
        if not isinstance(files, dict) or not files or not all(isinstance(k, str) and _is_hex64(v) for k, v in files.items()):
            problems.append(f"{name}: must map file paths to SHA-256 digests")
    if m["limitations"] != list(LIMITATIONS):
        problems.append("limitations: differ from the required statement")
    if m["selection"] != _plain(selection_section()):
        problems.append("selection: differs from the approved selection rules")
    if m["statistics"] != _plain(statistics_section()):
        problems.append("statistics: differs from the approved statistical protocol")
    if not isinstance(m["environment"], dict) or not {"python", "numpy", "platform", "threads"} <= set(m["environment"]):
        problems.append("environment: incomplete")
    for label, check in (("data", _check_data), ("arms", _check_arms), ("numerics", _check_numerics)):
        try:
            check(m, problems)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            problems.append(f"{label}: malformed or incomplete")
    version_expected = _plain({k: v for k, v in component_versions().items()})
    arms = m["arms"]
    if any(arms.get(k) != version_expected[k] for k in version_expected):
        problems.append("arms: component identifiers differ from the code")
    _scan_forbidden(m, "$", problems)
    if problems:
        raise ManifestSchemaError("; ".join(problems[:25]))


# -- independent verification ---------------------------------------------------------------------------------------------------------------


def _independent_checks(m: Mapping[str, Any], *, repo_root: Path, expected: ExpectedRun, source_files: Sequence[str],
                        config_files: Sequence[str], protocol_path: str, allow_environment_mismatch: bool) -> dict[str, Any]:
    failures: list[str] = []
    if m["purpose"] != expected.purpose:
        failures.append("purpose differs from the expected run")
    git = collect_git_state(repo_root)
    if not git["clean"]:
        raise ManifestVerificationError(f"the Git working tree is not clean ({git['dirty_entries']} entries)")
    if git["branch"] != expected.branch or m["git"]["branch"] != expected.branch:
        failures.append("git.branch differs from the expected branch")
    if git["commit"] != expected.commit or m["git"]["commit"] != expected.commit:
        failures.append("git.commit differs from the expected commit")
    live_sources = hash_files(repo_root, source_files)
    if m["sources"] != live_sources:
        changed = sorted(k for k in set(m["sources"]) | set(live_sources) if m["sources"].get(k) != live_sources.get(k))
        failures.append(f"sources differ from the repository ({len(changed)} file(s): {', '.join(changed[:5])})")
    live_config = hash_files(repo_root, config_files)
    if m["config_files"] != live_config:
        failures.append("config_files differ from the repository")
    protocol = collect_protocol(repo_root, protocol_path)
    for key in ("version", "status", "sha256"):
        if m["protocol"][key] != protocol[key]:
            failures.append(f"protocol.{key} differs from the protocol file")
    if expected.purpose == PURPOSE_CONFIRMATORY:
        if protocol["status"] != "FROZEN" or expected.approved_protocol_sha256 != protocol["sha256"]:
            raise ProtocolNotFrozenError("a confirmatory manifest needs a FROZEN protocol file whose SHA-256 equals the externally approved value")
    data = m["data"]
    for name, got, want in (("dataset_seed", data["dataset_seed"], expected.dataset_seed),
                            ("dataset_fingerprint", data["dataset_fingerprint"], expected.dataset_fingerprint),
                            ("generator.fingerprint", data["generator"]["fingerprint"], expected.generator_fingerprint),
                            ("split.seed", data["split"]["seed"], expected.split_seed),
                            ("split.manifest_sha256", data["split"]["manifest_sha256"], expected.split_manifest_sha256)):
        if got != want:
            failures.append(f"data.{name} differs from the expected value")
    if data["simulated_policy"] != {"catalog_version": policy_sim.CATALOG_VERSION, "catalog_sha256": policy_sim.CATALOG_SHA256}:
        failures.append("data.simulated_policy differs from the code")
    prov = m["numerics"]["provenance"]
    if prov != _plain(numerics.provenance()):
        failures.append("numerics.provenance differs from the running code")
    for rel, key in (("tokenization_ai/heldout_eval/numerics.py", "numerics_sha256"), ("tokenization_ai/model.py", "model_sha256")):
        if rel in m["sources"] and prov.get(key) != m["sources"][rel]:
            failures.append(f"numerics.provenance.{key} differs from the hashed source")
    env_diffs = environment_differences(m["environment"])
    if env_diffs and not allow_environment_mismatch:
        raise EnvironmentMismatchError(f"the running environment differs from the recorded one ({', '.join(env_diffs)})")
    if failures:
        raise ManifestVerificationError("; ".join(failures))
    return {"environment_differences": env_diffs, "git_clean": True}


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    manifest_sha256: str
    purpose: str
    protocol_status: str
    environment_differences: tuple[str, ...]
    checks: tuple[str, ...]


_CHECKS = ("expected_digest", "canonical_serialization", "schema_and_consistency", "git_clean_branch_commit", "source_and_config_hashes",
           "protocol_status_and_hash", "dataset_split_provenance", "numerical_configuration", "statistical_protocol", "environment")


def verify_manifest_bytes(data: bytes, *, expected_sha256: str, repo_root: str | os.PathLike[str], expected: ExpectedRun,
                          source_files: Sequence[str] = DEFAULT_SOURCE_FILES, config_files: Sequence[str] = DEFAULT_CONFIG_FILES,
                          protocol_path: str = DEFAULT_PROTOCOL_PATH, allow_environment_mismatch: bool = False) -> VerificationResult:
    """Verify exact manifest bytes against an externally supplied digest, the live repository and the trusted expectations."""
    if not isinstance(expected_sha256, str) or not _HEX64.fullmatch(expected_sha256):
        raise ManifestVerificationError("an expected SHA-256 from outside the manifest is required")
    if not isinstance(expected, ExpectedRun):
        raise ManifestVerificationError("trusted ExpectedRun values are required")
    if not isinstance(data, (bytes, bytearray)) or sha256_hex(bytes(data)) != expected_sha256:
        raise ManifestVerificationError("the manifest bytes do not match the expected SHA-256")
    manifest = parse_manifest_bytes(bytes(data))
    validate_manifest(manifest)
    outcome = _independent_checks(manifest, repo_root=Path(repo_root), expected=expected, source_files=tuple(source_files),
                                  config_files=tuple(config_files), protocol_path=protocol_path,
                                  allow_environment_mismatch=allow_environment_mismatch)
    return VerificationResult(ok=True, manifest_sha256=expected_sha256, purpose=manifest["purpose"], protocol_status=manifest["protocol"]["status"],
                              environment_differences=tuple(outcome["environment_differences"]), checks=_CHECKS)


def verify_manifest(path: str | os.PathLike[str], *, expected_sha256: str, **kwargs: Any) -> VerificationResult:
    """As :func:`verify_manifest_bytes`, reading the exact bytes of ``path``."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        raise ManifestVerificationError("the manifest file could not be read") from None
    return verify_manifest_bytes(data, expected_sha256=expected_sha256, **kwargs)


# -- write-once writer ---------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WrittenManifest:
    path: str
    sha256: str
    size: int
    purpose: str


def _outside(repo_root: Path, path: Path) -> bool:
    root = repo_root.resolve()
    target = path.resolve()
    return root != target and root not in target.parents


def write_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any], *, repo_root: str | os.PathLike[str], expected: ExpectedRun,
                   source_files: Sequence[str] = DEFAULT_SOURCE_FILES, config_files: Sequence[str] = DEFAULT_CONFIG_FILES,
                   protocol_path: str = DEFAULT_PROTOCOL_PATH, allow_environment_mismatch: bool = False) -> WrittenManifest:
    """Validate, independently check and write a manifest **once**; return the SHA-256 of the exact bytes written.

    Nothing is written unless every check of :func:`verify_manifest_bytes` (except the external digest, which does not exist yet) passes. An existing
    path is never overwritten.
    """
    target = Path(path)
    root = Path(repo_root)
    if not isinstance(expected, ExpectedRun):
        raise ManifestVerificationError("trusted ExpectedRun values are required")
    if not _outside(root, target):
        raise ManifestWriteError("the manifest path must be outside the repository (writing inside would dirty the tree it certifies)")
    if os.path.lexists(target):
        raise ManifestWriteError("a manifest already exists at this path; it is never overwritten")
    if not target.parent.is_dir():
        raise ManifestWriteError("the manifest directory does not exist")
    data = canonical_bytes(manifest)
    parsed = parse_manifest_bytes(data)  # round trip: the bytes that will be written are canonical and strictly parseable
    validate_manifest(parsed)
    _independent_checks(parsed, repo_root=root, expected=expected, source_files=tuple(source_files), config_files=tuple(config_files),
                        protocol_path=protocol_path, allow_environment_mismatch=allow_environment_mismatch)
    digest = sha256_hex(data)
    try:
        with open(target, "xb") as handle:  # exclusive creation: fails if the path appeared meanwhile
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise ManifestWriteError("a manifest already exists at this path; it is never overwritten") from None
    except OSError:
        raise ManifestWriteError("the manifest write failed; any partial file is left in place for review and is not replaced") from None
    try:
        if target.read_bytes() != data:
            raise ManifestWriteError("the written bytes differ from the intended bytes; the file is left in place for review")
        if os.name == "posix":
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.chmod(target, stat.S_IREAD)
    except ManifestWriteError:
        raise
    except OSError:
        raise ManifestWriteError("the manifest could not be finalized; the file is left in place for review") from None
    return WrittenManifest(path=str(target), sha256=digest, size=len(data), purpose=parsed["purpose"])


def source_sha256() -> str:
    """SHA-256 of this file with LF line endings (it is also listed in ``DEFAULT_SOURCE_FILES``)."""
    return file_sha256(Path(__file__))
