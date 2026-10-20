"""``validation-selection-record-1``: the write-once validation-selection record and its independent verifier (draft protocol heldout-eval-1.2.0-draft,
sections 12.11.4 and 12.12.3).

Infrastructure only. The record is written AFTER (future) authorized training and validation and BEFORE the one-time test gate. This module fits nothing,
scores nothing, reads **no test label, score or prediction**, and never authorizes test scoring (``VerificationResult.authorizes_test_scoring`` is always
``False``; the runner's gates are untouched).

What a record binds
-------------------
The exact pre-run manifest (by an **externally established SHA-256**, and the manifest must pass the real :func:`manifest.verify_manifest_bytes`); per arm the
selected candidate, all 27 approved candidates with full-precision and 10-decimal-rounded validation AP and a tie-break trace; the validation-only F1
threshold on the **raw-logit scale** with its confusion counts and score-tie diagnostics; the fitted model's artifact fingerprint; the frozen train-only
vocabulary and token budget; the model-source hash; and an explicit test-not-scored status. No request body, test label or test score is stored.

How verification works (trust model)
------------------------------------
A record never vouches for itself. Verification **rebuilds the whole record** from independent inputs and requires equality:

* the pre-run manifest bytes, their expected digest and the manifest verifier's own independent checks (clean Git tree, commit, source hashes, ...);
* a :class:`ValidationEvidence` supplied by the caller from outside the record: the validation ``ArmInputs`` encoded with the frozen vocabulary and budget,
  the validation labels by record id, the frozen vocabulary, the split result, and the **actual fitted model of each arm**;
* the approved selection rules, applied again by ``runner.select_candidate`` to the recorded candidate table;
* the selected candidate's validation AP, the F1 threshold, its confusion counts and the tie diagnostics, **recomputed** from the supplied model's raw logits on the
  validation inputs with ``detection_metrics``; the model artifact fingerprint is recomputed from the model's weights and bias.

Trust boundaries (documented, not hidden): the validation APs of the 26 non-selected candidates are checked for consistency with the selection rules but
cannot be recomputed without their models; they are claims of the training stage. A record whose evidence is not supplied is rejected (there is no mode that
skips the recomputation). This module is not listed in the manifest's default source list (the manifest predates it); the record therefore carries this
file's SHA-256 and the verifier compares it with the live file.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..model import LightweightAnomalyModel
from . import detection_metrics as dm
from . import manifest as mf
from . import numerics, runner
from .arms import ARM_A, ARM_B, ARM_C, ARMS, ArmInputs
from .detection_metrics import ThresholdSelection, average_precision, confusion_counts, select_threshold
from .runner import ValidationCandidateResult
from .splits import SplitResult

RECORD_VERSION = "validation-selection-record-1"
THRESHOLD_SCALE = "raw_logit"
AP_TOLERANCE_REL = 1e-9
AP_TOLERANCE_ABS = 1e-12
COMPATIBILITY_SAMPLE = 50
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_TEST_STATUS = {"test_scored": False, "test_labels_accessed": False, "test_scores_accessed": False, "test_data_in_record": False}
_TEST_KEYS_ALLOWED = frozenset({"test_status", "test_scored", "test_labels_accessed", "test_scores_accessed", "test_data_in_record",
                                "authorizes_test_scoring"})
_TOP_KEYS = {"record_version", "created_utc", "manifest", "validation", "arms", "rules", "statistics", "test_status", "code", "vocabulary", "budget",
             "limitations"}
_ARM_KEYS = {"selected", "candidates", "tie_break_trace", "threshold", "tie_diagnostics", "model"}
_CHOSEN_INPUT = {ARM_A: "arm_a", ARM_B: "arm_b", ARM_C: "arm_c"}
LIMITATIONS: tuple[str, ...] = (
    "the validation APs of non-selected candidates are claims of the training stage; only the selected candidate is recomputed",
    "thresholds are raw-logit-scale ranking cut-offs, not probability thresholds",
    "synthetic held-out evaluation only; no real-world detection claim",
    "this record does not authorize test scoring",
)


class SelectionRecordError(mf.ManifestError):
    """Base class. Messages hold field paths, categories and counts only."""


class SelectionRecordSchemaError(SelectionRecordError):
    """The record is malformed, ambiguous, incomplete or contains forbidden fields."""


class SelectionRecordWriteError(SelectionRecordError):
    """The record could not be written (existing path, I/O failure). Nothing was repaired or replaced."""


class SelectionRecordVerificationError(SelectionRecordError):
    """A check against the independently rebuilt truth failed."""


# -- evidence ---------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationEvidence:
    """Trusted, caller-supplied inputs that let verification recompute the validation-only results. None of it comes from the record."""

    arm_inputs: ArmInputs  # validation split, encoded with the frozen vocabulary and budget
    labels_by_id: Mapping[str, int]  # validation labels only
    frozen: Any  # runner/arms FrozenVocabulary (its .record["sha256"] is compared)
    split_result: SplitResult
    models: Mapping[str, LightweightAnomalyModel]  # the fitted model selected for each arm


def model_artifact_fingerprint(model: LightweightAnomalyModel) -> dict[str, Any]:
    """Robust binding of a fitted model: bucket count, SHA-256 of the little-endian float64 weight bytes, and the exact bias."""
    if not isinstance(model, LightweightAnomalyModel) or not model.is_trained:
        raise SelectionRecordVerificationError("a fitted LightweightAnomalyModel is required")
    weights = np.ascontiguousarray(model.weights, dtype="<f8")
    if weights.shape != (model.num_buckets,) or not np.isfinite(weights).all() or not math.isfinite(float(model.bias)):
        raise SelectionRecordVerificationError("the model artifact is malformed")
    return {"num_buckets": int(model.num_buckets), "weights_sha256": mf.sha256_hex(weights.tobytes()), "bias_hex": float(model.bias).hex(),
            "is_trained": True, "weights_dtype": "float64_little_endian"}


# -- helpers ------------------------------------------------------------------------------------------------------------------------------


def source_sha256() -> str:
    return mf.file_sha256(Path(__file__))


def _tie_trace(results: Sequence[ValidationCandidateResult]) -> list[dict[str, Any]]:
    rounded = {r.candidate_id: round(float(r.validation_ap), runner.AP_ROUND_DECIMALS) for r in results}
    top = max(rounded.values())
    tied = [r for r in results if rounded[r.candidate_id] == top]
    steps = [{"rule": "highest_rounded_validation_ap", "survivors": len(tied), "rounded_ap": top}]
    for rule, key in (("fewer_epochs", lambda r: r.epochs), ("smaller_num_buckets", lambda r: r.num_buckets),
                      ("smaller_relative_step_c", lambda r: r.relative_step_c), ("candidate_id_lexicographic", lambda r: r.candidate_id)):
        best = min(key(r) for r in tied)
        tied = [r for r in tied if key(r) == best]
        steps.append({"rule": rule, "survivors": len(tied)})
    return steps


def _tie_diagnostics(logits: Sequence[float]) -> dict[str, int]:
    counts = Counter(float(x) + 0.0 for x in logits)
    return {"validation_records": len(logits), "distinct_values": len(counts),
            "records_in_tied_blocks": sum(c for c in counts.values() if c > 1), "largest_tied_block": max(counts.values())}


def threshold_block(selection: ThresholdSelection, logits: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """The record's threshold section for a selection. ``value = None`` is the valid all-negative prediction. Raw-logit scale only."""
    counts = confusion_counts(logits, labels, selection.threshold)
    metrics = counts.metrics()
    return {
        "rule": selection.rule, "scale": THRESHOLD_SCALE, "predict_positive_when": "logit >= value (value null: nothing is predicted positive)",
        "value": selection.threshold, "predicts_all_negative": selection.threshold is None, "validation_f1": selection.f1,
        "counts": {"tp": counts.tp, "fp": counts.fp, "tn": counts.tn, "fn": counts.fn},
        "metrics": dict(metrics), "metric_status": {k: ("defined" if v is not None else "undefined") for k, v in metrics.items()},
        "candidates_considered": selection.candidates_considered, "basis": "validation_scores_and_labels_only",
        "probability_threshold": False,
    }


def _validate_evidence(manifest: Mapping[str, Any], evidence: ValidationEvidence) -> tuple[str, list[int]]:
    if not isinstance(evidence, ValidationEvidence):
        raise SelectionRecordVerificationError("trusted ValidationEvidence is required; there is no mode without it")
    if not isinstance(evidence.arm_inputs, ArmInputs) or evidence.arm_inputs.split != "validation":
        raise SelectionRecordVerificationError("the evidence inputs must be the validation split")
    ids = list(evidence.arm_inputs.record_ids)
    arms = manifest["arms"]
    if len(ids) != arms["alignment"]["validation_records"]:
        raise SelectionRecordVerificationError("the validation record count differs from the manifest")
    if evidence.arm_inputs.max_length != arms["budget"]["selected_max_length"]:
        raise SelectionRecordVerificationError("the validation inputs do not use the manifest's frozen token budget")
    record = getattr(evidence.frozen, "record", None)
    if not isinstance(record, Mapping) or record.get("sha256") != arms["vocabulary"]["sha256"]:
        raise SelectionRecordVerificationError("the frozen vocabulary differs from the manifest")
    split = evidence.split_result
    if not isinstance(split, SplitResult) or split.manifest_sha256 != manifest["data"]["split"]["manifest_sha256"] \
            or set(split.members("validation")) != set(ids):
        raise SelectionRecordVerificationError("the split result does not match the manifest and the validation records")
    try:
        labels = list(runner.aligned_labels(ids, evidence.labels_by_id))
    except runner.RunnerError:
        raise SelectionRecordVerificationError("the validation labels do not align with the validation records") from None
    if not isinstance(evidence.models, Mapping) or set(evidence.models) != set(ARMS):
        raise SelectionRecordVerificationError("exactly one fitted model per arm is required")
    return runner.ids_fingerprint(ids), labels


def _manifest_candidates(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {c["candidate_id"]: c for c in manifest["numerics"]["candidates"]}


def _arm_block(arm: str, results: Sequence[ValidationCandidateResult], manifest: Mapping[str, Any], evidence: ValidationEvidence,
               ids_fp: str, labels: list[int]) -> dict[str, Any]:
    try:
        decision = runner.select_candidate(list(results), arm=arm, validation_ids_sha256=ids_fp)
    except runner.RunnerError as exc:
        raise SelectionRecordVerificationError(f"arm {arm}: candidate selection input rejected ({exc})") from None
    by_id = _manifest_candidates(manifest)
    rows = []
    for r in sorted(results, key=lambda x: x.candidate_id):
        spec = by_id.get(r.candidate_id)
        if (spec is None or spec["arm"] != arm or spec["num_buckets"] != r.num_buckets or spec["epochs"] != r.epochs
                or spec["relative_step_c"] != r.relative_step_c):
            raise SelectionRecordVerificationError(f"arm {arm}: a candidate does not correspond to the pre-run manifest")
        rows.append({"candidate_id": r.candidate_id, "num_buckets": r.num_buckets, "epochs": r.epochs, "relative_step_c": r.relative_step_c,
                     "learning_rate": spec["learning_rate"], "learning_rate_hex": spec["learning_rate_hex"],
                     "validation_ap": float(r.validation_ap), "rounded_ap": round(float(r.validation_ap), runner.AP_ROUND_DECIMALS)})
    trace = _tie_trace(list(results))
    chosen = decision.selected
    model = evidence.models[arm]
    inputs = list(getattr(evidence.arm_inputs, _CHOSEN_INPUT[arm]))
    try:
        logits = numerics.raw_logits(model, inputs)
        step = max(1, len(inputs) // COMPATIBILITY_SAMPLE)  # an evenly spaced, deterministic sample is enough to check the readout against predict()
        compat = numerics.verify_logit_compatibility(model, inputs[::step][:COMPATIBILITY_SAMPLE])
    except (numerics.NumericsError, ValueError):
        raise SelectionRecordVerificationError(f"arm {arm}: the fitted model cannot be read out on the validation inputs") from None
    if model.num_buckets != chosen.num_buckets:
        raise SelectionRecordVerificationError(f"arm {arm}: the supplied model does not have the selected candidate's num_buckets")
    ap = average_precision(logits, labels)
    if ap is None or not math.isclose(ap, float(chosen.validation_ap), rel_tol=AP_TOLERANCE_REL, abs_tol=AP_TOLERANCE_ABS):
        raise SelectionRecordVerificationError(f"arm {arm}: the selected candidate's validation AP does not match its model's validation scores")
    try:
        selection = select_threshold(logits, labels)
    except dm.DetectionMetricsError:
        raise SelectionRecordVerificationError(f"arm {arm}: validation-only threshold selection is not possible") from None
    estimate = next(e for e in manifest["numerics"]["estimates"] if e["arm"] == arm and e["num_buckets"] == chosen.num_buckets)
    spec = by_id[chosen.candidate_id]
    return {
        "selected": {
            "candidate_id": chosen.candidate_id, "num_buckets": chosen.num_buckets, "epochs": chosen.epochs, "relative_step_c": chosen.relative_step_c,
            "learning_rate": spec["learning_rate"], "learning_rate_hex": spec["learning_rate_hex"], "validation_ap": float(chosen.validation_ap),
            "rounded_ap": decision.rounded_ap, "lambda_method": estimate["method"], "lambda_max": estimate["lambda_max"], "L": estimate["L"],
            "lambda_train_ids_sha256": estimate["record_ids_sha256"], "tied_at_rounded_ap": decision.tied_at_rounded_ap,
        },
        "candidates": rows, "tie_break_trace": trace, "threshold": threshold_block(selection, logits, labels), "tie_diagnostics": _tie_diagnostics(logits),
        "model": {**model_artifact_fingerprint(model), "model_sha256": manifest["numerics"]["provenance"]["model_sha256"],
                  "logit_method": numerics.LOGIT_METHOD_ID, "input_representation": {
                      "arms_version": manifest["arms"]["arms_version"], "arm": arm, "max_length": evidence.arm_inputs.max_length if arm != ARM_C else None,
                      "vocabulary_sha256": manifest["arms"]["vocabulary"]["sha256"] if arm != ARM_C else None,
                      "encoder": manifest["arms"]["arm_c_encoder"]["encoder_id"] if arm == ARM_C else None},
                  "readout_compatibility": {"method": compat["method"], "examples": compat["examples"], "checked": compat["checked"],
                                              "saturated_not_compared": compat["saturated_not_compared"],
                                              "max_abs_difference": compat["max_abs_difference"], "tolerance": compat["tolerance"],
                                              "comparison_limit_abs": compat["max_abs_logit_compared"]}},
    }


def _assemble(manifest: Mapping[str, Any], manifest_sha256: str, results_by_arm: Mapping[str, Sequence[ValidationCandidateResult]],
              evidence: ValidationEvidence, created_utc: str) -> dict[str, Any]:
    ids_fp, labels = _validate_evidence(manifest, evidence)
    if not isinstance(results_by_arm, Mapping) or set(results_by_arm) != set(ARMS):
        raise SelectionRecordVerificationError("candidate results are required for exactly the three arms")
    data, protocol, git = manifest["data"], manifest["protocol"], manifest["git"]
    return {
        "record_version": RECORD_VERSION, "created_utc": created_utc,
        "manifest": {"version": manifest["manifest_version"], "sha256": manifest_sha256, "purpose": manifest["purpose"],
                     "protocol_version": protocol["version"], "protocol_sha256": protocol["sha256"], "git_branch": git["branch"],
                     "git_commit": git["commit"], "dataset_seed": data["dataset_seed"], "dataset_fingerprint": data["dataset_fingerprint"],
                     "generator_fingerprint": data["generator"]["fingerprint"], "split_seed": data["split"]["seed"],
                     "split_manifest_sha256": data["split"]["manifest_sha256"]},
        "validation": {"split": "validation", "records": len(labels), "ids_sha256": ids_fp},
        "arms": {arm: _arm_block(arm, results_by_arm[arm], manifest, evidence, ids_fp, labels) for arm in ARMS},
        "rules": manifest["selection"], "statistics": manifest["statistics"], "test_status": dict(_TEST_STATUS),
        "code": {"selection_record_sha256": source_sha256(), "model_sha256": manifest["numerics"]["provenance"]["model_sha256"]},
        "vocabulary": {"sha256": manifest["arms"]["vocabulary"]["sha256"], "min_frequency": manifest["arms"]["vocabulary"]["min_frequency"],
                       "fitted_on_split": manifest["arms"]["vocabulary"]["fitted_on_split"]},
        "budget": {"rule_id": manifest["arms"]["budget"]["rule_id"], "selected_max_length": manifest["arms"]["budget"]["selected_max_length"]},
        "limitations": list(LIMITATIONS),
    }


def _load_manifest(manifest_bytes: bytes, expected_manifest_sha256: str) -> dict[str, Any]:
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise SelectionRecordVerificationError("an expected manifest SHA-256 from outside is required")
    if not isinstance(manifest_bytes, (bytes, bytearray)) or mf.sha256_hex(bytes(manifest_bytes)) != expected_manifest_sha256:
        raise SelectionRecordVerificationError("the manifest bytes do not match the expected manifest SHA-256")
    try:
        return mf.parse_manifest_bytes(bytes(manifest_bytes))
    except mf.ManifestError:
        raise SelectionRecordVerificationError("the manifest is malformed") from None


def build_selection_record(*, manifest_bytes: bytes, expected_manifest_sha256: str, results_by_arm: Mapping[str, Sequence[ValidationCandidateResult]],
                           evidence: ValidationEvidence, created_utc: str) -> dict[str, Any]:
    """Assemble a record by recomputing everything from the evidence (nothing is written or fully verified here; see the writer and verifier)."""
    if not isinstance(created_utc, str) or not _UTC.fullmatch(created_utc):
        raise SelectionRecordSchemaError("created_utc must look like 2026-01-01T00:00:00Z")
    manifest = _load_manifest(manifest_bytes, expected_manifest_sha256)
    return mf._plain(_assemble(manifest, expected_manifest_sha256, results_by_arm, evidence, created_utc))


# -- schema and forbidden content ---------------------------------------------------------------------------------------------------------------


def _scan(value: Any, path: str, problems: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            tokens = set(re.split(r"[^a-z0-9]+", key.lower()))
            data_bearing = isinstance(item, (dict, list)) or (isinstance(item, (int, float)) and not isinstance(item, bool))
            if tokens & mf._FORBIDDEN_KEY_TOKENS and data_bearing:
                problems.append(f"{path}.{key}: label, score, prediction, logit or body data is not allowed in a record")
            if "test" in tokens and key not in _TEST_KEYS_ALLOWED:
                problems.append(f"{path}.{key}: test-derived content is not allowed in a validation-selection record")
            _scan(item, f"{path}.{key}", problems)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _scan(item, f"{path}[{i}]", problems)


def validate_record_structure(record: Any) -> None:
    problems: list[str] = []
    if not isinstance(record, dict) or set(record) != _TOP_KEYS:
        raise SelectionRecordSchemaError("the record does not have exactly the required top-level fields")
    if record["record_version"] != RECORD_VERSION:
        problems.append("record_version")
    if not isinstance(record["created_utc"], str) or not _UTC.fullmatch(record["created_utc"]):
        problems.append("created_utc")
    if record["test_status"] != _TEST_STATUS:
        problems.append("test_status: must state that no test data was scored, read or included")
    if record["limitations"] != list(LIMITATIONS):
        problems.append("limitations")
    arms = record["arms"]
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        problems.append("arms: exactly the three recognized arms are required")
    else:
        for arm, block in arms.items():
            if not isinstance(block, dict) or set(block) != _ARM_KEYS:
                problems.append(f"arms.{arm}: missing or unexpected fields")
                continue
            threshold = block["threshold"]
            if not isinstance(threshold, dict) or threshold.get("scale") != THRESHOLD_SCALE or threshold.get("probability_threshold") is not False \
                    or threshold.get("basis") != "validation_scores_and_labels_only" or threshold.get("rule") != dm.THRESHOLD_RULE \
                    or threshold.get("predicts_all_negative") is not (threshold.get("value") is None):
                problems.append(f"arms.{arm}.threshold: must be a validation-only raw-logit threshold")
            elif threshold["value"] is not None and not (isinstance(threshold["value"], float) and math.isfinite(threshold["value"])):
                problems.append(f"arms.{arm}.threshold.value: must be a finite logit or null")
            if not isinstance(block["candidates"], list) or len(block["candidates"]) != 27:
                problems.append(f"arms.{arm}.candidates: exactly the 27 approved candidates are required")
    _scan(record, "$", problems)
    if problems:
        raise SelectionRecordSchemaError("; ".join(problems[:25]))


def _same(a: Any, b: Any, path: str, diffs: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                diffs.append(f"{path}.{key}")
            else:
                _same(a[key], b[key], f"{path}.{key}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(path)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _same(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, float) and isinstance(b, float):
        if not math.isclose(a, b, rel_tol=AP_TOLERANCE_REL, abs_tol=AP_TOLERANCE_ABS):
            diffs.append(path)
    elif type(a) is not type(b) or a != b:
        diffs.append(path)


def _results_from_record(record: Mapping[str, Any], evidence: ValidationEvidence) -> dict[str, list[ValidationCandidateResult]]:
    ids_fp = runner.ids_fingerprint(list(evidence.arm_inputs.record_ids))
    n = len(evidence.arm_inputs.record_ids)
    out = {}
    for arm in ARMS:
        out[arm] = [ValidationCandidateResult(r["candidate_id"], arm, r["num_buckets"], r["epochs"], r["relative_step_c"], r["validation_ap"],
                                              "validation", ids_fp, n) for r in record["arms"][arm]["candidates"]]
    return out


def check_record(record: Mapping[str, Any], *, manifest: Mapping[str, Any], expected_manifest_sha256: str, evidence: ValidationEvidence) -> None:
    """Structure plus a full independent rebuild of the record from the manifest and the evidence; raises on any difference (field paths only)."""
    if not isinstance(evidence, ValidationEvidence):
        raise SelectionRecordVerificationError("trusted ValidationEvidence is required; there is no mode without it")
    validate_record_structure(record)
    try:
        results = _results_from_record(record, evidence)
    except (KeyError, TypeError, AttributeError, ValueError):
        raise SelectionRecordSchemaError("the candidate tables are malformed") from None
    try:
        rebuilt = mf._plain(_assemble(manifest, expected_manifest_sha256, results, evidence, record["created_utc"]))
    except SelectionRecordError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, StopIteration, runner.RunnerError):
        raise SelectionRecordVerificationError("the record could not be rebuilt from the manifest and the evidence") from None
    diffs: list[str] = []
    _same(rebuilt, dict(record), "$", diffs)
    if diffs:
        raise SelectionRecordVerificationError(f"the record differs from the independently rebuilt record at: {', '.join(diffs[:10])}")


# -- verification --------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionVerification:
    ok: bool
    record_sha256: str
    manifest_sha256: str
    selected_candidates: Mapping[str, str]
    authorizes_test_scoring: bool = False  # never True: scoring needs the attempt marker, which does not exist
    test_scored: bool = False


def verify_selection_record_bytes(data: bytes, *, expected_sha256: str, manifest_bytes: bytes, expected_manifest_sha256: str,
                                  repo_root: str | os.PathLike[str], expected_run: mf.ExpectedRun, evidence: ValidationEvidence,
                                  **manifest_kwargs: Any) -> SelectionVerification:
    """Verify exact record bytes against an external digest, the actually verified manifest and independently supplied validation evidence."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise SelectionRecordVerificationError("an expected record SHA-256 from outside the record is required")
    if not isinstance(data, (bytes, bytearray)) or mf.sha256_hex(bytes(data)) != expected_sha256:
        raise SelectionRecordVerificationError("the record bytes do not match the expected SHA-256")
    try:
        record = mf.parse_manifest_bytes(bytes(data))
    except mf.ManifestError as exc:
        raise SelectionRecordSchemaError(str(exc)) from None
    _load_manifest(manifest_bytes, expected_manifest_sha256)
    try:
        mf.verify_manifest_bytes(bytes(manifest_bytes), expected_sha256=expected_manifest_sha256, repo_root=repo_root, expected=expected_run,
                                 **manifest_kwargs)
    except mf.ManifestError as exc:
        raise SelectionRecordVerificationError(f"the pre-run manifest did not verify: {exc}") from None
    manifest = mf.parse_manifest_bytes(bytes(manifest_bytes))
    check_record(record, manifest=manifest, expected_manifest_sha256=expected_manifest_sha256, evidence=evidence)
    return SelectionVerification(ok=True, record_sha256=expected_sha256, manifest_sha256=expected_manifest_sha256,
                                 selected_candidates={arm: record["arms"][arm]["selected"]["candidate_id"] for arm in ARMS})


def verify_selection_record(path: str | os.PathLike[str], *, expected_sha256: str, **kwargs: Any) -> SelectionVerification:
    try:
        data = Path(path).read_bytes()
    except OSError:
        raise SelectionRecordVerificationError("the record file could not be read") from None
    return verify_selection_record_bytes(data, expected_sha256=expected_sha256, **kwargs)


# -- write-once writer -------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WrittenSelectionRecord:
    path: str
    sha256: str
    size: int


def write_selection_record(path: str | os.PathLike[str], record: Mapping[str, Any], *, manifest_bytes: bytes, expected_manifest_sha256: str,
                           repo_root: str | os.PathLike[str], expected_run: mf.ExpectedRun, evidence: ValidationEvidence,
                           **manifest_kwargs: Any) -> WrittenSelectionRecord:
    """Verify everything, then create the record file **once** (exclusive creation, fsync, read-back, read-only) and return the digest of its bytes."""
    target = Path(path)
    root = Path(repo_root)
    resolved_root, resolved = root.resolve(), target.resolve()
    if resolved_root == resolved or resolved_root in resolved.parents:
        raise SelectionRecordWriteError("the record path must be outside the repository")
    if os.path.lexists(target):
        raise SelectionRecordWriteError("a record already exists at this path; it is never overwritten")
    if not target.parent.is_dir():
        raise SelectionRecordWriteError("the record directory does not exist")
    try:
        data = mf.canonical_bytes(record)
        parsed = mf.parse_manifest_bytes(data)
    except mf.ManifestError as exc:
        raise SelectionRecordSchemaError(str(exc)) from None
    _load_manifest(manifest_bytes, expected_manifest_sha256)
    try:
        mf.verify_manifest_bytes(bytes(manifest_bytes), expected_sha256=expected_manifest_sha256, repo_root=root, expected=expected_run,
                                 **manifest_kwargs)
    except mf.ManifestError as exc:
        raise SelectionRecordVerificationError(f"the pre-run manifest did not verify: {exc}") from None
    check_record(parsed, manifest=mf.parse_manifest_bytes(bytes(manifest_bytes)), expected_manifest_sha256=expected_manifest_sha256, evidence=evidence)
    digest = mf.sha256_hex(data)
    try:
        with open(target, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise SelectionRecordWriteError("a record already exists at this path; it is never overwritten") from None
    except OSError:
        raise SelectionRecordWriteError("the record write failed; any partial file is left in place for review and is not replaced") from None
    try:
        if target.read_bytes() != data:
            raise SelectionRecordWriteError("the written bytes differ from the intended bytes; the file is left in place for review")
        os.chmod(target, stat.S_IREAD)
    except SelectionRecordWriteError:
        raise
    except OSError:
        raise SelectionRecordWriteError("the record could not be finalized; the file is left in place for review") from None
    return WrittenSelectionRecord(path=str(target), sha256=digest, size=len(data))
