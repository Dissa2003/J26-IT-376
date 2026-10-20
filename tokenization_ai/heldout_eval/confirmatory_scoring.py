"""``confirmatory-scoring-1``: the one-time held-out test scoring of the selected A/B/C models (draft protocol heldout-eval-1.2.0-draft, sections 12.11.4
and 12.12.10).

:func:`score_test_once` is the only test-scoring path. It never touches a test label itself before the one-time gate has opened:

1. Before the gate (nothing is created): the training configuration must be the one bound by the manifest (frozen vocabulary hash, token budget, training
   record ids), the results directory must be a trusted directory outside the repository, and no result may exist yet.
2. ``attempt_gate.open_test_attempt`` verifies the frozen pre-run manifest and the validation-selection record with the real verifiers (including the
   recomputation of all 81 validation APs), creates the atomic one-time marker, and only then recovers the authentic test labels.
3. After the marker: exactly the three candidates selected on validation (read from the verified record) are loaded through the strict artifact loader; the
   blinded test requests are encoded with the frozen vocabulary and budget; raw logits are computed; per arm the Average Precision, ROC-AUC and the confusion
   matrix at the **frozen validation threshold** (``None`` = all-negative) are computed; the primary contrast ``AP(B) - AP(A)`` uses the existing paired
   template-group bootstrap with its fixed settings (2,000 replicates, seed 8201, 95% percentile interval, 95% valid-replicate convention); contrasts involving
   Arm C are computed the same way and marked secondary and descriptive. The decision category follows the pre-specified rule (interval entirely above zero:
   evidence of synthetic improvement; entirely below: possible worse; otherwise no improvement claimed; not estimable: inconclusive).
4. The aggregate result (no per-record label, score or request content) is written once, canonically, outside the repository.

There is no retraining, tuning, threshold change or second attempt: any failure after the marker leaves the marker in place, writes a redacted failure report
(stage ``test_scoring``) and raises; nothing is retried. If an analysis cannot be completed (for example an undefined AP or a non-estimable interval), that is
reported as such; no other statistic is substituted. All data are synthetic; Arm B uses simulated rules, not Member 2 verified rules; scores are uncalibrated logits.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import attempt_gate as gate
from . import detection_metrics as dm
from . import manifest as mf
from . import model_artifacts as ma
from . import numerics, runner
from . import selection_record as sr
from .arms import ARM_A, ARM_B, ARM_C, ARMS, build_arm_inputs, prepare_split
from .splits import EvalRecord

SCORING_VERSION = "confirmatory-scoring-1"
RESULT_VERSION = "confirmatory-test-result-1"
_INPUT_ATTR = {ARM_A: "arm_a", ARM_B: "arm_b", ARM_C: "arm_c"}
PRIMARY_CONTRAST = (ARM_A, ARM_B)  # AP(B) - AP(A)
SECONDARY_CONTRASTS = ((ARM_A, ARM_C), (ARM_B, ARM_C))  # AP(C) - AP(A), AP(C) - AP(B): descriptive only
LIMITATIONS: tuple[str, ...] = (
    "synthetic held-out evaluation only; no real-world detection claim",
    "Arm B uses simulated simpol-1 rules, not Member 2 formally verified rules",
    "scores are uncalibrated raw logits; thresholds are logit-scale",
    "bootstrap intervals describe synthetic-generator sampling uncertainty over template groups for one dataset and one split",
    "contrasts involving Arm C are secondary and descriptive",
)


class ScoringError(RuntimeError):
    """Scoring could not start (nothing was created). Messages name checks only."""


def result_filename(experiment_id: str) -> str:
    return f"test-result-{experiment_id}.json"


def decision_category(bootstrap: Mapping[str, Any]) -> str:
    """The pre-specified evidence rule (zero margin) applied to a paired bootstrap result dictionary."""
    if bootstrap.get("status") != dm.STATUS_OK or bootstrap.get("interval") is None:
        return "inconclusive"
    low, high = bootstrap["interval"]
    if low > 0:
        return "evidence_of_synthetic_improvement"
    if high < 0:
        return "possible_worse_synthetic_performance"
    return "no_improvement_claimed"


@dataclass(frozen=True)
class ScoringOutcome:
    result_path: str
    result_sha256: str
    experiment_id: str
    primary_decision: str
    marker_path: str


def _pre_gate_checks(manifest_bytes: bytes, train_config: runner.TrainConfiguration, results_dir: Path, repo_root: Any) -> None:
    if not isinstance(train_config, runner.TrainConfiguration):
        raise ScoringError("the runner's TrainConfiguration is required to encode the test requests")
    try:
        manifest = mf.parse_manifest_bytes(bytes(manifest_bytes))
    except (mf.ManifestError, TypeError):
        raise ScoringError("the manifest is malformed") from None
    cfg = train_config.to_dict()
    arms = manifest.get("arms", {})
    if (cfg["vocabulary"].get("sha256") != arms.get("vocabulary", {}).get("sha256") or cfg["train_ids_sha256"] != arms.get("alignment", {}).get("train_ids_sha256")
            or train_config.decision.selected_max_length != arms.get("budget", {}).get("selected_max_length")):
        raise ScoringError("the training configuration is not the one bound by the manifest")
    if not results_dir.is_dir():
        raise ScoringError("the results directory does not exist")
    resolved, root = results_dir.resolve(), Path(repo_root).resolve()
    if resolved == root or root in resolved.parents:
        raise ScoringError("the results directory must be outside the repository")


def _per_arm(logits: list[float], labels: list[int], threshold: float | None) -> dict[str, Any]:
    counts = dm.confusion_counts(logits, labels, threshold)
    return {"average_precision": dm.average_precision(logits, labels), "roc_auc": dm.roc_auc(logits, labels),
            "threshold": threshold, "threshold_scale": "raw_logit", "threshold_source": "frozen validation-only F1 threshold",
            "confusion": counts.to_dict()}


def _write_result(path: Path, result: Mapping[str, Any]) -> str:
    data = mf.canonical_bytes(result)
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != data:
        raise OSError("result read back differently")
    try:
        os.chmod(path, stat.S_IREAD)
    except OSError:
        pass
    return mf.sha256_hex(data)


def score_test_once(*, attempt_root: str | os.PathLike[str], results_dir: str | os.PathLike[str], manifest_bytes: bytes, expected_manifest_sha256: str,
                    record_bytes: bytes, expected_record_sha256: str, repo_root: str | os.PathLike[str], expected_run: mf.ExpectedRun,
                    evidence: sr.ValidationEvidence, sealed_test: runner.SealedTestSet, label_provider: Callable[[], Iterable[EvalRecord]],
                    train_config: runner.TrainConfiguration, created_utc: str | None = None) -> ScoringOutcome:
    """Open the one-time gate and score the three validation-selected models on the test split exactly once. See the module docstring."""
    results = Path(results_dir)
    _pre_gate_checks(manifest_bytes, train_config, results, repo_root)
    outcome = gate.open_test_attempt(attempt_root=attempt_root, manifest_bytes=manifest_bytes, expected_manifest_sha256=expected_manifest_sha256,
                                     record_bytes=record_bytes, expected_record_sha256=expected_record_sha256, repo_root=repo_root, expected_run=expected_run,
                                     evidence=evidence, sealed_test=sealed_test, label_provider=label_provider, created_utc=created_utc)
    try:
        result_path = results / result_filename(outcome.experiment_id)
        if os.path.lexists(result_path):
            raise OSError("a result already exists")
        manifest = mf.parse_manifest_bytes(bytes(manifest_bytes))
        record = mf.parse_manifest_bytes(bytes(record_bytes))
        rates = ma.learning_rates_from_manifest(manifest)
        selected = {arm: record["arms"][arm]["selected"]["candidate_id"] for arm in ARMS}
        thresholds = {arm: record["arms"][arm]["threshold"]["value"] for arm in ARMS}
        models = {}
        for arm, cid in selected.items():
            loaded = ma.load_model_artifact(evidence.artifact_dir, expected_sha256=evidence.artifact_digests[cid], expected_arm=arm, expected_candidate_id=cid,
                                            expected_manifest_sha256=expected_manifest_sha256, expected_learning_rate=rates[cid],
                                            expected_model_sha256=manifest["numerics"]["provenance"]["model_sha256"])
            if loaded.fingerprint["weights_sha256"] != record["arms"][arm]["model"]["weights_sha256"]:
                raise ValueError("selected model differs from the record")
            models[arm] = loaded.model
        prepared = prepare_split(sealed_test.blinded_records(), "test")
        test_inputs = build_arm_inputs(prepared, train_config.frozen, train_config.decision)
        if tuple(test_inputs.record_ids) != tuple(outcome.labels.record_ids):
            raise ValueError("test inputs and recovered labels are not aligned")
        labels = list(outcome.labels.labels_in_order())
        ids = list(test_inputs.record_ids)
        logits = {arm: numerics.raw_logits(models[arm], list(getattr(test_inputs, _INPUT_ATTR[arm]))) for arm in ARMS}
        per_arm = {arm: _per_arm(logits[arm], labels, thresholds[arm]) for arm in ARMS}
        by_id = {arm: dict(zip(ids, logits[arm])) for arm in ARMS}
        labels_by_id = dict(zip(ids, labels))
        groups = dict(sealed_test.group_by_id)

        def contrast(base: str, other: str) -> dict[str, Any]:
            boot = dm.paired_group_bootstrap(by_id[base], by_id[other], labels_by_id, groups).to_dict()
            boot["statistic"] = f"AP({other}) - AP({base})"
            return boot

        primary = contrast(*PRIMARY_CONTRAST)
        secondary = [{**contrast(a, b), "role": "secondary_descriptive"} for a, b in SECONDARY_CONTRASTS]
        result = {
            "result_version": RESULT_VERSION, "scoring_version": SCORING_VERSION, "experiment_id": outcome.experiment_id,
            "manifest_sha256": expected_manifest_sha256, "selection_record_sha256": expected_record_sha256, "marker_sha256": outcome.marker_sha256,
            "selected_candidates": selected, "test_records": len(ids), "test_groups": len(set(groups.values())),
            "per_arm": per_arm, "primary": {**primary, "decision": decision_category(primary), "rule": "whole 95% interval strictly above zero; margin 0"},
            "secondary": secondary, "post_test_changes": "none: no retraining, tuning, threshold change or second attempt",
            "limitations": list(LIMITATIONS),
        }
        digest = _write_result(result_path, result)
    except Exception:  # noqa: BLE001 - any failure after the marker is reported by category only and never retried
        report = gate.report_post_marker_failure(attempt_root, outcome, "test_scoring", "scoring_failed")
        raise gate.PostMarkerFailure("scoring_failed", "test_scoring", report) from None
    return ScoringOutcome(result_path=str(result_path), result_sha256=digest, experiment_id=outcome.experiment_id,
                          primary_decision=result["primary"]["decision"], marker_path=outcome.marker_path)


def source_sha256() -> str:
    return mf.file_sha256(Path(__file__))
