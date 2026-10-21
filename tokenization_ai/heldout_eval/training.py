"""``training-stage-1``: the authorized training and validation stage of the held-out A/B/C evaluation (draft protocol heldout-eval-1.2.0-draft,
sections 12.11.2-12.11.4 and 12.12.9).

Three steps, each requiring the previous one's output object (constructible only inside this module):

1. :func:`build_training_plan` takes a genuine :class:`runner.TrainingAuthorization` (checked with ``runner.require_training_authorization``) and the
   runner's prepared objects, and re-derives everything it relies on instead of trusting it: the split result's own hash, the dataset content fingerprint
   (recomputed from the partition's request content, labels excluded), train/validation/test membership, disjointness and group isolation, the training
   configuration (recomputed from the training records: vocabulary, token budget and encoded inputs must be identical), the numerical configuration
   (recomputed: eigenvalue estimates, learning-rate table, candidates), and the identities bound into the authorization. It produces a deterministic plan for
   exactly the 81 approved candidates (3 arms x ``num_buckets`` {1024, 2048, 4096} x ``epochs`` {50, 150, 400} x ``c`` {0.1, 0.3, 0.8}) with the authorized
   train-only learning rates. It holds the training labels (aligned by record id) and nothing else.
2. :func:`train_candidates` fits one fresh ``LightweightAnomalyModel`` per candidate with the model's own ``fit`` on the **training** inputs and **training**
   labels only, with the approved epochs and learning rate, persists each with ``model_artifacts.write_model_artifact`` (write-once, outside the repository),
   reloads each through the strict loader and checks bit-exact weights, bias, fingerprint and raw-logit readout, and writes a write-once training ledger of the 81
   artifact digests. Any failure stops the stage; nothing is retried, overwritten or deleted.
3. :func:`select_on_validation` encodes the **validation** records with the frozen vocabulary and budget, loads all 81 artifacts through the strict loader,
   computes every candidate's validation AP from raw logits, and writes the ``validation-selection-record-2`` through the real writer, which re-verifies the
   manifest, re-loads the artifacts, recomputes all 81 APs, re-runs the approved selection rule and recomputes the validation-only F1 thresholds; the written bytes
   are then verified again with the real verifier.

No step reads a test label (the sealed test set holds none), scores a test record, creates or consumes an attempt marker, or reports a metric. All three
are deterministic for fixed inputs. Python object restrictions here prevent accidental misuse by honest code; they are not a boundary against malicious code
running in the same process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from . import manifest as mf
from . import model_artifacts as ma
from . import numerics, runner
from . import selection_record as sr
from .arms import ARM_A, ARM_B, ARM_C, ARMS, new_untrained_model
from .detection_metrics import average_precision
from .generator import SyntheticDataset
from .splits import SplitResult, _manifest_hash, exact_fingerprint

TRAINING_STAGE_ID = "training-stage-1"
LEDGER_VERSION = "training-ledger-1"
EXPECTED_CANDIDATES = 81
_INPUT_ATTR = {ARM_A: "arm_a", ARM_B: "arm_b", ARM_C: "arm_c"}
_KEY = object()  # only this module creates TrainingPlan / TrainedCandidates objects
_READOUT_CHECK = 8  # training inputs re-scored after reload to confirm identical raw logits


class TrainingStageError(RuntimeError):
    """The training stage cannot proceed. Messages hold check names, categories, positions and counts only, never values, labels or request content."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _content_fingerprint(records) -> str:
    """The splitter's dataset fingerprint (sorted exact request fingerprints), recomputed from request content only; labels do not enter it."""
    prints = sorted(exact_fingerprint(r.features) for r in records)
    return hashlib.sha256("\n".join(prints).encode("ascii")).hexdigest()


# -- 1. plan ----------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedCandidate:
    index: int
    candidate_id: str
    arm: str
    num_buckets: int
    epochs: int
    relative_step_c: float
    learning_rate: float
    lambda_method: str
    lambda_max: float
    L: float


class TrainingPlan:
    """A verified, deterministic plan for the 81 candidates. Holds training labels only; created only by :func:`build_training_plan`."""

    __slots__ = ("authorization", "candidates", "train_record_ids", "train_labels", "validation_ids", "validation_labels_sha256", "test_ids",
                 "train_config", "numerical", "split_result", "plan_sha256")

    def __init__(self, key: object, **fields: Any) -> None:
        if key is not _KEY:
            raise TrainingStageError("a TrainingPlan can only be produced by build_training_plan")
        for name in self.__slots__:
            object.__setattr__(self, name, fields[name])

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a TrainingPlan is immutable")

    def __repr__(self) -> str:
        return f"TrainingPlan(candidates={len(self.candidates)}, train_records={len(self.train_record_ids)}, plan={self.plan_sha256[:12]}...)"

    def __reduce__(self) -> Any:
        raise TypeError("a TrainingPlan is not serializable")

    def to_dict(self) -> dict[str, Any]:
        """Identities and the candidate schedule only (no label)."""
        return {"stage": TRAINING_STAGE_ID, "plan_sha256": self.plan_sha256, "manifest_sha256": self.authorization.manifest_sha256,
                "train_records": len(self.train_record_ids), "candidates": [vars_of(c) for c in self.candidates]}


def vars_of(candidate: PlannedCandidate) -> dict[str, Any]:
    return {"index": candidate.index, "candidate_id": candidate.candidate_id, "arm": candidate.arm, "num_buckets": candidate.num_buckets,
            "epochs": candidate.epochs, "relative_step_c": candidate.relative_step_c, "learning_rate": candidate.learning_rate,
            "learning_rate_hex": candidate.learning_rate.hex(), "lambda_method": candidate.lambda_method, "lambda_max": candidate.lambda_max, "L": candidate.L}


def _same_inputs(a: runner.ArmInputs, b: runner.ArmInputs) -> bool:
    return (a.split == b.split and tuple(a.record_ids) == tuple(b.record_ids) and a.max_length == b.max_length and a.arm_a == b.arm_a
            and a.arm_b == b.arm_b and a.arm_c == b.arm_c)


def build_training_plan(authorization: object, *, dataset: SyntheticDataset, split_result: SplitResult, partition: runner.Partition,
                        train_config: runner.TrainConfiguration, numerical: runner.NumericalConfiguration) -> TrainingPlan:
    """Verify the authorization against the prepared run and fix the deterministic 81-candidate schedule. Fits nothing."""
    try:
        auth = runner.require_training_authorization(authorization)
    except runner.TrainingAuthorizationError as exc:
        raise TrainingStageError(f"training requires a genuine, unmodified TrainingAuthorization ({exc})") from None
    if not (isinstance(dataset, SyntheticDataset) and isinstance(split_result, SplitResult) and isinstance(partition, runner.Partition)
            and isinstance(train_config, runner.TrainConfiguration) and isinstance(numerical, runner.NumericalConfiguration)):
        raise TrainingStageError("the dataset, split, partition, training and numerical configuration must be the runner's own objects")
    problems: list[str] = []

    # provenance of the split and the data
    if _manifest_hash(split_result) != split_result.manifest_sha256 or split_result.manifest_sha256 != auth.split_manifest_sha256:
        problems.append("split result does not hash to the authorized split manifest")
    if dataset.fingerprint != auth.generator_fingerprint or dataset.config.seed != auth.dataset_seed:
        problems.append("dataset is not the authorized generator output")
    sealed = partition.sealed_test
    all_records = [*partition.train.records, *partition.validation.records, *sealed.blinded_records()]
    if (_content_fingerprint(all_records) != split_result.report.dataset_fingerprint or split_result.report.dataset_fingerprint != auth.dataset_fingerprint
            or _content_fingerprint(dataset.records) != auth.dataset_fingerprint):
        problems.append("request content differs from the authorized dataset fingerprint")

    # membership, disjointness, group isolation
    train_ids, val_ids, test_ids = set(partition.train.record_ids), set(partition.validation.record_ids), set(sealed.record_ids)
    if (train_ids != set(split_result.members("train")) or val_ids != set(split_result.members("validation"))
            or test_ids != set(split_result.members("test"))):
        problems.append("partition membership differs from the split")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids or len(partition.train.record_ids) != len(train_ids):
        problems.append("a record occurs in more than one split or twice")
    groups = {name: {split_result.group_of.get(i) for i in ids} for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids))}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"] or None in groups["train"]:
        problems.append("a template group spans splits")
    if any(sealed.group_by_id.get(i) != split_result.group_of.get(i) for i in test_ids) or runner.ids_fingerprint(sorted(test_ids)) != auth.test_ids_sha256:
        problems.append("sealed test binding")

    # labels: training and validation labels must be the authorized dataset's labels for those ids (test labels are never looked up here)
    wanted = train_ids | val_ids
    source_labels = {r.record_id: r.meta.label for r in dataset.records if r.record_id in wanted}
    if (any(source_labels.get(r.record_id) != r.meta.label for r in (*partition.train.records, *partition.validation.records))
            or partition.summary["train"]["positives"] != sum(r.meta.label for r in partition.train.records)
            or partition.summary["validation"]["positives"] != sum(r.meta.label for r in partition.validation.records)):
        problems.append("training or validation labels differ from the authorized dataset")

    # training configuration: recomputed from the training records only
    recomputed = runner.prepare_train_configuration(partition.train)
    if (recomputed.to_dict() != train_config.to_dict() or not _same_inputs(recomputed.inputs, train_config.inputs) or train_config.inputs.split != "train"):
        problems.append("training configuration is not the one derived from the training records")
    if (tuple(train_config.inputs.record_ids) != tuple(sorted(train_ids)) or runner.ids_fingerprint(sorted(train_ids)) != auth.train_ids_sha256):
        problems.append("training record ids")
    if (train_config.frozen.record.get("sha256") != auth.vocabulary_sha256 or train_config.frozen.record.get("fitted_on_split") != "train"
            or train_config.decision.selected_max_length != auth.max_length or train_config.inputs.max_length != auth.max_length):
        problems.append("frozen vocabulary or token budget")

    # numerical configuration: recomputed and bound to the authorization
    re_numerical = runner.prepare_numerical_configuration(train_config)
    if re_numerical.to_dict() != numerical.to_dict():
        problems.append("numerical configuration is not the one derived from the training inputs")
    identity = runner._canonical_digest({"estimates": [runner._norm(e.to_dict()) for e in numerical.estimates],
                                         "learning_rates": runner._norm(dict(numerical.learning_rates))})
    if identity != auth.numerics_identity_sha256:
        problems.append("numerical configuration identity")
    ids = sorted(c.candidate_id for c in numerical.candidates)
    expected_ids = sorted(cid for arm in ARMS for cid in runner.approved_candidate_ids(arm))
    if ids != expected_ids or len(ids) != EXPECTED_CANDIDATES or runner._canonical_digest(ids) != auth.candidate_grid_sha256:
        problems.append("candidate grid")
    if problems:
        raise TrainingStageError("the prepared run does not match the authorization (" + "; ".join(problems) + ")")

    estimates = {(e.arm, e.num_buckets): e for e in numerical.estimates}
    planned = []
    for index, cand in enumerate(sorted(numerical.candidates, key=lambda c: c.candidate_id)):
        est = estimates[(cand.arm, cand.num_buckets)]
        if (cand.candidate_id != runner.canonical_candidate_id(cand.arm, cand.num_buckets, cand.epochs, cand.relative_step_c)
                or cand.learning_rate != numerics.relative_learning_rate(cand.relative_step_c, est.L) or est.split != "train" or not est.converged):
            raise TrainingStageError(f"candidate {index} does not carry the authorized train-only learning rate")
        planned.append(PlannedCandidate(index, cand.candidate_id, cand.arm, cand.num_buckets, cand.epochs, cand.relative_step_c, cand.learning_rate,
                                        est.method, est.lambda_max, est.L))
    counts = {arm: sum(1 for c in planned if c.arm == arm) for arm in ARMS}
    if counts != {arm: 27 for arm in ARMS}:
        raise TrainingStageError("the plan must hold exactly 27 candidates per arm")
    try:
        labels = runner.aligned_labels(train_config.inputs.record_ids, partition.train.labels_by_id())
    except runner.RunnerError:
        raise TrainingStageError("training labels do not align with the training inputs") from None
    if set(labels) != {0, 1}:
        raise TrainingStageError("the training split needs both classes")
    validation_labels_sha256 = _digest(sorted(partition.validation.labels_by_id().items()))
    plan_sha = _digest({"stage": TRAINING_STAGE_ID, "authorization_seal": auth._seal, "candidates": [vars_of(c) for c in planned],
                        "train_ids_sha256": auth.train_ids_sha256, "validation_ids_sha256": runner.ids_fingerprint(sorted(val_ids)),
                        "test_ids_sha256": auth.test_ids_sha256})
    return TrainingPlan(_KEY, authorization=auth, candidates=tuple(planned), train_record_ids=tuple(train_config.inputs.record_ids), train_labels=labels,
                        validation_ids=tuple(sorted(val_ids)), validation_labels_sha256=validation_labels_sha256, test_ids=tuple(sorted(test_ids)), train_config=train_config, numerical=numerical,
                        split_result=split_result, plan_sha256=plan_sha)


def check_live_repository(authorization: runner.TrainingAuthorization, repo_root: str | os.PathLike[str]) -> None:
    """Re-check, at execution time, that the repository is still the authorized one: clean tree, authorized branch and commit, unchanged required sources."""
    try:
        state = mf.collect_git_state(repo_root)
        live_sources = mf.hash_files(repo_root, tuple(mf.REQUIRED_SOURCE_FILES))
    except mf.ManifestError:
        raise TrainingStageError("the repository state could not be read") from None
    if not state["clean"] or state["commit"] != authorization.git_commit or state["branch"] != authorization.git_branch:
        raise TrainingStageError("the repository is no longer the authorized clean commit")
    if runner._canonical_digest(live_sources) != authorization.sources_sha256:
        raise TrainingStageError("a required source differs from the authorized source table")


def _require_plan(plan: object) -> TrainingPlan:
    if type(plan) is not TrainingPlan:
        raise TrainingStageError("a TrainingPlan from build_training_plan is required")
    try:
        runner.require_training_authorization(plan.authorization)
    except runner.TrainingAuthorizationError:
        raise TrainingStageError("the plan's authorization is no longer intact") from None
    return plan


# -- 2. training ------------------------------------------------------------------------------------------------------------------------------


class TrainedCandidates:
    """The 81 persisted candidates of one plan: artifact directory, digests and weight hashes. Created only by :func:`train_candidates`."""

    __slots__ = ("plan_sha256", "manifest_sha256", "artifact_dir", "digests", "weights_sha256", "ledger_path", "ledger_sha256")

    def __init__(self, key: object, **fields: Any) -> None:
        if key is not _KEY:
            raise TrainingStageError("TrainedCandidates can only be produced by train_candidates")
        for name in self.__slots__:
            object.__setattr__(self, name, fields[name])

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TrainedCandidates are immutable")

    def __reduce__(self) -> Any:
        raise TypeError("TrainedCandidates are not serializable")

    def __repr__(self) -> str:
        return f"TrainedCandidates(candidates={len(self.digests)}, plan={self.plan_sha256[:12]}...)"


def ledger_filename(manifest_sha256: str) -> str:
    return f"training-ledger-{manifest_sha256}.json"


def _training_provenance(plan: TrainingPlan, cand: PlannedCandidate) -> dict[str, Any]:
    auth = plan.authorization
    is_c = cand.arm == ARM_C
    return {"train_ids_sha256": auth.train_ids_sha256, "vocabulary_sha256": None if is_c else auth.vocabulary_sha256,
            "max_length": None if is_c else auth.max_length, "lambda_method": cand.lambda_method, "lambda_max": cand.lambda_max, "L": cand.L}


def _write_once(path: Path, data: bytes) -> None:
    try:
        with open(path, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise TrainingStageError("the training ledger already exists; it is never overwritten") from None
    except OSError:
        raise TrainingStageError("the training ledger could not be written; any partial file is left for review") from None
    if path.read_bytes() != data:
        raise TrainingStageError("the training ledger read back differently; it is left for review")
    try:
        os.chmod(path, stat.S_IREAD)
    except OSError:
        pass


def train_candidates(plan: object, *, artifact_dir: str | os.PathLike[str], repo_root: str | os.PathLike[str]) -> TrainedCandidates:
    """Fit, persist and re-verify all 81 candidates on the training split only. Stops at the first failure; never retries or overwrites."""
    plan = _require_plan(plan)
    auth = plan.authorization
    check_live_repository(auth, repo_root)
    directory = Path(artifact_dir)
    if not directory.is_dir():
        raise TrainingStageError("the artifact directory does not exist")
    existing = [name for name in (*ma.expected_artifact_names(auth.manifest_sha256).values(), ledger_filename(auth.manifest_sha256))
                if os.path.lexists(directory / name)]
    if existing:
        raise TrainingStageError(f"{len(existing)} artifact(s) for this manifest already exist; there is no silent retry or overwrite")
    labels = list(plan.train_labels)
    digests: dict[str, str] = {}
    weights: dict[str, str] = {}
    for cand in plan.candidates:
        inputs = list(getattr(plan.train_config.inputs, _INPUT_ATTR[cand.arm]))
        model = new_untrained_model(cand.num_buckets)  # a fresh instance: fit warm-starts from existing weights
        try:
            model.fit(inputs, labels, epochs=cand.epochs, learning_rate=cand.learning_rate)
        except (ValueError, FloatingPointError, MemoryError) as exc:
            raise TrainingStageError(f"candidate {cand.index} could not be fitted ({type(exc).__name__})") from None
        if not model.is_trained or not np.isfinite(model.weights).all() or not math.isfinite(float(model.bias)):
            raise TrainingStageError(f"candidate {cand.index} produced non-finite parameters; the stage stops")
        try:
            written = ma.write_model_artifact(directory, model, repo_root=repo_root, arm=cand.arm, num_buckets=cand.num_buckets, epochs=cand.epochs,
                                              relative_step_c=cand.relative_step_c, learning_rate=cand.learning_rate,
                                              manifest_sha256=auth.manifest_sha256, training=_training_provenance(plan, cand))
            loaded = ma.load_model_artifact(directory, expected_sha256=written.sha256, expected_arm=cand.arm, expected_candidate_id=cand.candidate_id,
                                            expected_manifest_sha256=auth.manifest_sha256, expected_learning_rate=cand.learning_rate,
                                            expected_model_sha256=auth.model_source_sha256, expected_fingerprint=sr.model_artifact_fingerprint(model))
        except mf.ManifestError as exc:
            raise TrainingStageError(f"candidate {cand.index} could not be persisted and re-verified ({type(exc).__name__}); nothing is retried") from None
        if (loaded.model.weights.tobytes() != np.ascontiguousarray(model.weights, dtype="<f8").tobytes() or loaded.model.bias != model.bias
                or numerics.raw_logits(loaded.model, inputs[:_READOUT_CHECK]) != numerics.raw_logits(model, inputs[:_READOUT_CHECK])):
            raise TrainingStageError(f"candidate {cand.index} does not reload bit-exactly")
        digests[cand.candidate_id] = written.sha256
        weights[cand.candidate_id] = loaded.fingerprint["weights_sha256"]
    if len(digests) != EXPECTED_CANDIDATES or len(set(digests.values())) != EXPECTED_CANDIDATES:
        raise TrainingStageError("the stage did not produce 81 distinct artifacts")
    ledger = {"ledger_version": LEDGER_VERSION, "stage": TRAINING_STAGE_ID, "manifest_sha256": auth.manifest_sha256, "plan_sha256": plan.plan_sha256,
              "authorization_seal": auth._seal, "candidates": [{"candidate_id": c.candidate_id, "artifact_sha256": digests[c.candidate_id],
                                                                 "weights_sha256": weights[c.candidate_id]} for c in plan.candidates],
              "training_data": "training split only", "labels_recorded": False}
    data = mf.canonical_bytes(ledger)
    ledger_path = directory / ledger_filename(auth.manifest_sha256)
    _write_once(ledger_path, data)
    return TrainedCandidates(_KEY, plan_sha256=plan.plan_sha256, manifest_sha256=auth.manifest_sha256, artifact_dir=str(directory),
                             digests=MappingProxyType(dict(digests)), weights_sha256=MappingProxyType(dict(weights)), ledger_path=str(ledger_path),
                             ledger_sha256=mf.sha256_hex(data))


# -- 3. validation scoring and selection -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationSelectionOutcome:
    record_path: str
    record_sha256: str
    selected_candidates: Mapping[str, str]
    candidates_scored: int
    verification: sr.SelectionVerification


def select_on_validation(plan: object, trained: object, *, validation: runner.ValidationData, manifest_bytes: bytes, expected_manifest_sha256: str,
                         repo_root: str | os.PathLike[str], expected_run: mf.ExpectedRun, record_path: str | os.PathLike[str],
                         created_utc: str) -> ValidationSelectionOutcome:
    """Score all 81 persisted candidates on the validation split, write the selection record through the real writer, and verify it again."""
    plan = _require_plan(plan)
    if type(trained) is not TrainedCandidates or trained.plan_sha256 != plan.plan_sha256:
        raise TrainingStageError("TrainedCandidates from train_candidates for this plan are required")
    auth = plan.authorization
    if expected_manifest_sha256 != auth.manifest_sha256 or not isinstance(manifest_bytes, (bytes, bytearray)) \
            or mf.sha256_hex(bytes(manifest_bytes)) != auth.manifest_sha256:
        raise TrainingStageError("the manifest is not the authorized manifest")
    if (not isinstance(validation, runner.ValidationData) or tuple(validation.record_ids) != plan.validation_ids
            or _digest(sorted(validation.labels_by_id().items())) != plan.validation_labels_sha256):
        raise TrainingStageError("the validation data is not the plan's validation split")
    ledger = Path(trained.ledger_path)
    if not ledger.is_file() or mf.sha256_hex(ledger.read_bytes()) != trained.ledger_sha256:
        raise TrainingStageError("the training ledger is missing or changed")
    val_inputs = runner.encode_validation(plan.train_config, validation)
    labels = list(runner.aligned_labels(val_inputs.record_ids, validation.labels_by_id()))
    try:
        loaded = ma.load_candidate_set(trained.artifact_dir, expected_manifest_sha256=auth.manifest_sha256, expected_digests=dict(trained.digests),
                                       expected_learning_rates={c.candidate_id: c.learning_rate for c in plan.candidates},
                                       expected_model_sha256=auth.model_source_sha256)
    except mf.ManifestError as exc:
        raise TrainingStageError(f"the persisted candidates did not verify ({type(exc).__name__})") from None
    ids_fp = runner.ids_fingerprint(list(val_inputs.record_ids))
    results: dict[str, list[runner.ValidationCandidateResult]] = {arm: [] for arm in ARMS}
    for cand in plan.candidates:
        logits = numerics.raw_logits(loaded[cand.candidate_id].model, list(getattr(val_inputs, _INPUT_ATTR[cand.arm])))
        ap = average_precision(logits, labels)
        if ap is None:
            raise TrainingStageError("validation AP is undefined (no positive validation record)")
        results[cand.arm].append(runner.ValidationCandidateResult(cand.candidate_id, cand.arm, cand.num_buckets, cand.epochs, cand.relative_step_c, ap,
                                                                  "validation", ids_fp, len(labels)))
    evidence = sr.ValidationEvidence(arm_inputs=val_inputs, labels_by_id=validation.labels_by_id(), frozen=plan.train_config.frozen,
                                     split_result=plan.split_result, artifact_dir=trained.artifact_dir, artifact_digests=dict(trained.digests))
    try:
        record = sr.build_selection_record(manifest_bytes=bytes(manifest_bytes), expected_manifest_sha256=auth.manifest_sha256, results_by_arm=results,
                                           evidence=evidence, created_utc=created_utc)
        written = sr.write_selection_record(record_path, record, manifest_bytes=bytes(manifest_bytes), expected_manifest_sha256=auth.manifest_sha256,
                                            repo_root=repo_root, expected_run=expected_run, evidence=evidence)
        verification = sr.verify_selection_record(written.path, expected_sha256=written.sha256, manifest_bytes=bytes(manifest_bytes),
                                                  expected_manifest_sha256=auth.manifest_sha256, repo_root=repo_root, expected_run=expected_run,
                                                  evidence=evidence)
    except mf.ManifestError as exc:
        raise TrainingStageError(f"the validation-selection record could not be written and verified ({type(exc).__name__})") from None
    return ValidationSelectionOutcome(record_path=written.path, record_sha256=written.sha256,
                                      selected_candidates=MappingProxyType(dict(verification.selected_candidates)),
                                      candidates_scored=sum(len(v) for v in results.values()), verification=verification)


def source_sha256() -> str:
    return mf.file_sha256(Path(__file__))
