"""``test-attempt-gate-1``: the fail-closed, atomic, one-time test attempt gate (draft protocol heldout-eval-1.2.0-draft, sections 12.11.4 and 12.12.4).

Infrastructure only. **Nothing here scores a test set, trains a model or touches confirmatory data.** The gate is a strictly ordered, guarded primitive
that a future runner may call; the runner's own training and test gates (``runner.request_training_authorization`` / ``request_test_access``) stay blocked
and are not wired to this module. Because the protocol is a DRAFT, a real confirmatory manifest cannot verify, so the gate cannot open for a real run.

Order of operations (each step must succeed before the next)
--------------------------------------------------------------
1. :func:`verify_attempt_inputs`: the **real** manifest verifier (external digest, FROZEN and externally approved protocol, clean tree, commit, source and
   config hashes, dataset/split provenance, environment) and the **real** validation-selection verifier (external digest, binding to that exact manifest,
   recomputation from trusted validation evidence). The expected run must be ``confirmatory``. There is no flag, callback or "already verified" object a
   caller can pass; the :class:`VerifiedAttempt` it returns can only be made by this function. The sealed test set must match the manifest (id hash, count, group
   count, split hash) and the supplied split result.
2. :func:`create_attempt_marker`: creates the one-time marker with ``O_CREAT | O_EXCL`` in a **trusted attempt directory outside the repository**, flushes and
   fsyncs, reads the bytes back and compares. The path is derived only from the experiment identity of the verified manifest, never from record metadata. An
   existing marker, even a partial or malformed one, blocks the attempt. Markers are never overwritten, reset or deleted by code.
3. :func:`recover_test_labels`: only now is the caller-supplied ``label_provider`` invoked. The authentic labels are recovered from the original source
   records and validated (exact id alignment, request-content equality with the sealed set, split and group binding, binary labels; the blinded placeholder
   ``-1`` is refused). The labels are returned in a redacted holder; nothing is scored.
:func:`open_test_attempt` runs 1-3 and, for any failure after the marker exists, writes a versioned failure report that holds only stage, category and the
attempt binding. A failure never retries, regenerates, re-seeds or unlocks anything; recovery needs a separate explicit methodological review and a future
authorization mechanism that does not exist.

Trust boundaries: the gate cannot know what unrelated code already read (the caller owns the source records); filesystem atomicity is that of
``O_EXCL`` on a local filesystem (not guaranteed on some network filesystems); in-process code can bypass any in-process check, so the guards prevent
accidents and make misuse visible, not malicious control of the process.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import manifest as mf
from . import runner
from . import selection_record as sr
from .splits import EvalRecord, SplitResult

GATE_VERSION = "test-attempt-gate-1"
MARKER_VERSION = "test-attempt-marker-1"
FAILURE_REPORT_VERSION = "test-attempt-failure-report-1"
LIFECYCLE_STARTED = "attempt_started"
MAX_MARKER_BYTES = 4096
MAX_FAILURE_REPORTS = 99
BLINDED_LABEL = -1  # the runner's sentinel for blinded records; never ground truth
_HEX64 = re.compile(r"[0-9a-f]{64}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_KEY = object()  # only this module creates VerifiedAttempt / MarkerReceipt objects
_MARKER_KEYS = {"marker_version", "gate_version", "lifecycle_state", "experiment_id", "manifest_sha256", "selection_record_sha256", "dataset_seed",
                "dataset_fingerprint", "generator_fingerprint", "split_manifest_sha256", "test_ids_sha256", "test_records", "created_utc"}
STAGES = ("verification", "marker_creation", "marker_persistence", "label_recovery")
CATEGORIES = ("verification_failed", "marker_exists", "marker_creation_failed", "marker_persistence_failed", "marker_unverifiable", "binding_mismatch",
              "provider_failed", "records_misaligned", "placeholder_label", "invalid_label", "content_mismatch", "group_mismatch", "unexpected_error")


class AttemptGateError(mf.ManifestError):
    """Base class. Messages hold stages, categories and counts only: never labels, scores, request content or exception text."""


class GateVerificationError(AttemptGateError):
    """The manifest, the selection record, the expected run or the sealed set did not verify. No marker was created."""


class AttemptAlreadyConsumed(AttemptGateError):
    """A marker (valid, partial or malformed) already exists for this experiment. The attempt is spent; no automatic retry exists."""


class MarkerCreationError(AttemptGateError):
    """The marker could not be created (nothing exists on disk)."""


class MarkerPersistenceError(AttemptGateError):
    """The marker exists but its persistence could not be confirmed. It is left in place and counts as a consumed attempt requiring review."""


class LabelRecoveryError(AttemptGateError):
    """The authentic labels could not be recovered and validated. ``category`` is one of ``CATEGORIES``."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class PostMarkerFailure(AttemptGateError):
    """A failure after the marker exists. The marker is intact; ``report_path`` names the failure report (or ``None`` if it could not be written)."""

    def __init__(self, category: str, stage: str, report_path: str | None) -> None:
        super().__init__(f"the attempt failed after the marker was created (stage {stage}, category {category}); "
                         "the marker stays in place and manual methodological review is required")
        self.category, self.stage, self.report_path = category, stage, report_path


# -- identity, markers -------------------------------------------------------------------------------------------------------------------


def experiment_identity(manifest: dict[str, Any]) -> str:
    """SHA-256 of the dataset and split identity of a verified manifest. The marker path depends on this (and nothing a record can choose)."""
    data = manifest["data"]
    identity = {"dataset_seed": data["dataset_seed"], "generator_fingerprint": data["generator"]["fingerprint"],
                "dataset_fingerprint": data["dataset_fingerprint"], "split_seed": data["split"]["seed"],
                "split_manifest_sha256": data["split"]["manifest_sha256"]}
    return mf.sha256_hex(mf.canonical_bytes(identity))


class VerifiedAttempt:
    """Evidence that :func:`verify_attempt_inputs` succeeded. Constructible only with this module's private key."""

    __slots__ = ("experiment_id", "manifest_sha256", "selection_record_sha256", "dataset_seed", "dataset_fingerprint", "generator_fingerprint",
                 "split_manifest_sha256", "test_ids_sha256", "test_records")

    def __init__(self, key: object, **fields: Any) -> None:
        if key is not _KEY:
            raise GateVerificationError("a VerifiedAttempt can only be produced by the gate's verification")
        for name in self.__slots__:
            setattr(self, name, fields[name])

    def __repr__(self) -> str:
        return f"VerifiedAttempt(experiment_id={self.experiment_id[:12]}...)"


class MarkerReceipt:
    """Proof that this process created and read back the marker. Constructible only with this module's private key."""

    __slots__ = ("verified", "path", "marker_bytes", "marker_sha256")

    def __init__(self, key: object, verified: VerifiedAttempt, path: str, marker_bytes: bytes) -> None:
        if key is not _KEY:
            raise AttemptGateError("a MarkerReceipt can only be produced by marker creation")
        self.verified, self.path, self.marker_bytes, self.marker_sha256 = verified, path, marker_bytes, mf.sha256_hex(marker_bytes)

    def __repr__(self) -> str:
        return f"MarkerReceipt(experiment_id={self.verified.experiment_id[:12]}...)"


def marker_path(attempt_root: str | os.PathLike[str], experiment_id: str) -> Path:
    if not isinstance(experiment_id, str) or not _HEX64.fullmatch(experiment_id):
        raise AttemptGateError("the experiment identity must be a SHA-256 hex string")
    return Path(attempt_root) / f"test-attempt-{experiment_id}.json"


def _check_attempt_root(attempt_root: str | os.PathLike[str], repo_root: str | os.PathLike[str] | None) -> Path:
    root = Path(attempt_root)
    if not root.is_dir():
        raise MarkerCreationError("the attempt directory does not exist")
    resolved = root.resolve()
    if repo_root is not None:
        repo = Path(repo_root).resolve()
        if resolved == repo or repo in resolved.parents:
            raise MarkerCreationError("the attempt directory must be outside the repository")
    return resolved


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _marker_content(verified: VerifiedAttempt, created_utc: str) -> dict[str, Any]:
    return {"marker_version": MARKER_VERSION, "gate_version": GATE_VERSION, "lifecycle_state": LIFECYCLE_STARTED,
            "experiment_id": verified.experiment_id, "manifest_sha256": verified.manifest_sha256,
            "selection_record_sha256": verified.selection_record_sha256, "dataset_seed": verified.dataset_seed,
            "dataset_fingerprint": verified.dataset_fingerprint, "generator_fingerprint": verified.generator_fingerprint,
            "split_manifest_sha256": verified.split_manifest_sha256, "test_ids_sha256": verified.test_ids_sha256,
            "test_records": verified.test_records, "created_utc": created_utc}


def parse_marker_bytes(data: bytes) -> dict[str, Any]:
    """Strict parse of marker bytes (canonical JSON, exact field set, bounded size). Raises :class:`AttemptGateError` on anything else."""
    if not isinstance(data, (bytes, bytearray)) or not data or len(data) > MAX_MARKER_BYTES:
        raise AttemptGateError("the marker is empty or too large")
    try:
        marker = mf.parse_manifest_bytes(bytes(data))
    except mf.ManifestError:
        raise AttemptGateError("the marker is partial, malformed or not canonical") from None
    ok = (set(marker) == _MARKER_KEYS and marker["marker_version"] == MARKER_VERSION and marker["lifecycle_state"] == LIFECYCLE_STARTED
          and all(isinstance(marker[k], str) and _HEX64.fullmatch(marker[k]) for k in ("experiment_id", "manifest_sha256", "selection_record_sha256",
                                                                                      "dataset_fingerprint", "generator_fingerprint",
                                                                                      "split_manifest_sha256", "test_ids_sha256"))
          and isinstance(marker["created_utc"], str) and _UTC.fullmatch(marker["created_utc"])
          and isinstance(marker["dataset_seed"], int) and isinstance(marker["test_records"], int))
    if not ok:
        raise AttemptGateError("the marker does not match the marker schema")
    return marker


@dataclass(frozen=True)
class AttemptStatus:
    state: str  # "absent", "consumed" (a valid marker) or "invalid_marker" (partial, malformed or substituted: also a consumed attempt)
    path: str
    marker: dict[str, Any] | None = None

    @property
    def blocks_new_attempt(self) -> bool:
        return self.state != "absent"


def inspect_attempt(attempt_root: str | os.PathLike[str], experiment_id: str) -> AttemptStatus:
    """What is on disk for this experiment. Any file at the marker path blocks a new attempt, valid or not. Reports never unlock anything."""
    path = marker_path(attempt_root, experiment_id)
    if not os.path.lexists(path):
        return AttemptStatus("absent", str(path))
    try:
        marker = parse_marker_bytes(path.read_bytes())
        if marker["experiment_id"] != experiment_id:
            raise AttemptGateError("substituted marker")
    except (AttemptGateError, OSError):
        return AttemptStatus("invalid_marker", str(path))
    return AttemptStatus("consumed", str(path), marker)


def _read_back(path: Path) -> bytes:
    return path.read_bytes()


def _write_exclusive(path: Path, data: bytes) -> None:
    """Create ``path`` atomically and exclusively, write, fsync and read back. A file that exists afterwards is never removed."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        raise AttemptAlreadyConsumed("a marker already exists for this experiment; the one-time attempt is spent") from None
    except OSError:
        raise MarkerCreationError("the marker could not be created; nothing was written") from None
    try:
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if _read_back(path) != data:
            raise MarkerPersistenceError("the marker read back differently from what was written; it is left in place for review")
        if os.name == "posix":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        try:
            os.chmod(path, stat.S_IREAD)
        except OSError:
            pass
    except MarkerPersistenceError:
        raise
    except OSError:
        raise MarkerPersistenceError("persistence of the marker could not be confirmed; it is left in place for review") from None


def create_attempt_marker(attempt_root: str | os.PathLike[str], verified: VerifiedAttempt, *, repo_root: str | os.PathLike[str] | None = None,
                          created_utc: str | None = None) -> MarkerReceipt:
    """Atomically create the one-time marker for a verified experiment. Fails if any file already exists at the marker path."""
    if not isinstance(verified, VerifiedAttempt):
        raise GateVerificationError("a VerifiedAttempt from verify_attempt_inputs is required")
    stamp = _utc_now() if created_utc is None else created_utc
    if not isinstance(stamp, str) or not _UTC.fullmatch(stamp):
        raise MarkerCreationError("created_utc must look like 2026-01-01T00:00:00Z")
    root = _check_attempt_root(attempt_root, repo_root)
    path = marker_path(root, verified.experiment_id)
    data = mf.canonical_bytes(_marker_content(verified, stamp))
    _write_exclusive(path, data)
    return MarkerReceipt(_KEY, verified, str(path), data)


# -- verification ---------------------------------------------------------------------------------------------------------------------------------


def verify_attempt_inputs(*, manifest_bytes: bytes, expected_manifest_sha256: str, record_bytes: bytes, expected_record_sha256: str,
                          repo_root: str | os.PathLike[str], expected_run: mf.ExpectedRun, evidence: sr.ValidationEvidence,
                          sealed_test: runner.SealedTestSet, **manifest_kwargs: Any) -> VerifiedAttempt:
    """Run the real manifest and selection-record verifiers and the sealed-set bindings. Raises :class:`GateVerificationError`; creates nothing."""
    if not isinstance(expected_run, mf.ExpectedRun) or expected_run.purpose != mf.PURPOSE_CONFIRMATORY:
        raise GateVerificationError("the gate only opens for an expected confirmatory run")
    if not isinstance(manifest_bytes, (bytes, bytearray)) or not isinstance(record_bytes, (bytes, bytearray)):
        raise GateVerificationError("the manifest and the selection record must be supplied as their exact bytes")
    if not isinstance(sealed_test, runner.SealedTestSet) or not isinstance(evidence, sr.ValidationEvidence):
        raise GateVerificationError("a SealedTestSet and ValidationEvidence are required")
    try:
        mf.verify_manifest_bytes(bytes(manifest_bytes), expected_sha256=expected_manifest_sha256, repo_root=repo_root, expected=expected_run,
                                 **manifest_kwargs)
    except mf.ManifestError as exc:
        raise GateVerificationError(f"the pre-run manifest did not verify ({type(exc).__name__})") from None
    try:
        verification = sr.verify_selection_record_bytes(bytes(record_bytes), expected_sha256=expected_record_sha256, manifest_bytes=bytes(manifest_bytes),
                                                        expected_manifest_sha256=expected_manifest_sha256, repo_root=repo_root,
                                                        expected_run=expected_run, evidence=evidence, **manifest_kwargs)
    except mf.ManifestError as exc:
        raise GateVerificationError(f"the validation-selection record did not verify ({type(exc).__name__})") from None
    if not verification.ok or verification.manifest_sha256 != expected_manifest_sha256 or verification.authorizes_test_scoring is not False:
        raise GateVerificationError("the validation-selection verification result is not acceptable")
    manifest = mf.parse_manifest_bytes(bytes(manifest_bytes))
    record = mf.parse_manifest_bytes(bytes(record_bytes))
    if record["test_status"] != {"test_scored": False, "test_labels_accessed": False, "test_scores_accessed": False, "test_data_in_record": False}:
        raise GateVerificationError("the selection record reports prior test access")
    _check_sealed(manifest, evidence.split_result, sealed_test)
    return VerifiedAttempt(_KEY, experiment_id=experiment_identity(manifest), manifest_sha256=expected_manifest_sha256,
                           selection_record_sha256=expected_record_sha256, dataset_seed=manifest["data"]["dataset_seed"],
                           dataset_fingerprint=manifest["data"]["dataset_fingerprint"], generator_fingerprint=manifest["data"]["generator"]["fingerprint"],
                           split_manifest_sha256=manifest["data"]["split"]["manifest_sha256"],
                           test_ids_sha256=manifest["arms"]["alignment"]["test_ids_sha256"], test_records=manifest["arms"]["alignment"]["test_records"])


def _check_sealed(manifest: dict[str, Any], split_result: SplitResult, sealed: runner.SealedTestSet) -> None:
    binding = sealed.binding()
    alignment, counts = manifest["arms"]["alignment"], manifest["data"]["counts"]["test"]
    ids = set(sealed.record_ids)
    groups = sealed.group_by_id
    problems = []
    if binding["ids_sha256"] != alignment["test_ids_sha256"] or binding["records"] != alignment["test_records"] or binding["records"] != counts["records"]:
        problems.append("test record ids or count")
    if len(set(groups.values())) != counts["groups"]:
        problems.append("test group count")
    if binding["split_manifest_sha256"] != manifest["data"]["split"]["manifest_sha256"]:
        problems.append("split manifest hash")
    if not isinstance(split_result, SplitResult) or split_result.manifest_sha256 != manifest["data"]["split"]["manifest_sha256"] \
            or set(split_result.members("test")) != ids or any(split_result.group_of.get(i) != groups.get(i) for i in ids):
        problems.append("split result membership or groups")
    elif binding["groups_sha256"] != runner.ids_fingerprint(sorted({split_result.group_of[i] for i in ids})):
        problems.append("group fingerprint")
    if problems:
        raise GateVerificationError(f"the sealed test set does not match the verified manifest ({', '.join(problems)})")


# -- label recovery ---------------------------------------------------------------------------------------------------------------------------------


class RecoveredTestLabels:
    """Authentic test labels in the sealed set's sorted id order. Redacted in repr, not serializable, no mapping or dict export."""

    __slots__ = ("record_ids", "_labels")

    def __init__(self, key: object, record_ids: tuple[str, ...], labels: tuple[int, ...]) -> None:
        if key is not _KEY:
            raise AttemptGateError("recovered labels can only be produced by label recovery")
        self.record_ids, self._labels = record_ids, labels

    def labels_in_order(self) -> tuple[int, ...]:
        return self._labels

    def __repr__(self) -> str:
        return f"RecoveredTestLabels(records={len(self.record_ids)}, labels=<withheld>)"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("recovered test labels are not serializable")


def recover_test_labels(receipt: MarkerReceipt, sealed_test: runner.SealedTestSet, split_result: SplitResult,
                        label_provider: Callable[[], Iterable[EvalRecord]]) -> RecoveredTestLabels:
    """Invoke ``label_provider`` (only now) and validate the authentic labels it returns. Requires a receipt for an existing, intact marker."""
    if not isinstance(receipt, MarkerReceipt):
        raise LabelRecoveryError("marker_unverifiable", "a marker receipt is required before test labels can be recovered")
    try:
        on_disk = Path(receipt.path).read_bytes()
    except OSError:
        raise LabelRecoveryError("marker_unverifiable", "the attempt marker could not be read back") from None
    if on_disk != receipt.marker_bytes:
        raise LabelRecoveryError("marker_unverifiable", "the attempt marker on disk differs from the created marker")
    verified = receipt.verified
    binding = sealed_test.binding() if isinstance(sealed_test, runner.SealedTestSet) else {}
    if (binding.get("ids_sha256") != verified.test_ids_sha256 or binding.get("split_manifest_sha256") != verified.split_manifest_sha256
            or not isinstance(split_result, SplitResult) or split_result.manifest_sha256 != verified.split_manifest_sha256
            or set(split_result.members("test")) != set(sealed_test.record_ids)):
        raise LabelRecoveryError("binding_mismatch", "the sealed set or split result does not match the verified attempt")
    try:
        rows = list(label_provider())
    except Exception as exc:  # noqa: BLE001 - only the exception class name is reported, never its text
        raise LabelRecoveryError("provider_failed", f"the label provider failed ({type(exc).__name__})") from None
    if not all(isinstance(r, EvalRecord) for r in rows):
        raise LabelRecoveryError("records_misaligned", "the label provider must return EvalRecord objects")
    ids = [r.record_id for r in rows]
    wanted = set(sealed_test.record_ids)
    if len(set(ids)) != len(ids):
        raise LabelRecoveryError("records_misaligned", "the label source holds duplicate record ids")
    if set(ids) != wanted:
        raise LabelRecoveryError("records_misaligned", f"the label source does not match the sealed ids ({len(wanted - set(ids))} missing, "
                                                       f"{len(set(ids) - wanted)} extra)")
    if any(r.meta.label == BLINDED_LABEL for r in rows):
        raise LabelRecoveryError("placeholder_label", "the label source holds the blinded placeholder, which is not ground truth")
    if any(isinstance(r.meta.label, bool) or r.meta.label not in (0, 1) for r in rows):
        raise LabelRecoveryError("invalid_label", "a recovered label is not the integer 0 or 1")
    blinded = {r.record_id: r for r in sealed_test.blinded_records()}
    groups = sealed_test.group_by_id
    by_id = {r.record_id: r for r in rows}
    if any(by_id[i].features != blinded[i].features for i in wanted):
        raise LabelRecoveryError("content_mismatch", "a source record's request content differs from the sealed record")
    if any(split_result.group_of.get(i) != groups[i] or split_result.assignments.get(i) != "test" for i in wanted):
        raise LabelRecoveryError("group_mismatch", "a source record's split or group differs from the sealed set")
    order = tuple(sorted(wanted))
    return RecoveredTestLabels(_KEY, order, tuple(int(by_id[i].meta.label) for i in order))


# -- failure reports ----------------------------------------------------------------------------------------------------------------------------------


def build_failure_report(*, verified: VerifiedAttempt, stage: str, category: str, created_utc: str | None = None) -> dict[str, Any]:
    """The versioned, redacted failure report: stage, category and attempt binding only. It authorizes nothing."""
    if stage not in STAGES or category not in CATEGORIES:
        raise AttemptGateError("unknown failure stage or category")
    return {
        "report_version": FAILURE_REPORT_VERSION, "gate_version": GATE_VERSION, "experiment_id": verified.experiment_id,
        "manifest_sha256": verified.manifest_sha256, "selection_record_sha256": verified.selection_record_sha256, "stage": stage, "category": category,
        "marker_present": True, "retry_authorized": False, "automatic_retry": False,
        "required_next_step": "separate explicit methodological review and a future authorization mechanism (none is implemented)",
        "created_utc": _utc_now() if created_utc is None else created_utc,
    }


def write_failure_report(attempt_root: str | os.PathLike[str], report: dict[str, Any]) -> str | None:
    """Write the report exclusively as ``test-attempt-<id>.failure-<n>.json``; returns its path or ``None`` if it could not be written."""
    try:
        data = mf.canonical_bytes(report)
        for n in range(1, MAX_FAILURE_REPORTS + 1):
            path = Path(attempt_root) / f"test-attempt-{report['experiment_id']}.failure-{n}.json"
            try:
                with open(path, "xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                return str(path)
            except FileExistsError:
                continue
    except (OSError, KeyError, mf.ManifestError):
        return None
    return None


# -- orchestration ---------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptOutcome:
    """Result of a successful gate run: the redacted labels holder and the marker identity. The test set has NOT been scored."""

    labels: RecoveredTestLabels
    experiment_id: str
    marker_path: str
    marker_sha256: str
    test_scored: bool = False


def open_test_attempt(*, attempt_root: str | os.PathLike[str], manifest_bytes: bytes, expected_manifest_sha256: str, record_bytes: bytes,
                      expected_record_sha256: str, repo_root: str | os.PathLike[str], expected_run: mf.ExpectedRun, evidence: sr.ValidationEvidence,
                      sealed_test: runner.SealedTestSet, label_provider: Callable[[], Iterable[EvalRecord]], created_utc: str | None = None,
                      **manifest_kwargs: Any) -> AttemptOutcome:
    """Verify both records, atomically create the one-time marker, and only then recover the authentic test labels. Scores nothing."""
    verified = verify_attempt_inputs(manifest_bytes=manifest_bytes, expected_manifest_sha256=expected_manifest_sha256, record_bytes=record_bytes,
                                     expected_record_sha256=expected_record_sha256, repo_root=repo_root, expected_run=expected_run, evidence=evidence,
                                     sealed_test=sealed_test, **manifest_kwargs)
    try:
        receipt = create_attempt_marker(attempt_root, verified, repo_root=repo_root, created_utc=created_utc)
    except MarkerPersistenceError:
        _report(attempt_root, verified, "marker_persistence", "marker_persistence_failed")
        raise
    try:
        labels = recover_test_labels(receipt, sealed_test, evidence.split_result, label_provider)
    except LabelRecoveryError as exc:
        raise PostMarkerFailure(exc.category, "label_recovery", _report(attempt_root, verified, "label_recovery", exc.category)) from None
    except Exception:  # noqa: BLE001 - any other failure after the marker is reported by category only
        raise PostMarkerFailure("unexpected_error", "label_recovery", _report(attempt_root, verified, "label_recovery", "unexpected_error")) from None
    return AttemptOutcome(labels=labels, experiment_id=verified.experiment_id, marker_path=receipt.path, marker_sha256=receipt.marker_sha256)


def _report(attempt_root: str | os.PathLike[str], verified: VerifiedAttempt, stage: str, category: str) -> str | None:
    return write_failure_report(attempt_root, build_failure_report(verified=verified, stage=stage, category=category))


def source_sha256() -> str:
    return mf.file_sha256(Path(__file__))
