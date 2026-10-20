"""``model-artifact-1``: reproducible, write-once persistence of fitted ``LightweightAnomalyModel`` candidates (draft protocol heldout-eval-1.2.0-draft,
section 12.12.5).

Infrastructure only. **This module trains nothing, scores nothing, authorizes nothing** and calls neither ``fit`` nor ``predict``. Its job is to let the
future authorized runner save every candidate model (81 = 3 arms x 27) and let a verifier later reload each one bit-exactly.

Format. One bounded file per candidate: canonical JSON (sorted keys, compact separators, ASCII, no NaN or infinity, one trailing newline). The float64 weight
vector is stored as base64 of its little-endian bytes (lossless, deterministic; split into fixed 4000-character chunks), with its SHA-256; the bias is stored as ``float.hex``. Loading never executes
code: only JSON parsing and ``base64`` / ``numpy.frombuffer`` on checked lengths (no pickle-style loaders, no eval).

What an artifact records: format version; arm; candidate id, ``num_buckets``, ``epochs``, ``relative_step_c`` and the train-only learning rate (which must equal
``c / L`` exactly, ``L = lambda_max / 4``); input representation; the model-source SHA-256; the pre-run manifest digest it is *bound to*; allow-listed train-only
provenance (training id hash, vocabulary hash, token budget, eigenvalue estimate); the fitted weights and bias; and the model fingerprint of
``selection_record.model_artifact_fingerprint``. It holds no request content, no label of any split, no score, prediction or metric, no loss history, and it
**does not claim** that an authorized experiment trained it (``authorization.asserted`` is ``false``: whether training was authorized is established only by
verifying the manifest, which this module does not do).

Trust model. The loader takes everything that matters from the caller, never from the artifact: the expected file SHA-256 (from outside), the expected arm,
candidate id, manifest digest and learning rate (from a verified manifest), and optionally the expected model-source hash and model fingerprint. The artifact
is checked against them. Binding to a manifest digest is *not* proof that the manifest was verified; the caller must verify the manifest first.
"""

from __future__ import annotations

import base64
import binascii
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import model as _model_module
from ..model import LightweightAnomalyModel
from . import manifest as mf
from . import numerics, runner
from . import selection_record as sr
from .arms import ARMS, ARMS_VERSION

ARTIFACT_VERSION = "model-artifact-1"
ARTIFACT_KIND = "fitted_candidate_model"
MAX_ARTIFACT_BYTES = 1024 * 1024
CHUNK_CHARS = 4000  # base64 text is split into fixed 4000-character chunks (a multiple of 4) because canonical strings are bounded
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = {"artifact_version", "kind", "arm", "candidate", "input_representation", "model_source_sha256", "manifest_sha256", "training",
             "authorization", "model", "limitations"}
_CANDIDATE_KEYS = {"candidate_id", "num_buckets", "epochs", "relative_step_c", "learning_rate", "learning_rate_hex"}
_TRAINING_KEYS = {"train_ids_sha256", "vocabulary_sha256", "max_length", "lambda_method", "lambda_max", "L"}
_MODEL_KEYS = {"num_buckets", "weights_dtype", "weights_b64_chunks", "weights_sha256", "bias_hex", "is_trained", "fingerprint"}
_AUTHORIZATION = {"asserted": False, "note": "this artifact does not assert that an authorized experiment trained it; authorization is established "
                                              "only by verifying the pre-run manifest"}
LIMITATIONS: tuple[str, ...] = (
    "the manifest digest is a binding, not proof that the manifest was verified",
    "the artifact stores parameters only: no loss history, labels, scores, predictions or metrics",
    "synthetic held-out evaluation infrastructure; no real-world claim",
)


class ArtifactError(mf.ManifestError):
    """Base class. Messages hold field names and categories only, never parameter values, labels, scores or request content."""


class ArtifactSchemaError(ArtifactError):
    """The artifact or its inputs are malformed, ambiguous or contain forbidden content."""


class ArtifactWriteError(ArtifactError):
    """The artifact could not be written. Nothing was repaired or replaced."""


class ArtifactVerificationError(ArtifactError):
    """The artifact does not match the independently supplied expectations."""


# -- identity and paths ------------------------------------------------------------------------------------------------------------------


def input_representation(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ArtifactSchemaError(f"arm must be one of {ARMS}")
    return {"arms_version": ARMS_VERSION, "arm": arm, "model_input": "arm_c_hashed_char_trigram_ids" if arm == ARMS[2] else "tensorized_token_ids"}


def model_source_sha256() -> str:
    """SHA-256 (LF-normalized) of the live ``model.py``, the implementation an artifact must have been written for."""
    return mf.file_sha256(Path(_model_module.__file__))


def artifact_filename(manifest_sha256: str, candidate_id: str) -> str:
    """Deterministic file name from the trusted manifest digest and candidate id; nothing from the artifact content is used."""
    if not isinstance(manifest_sha256, str) or not _HEX64.fullmatch(manifest_sha256):
        raise ArtifactSchemaError("manifest_sha256 must be a SHA-256 hex string")
    arm = next((a for a in ARMS if isinstance(candidate_id, str) and candidate_id.startswith(a + "|")), None)
    if arm is None or candidate_id not in runner.approved_candidate_ids(arm):
        raise ArtifactSchemaError("candidate_id is not one of the approved candidates")
    return f"model-artifact-{mf.sha256_hex((manifest_sha256 + '|' + candidate_id).encode('ascii'))}.json"


def expected_artifact_names(manifest_sha256: str) -> dict[str, str]:
    """File names of all 81 approved candidates (27 per arm) for one manifest."""
    return {cid: artifact_filename(manifest_sha256, cid) for arm in ARMS for cid in runner.approved_candidate_ids(arm)}


def learning_rates_from_manifest(manifest: Mapping[str, Any]) -> dict[str, float]:
    """candidate id -> train-only learning rate from an already **verified** manifest dictionary (the caller must have verified it)."""
    try:
        return {c["candidate_id"]: c["learning_rate"] for c in manifest["numerics"]["candidates"]}
    except (KeyError, TypeError):
        raise ArtifactSchemaError("the manifest has no candidate table") from None


# -- validation helpers -------------------------------------------------------------------------------------------------------------------------------


def _check_model(model: object) -> LightweightAnomalyModel:
    if not isinstance(model, LightweightAnomalyModel) or not model.is_trained:
        raise ArtifactVerificationError("a fitted LightweightAnomalyModel is required")
    w = model.weights
    if not isinstance(w, np.ndarray) or w.dtype != np.float64 or w.shape != (model.num_buckets,) or not np.isfinite(w).all():
        raise ArtifactVerificationError("the model weights are malformed or not finite float64 of length num_buckets")
    b = model.bias
    if isinstance(b, bool) or not isinstance(b, (int, float, np.floating)) or not math.isfinite(float(b)):
        raise ArtifactVerificationError("the model bias is not a finite number")
    return model


def _check_candidate(arm: str, num_buckets: object, epochs: object, c: object) -> tuple[str, int, int, float]:
    if arm not in ARMS:
        raise ArtifactSchemaError(f"arm must be one of {ARMS}")
    for value, grid, name in ((num_buckets, numerics.NUM_BUCKETS_GRID, "num_buckets"), (epochs, runner.EPOCHS_GRID, "epochs")):
        if isinstance(value, bool) or not isinstance(value, int) or value not in grid:
            raise ArtifactSchemaError(f"{name} is not an approved value")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or c not in numerics.RELATIVE_STEP_GRID:
        raise ArtifactSchemaError("relative_step_c is not an approved value")
    return arm, int(num_buckets), int(epochs), float(c)  # type: ignore[arg-type]


def _check_training(training: object, c: float, learning_rate: float) -> dict[str, Any]:
    if not isinstance(training, Mapping) or set(training) != _TRAINING_KEYS:
        raise ArtifactSchemaError("training provenance must hold exactly " + ", ".join(sorted(_TRAINING_KEYS)))
    t = dict(training)
    if not (isinstance(t["train_ids_sha256"], str) and _HEX64.fullmatch(t["train_ids_sha256"])):
        raise ArtifactSchemaError("training.train_ids_sha256")
    if t["vocabulary_sha256"] is not None and not (isinstance(t["vocabulary_sha256"], str) and _HEX64.fullmatch(t["vocabulary_sha256"])):
        raise ArtifactSchemaError("training.vocabulary_sha256")
    if t["max_length"] is not None and (isinstance(t["max_length"], bool) or not isinstance(t["max_length"], int) or t["max_length"] < 1):
        raise ArtifactSchemaError("training.max_length")
    if t["lambda_method"] != numerics.LAMBDA_METHOD_ID:
        raise ArtifactSchemaError("training.lambda_method")
    lam, L = t["lambda_max"], t["L"]
    for v in (lam, L):
        if isinstance(v, bool) or not isinstance(v, float) or not math.isfinite(v) or v <= 0:
            raise ArtifactSchemaError("training.lambda_max and training.L must be finite positive floats")
    if L != lam / 4:
        raise ArtifactSchemaError("training.L must equal lambda_max / 4")
    try:
        expected = numerics.relative_learning_rate(c, L)
    except numerics.NumericsError:
        raise ArtifactSchemaError("the learning rate cannot be derived from c and L") from None
    if expected != learning_rate:
        raise ArtifactSchemaError("the learning rate is not c / L of the recorded eigenvalue estimate")
    return t


def _chunks(text: str) -> list[str]:
    return [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]


def _build_artifact(model: LightweightAnomalyModel, arm: str, candidate: tuple[int, int, float, float], manifest_sha256: str,
                    training: Mapping[str, Any]) -> dict[str, Any]:
    num_buckets, epochs, c, learning_rate = candidate
    checked_arm, nb, ep, step = _check_candidate(arm, num_buckets, epochs, c)
    if model.num_buckets != nb:
        raise ArtifactVerificationError("the model's num_buckets differs from the candidate's")
    if not isinstance(manifest_sha256, str) or not _HEX64.fullmatch(manifest_sha256):
        raise ArtifactSchemaError("manifest_sha256 must be a SHA-256 hex string")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ArtifactSchemaError("learning_rate must be a finite positive float")
    t = _check_training(training, step, learning_rate)
    weights = np.ascontiguousarray(model.weights, dtype="<f8")
    cid = runner.canonical_candidate_id(checked_arm, nb, ep, step)
    return {
        "artifact_version": ARTIFACT_VERSION, "kind": ARTIFACT_KIND, "arm": checked_arm,
        "candidate": {"candidate_id": cid, "num_buckets": nb, "epochs": ep, "relative_step_c": step, "learning_rate": learning_rate,
                      "learning_rate_hex": learning_rate.hex()},
        "input_representation": input_representation(checked_arm), "model_source_sha256": model_source_sha256(), "manifest_sha256": manifest_sha256,
        "training": t, "authorization": dict(_AUTHORIZATION),
        "model": {"num_buckets": nb, "weights_dtype": "float64_little_endian", "weights_b64_chunks": _chunks(base64.b64encode(weights.tobytes()).decode("ascii")),
                  "weights_sha256": mf.sha256_hex(weights.tobytes()), "bias_hex": float(model.bias).hex(), "is_trained": True,
                  "fingerprint": sr.model_artifact_fingerprint(model)},
        "limitations": list(LIMITATIONS),
    }


def build_artifact(model: LightweightAnomalyModel, *, arm: str, num_buckets: int, epochs: int, relative_step_c: float, learning_rate: float,
                   manifest_sha256: str, training: Mapping[str, Any]) -> dict[str, Any]:
    """The artifact dictionary for one fitted candidate (nothing is written)."""
    return _build_artifact(_check_model(model), arm, (num_buckets, epochs, relative_step_c, learning_rate), manifest_sha256, training)


# -- writer ----------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WrittenArtifact:
    path: str
    sha256: str
    size: int
    candidate_id: str


def write_model_artifact(output_dir: str | os.PathLike[str], model: LightweightAnomalyModel, *, repo_root: str | os.PathLike[str], arm: str,
                         num_buckets: int, epochs: int, relative_step_c: float, learning_rate: float, manifest_sha256: str,
                         training: Mapping[str, Any]) -> WrittenArtifact:
    """Create the candidate's artifact **once** in a trusted directory outside the repository; returns the SHA-256 of the exact bytes written."""
    artifact = build_artifact(model, arm=arm, num_buckets=num_buckets, epochs=epochs, relative_step_c=relative_step_c, learning_rate=learning_rate,
                              manifest_sha256=manifest_sha256, training=training)
    directory = Path(output_dir)
    if not directory.is_dir():
        raise ArtifactWriteError("the artifact directory does not exist")
    resolved, root = directory.resolve(), Path(repo_root).resolve()
    if resolved == root or root in resolved.parents:
        raise ArtifactWriteError("the artifact directory must be outside the repository")
    cid = artifact["candidate"]["candidate_id"]
    path = directory / artifact_filename(manifest_sha256, cid)
    try:
        data = mf.canonical_bytes(artifact)
    except mf.ManifestError:
        raise ArtifactSchemaError("the artifact could not be serialized canonically") from None
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ArtifactSchemaError("the artifact is too large")
    _validate_structure(mf.parse_manifest_bytes(data))
    if os.path.lexists(path):
        raise ArtifactWriteError("an artifact already exists for this candidate; it is never overwritten")
    try:
        with open(path, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise ArtifactWriteError("an artifact already exists for this candidate; it is never overwritten") from None
    except OSError:
        raise ArtifactWriteError("the artifact write failed; any partial file is left in place for review and is not replaced") from None
    try:
        if path.read_bytes() != data:
            raise ArtifactWriteError("the written bytes differ from the intended bytes; the file is left in place for review")
        os.chmod(path, stat.S_IREAD)
    except ArtifactWriteError:
        raise
    except OSError:
        raise ArtifactWriteError("the artifact could not be finalized; the file is left in place for review") from None
    return WrittenArtifact(path=str(path), sha256=mf.sha256_hex(data), size=len(data), candidate_id=cid)


# -- loader ------------------------------------------------------------------------------------------------------------------------------------------


def _validate_structure(a: Any) -> None:
    problems: list[str] = []
    if not isinstance(a, dict) or set(a) != _TOP_KEYS:
        raise ArtifactSchemaError("the artifact does not have exactly the required fields")
    if a["artifact_version"] != ARTIFACT_VERSION or a["kind"] != ARTIFACT_KIND:
        problems.append("artifact_version or kind")
    if a["authorization"] != _AUTHORIZATION:
        problems.append("authorization: must state that authorization is not asserted")
    if a["limitations"] != list(LIMITATIONS):
        problems.append("limitations")
    if not isinstance(a["candidate"], dict) or set(a["candidate"]) != _CANDIDATE_KEYS or not isinstance(a["model"], dict) \
            or set(a["model"]) != _MODEL_KEYS or not isinstance(a["training"], dict) or set(a["training"]) != _TRAINING_KEYS:
        raise ArtifactSchemaError("; ".join(problems + ["candidate, training or model fields"]))
    try:
        _, nb, ep, c = _check_candidate(a["arm"], a["candidate"]["num_buckets"], a["candidate"]["epochs"], a["candidate"]["relative_step_c"])
        lr = a["candidate"]["learning_rate"]
        if a["candidate"]["candidate_id"] != runner.canonical_candidate_id(a["arm"], nb, ep, c) or a["candidate"]["learning_rate_hex"] != float(lr).hex():
            problems.append("candidate identity")
        _check_training(a["training"], c, lr)
        if a["input_representation"] != input_representation(a["arm"]):
            problems.append("input_representation")
        if a["model"]["num_buckets"] != nb or a["model"]["weights_dtype"] != "float64_little_endian" or a["model"]["is_trained"] is not True:
            problems.append("model header")
    except (ArtifactSchemaError, TypeError, ValueError):
        problems.append("candidate, training or model values")
    for key in ("model_source_sha256", "manifest_sha256"):
        if not (isinstance(a[key], str) and _HEX64.fullmatch(a[key])):
            problems.append(key)
    scan: list[str] = []
    sr._scan(a, "$", scan)
    problems += scan
    if problems:
        raise ArtifactSchemaError("; ".join(problems[:15]))


@dataclass(frozen=True)
class LoadedArtifact:
    """A reconstructed fitted model plus the (descriptive, already checked) artifact metadata."""

    model: LightweightAnomalyModel
    candidate_id: str
    arm: str
    sha256: str
    fingerprint: Mapping[str, Any]
    metadata: Mapping[str, Any]


def load_model_artifact(output_dir: str | os.PathLike[str], *, expected_sha256: str, expected_arm: str, expected_candidate_id: str,
                        expected_manifest_sha256: str, expected_learning_rate: float, expected_model_sha256: str | None = None,
                        expected_fingerprint: Mapping[str, Any] | None = None) -> LoadedArtifact:
    """Load, verify and reconstruct one candidate model. All expectations come from the caller (a verified manifest, the file digest from outside)."""
    if not isinstance(expected_sha256, str) or not _HEX64.fullmatch(expected_sha256):
        raise ArtifactVerificationError("an expected artifact SHA-256 from outside is required")
    if expected_arm not in ARMS:
        raise ArtifactVerificationError("the expected arm is not recognized")
    path = Path(output_dir) / artifact_filename(expected_manifest_sha256, expected_candidate_id)  # path from trusted values only
    if not expected_candidate_id.startswith(expected_arm + "|"):
        raise ArtifactVerificationError("the expected candidate does not belong to the expected arm")
    if isinstance(expected_learning_rate, bool) or not isinstance(expected_learning_rate, float) or not math.isfinite(expected_learning_rate):
        raise ArtifactVerificationError("an expected learning rate from the verified manifest is required")
    try:
        data = path.read_bytes()
    except OSError:
        raise ArtifactVerificationError("the artifact file is missing or unreadable") from None
    if len(data) > MAX_ARTIFACT_BYTES or mf.sha256_hex(data) != expected_sha256:
        raise ArtifactVerificationError("the artifact bytes do not match the expected SHA-256")
    try:
        artifact = mf.parse_manifest_bytes(data)
    except mf.ManifestError:
        raise ArtifactSchemaError("the artifact is malformed, ambiguous or not canonical") from None
    _validate_structure(artifact)
    cand = artifact["candidate"]
    mismatches = []
    if artifact["arm"] != expected_arm:
        mismatches.append("arm")
    if cand["candidate_id"] != expected_candidate_id:
        mismatches.append("candidate_id")
    if artifact["manifest_sha256"] != expected_manifest_sha256:
        mismatches.append("manifest_sha256")
    if cand["learning_rate"] != expected_learning_rate:
        mismatches.append("learning_rate")
    live = model_source_sha256()
    if artifact["model_source_sha256"] != live or (expected_model_sha256 is not None and expected_model_sha256 != live):
        mismatches.append("model_source_sha256")
    if mismatches:
        raise ArtifactVerificationError(f"the artifact differs from the expected identity ({', '.join(mismatches)})")

    nb = cand["num_buckets"]
    chunks = artifact["model"]["weights_b64_chunks"]
    if not isinstance(chunks, list) or not chunks or not all(isinstance(c, str) for c in chunks)             or any(len(c) != CHUNK_CHARS for c in chunks[:-1]) or not 0 < len(chunks[-1]) <= CHUNK_CHARS:
        raise ArtifactSchemaError("the weight encoding is not in the canonical chunk layout")
    try:
        raw = base64.b64decode("".join(chunks).encode("ascii"), validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError):
        raise ArtifactSchemaError("the weight encoding is invalid") from None
    if len(raw) != nb * 8 or mf.sha256_hex(raw) != artifact["model"]["weights_sha256"]:
        raise ArtifactVerificationError("the weights do not match their recorded length or hash")
    weights = np.frombuffer(raw, dtype="<f8").astype(np.float64)  # a private native-endian copy
    try:
        bias = float.fromhex(artifact["model"]["bias_hex"])
    except (ValueError, TypeError):
        raise ArtifactSchemaError("the bias encoding is invalid") from None
    if not np.isfinite(weights).all() or not math.isfinite(bias):
        raise ArtifactVerificationError("the stored parameters are not finite")

    model = LightweightAnomalyModel(num_buckets=nb)
    if not isinstance(model.weights, np.ndarray) or model.weights.shape != (nb,) or model.weights.dtype != np.float64 or model.is_trained:
        raise ArtifactVerificationError("the production model's construction contract changed; refusing to reconstruct")
    model.weights, model.bias, model.is_trained = weights, bias, True  # the model's own representation of a fitted state
    fingerprint = sr.model_artifact_fingerprint(model)
    if fingerprint != artifact["model"]["fingerprint"] or (expected_fingerprint is not None and dict(expected_fingerprint) != fingerprint):
        raise ArtifactVerificationError("the reconstructed model fingerprint differs from the recorded or expected fingerprint")
    return LoadedArtifact(model=model, candidate_id=expected_candidate_id, arm=expected_arm, sha256=expected_sha256, fingerprint=fingerprint,
                          metadata={k: artifact[k] for k in ("candidate", "training", "input_representation", "model_source_sha256", "manifest_sha256")})


def load_candidate_set(output_dir: str | os.PathLike[str], *, expected_manifest_sha256: str, expected_digests: Mapping[str, str],
                       expected_learning_rates: Mapping[str, float], expected_model_sha256: str | None = None) -> dict[str, LoadedArtifact]:
    """Load exactly the 81 approved candidates (27 per arm). Missing, extra or substituted artifacts raise; each model is an independent object."""
    names = expected_artifact_names(expected_manifest_sha256)
    if set(expected_digests) != set(names) or set(expected_learning_rates) != set(names):
        raise ArtifactVerificationError(f"expected digests and learning rates must cover exactly the {len(names)} approved candidates")
    out = {}
    for cid in sorted(names):
        arm = next(a for a in ARMS if cid.startswith(a + "|"))
        out[cid] = load_model_artifact(output_dir, expected_sha256=expected_digests[cid], expected_arm=arm, expected_candidate_id=cid,
                                       expected_manifest_sha256=expected_manifest_sha256, expected_learning_rate=expected_learning_rates[cid],
                                       expected_model_sha256=expected_model_sha256)
    return out


def source_sha256() -> str:
    return mf.file_sha256(Path(__file__))
