"""Numerical helpers for the held-out evaluation (draft protocol heldout-eval-1.2.0-draft, section 12.11.2).

Two components, both evaluation infrastructure only. Neither fits a model, scores a dataset, reads a label or changes the production model.

``lambda-power-1`` (training-only eigenvalue estimate and arm-relative learning rates)
-------------------------------------------------------------------------------------
For one arm and one ``num_buckets`` setting, the estimator builds the training feature matrix ``X`` with the model's own feature construction
(``LightweightAnomalyModel._features``, the routine ``fit`` uses, on an untrained instance) and estimates the largest eigenvalue of
``[X | 1]^T [X | 1] / n``; the column of ones stands for the bias. ``L = lambda_max / 4`` bounds the curvature of the logistic loss and
``learning_rate = c / L`` for the relative steps ``c`` in (0.1, 0.3, 0.8). A step ``c <= 1`` is the conservative range and ``c < 2`` the monotone
descent range; this is a numerical-stability statement, not evidence of predictive quality.

* Float64 throughout; the matrix ``[X | 1]^T [X | 1] / n`` is never formed (two matrix-vector products per iteration).
* Deterministic power iteration. Start vector: uniform, every entry ``1 / sqrt(d)`` with ``d = num_buckets + 1``. No random numbers. The matrix is
  entrywise non-negative, so by Perron-Frobenius the dominant eigenvector is non-negative and cannot be orthogonal to the uniform start.
* Estimate: the Rayleigh quotient of the current unit vector. Convergence: relative change of the Rayleigh quotient between successive iterations
  at most ``1e-10``; at most ``5000`` iterations.
* No fallback. A non-finite or non-positive estimate, or a run that does not converge, raises (:class:`EigenvalueError` /
  :class:`EigenvalueConvergenceError`); there is no default learning rate.
* Training only: the split must be named ``"train"`` explicitly. Split identity is never inferred from request content, and
  :func:`estimate_from_arm_inputs` additionally requires ``ArmInputs.split == "train"``.
* Limits: the Rayleigh quotient is a lower bound on the true largest eigenvalue; a small relative change between iterations is evidence of, not
  proof of, convergence when the two largest eigenvalues are close; the estimate varies by a few percent with the dataset.

``logit-readout-1`` (raw linear logit for ranking)
--------------------------------------------------
``logit = weights . features + bias``, with ``features`` from the model's own ``_features`` and the same ``numpy.dot`` expression as
``predict`` (``predict`` is the stable sigmoid of exactly that number). It exists because ``predict`` rounds to exactly 1.0 or 0.0 for large
|logit|, which creates artificial ties. The production model is not changed; this module only reads ``weights``, ``bias``, ``is_trained``,
``num_buckets``, ``_features`` and ``_validate_model_input``. The logit is an uncalibrated ranking score, not a probability, and a threshold chosen
on logits is a logit-scale threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..model import LightweightAnomalyModel
from ..models import ModelInput
from .arms import ARMS, ArmInputs

LAMBDA_METHOD_ID = "lambda-power-1"
LOGIT_METHOD_ID = "logit-readout-1"
TRAIN_SPLIT = "train"
POWER_RELATIVE_TOLERANCE = 1e-10
POWER_MAX_ITERATIONS = 5000
RELATIVE_STEP_GRID: tuple[float, ...] = (0.1, 0.3, 0.8)
NUM_BUCKETS_GRID: tuple[int, ...] = (1024, 2048, 4096)
SIGMOID_COMPARISON_TOLERANCE = 1e-12
SIGMOID_COMPARISON_MAX_ABS_LOGIT = 30.0


class NumericsError(ValueError):
    """An input, model or estimate is invalid. Messages hold positions, counts and identifiers only, never request or model values."""


class EigenvalueError(NumericsError):
    """The eigenvalue estimate is not finite and positive."""


class EigenvalueConvergenceError(RuntimeError):
    """Power iteration did not converge within the iteration limit. No fallback is used."""


# -- validation helpers -------------------------------------------------------------------------------------------------------


def _check_bucket_count(num_buckets: object) -> int:
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int) or num_buckets < 1:
        raise NumericsError("num_buckets must be a positive integer")
    return num_buckets


def _check_arm(arm: object) -> str:
    if arm not in ARMS:
        raise NumericsError(f"arm must be one of {ARMS}")
    return arm  # type: ignore[return-value]


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_sha256() -> str:
    """SHA-256 of this file with LF line endings, for the pre-run manifest."""
    return _source_sha256(Path(__file__))


def _model_source_sha256() -> str:
    import tokenization_ai.model as model_module
    return _source_sha256(Path(model_module.__file__))


def _check_model_input(model: LightweightAnomalyModel, item: object, position: int) -> ModelInput:
    if not isinstance(item, ModelInput):
        raise NumericsError(f"input at position {position} is not a ModelInput")
    for values in (item.input_ids, item.attention_mask):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise NumericsError(f"input at position {position} holds a non-integer id or mask value")
    try:
        model._validate_model_input(item)  # the production fit/predict validation
    except ValueError:
        raise NumericsError(f"input at position {position} is malformed") from None
    return item


# -- lambda-power-1 --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LambdaEstimate:
    """Training-only estimate for one arm and ``num_buckets``. JSON-serializable through ``to_dict``."""

    method: str
    arm: str
    split: str
    num_buckets: int
    training_records: int
    dimension: int
    lambda_max: float
    L: float
    iterations: int
    converged: bool
    relative_tolerance: float
    max_iterations: int
    final_relative_change: float
    residual_relative: float
    feature_mass_mean: float
    feature_mass_max: float
    feature_sq_norm_mean: float
    trace: float
    record_ids_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method, "arm": self.arm, "split": self.split, "num_buckets": self.num_buckets,
            "training_records": self.training_records, "dimension": self.dimension, "matrix": "[X | 1]^T [X | 1] / n",
            "precision": "float64", "initial_vector": "uniform, entries 1/sqrt(dimension)", "random_numbers": False,
            "lambda_max": self.lambda_max, "L": self.L, "L_definition": "lambda_max / 4",
            "iterations": self.iterations, "converged": self.converged, "relative_tolerance": self.relative_tolerance,
            "max_iterations": self.max_iterations, "final_relative_change": self.final_relative_change,
            "residual_relative": self.residual_relative,
            "feature_mass_mean": self.feature_mass_mean, "feature_mass_max": self.feature_mass_max,
            "feature_sq_norm_mean": self.feature_sq_norm_mean, "trace": self.trace,
            "record_ids_sha256": self.record_ids_sha256,
            "lambda_max_is_lower_bound": True, "fallback_learning_rate": None,
        }


def _validated_estimate(est: LambdaEstimate) -> LambdaEstimate:
    if est.method != LAMBDA_METHOD_ID or est.split != TRAIN_SPLIT or not est.converged:
        raise NumericsError("not a converged training-split estimate of the approved method")
    if not (math.isfinite(est.lambda_max) and est.lambda_max > 0 and math.isfinite(est.L) and est.L > 0):
        raise EigenvalueError("the eigenvalue estimate is not finite and positive")
    if est.L != est.lambda_max / 4:
        raise NumericsError("L is inconsistent with lambda_max / 4")
    return est


def _power_iteration(X: np.ndarray) -> tuple[float, int, bool, float, float]:
    """Largest eigenvalue of ``[X | 1]^T [X | 1] / n``; returns (estimate, iterations, converged, last relative change, residual)."""
    n, nb = X.shape
    d = nb + 1
    v = np.full(d, 1.0 / math.sqrt(d), dtype=np.float64)
    previous = 0.0
    change = math.inf

    def apply(vec: np.ndarray) -> np.ndarray:
        xv = X @ vec[:-1] + vec[-1]
        out = np.empty(d, dtype=np.float64)
        out[:-1] = X.T @ xv / n
        out[-1] = xv.sum() / n
        return out

    for iteration in range(1, POWER_MAX_ITERATIONS + 1):
        w = apply(v)
        estimate = float(v @ w)  # Rayleigh quotient of the unit vector v
        norm = float(np.linalg.norm(w))
        if not math.isfinite(estimate) or not math.isfinite(norm) or norm == 0.0:
            return math.nan, iteration, False, math.inf, math.inf
        if iteration > 1:
            change = abs(estimate - previous) / abs(estimate) if estimate != 0 else math.inf
            if change <= POWER_RELATIVE_TOLERANCE:
                residual = float(np.linalg.norm(apply(v) - estimate * v)) / abs(estimate)
                return estimate, iteration, True, change, residual
        previous = estimate
        v = w / norm
    return previous, POWER_MAX_ITERATIONS, False, change, math.inf


def estimate_lambda_max(inputs: Sequence[ModelInput], *, split: str, arm: str, num_buckets: int) -> LambdaEstimate:
    """Training-only ``lambda_max`` of ``[X | 1]^T [X | 1] / n`` for one arm and one ``num_buckets`` (``lambda-power-1``)."""
    if not isinstance(split, str) or split != TRAIN_SPLIT:
        raise NumericsError("only the training split is accepted")
    arm = _check_arm(arm)
    nb = _check_bucket_count(num_buckets)
    if isinstance(inputs, (str, bytes, Mapping)):
        raise NumericsError("inputs must be a sequence of ModelInput")
    items = list(inputs)
    if not items:
        raise NumericsError("the training inputs must not be empty")
    model = LightweightAnomalyModel(num_buckets=nb)  # untrained; used only for its feature construction
    X = np.empty((len(items), nb), dtype=np.float64)
    ids = []
    for k, raw in enumerate(items):
        item = _check_model_input(model, raw, k)
        X[k] = model._features(item.input_ids, item.attention_mask)
        ids.append(item.event_id)
    if len(set(ids)) != len(ids):
        raise NumericsError("duplicate record ids in the training inputs")
    if not np.isfinite(X).all():
        raise NumericsError("non-finite feature values")

    estimate, iterations, converged, change, residual = _power_iteration(X)
    if not math.isfinite(estimate) or estimate <= 0:
        raise EigenvalueError("the eigenvalue estimate is not finite and positive")
    if not converged:
        raise EigenvalueConvergenceError(
            f"{LAMBDA_METHOD_ID}: no convergence within {POWER_MAX_ITERATIONS} iterations (relative change {change:.3g}); "
            "no fallback learning rate is used")
    mass = X.sum(axis=1)
    sq = np.einsum("ij,ij->i", X, X)
    fingerprint = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    return LambdaEstimate(
        method=LAMBDA_METHOD_ID, arm=arm, split=TRAIN_SPLIT, num_buckets=nb, training_records=len(items), dimension=nb + 1,
        lambda_max=estimate, L=estimate / 4, iterations=iterations, converged=True, relative_tolerance=POWER_RELATIVE_TOLERANCE,
        max_iterations=POWER_MAX_ITERATIONS, final_relative_change=change, residual_relative=residual,
        feature_mass_mean=float(mass.mean()), feature_mass_max=float(mass.max()), feature_sq_norm_mean=float(sq.mean()),
        trace=float(sq.mean()) + 1.0, record_ids_sha256=fingerprint,
    )


def estimate_from_arm_inputs(arm_inputs: ArmInputs, *, arm: str, num_buckets: int) -> LambdaEstimate:
    """As :func:`estimate_lambda_max`, taking one arm of a **training** ``ArmInputs`` (its declared split must be ``"train"``)."""
    if not isinstance(arm_inputs, ArmInputs):
        raise NumericsError("an ArmInputs is required")
    arm = _check_arm(arm)
    chosen = {"A_request_only": arm_inputs.arm_a, "B_simulated_rules": arm_inputs.arm_b, "C_char3": arm_inputs.arm_c}[arm]
    return estimate_lambda_max(chosen, split=arm_inputs.split, arm=arm, num_buckets=num_buckets)


# -- arm-relative learning rates -------------------------------------------------------------------------------------------------


def _check_steps(steps: object) -> tuple[float, ...]:
    if not isinstance(steps, (tuple, list)) or not steps:
        raise NumericsError("the relative-step grid must be a non-empty tuple")
    for c in steps:
        if isinstance(c, bool) or not isinstance(c, numbers.Real) or not math.isfinite(c) or not 0 < c < 2:
            raise NumericsError("each relative step must be a finite number in (0, 2)")
    values = tuple(float(c) for c in steps)
    if any(b <= a for a, b in zip(values, values[1:])):
        raise NumericsError("the relative-step grid must be strictly ascending")
    return values


def relative_learning_rate(c: float, L: float) -> float:
    """``learning_rate = c / L`` for a finite ``L > 0`` and a relative step ``0 < c < 2``."""
    (step,) = _check_steps((c,))
    if isinstance(L, bool) or not isinstance(L, numbers.Real) or not math.isfinite(L) or L <= 0:
        raise EigenvalueError("L must be finite and positive")
    rate = step / float(L)
    if not math.isfinite(rate) or rate <= 0:
        raise NumericsError("the learning rate is not finite and positive")
    return rate


def learning_rates(estimate: LambdaEstimate, steps: Sequence[float] = RELATIVE_STEP_GRID) -> dict[str, Any]:
    """All relative-step learning rates for one validated training estimate. JSON-serializable; ``hex`` reproduces the double exactly."""
    est = _validated_estimate(estimate)
    grid = _check_steps(tuple(steps))
    rows = []
    for c in grid:
        rate = relative_learning_rate(c, est.L)
        rows.append({"relative_step_c": c, "learning_rate": rate, "learning_rate_hex": rate.hex(), "rate_times_L": rate * est.L})
    return {"method": LAMBDA_METHOD_ID, "arm": est.arm, "num_buckets": est.num_buckets, "split": est.split,
            "lambda_max": est.lambda_max, "L": est.L, "rates": rows,
            "note": "computed from training features only; not selected or tuned on labels or performance"}


def learning_rate_table(estimates: Sequence[LambdaEstimate], *, arms: Sequence[str] = ARMS,
                        bucket_grid: Sequence[int] = NUM_BUCKETS_GRID,
                        steps: Sequence[float] = RELATIVE_STEP_GRID) -> dict[str, Any]:
    """Rates for every (arm, num_buckets) of the approved grid; exactly one estimate per cell is required (no gaps, no duplicates)."""
    by_cell: dict[tuple[str, int], LambdaEstimate] = {}
    for est in estimates:
        key = (est.arm, est.num_buckets)
        if key in by_cell:
            raise NumericsError("more than one estimate for the same arm and num_buckets")
        by_cell[key] = _validated_estimate(est)
    wanted = {(a, b) for a in arms for b in bucket_grid}
    if set(by_cell) != wanted:
        raise NumericsError(f"estimates must cover exactly the grid cells ({len(wanted) - len(set(by_cell) & wanted)} missing, "
                            f"{len(set(by_cell) - wanted)} unexpected)")
    return {"method": LAMBDA_METHOD_ID, "relative_step_grid": list(_check_steps(tuple(steps))),
            "num_buckets_grid": list(bucket_grid),
            "cells": [learning_rates(by_cell[(a, b)], steps) for a in arms for b in bucket_grid]}


# -- logit-readout-1 ---------------------------------------------------------------------------------------------------------------


def _check_fitted_model(model: object) -> LightweightAnomalyModel:
    if not isinstance(model, LightweightAnomalyModel):
        raise NumericsError("a LightweightAnomalyModel is required")
    if not model.is_trained:
        raise NumericsError("the model is not fitted")
    weights = model.weights
    if not isinstance(weights, np.ndarray) or weights.dtype != np.float64 or weights.shape != (model.num_buckets,):
        raise NumericsError("the weight vector does not match num_buckets")
    if not np.isfinite(weights).all():
        raise NumericsError("the model weights are not finite")
    bias = model.bias
    if isinstance(bias, bool) or not isinstance(bias, numbers.Real) or not math.isfinite(bias):
        raise NumericsError("the model bias is not a finite number")
    return model


def raw_logit(model: LightweightAnomalyModel, model_input: ModelInput) -> float:
    """``weights . features + bias`` for one input (``logit-readout-1``); finite, uncalibrated, never passed through a sigmoid."""
    _check_fitted_model(model)
    item = _check_model_input(model, model_input, 0)
    with np.errstate(over="ignore", invalid="ignore"):  # an overflow must surface as the explicit error below, not as a warning
        logit = float(np.dot(model._features(item.input_ids, item.attention_mask), model.weights) + model.bias)
    if not math.isfinite(logit):
        raise NumericsError("the raw logit is not finite")
    return logit


def raw_logits(model: LightweightAnomalyModel, inputs: Sequence[ModelInput]) -> list[float]:
    """Raw logits in input order. The model is validated once; an invalid input names its position only."""
    _check_fitted_model(model)
    if isinstance(inputs, (str, bytes, Mapping)):
        raise NumericsError("inputs must be a sequence of ModelInput")
    out = []
    for k, raw in enumerate(inputs):
        item = _check_model_input(model, raw, k)
        with np.errstate(over="ignore", invalid="ignore"):
            logit = float(np.dot(model._features(item.input_ids, item.attention_mask), model.weights) + model.bias)
        if not math.isfinite(logit):
            raise NumericsError(f"the raw logit at position {k} is not finite")
        out.append(logit)
    return out


def _stable_sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def verify_logit_compatibility(model: LightweightAnomalyModel, inputs: Sequence[ModelInput], *,
                               tolerance: float = SIGMOID_COMPARISON_TOLERANCE,
                               max_abs_logit: float = SIGMOID_COMPARISON_MAX_ABS_LOGIT) -> dict[str, Any]:
    """Check ``sigmoid(raw_logit) == predict()`` on the non-saturated inputs (``|logit| <= max_abs_logit``).

    Raises if any checked example disagrees by more than ``tolerance`` or if no example is non-saturated (nothing could be verified).
    Saturated examples are counted, not compared. This calls ``predict`` only to compare, never ``fit``, and changes nothing.
    """
    logits = raw_logits(model, inputs)
    checked = saturated = 0
    worst = 0.0
    for item, logit in zip(inputs, logits):
        if abs(logit) > max_abs_logit:
            saturated += 1
            continue
        diff = abs(model.predict(item) - _stable_sigmoid(logit))
        worst = max(worst, diff)
        checked += 1
        if diff > tolerance:
            raise NumericsError(f"raw logit and predict() disagree beyond {tolerance:g}; faithful logit extraction is not safe")
    if checked == 0:
        raise NumericsError("no non-saturated example is available, so logit compatibility could not be verified")
    return {"method": LOGIT_METHOD_ID, "examples": len(logits), "checked": checked, "saturated_not_compared": saturated,
            "max_abs_difference": worst, "tolerance": tolerance, "max_abs_logit_compared": max_abs_logit}


def provenance() -> dict[str, Any]:
    """JSON-safe identifiers, settings and source hashes of both components, for the future pre-run manifest."""
    return {
        "lambda_method": LAMBDA_METHOD_ID, "logit_method": LOGIT_METHOD_ID, "relative_step_grid": list(RELATIVE_STEP_GRID),
        "num_buckets_grid": list(NUM_BUCKETS_GRID), "relative_tolerance": POWER_RELATIVE_TOLERANCE,
        "max_iterations": POWER_MAX_ITERATIONS, "precision": "float64", "initial_vector": "uniform 1/sqrt(dimension)",
        "numpy_version": np.__version__, "numerics_sha256": source_sha256(), "model_sha256": _model_source_sha256(),
        "private_model_interfaces_used": ["_features", "_validate_model_input"],
        "public_model_attributes_read": ["weights", "bias", "is_trained", "num_buckets"],
        "scores_are_uncalibrated": True,
    }
