"""``detection-metrics-1``: deterministic detection metrics and a paired template-group bootstrap for the A/B/C study.

Evaluation infrastructure only (draft protocol heldout-eval-1.2.0-draft, sections 7 and 12.11). Nothing here trains, fits or
scores a model, reads a request, or touches a dataset: every function takes already-computed scores and labels as plain numbers.
Scores are **uncalibrated ranking scores, not probabilities**. Simulated rules (Arm B) are not Member 2 verified rules, and a
bootstrap interval describes only the sampling variability of the synthetic generator under the chosen sampling scheme (whole
normalized-template groups drawn with replacement); it says nothing about real traffic or real attackers.

Inputs and errors
-----------------
Scores must be finite real numbers (``bool``, NaN and +/-inf are rejected) and labels must be the integers 0 (normal) or 1
(suspicious). Invalid input raises :class:`DetectionMetricsError` (messages never echo more than counts and positions). A metric that
is mathematically undefined for *valid* input (for example AP with no positives) is returned as ``None`` and is never replaced by 0.

Average Precision (primary, ``average_precision``)
--------------------------------------------------
Scores are sorted descending and tied scores form one block, so the result cannot depend on input order or on how ties happen to be
listed. After each block ``k`` let ``TP_k`` and ``n_k`` be the cumulative positives and cumulative records. The estimator is the
step-wise sum without interpolation::

    AP = sum_k (R_k - R_{k-1}) * P_k = (1 / P) * sum_k pos_k * TP_k / n_k

with ``R`` recall, ``P_k`` precision, ``pos_k`` the positives inside block ``k`` and ``P`` the total positives. Undefined (``None``)
when ``P == 0``. A single block (all scores equal) gives ``AP = P / n``, the prevalence.

ROC-AUC (secondary, ``roc_auc``)
--------------------------------
The Mann-Whitney statistic with ties counted as one half, computed exactly in integers and divided once. Undefined (``None``) when
either class is absent.

Threshold metrics (secondary)
-----------------------------
``confusion_counts`` predicts positive when ``score >= threshold`` (``threshold=None`` is the all-negative prediction). Precision
is ``TP / (TP + FP)`` and is ``None`` when nothing is predicted positive; recall is ``TP / (TP + FN)`` and is ``None`` without positives;
F1 is ``2 TP / (2 TP + FP + FN)`` and is ``None`` only when that denominator is 0 (no positives and no predicted positives). So with
positives present and nothing predicted positive F1 is 0 (and precision undefined). FPR is ``FP / (FP + TN)`` and is ``None`` without
negatives; FNR is ``FN / (FN + TP)`` and is ``None`` without positives.

Validation-only threshold (``select_threshold``)
------------------------------------------------
The function receives validation scores and labels and nothing else. Candidates are every distinct validation score (predict positive
when ``score >= t``) plus the all-negative prediction. The candidate with the largest F1 (compared exactly as fractions) wins; ties go
to the highest threshold, and the all-negative candidate counts as the highest. Both classes must be present in validation. The test
split enters only afterwards, through ``confusion_counts`` with the already fixed threshold.

Paired group bootstrap (``paired_group_bootstrap``)
---------------------------------------------------
Statistic: ``AP(B) - AP(A)`` on the same records. The unit of resampling is the normalized template group: ``G`` groups are drawn with
replacement from the ``G`` observed groups, a drawn group contributes **all** its records (mixed-label groups stay intact), and a group
drawn ``m`` times contributes each of its records ``m`` times. The same draw is used for both arms, so the two arms are always scored on
identical resampled records, and Arms A and B are joined by record id (the id sets of scores A, scores B, labels and groups must match
exactly). Settings: 2,000 replicates, ``random.Random(8201)`` (a local generator; global random state is never used or changed), 95%
percentile interval by linear interpolation between order statistics (position ``(m - 1) * alpha``).

Undefined replicates: a resample can contain no positive record (labels are shared, so then AP is undefined for both arms at once).
Such replicates are counted and **never redrawn or replaced**; the interval is computed from the valid replicates only (and is therefore
conditional on the resample containing a positive). Draft minimum-valid rule: at least 95% of the replicates must be valid (1,900 of
2,000), otherwise ``status="insufficient_valid_replicates"`` and no interval is reported. The 95% level mirrors the interval level so
that discarding cannot move more probability mass than the interval's own tail allowance; it is a draft convention, not tuned on any
result. If the full-sample AP is itself undefined (no positives) the status is ``"undefined_point_estimate"`` and no replicate is run.

Limits: with few groups the percentile interval is rough; the interval ignores uncertainty from model fitting, threshold, hyper-
parameters and dataset seed; it is conditional on one generated dataset and one split; and no success margin is defined here.
"""

from __future__ import annotations

import math
import numbers
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

METRICS_ID = "detection-metrics-1"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 8201
BOOTSTRAP_CONFIDENCE = 0.95
MIN_VALID_PERCENT = 95
STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_valid_replicates"
STATUS_UNDEFINED_POINT = "undefined_point_estimate"
REASON_NO_POSITIVE = "no_positive_in_resample"


class DetectionMetricsError(ValueError):
    """Scores, labels, groups or settings are invalid. Messages hold counts and positions only."""


# -- input validation -------------------------------------------------------------------------------------------------


def _check_scores_labels(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[int]]:
    if isinstance(scores, (str, bytes, Mapping)) or isinstance(labels, (str, bytes, Mapping)):
        raise DetectionMetricsError("scores and labels must be sequences of numbers")
    try:
        score_list, label_list = list(scores), list(labels)
    except TypeError:
        raise DetectionMetricsError("scores and labels must be sequences of numbers") from None
    if len(score_list) != len(label_list):
        raise DetectionMetricsError(f"scores ({len(score_list)}) and labels ({len(label_list)}) differ in length")
    if not score_list:
        raise DetectionMetricsError("scores and labels must not be empty")
    return [_check_score(s, i) for i, s in enumerate(score_list)], [_check_label(y, i) for i, y in enumerate(label_list)]


def _check_score(value: object, position: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise DetectionMetricsError(f"score at {position} is not a real number")
    number = float(value)
    if not math.isfinite(number):
        raise DetectionMetricsError(f"score at {position} is not finite")
    return number


def _check_label(value: object, position: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) not in (0, 1):
        raise DetectionMetricsError(f"label at {position} is not the integer 0 or 1")
    return int(value)


def _blocks(scores: list[float], labels: list[int]) -> list[tuple[float, int, int]]:
    """Tied-score blocks as ``(score, positives, negatives)``, highest score first."""
    counts: dict[float, list[int]] = {}
    for s, y in zip(scores, labels):
        counts.setdefault(s + 0.0, [0, 0])[0 if y == 1 else 1] += 1  # + 0.0 folds -0.0 into 0.0
    return [(s, counts[s][0], counts[s][1]) for s in sorted(counts, reverse=True)]


# -- AP and ROC-AUC ---------------------------------------------------------------------------------------------------


def _ap_from_blocks(blocks: Sequence[tuple[int, int]]) -> float | None:
    """AP from ``(positives, negatives)`` per tied block, highest score first. ``None`` without positives."""
    total_pos = sum(p for p, _ in blocks)
    if total_pos == 0:
        return None
    tp = seen = 0
    terms = []
    for pos, neg in blocks:
        tp += pos
        seen += pos + neg
        if pos:
            terms.append(pos * tp / seen)
    return math.fsum(terms) / total_pos


def _auc_from_blocks(blocks: Sequence[tuple[int, int]]) -> float | None:
    """ROC-AUC from ``(positives, negatives)`` per tied block, highest score first. ``None`` unless both classes occur."""
    total_pos = sum(p for p, _ in blocks)
    total_neg = sum(n for _, n in blocks)
    if total_pos == 0 or total_neg == 0:
        return None
    neg_below = total_neg
    twice_wins = 0
    for pos, neg in blocks:
        neg_below -= neg
        twice_wins += pos * (2 * neg_below + neg)  # a positive beats every lower negative and ties (half) the same-block ones
    return twice_wins / (2 * total_pos * total_neg)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Step-wise Average Precision with tied scores grouped; ``None`` when there is no positive record."""
    score_list, label_list = _check_scores_labels(scores, labels)
    return _ap_from_blocks([(p, n) for _, p, n in _blocks(score_list, label_list)])


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """ROC-AUC with ties counted as one half; ``None`` unless both classes are present."""
    score_list, label_list = _check_scores_labels(scores, labels)
    return _auc_from_blocks([(p, n) for _, p, n in _blocks(score_list, label_list)])


# -- confusion matrix and threshold metrics ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.fp + self.tn

    @property
    def records(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def metrics(self) -> dict[str, float | None]:
        """Precision, recall, F1, FPR and FNR; ``None`` where the denominator is 0 (see the module docstring)."""

        def ratio(num: int, den: int) -> float | None:
            return num / den if den else None

        return {
            "precision": ratio(self.tp, self.tp + self.fp),
            "recall": ratio(self.tp, self.tp + self.fn),
            "f1": ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "false_positive_rate": ratio(self.fp, self.fp + self.tn),
            "false_negative_rate": ratio(self.fn, self.fn + self.tp),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn, "records": self.records, **self.metrics()}


def confusion_counts(scores: Sequence[float], labels: Sequence[int], threshold: float | None) -> ConfusionCounts:
    """Counts when predicting positive for ``score >= threshold``; ``threshold=None`` predicts no record positive."""
    score_list, label_list = _check_scores_labels(scores, labels)
    cut = None if threshold is None else _check_score(threshold, "threshold")
    tp = fp = tn = fn = 0
    for s, y in zip(score_list, label_list):
        predicted = cut is not None and s >= cut
        if predicted:
            tp, fp = tp + (y == 1), fp + (y == 0)
        else:
            fn, tn = fn + (y == 1), tn + (y == 0)
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


# -- validation-only threshold selection ------------------------------------------------------------------------------

THRESHOLD_RULE = "max_validation_f1_ties_to_highest_threshold"


@dataclass(frozen=True)
class ThresholdSelection:
    """A threshold fixed on validation data. ``threshold=None`` means the all-negative prediction won."""

    rule: str
    threshold: float | None
    f1: float
    validation_counts: ConfusionCounts
    candidates_considered: int  # distinct validation scores plus the all-negative candidate

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "threshold": self.threshold, "predicts_all_negative": self.threshold is None,
                "validation_f1": self.f1, "validation_counts": self.validation_counts.to_dict(),
                "candidates_considered": self.candidates_considered, "predict_positive_when": "score >= threshold",
                "basis": "validation_scores_and_labels_only"}


def select_threshold(validation_scores: Sequence[float], validation_labels: Sequence[int]) -> ThresholdSelection:
    """Pick the threshold that maximizes validation F1 (ties to the highest). Uses validation data only, by construction."""
    scores, labels = _check_scores_labels(validation_scores, validation_labels)
    blocks = _blocks(scores, labels)
    total_pos = sum(p for _, p, _ in blocks)
    total_neg = sum(n for _, _, n in blocks)
    if total_pos == 0 or total_neg == 0:
        raise DetectionMetricsError("threshold selection needs both classes in the validation data")
    best_threshold: float | None = None  # the all-negative candidate is the highest threshold; F1 = 0 as positives exist
    best_f1 = Fraction(0)
    tp = fp = 0
    for score, pos, neg in blocks:  # descending, so a tie never displaces the earlier (higher) threshold
        tp, fp = tp + pos, fp + neg
        f1 = Fraction(2 * tp, 2 * tp + fp + (total_pos - tp))
        if f1 > best_f1:
            best_threshold, best_f1 = score, f1
    counts = confusion_counts(scores, labels, best_threshold)
    return ThresholdSelection(rule=THRESHOLD_RULE, threshold=best_threshold, f1=float(best_f1), validation_counts=counts,
                              candidates_considered=len(blocks) + 1)


def evaluate_at_threshold(scores: Sequence[float], labels: Sequence[int], selection: ThresholdSelection) -> ConfusionCounts:
    """Apply an already fixed threshold to any split. It cannot change the threshold."""
    return confusion_counts(scores, labels, selection.threshold)


# -- paired group bootstrap -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Result of ``paired_group_bootstrap``. JSON-serializable through ``to_dict`` (no NaN or infinity)."""

    status: str
    statistic: str
    seed: int
    replicates_requested: int
    replicates_valid: int
    replicates_undefined: int
    undefined_reasons: Mapping[str, int]
    min_valid_replicates: int
    confidence: float
    ap_a: float | None
    ap_b: float | None
    difference: float | None  # AP(B) - AP(A) on the full sample
    interval: tuple[float, float] | None
    records: int
    groups: int
    mixed_label_groups: int
    positive_records: int
    groups_with_positive: int
    replicate_differences: tuple[float | None, ...] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "metrics_id": METRICS_ID, "status": self.status, "statistic": self.statistic,
            "resampling_unit": "normalized_template_group", "resampling": "groups drawn with replacement; same draw for both arms",
            "seed": self.seed, "replicates_requested": self.replicates_requested, "replicates_valid": self.replicates_valid,
            "replicates_undefined": self.replicates_undefined, "undefined_reasons": dict(self.undefined_reasons),
            "undefined_replicates_replaced": False, "min_valid_replicates": self.min_valid_replicates,
            "confidence": self.confidence, "interval_method": "percentile_linear_interpolation_valid_replicates_only",
            "ap_a": self.ap_a, "ap_b": self.ap_b, "difference": self.difference,
            "interval": None if self.interval is None else list(self.interval),
            "sample": {"records": self.records, "groups": self.groups, "mixed_label_groups": self.mixed_label_groups,
                       "positive_records": self.positive_records, "groups_with_positive": self.groups_with_positive},
            "scores_are_uncalibrated": True,
            "scope": "synthetic-generator sampling uncertainty only; not a real-world detection claim",
        }
        if self.replicate_differences is not None:
            out["replicate_differences"] = list(self.replicate_differences)
        return out


def _check_ids(name: str, mapping: object) -> dict[str, Any]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise DetectionMetricsError(f"{name} must be a non-empty mapping keyed by record id")
    for key in mapping:
        if not isinstance(key, str) or not key or key != key.strip():
            raise DetectionMetricsError(f"{name} has an empty, padded or non-string record id")
    return dict(mapping)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def paired_group_bootstrap(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    labels: Mapping[str, int],
    groups: Mapping[str, str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    keep_replicates: bool = False,
) -> BootstrapResult:
    """Paired template-group bootstrap of ``AP(B) - AP(A)``; every mapping is keyed by opaque record id and the id sets must match."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise DetectionMetricsError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DetectionMetricsError("seed must be an integer")
    if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real) or not 0 < confidence < 1:
        raise DetectionMetricsError("confidence must be strictly between 0 and 1")

    maps = {"scores_a": _check_ids("scores_a", scores_a), "scores_b": _check_ids("scores_b", scores_b),
            "labels": _check_ids("labels", labels), "groups": _check_ids("groups", groups)}
    id_set = set(maps["labels"])
    for name, mapping in maps.items():
        if set(mapping) != id_set:
            missing, extra = len(id_set - set(mapping)), len(set(mapping) - id_set)
            raise DetectionMetricsError(f"{name} does not pair with labels by record id ({missing} missing, {extra} extra)")
    ids = sorted(id_set)
    a = {i: _check_score(maps["scores_a"][i], "scores_a") for i in ids}
    b = {i: _check_score(maps["scores_b"][i], "scores_b") for i in ids}
    y = {i: _check_label(maps["labels"][i], "labels") for i in ids}
    group_of: dict[str, str] = {}
    for i in ids:
        g = maps["groups"][i]
        if not isinstance(g, str) or not g or g != g.strip():
            raise DetectionMetricsError("groups holds an empty, padded or non-string group id")
        group_of[i] = g
    group_ids = sorted(set(group_of.values()))
    if len(group_ids) < 2:
        raise DetectionMetricsError("a group bootstrap needs at least 2 groups")
    index = {g: k for k, g in enumerate(group_ids)}
    n_groups = len(group_ids)
    member_group = {i: index[group_of[i]] for i in ids}

    labels_by_group: dict[int, set[int]] = {}
    for i in ids:
        labels_by_group.setdefault(member_group[i], set()).add(y[i])
    mixed = sum(1 for s in labels_by_group.values() if len(s) == 2)
    groups_with_positive = sum(1 for s in labels_by_group.values() if 1 in s)
    positives = sum(y.values())

    def arm_blocks(score: Mapping[str, float]) -> list[list[tuple[int, int]]]:
        by_score: dict[float, list[tuple[int, int]]] = {}
        for i in ids:
            by_score.setdefault(score[i] + 0.0, []).append((member_group[i], y[i]))
        return [by_score[s] for s in sorted(by_score, reverse=True)]

    blocks_a, blocks_b = arm_blocks(a), arm_blocks(b)

    def weighted_ap(blocks: Sequence[Sequence[tuple[int, int]]], weight: Sequence[int]) -> float | None:
        counts = []
        for members in blocks:
            pos = sum(weight[g] for g, label in members if label == 1)
            counts.append((pos, sum(weight[g] for g, _ in members) - pos))
        return _ap_from_blocks(counts)

    full = [1] * n_groups
    ap_a, ap_b = weighted_ap(blocks_a, full), weighted_ap(blocks_b, full)
    difference = None if ap_a is None or ap_b is None else ap_b - ap_a
    min_valid = -(-replicates * MIN_VALID_PERCENT // 100)
    common = dict(statistic="AP(B) - AP(A)", seed=seed, replicates_requested=replicates, min_valid_replicates=min_valid,
                  confidence=float(confidence), ap_a=ap_a, ap_b=ap_b, difference=difference, records=len(ids), groups=n_groups,
                  mixed_label_groups=mixed, positive_records=positives, groups_with_positive=groups_with_positive)
    if difference is None:
        return BootstrapResult(status=STATUS_UNDEFINED_POINT, replicates_valid=0, replicates_undefined=0, undefined_reasons={},
                               interval=None, replicate_differences=() if keep_replicates else None, **common)

    rng = random.Random(seed)  # local generator: global random state is neither read nor changed
    valid: list[float] = []
    kept: list[float | None] = []
    undefined = Counter()
    for _ in range(replicates):
        weight = [0] * n_groups
        for _ in range(n_groups):
            weight[rng.randrange(n_groups)] += 1
        ra, rb = weighted_ap(blocks_a, weight), weighted_ap(blocks_b, weight)
        if ra is None or rb is None:  # shared labels make this a property of the resample, not of an arm
            undefined[REASON_NO_POSITIVE] += 1
            kept.append(None)
            continue
        valid.append(rb - ra)
        kept.append(rb - ra)
    n_valid = len(valid)
    status, interval = STATUS_INSUFFICIENT, None
    if n_valid >= min_valid:
        ordered = sorted(valid)
        alpha = round((1 - confidence) / 2, 12)
        status, interval = STATUS_OK, (_percentile(ordered, alpha), _percentile(ordered, 1 - alpha))
    return BootstrapResult(status=status, replicates_valid=n_valid, replicates_undefined=replicates - n_valid,
                           undefined_reasons=dict(undefined), interval=interval,
                           replicate_differences=tuple(kept) if keep_replicates else None, **common)
