"""``budget-rule-1``: a shared, training-only ``max_length`` for Arms A and B, plus post-selection exposure reporting.

Evaluation infrastructure only (draft protocol heldout-eval-1.2.0-draft, section 12.8). Nothing here trains or scores a
model, reads a label, or touches a validation or test sequence when choosing the budget.

What a "length" is
------------------
The length of a request is ``len(TokenSequence.tokens)`` exactly as the tokenizer emits it and as ``Tensorizer`` receives it
(``Vocabulary.encode`` maps tokens to ids one-to-one). It therefore already contains ``[SEQ_START]``/``[SEQ_END]``, the method,
path and header tokens, the payload tokens, and the complete rules segment: ``[RULES_START] [RULES_END]`` even when a request has no
rules (Arm A), plus 20 tokens per simulated rule (Arm B). No framing overhead is estimated or added.

What the real tensorizer does with a sequence longer than ``max_length``
-----------------------------------------------------------------------
It never cuts framing or rules. It drops whole payload fields to fit (``payload truncated``); if the framing plus the complete rules
segment cannot fit even with an empty payload it raises ``ValueError`` (the request is *rejected*). ``tensorizer_outcome`` reports
``fits`` / ``truncated`` / ``rejected`` by running that same logic.

The rule (``budget-rule-1``)
----------------------------
Input: one length per **training-split** Arm B sequence, keyed by opaque record id. Output: the smallest value of the candidate grid
(128, 192, 256, 384, 512) that is greater than or equal to the largest of those lengths (a length equal to the candidate fits).
Arm B is used because rule tokens make it the longer arm. If no candidate fits, the decision has ``status="no_candidate_fits"`` and
no ``selected_max_length``: nothing is truncated or chosen silently, and the rule must be revised under a new protocol version.
The same value is then used for Arm A and Arm B. The decision is independent of input order.

Statistics (explicit definitions): ``mean`` = sum / n rounded to 6 decimals; ``median`` = middle value (mean of the two middle
values for even n); ``p95`` / ``p99`` = nearest rank, the value at 1-based rank ``ceil(p * n)`` of the sorted lengths (always an
observed length).

Exposure reporting (``exposure_report``) takes a *fixed* decision and the lengths of any split and arm. It never changes the budget,
excludes or reorders nothing, and returns explicit findings when a validation or test record exceeds the frozen budget.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..models import TokenSequence
from ..tensorizer import Tensorizer
from ..vocabulary import Vocabulary

RULE_ID = "budget-rule-1"
CANDIDATE_GRID: tuple[int, ...] = (128, 192, 256, 384, 512)
STATUS_SELECTED = "selected"
STATUS_NO_CANDIDATE = "no_candidate_fits"
SPLITS = ("train", "validation", "test")
_MAX_LISTED_IDS = 20


class BudgetInputError(ValueError):
    """The lengths, grid or decision supplied are invalid. Messages never echo record content."""


class BudgetNotFeasibleError(RuntimeError):
    """No candidate in the grid fits every training sequence."""


@dataclass(frozen=True)
class BudgetDecision:
    """Result of ``select_shared_max_length``. JSON-serializable through ``to_dict``."""

    status: str
    selected_max_length: int | None
    rule_id: str
    candidate_grid: tuple[int, ...]
    train_records: int
    min: int
    mean: float
    median: float
    p95: int
    p99: int
    max: int
    headroom: int | None
    records_over_candidate: Mapping[str, int]
    lengths_sha256: str
    basis: str = "arm_b_training_split_token_sequence_lengths"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "selected_max_length": self.selected_max_length, "rule_id": self.rule_id,
            "candidate_grid": list(self.candidate_grid), "basis": self.basis, "train_records": self.train_records,
            "length_statistics": {"min": self.min, "mean": self.mean, "median": self.median, "p95": self.p95,
                                  "p99": self.p99, "max": self.max},
            "headroom": self.headroom,
            "records_over_candidate": dict(self.records_over_candidate),
            "lengths_sha256": self.lengths_sha256,
        }

    def require_selected(self) -> int:
        """The selected ``max_length``, or :class:`BudgetNotFeasibleError` if no candidate fits."""
        if self.status != STATUS_SELECTED or self.selected_max_length is None:
            raise BudgetNotFeasibleError(
                f"{self.rule_id}: no candidate in {list(self.candidate_grid)} fits the longest training sequence ({self.max})")
        return self.selected_max_length


# -- validation helpers -----------------------------------------------------------------------------------------------


def _check_grid(grid: object) -> tuple[int, ...]:
    if not isinstance(grid, (tuple, list)) or not grid:
        raise BudgetInputError("the candidate grid must be a non-empty tuple of integers")
    for g in grid:
        if isinstance(g, bool) or not isinstance(g, int) or g < 1:
            raise BudgetInputError("candidate values must be positive integers")
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise BudgetInputError("the candidate grid must be strictly ascending")
    return tuple(grid)


def _as_lengths(lengths: Mapping[str, int] | Iterable[tuple[str, int]], what: str) -> dict[str, int]:
    pairs = lengths.items() if isinstance(lengths, Mapping) else lengths
    out: dict[str, int] = {}
    try:
        for rid, n in pairs:
            if not isinstance(rid, str) or not rid or rid != rid.strip():
                raise BudgetInputError(f"{what}: record ids must be non-empty strings without surrounding whitespace")
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise BudgetInputError(f"{what}: lengths must be integers of at least 1")
            if rid in out:
                raise BudgetInputError(f"{what}: duplicate record id")
            out[rid] = n
    except (TypeError, ValueError) as exc:
        if isinstance(exc, BudgetInputError):
            raise
        raise BudgetInputError(f"{what}: expected a mapping or pairs of (record id, length)") from exc
    return out


def _nearest_rank(sorted_values: list[int], p: float) -> int:
    return sorted_values[max(1, math.ceil(p * len(sorted_values))) - 1]


def _median(sorted_values: list[int]) -> float:
    n = len(sorted_values)
    mid = n // 2
    return float(sorted_values[mid]) if n % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2


def lengths_fingerprint(lengths: Mapping[str, int]) -> str:
    """SHA-256 of the canonical, id-sorted ``[[record_id, length], ...]`` JSON (compact separators, ASCII)."""
    body = json.dumps([[rid, lengths[rid]] for rid in sorted(lengths)], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# -- selection ---------------------------------------------------------------------------------------------------------


def select_shared_max_length(
    train_arm_b_lengths: Mapping[str, int] | Iterable[tuple[str, int]],
    grid: tuple[int, ...] = CANDIDATE_GRID,
) -> BudgetDecision:
    """``budget-rule-1``: the smallest grid value that fits every training-split Arm B sequence (see module docstring).

    Pass **only** training-split Arm B lengths. The function cannot see validation or test data, labels or any metadata.
    """
    candidates = _check_grid(grid)
    lengths = _as_lengths(train_arm_b_lengths, "training lengths")
    if not lengths:
        raise BudgetInputError("training lengths must not be empty")
    values = sorted(lengths.values())
    n, longest = len(values), values[-1]
    chosen = next((c for c in candidates if c >= longest), None)
    return BudgetDecision(
        status=STATUS_SELECTED if chosen is not None else STATUS_NO_CANDIDATE,
        selected_max_length=chosen, rule_id=RULE_ID, candidate_grid=candidates, train_records=n,
        min=values[0], mean=round(sum(values) / n, 6), median=_median(values),
        p95=_nearest_rank(values, 0.95), p99=_nearest_rank(values, 0.99), max=longest,
        headroom=None if chosen is None else chosen - longest,
        records_over_candidate={str(c): sum(1 for v in values if v > c) for c in candidates},
        lengths_sha256=lengths_fingerprint(lengths),
    )


# -- real tensorizer semantics -----------------------------------------------------------------------------------------


def sequence_lengths(sequences: Mapping[str, TokenSequence]) -> dict[str, int]:
    """``record id -> len(tokens)`` for token sequences exactly as they would be passed to ``Tensorizer``."""
    return {rid: len(seq.tokens) for rid, seq in sequences.items()}


def tensorizer_outcome(sequence: TokenSequence, max_length: int) -> str:
    """``fits``, ``truncated`` (payload fields dropped) or ``rejected`` (framing plus rules cannot fit), as the real tensorizer decides.

    Uses the tensorizer's own payload-truncation routine (which needs no fitted vocabulary); it does not modify the sequence.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise BudgetInputError("max_length must be a positive integer")
    if len(sequence.tokens) <= max_length:
        return "fits"
    try:
        Tensorizer(Vocabulary(), max_length=max_length)._truncate_payload(sequence)
    except ValueError:
        return "rejected"
    return "truncated"


def tensorizer_outcomes(sequences: Mapping[str, TokenSequence], max_length: int) -> dict[str, str]:
    return {rid: tensorizer_outcome(seq, max_length) for rid, seq in sequences.items()}


# -- post-selection exposure --------------------------------------------------------------------------------------------


def exposure_report(
    decision: BudgetDecision,
    split: str,
    arm_a_lengths: Mapping[str, int] | Iterable[tuple[str, int]],
    arm_b_lengths: Mapping[str, int] | Iterable[tuple[str, int]],
    *,
    arm_a_outcomes: Mapping[str, str] | None = None,
    arm_b_outcomes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """How the lengths of one split and both arms sit against a **fixed** budget. Pure; changes nothing.

    ``arm_*_outcomes`` (from ``tensorizer_outcomes``) are optional; when given they add the real tensorizer's
    ``fits``/``truncated``/``rejected`` counts. No record is excluded, reordered or capped here: a record the tensorizer would
    reject is only reported, and removing it from all arms (protocol section 3) is a later, explicit step.
    """
    if not isinstance(decision, BudgetDecision) or decision.status != STATUS_SELECTED or decision.selected_max_length is None:
        raise BudgetInputError("exposure_report needs a decision with a selected max_length")
    if split not in SPLITS:
        raise BudgetInputError(f"split must be one of {SPLITS}")
    budget = decision.selected_max_length
    a = _as_lengths(arm_a_lengths, "arm A lengths")
    b = _as_lengths(arm_b_lengths, "arm B lengths")
    ids_a, ids_b = set(a), set(b)
    paired = sorted(ids_a & ids_b)
    missing_in_a, missing_in_b = sorted(ids_b - ids_a), sorted(ids_a - ids_b)

    def arm(lengths: dict[str, int], outcomes: Mapping[str, str] | None) -> dict[str, Any]:
        over = sorted(rid for rid, n in lengths.items() if n > budget)
        entry: dict[str, Any] = {
            "records": len(lengths), "over_budget": len(over),
            "over_budget_share": round(len(over) / len(lengths), 6) if lengths else 0.0,
            "longest": max(lengths.values()) if lengths else None,
            "largest_excess": max((n - budget for n in lengths.values()), default=0) if over else 0,
            "truncation_risk": bool(over), "over_budget_ids": over[:_MAX_LISTED_IDS],
        }
        if outcomes is not None:
            unknown = set(outcomes) ^ set(lengths)
            if unknown:
                raise BudgetInputError("tensorizer outcomes must cover exactly the supplied records")
            bad = [v for v in outcomes.values() if v not in ("fits", "truncated", "rejected")]
            if bad:
                raise BudgetInputError("unknown tensorizer outcome")
            entry["tensorizer"] = {k: sum(1 for v in outcomes.values() if v == k) for k in ("fits", "truncated", "rejected")}
            entry["rejected_ids"] = sorted(r for r, v in outcomes.items() if v == "rejected")[:_MAX_LISTED_IDS]
        return entry

    arm_a, arm_b = arm(a, arm_a_outcomes), arm(b, arm_b_outcomes)
    over_a = {r for r in paired if a[r] > budget}
    over_b = {r for r in paired if b[r] > budget}
    diffs = [b[r] - a[r] for r in paired]
    a_longer = sorted(r for r in paired if a[r] > b[r])
    report: dict[str, Any] = {
        "rule_id": decision.rule_id, "split": split, "max_length": budget, "lengths_sha256_of_selection": decision.lengths_sha256,
        "budget_changed_by_this_report": False, "records_excluded_by_this_report": 0,
        "pairing": {
            "arm_a_records": len(a), "arm_b_records": len(b), "paired": len(paired),
            "missing_in_arm_a": len(missing_in_a), "missing_in_arm_b": len(missing_in_b),
            "missing_in_arm_a_ids": missing_in_a[:_MAX_LISTED_IDS], "missing_in_arm_b_ids": missing_in_b[:_MAX_LISTED_IDS],
        },
        "arm_a": arm_a, "arm_b": arm_b,
        "paired_exposure": {
            "over_budget_in_either_arm": len(over_a | over_b), "over_budget_only_in_arm_b": len(over_b - over_a),
            "over_budget_only_in_arm_a": len(over_a - over_b), "over_budget_in_both": len(over_a & over_b),
            "mean_length_difference_b_minus_a": round(sum(diffs) / len(diffs), 6) if diffs else None,
            "max_length_difference_b_minus_a": max(diffs) if diffs else None,
            "records_where_arm_a_longer_than_arm_b": len(a_longer), "arm_a_longer_ids": a_longer[:_MAX_LISTED_IDS],
        },
    }
    findings: list[str] = []
    if missing_in_a or missing_in_b:
        findings.append("paired-record mismatch: some records are present in only one arm")
    if a_longer:
        findings.append("unexpected: Arm A is longer than Arm B for some paired records")
    if split == "train":
        if arm_b["over_budget"]:
            findings.append("unexpected: an Arm B training sequence exceeds the budget selected from the training split")
        if arm_a["over_budget"]:
            findings.append("unexpected: an Arm A training sequence exceeds the shared budget (the rule is based on Arm B only)")
    elif arm_a["over_budget"] or arm_b["over_budget"]:
        findings.append(f"{split} records exceed the frozen budget: payload truncation applies; protocol review required, "
                        "the budget is not changed and no record is discarded")
    rejected = (arm_a.get("tensorizer", {}).get("rejected", 0), arm_b.get("tensorizer", {}).get("rejected", 0))
    if any(rejected):
        findings.append("records the tensorizer would reject exist; protocol section 3 requires removing them from all arms, "
                        "which is not done here")
    report["findings"] = findings
    report["protocol_review_required"] = bool(findings)
    return report
