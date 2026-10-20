"""``runner-foundation-1``: the fail-closed orchestration foundation of the held-out A/B/C evaluation.

Draft protocol heldout-eval-1.2.0-draft, sections 12.11.2-12.11.4 and 12.12. **This module fits no detection model, scores nothing and reads no
test label.** It connects existing components (splits, arms, token budget, numerics, detection metrics) through small, typed, validating steps:

A. :func:`partition_records` validates the dataset and split metadata and separates the data into :class:`TrainData`, :class:`ValidationData`
   (the only two parts that keep labels) and a label-free :class:`SealedTestSet`.
B. :func:`aligned_labels` maps labels to the verified three-arm record order **by record id**, never by list position.
C. :func:`prepare_train_configuration` prepares the training-only vocabulary and shared token budget (``arms-1``, ``budget-rule-1``).
D. :func:`preflight_split` / :func:`preflight_report` check, label-free, that the frozen budget represents every Arm A/B request in full; any outcome
   other than ``fits`` fails closed (:class:`RepresentabilityError`). Nothing is excluded, truncated or re-budgeted.
E. :func:`prepare_numerical_configuration` derives the training-only eigenvalue estimates, the arm-relative learning rates and the 27 candidates
   per arm (``lambda-power-1``).
F. :func:`select_candidate` is a pure validation-only selection over externally supplied, already aligned results.
G. :func:`request_training_authorization` and :func:`request_test_access` are the explicit **unavailable** states of the gates that need the future
   manifest, selection-record and attempt-marker modules. There is no permissive path.

Split isolation. Train records and labels are used for fitting only, validation records and labels for selection only. The test split is held as
*blinded* records: the runner keeps the request content (for the label-free preflight), the record ids and bindings to the dataset split, but **drops
the labels at partition time**, so no selection interface can reach a test label, and no test score exists. Blinded records carry the sentinel
``label=-1`` because the ``EvalRecord`` type requires a value; it is *not* a class, every label consumer rejects it (``aligned_labels``,
``TrainData``, ``ValidationData``, model training) and the arm builders never read it (tested), so it can never be mistaken for ground truth.

Where the authentic test labels live. The runner retains none. They remain only in the caller's source dataset (``EvalRecord.meta.label``, rebuilt
deterministically from the recorded generator seed, whose fingerprint and split manifest hash go into the pre-run manifest). The future one-time test gate
(not implemented) must recover them from that source by record id **after** both records verify and the attempt marker exists, check them against
``SealedTestSet.binding()`` (record count, id-list hash, group-list hash, split manifest hash), and align them with ``aligned_labels``.

Test scoring inputs are never encoded here. The test preflight builds only transient, vocabulary-free Arm A and Arm B token sequences to read their
lengths and tensorizer outcomes; it builds no ``ModelInput`` (so no Arm C encoding), keeps nothing and returns only counts and ids.

Not implemented (pending): the pre-run manifest and selection record writers and verifiers, the persistent one-time attempt marker, model fitting,
threshold selection on real scores, test unsealing and scoring. The protocol remains a draft.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from ..fusion import InputFusion
from ..masking import PrivacyMasker
from ..models import TokenSequence, VerifiedRuleInfo
from ..normalization import Normalizer
from ..tokenizer import HybridTokenizer
from . import arms as _arms
from .arms import (
    ARM_A,
    ARM_B,
    ARMS,
    MIN_FREQUENCY_PROPOSED,
    ArmInputs,
    FrozenVocabulary,
    PreparedSplit,
    build_arm_inputs,
    fit_shared_vocabulary,
    prepare_split,
    training_arm_b_lengths,
)
from .numerics import (
    NUM_BUCKETS_GRID,
    RELATIVE_STEP_GRID,
    LambdaEstimate,
    estimate_from_arm_inputs,
    learning_rate_table,
)
from .policy_sim import simulate_rules_for_record
from .splits import SPLIT_NAMES, EvalRecord, SplitMetadata, SplitResult, verify_split_isolation
from .token_budget import BudgetDecision, select_shared_max_length, tensorizer_outcome

RUNNER_ID = "runner-foundation-1"
EPOCHS_GRID: tuple[int, ...] = (50, 150, 400)
MIN_FREQUENCY = MIN_FREQUENCY_PROPOSED
MAX_LITERAL_VOCAB_SIZE: int | None = None
AP_ROUND_DECIMALS = 10
TEST_SCORING_AVAILABLE = False
TRAINING_AUTHORIZATION_AVAILABLE = False
_BLINDED_LABEL = -1  # sentinel required by the EvalRecord type; NOT a class: every label consumer rejects it, the arm builders never read it
_MAX_LISTED_IDS = 20
_BUDGETED_ARMS = (ARM_A, ARM_B)


class RunnerError(ValueError):
    """Invalid dataset, split, alignment or selection input. Messages hold counts and categories, never request content or labels."""


class RepresentabilityError(RuntimeError):
    """An Arm A or B request is not represented in full by the frozen budget. Stop for protocol review; nothing was changed."""

    def __init__(self, message: str, report: "RepresentabilityReport") -> None:
        super().__init__(message)
        self.report = report


class AuthorizationUnavailable(RuntimeError):
    """A gate that needs the future manifest, selection-record or attempt-marker modules. Always raised in this phase."""

    def __init__(self, action: str, missing: Sequence[str]) -> None:
        super().__init__(f"{action} is unavailable: {'; '.join(missing)}")
        self.action = action
        self.missing = tuple(missing)


# -- A. dataset and split validation ---------------------------------------------------------------------------------------------


def ids_fingerprint(record_ids: Sequence[str]) -> str:
    """SHA-256 of the sorted record-id list (compact ASCII JSON)."""
    return hashlib.sha256(json.dumps(sorted(record_ids), separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _blind(record: EvalRecord) -> EvalRecord:
    return EvalRecord(record_id=record.record_id, features=record.features, meta=SplitMetadata(label=_BLINDED_LABEL))


def _require_real_labels(records: Sequence[EvalRecord], what: str) -> None:
    if any(isinstance(r.meta.label, bool) or r.meta.label not in (0, 1) for r in records):
        raise RunnerError(f"{what} needs real 0/1 labels; blinded or invalid records are refused")


@dataclass(frozen=True)
class TrainData:
    """Training records with labels: for fitting only."""

    records: tuple[EvalRecord, ...]

    def __post_init__(self) -> None:
        _require_real_labels(self.records, "TrainData")

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(r.record_id for r in self.records)

    def labels_by_id(self) -> dict[str, int]:
        return {r.record_id: r.meta.label for r in self.records}


@dataclass(frozen=True)
class ValidationData:
    """Validation records with labels: for model and threshold selection only."""

    records: tuple[EvalRecord, ...]

    def __post_init__(self) -> None:
        _require_real_labels(self.records, "ValidationData")

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(r.record_id for r in self.records)

    def labels_by_id(self) -> dict[str, int]:
        return {r.record_id: r.meta.label for r in self.records}


class SealedTestSet:
    """The test split without labels: blinded request records, record ids and group ids only.

    The labels were dropped when this object was created and no method returns one. It offers nothing that scores, encodes or unseals the split.
    ``binding()`` ties the set to its dataset split so that the future gate can check recovered labels against it.
    """

    def __init__(self, blinded: Sequence[EvalRecord], group_by_id: Mapping[str, str], *, split_manifest_sha256: str) -> None:
        self._records = tuple(sorted(blinded, key=lambda r: r.record_id))
        self._groups = dict(group_by_id)
        if any(r.meta.label != _BLINDED_LABEL for r in self._records) or set(self._groups) != {r.record_id for r in self._records}:
            raise RunnerError("a sealed test set needs blinded records and one group id per record")
        if not isinstance(split_manifest_sha256, str):
            raise RunnerError("the split manifest hash must be a string")
        self._split_manifest_sha256 = split_manifest_sha256

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"SealedTestSet(records={len(self._records)})"

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(r.record_id for r in self._records)

    @property
    def group_by_id(self) -> dict[str, str]:
        return dict(self._groups)

    def blinded_records(self) -> tuple[EvalRecord, ...]:
        """Request content for the label-free representability preflight. The ``label`` field is the sentinel -1, not a class."""
        return self._records

    def binding(self) -> dict[str, Any]:
        """Hashes and counts that bind this set to its dataset split (no label, no request content)."""
        groups = sorted(set(self._groups.values()))
        return {"records": len(self._records), "ids_sha256": ids_fingerprint(self.record_ids),
                "groups_sha256": ids_fingerprint(groups), "split_manifest_sha256": self._split_manifest_sha256}


@dataclass(frozen=True)
class Partition:
    train: TrainData
    validation: ValidationData
    sealed_test: SealedTestSet
    group_by_id: Mapping[str, str]  # all splits; group ids are not labels
    summary: Mapping[str, Any]


def partition_records(records: Sequence[EvalRecord], split_result: SplitResult, *, verify_isolation: bool = True) -> Partition:
    """Validate the dataset against its split and separate train, validation and the blinded test split. Fails closed on any irregularity."""
    if isinstance(records, (str, bytes, Mapping)):
        raise RunnerError("records must be a sequence of EvalRecord")
    items = list(records)
    if not items or not all(isinstance(r, EvalRecord) for r in items):
        raise RunnerError("records must be a non-empty sequence of EvalRecord")
    ids = [r.record_id for r in items]
    if len(set(ids)) != len(ids):
        raise RunnerError("duplicate record ids")
    if not isinstance(split_result, SplitResult):
        raise RunnerError("a SplitResult is required")
    if not split_result.report.meets_requirements:
        raise RunnerError("the split does not meet its requirements; no evaluation on an infeasible split")
    if split_result.excluded:
        raise RunnerError(f"the splitter excluded {len(split_result.excluded)} record(s); the runner does not accept silent exclusions")
    id_set = set(ids)
    if set(split_result.assignments) != id_set:
        missing = len(id_set - set(split_result.assignments))
        extra = len(set(split_result.assignments) - id_set)
        raise RunnerError(f"records and split assignments differ ({missing} unassigned, {extra} unknown)")
    if set(split_result.group_of) != id_set:
        raise RunnerError("every record needs exactly one group id")
    if any(name not in SPLIT_NAMES for name in split_result.assignments.values()):
        raise RunnerError("an assignment names an unknown split")
    group_splits: dict[str, set[str]] = {}
    for rid, group in split_result.group_of.items():
        group_splits.setdefault(group, set()).add(split_result.assignments[rid])
    if any(len(s) > 1 for s in group_splits.values()):
        raise RunnerError("a template group spans more than one split")
    if verify_isolation:
        problems = verify_split_isolation(split_result, items)
        if problems:
            raise RunnerError(f"split isolation violated ({len(problems)} problem(s))")

    by_split: dict[str, list[EvalRecord]] = {name: [] for name in SPLIT_NAMES}
    for rec in items:
        by_split[split_result.assignments[rec.record_id]].append(rec)
    if any(not by_split[name] for name in SPLIT_NAMES):
        raise RunnerError("every split needs at least one record")
    train = TrainData(tuple(sorted(by_split["train"], key=lambda r: r.record_id)))
    validation = ValidationData(tuple(sorted(by_split["validation"], key=lambda r: r.record_id)))
    for name, part in (("train", train), ("validation", validation)):
        labels = {r.meta.label for r in part.records}
        if labels != {0, 1}:
            raise RunnerError(f"the {name} split needs both classes")
    test_ids = {r.record_id for r in by_split["test"]}  # the labels of these records are not carried into any returned structure
    sealed = SealedTestSet([_blind(r) for r in by_split["test"]], {rid: split_result.group_of[rid] for rid in test_ids},
                           split_manifest_sha256=split_result.manifest_sha256)
    summary = {
        "runner": RUNNER_ID, "split_manifest_sha256": split_result.manifest_sha256,
        "train": {"records": len(train.records), "groups": len({split_result.group_of[r] for r in train.record_ids}),
                  "positives": sum(r.meta.label for r in train.records)},
        "validation": {"records": len(validation.records), "groups": len({split_result.group_of[r] for r in validation.record_ids}),
                       "positives": sum(r.meta.label for r in validation.records)},
        "test": {"records": len(sealed), "groups": len(set(sealed.group_by_id.values())), "labels": "sealed_not_retained_by_runner",
                 "ids_sha256": sealed.binding()["ids_sha256"]},
    }
    return Partition(train=train, validation=validation, sealed_test=sealed, group_by_id=dict(split_result.group_of), summary=summary)


# -- B. paired record alignment -----------------------------------------------------------------------------------------------------


def aligned_labels(record_ids: Sequence[str], labels_by_id: Mapping[str, int]) -> tuple[int, ...]:
    """Labels in the order of ``record_ids`` (the verified three-arm order), looked up **by id**. Missing, extra or invalid labels raise."""
    wanted = list(record_ids)
    if len(set(wanted)) != len(wanted):
        raise RunnerError("duplicate record ids")
    if set(labels_by_id) != set(wanted):
        missing = len(set(wanted) - set(labels_by_id))
        extra = len(set(labels_by_id) - set(wanted))
        raise RunnerError(f"labels do not match the record ids ({missing} missing, {extra} extra)")
    out = []
    for rid in wanted:
        label = labels_by_id[rid]
        if isinstance(label, bool) or not isinstance(label, numbers.Integral) or int(label) not in (0, 1):
            raise RunnerError("a label is not the integer 0 or 1")
        out.append(int(label))
    return tuple(out)


# -- C. train-only preparation ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfiguration:
    """Vocabulary, shared budget and training inputs, all derived from the training split only."""

    prepared: PreparedSplit
    frozen: FrozenVocabulary
    decision: BudgetDecision
    inputs: ArmInputs

    def to_dict(self) -> dict[str, Any]:
        return {"runner": RUNNER_ID, "derived_from": "train_split_only", "train_records": len(self.inputs.record_ids),
                "budget": self.decision.to_dict(), "vocabulary": dict(self.frozen.record),
                "train_ids_sha256": ids_fingerprint(self.inputs.record_ids), "min_frequency": MIN_FREQUENCY,
                "max_literal_vocab_size": MAX_LITERAL_VOCAB_SIZE, "summary": self.inputs.summary()}


def prepare_train_configuration(train: TrainData) -> TrainConfiguration:
    """Shared ``max_length`` (``budget-rule-1``) and the A/B vocabulary from the training records only; fits no model."""
    if not isinstance(train, TrainData):
        raise RunnerError("only TrainData is accepted")
    prepared = prepare_split(train.records, "train")
    decision = select_shared_max_length(training_arm_b_lengths(prepared))
    decision.require_selected()  # no_candidate_fits stops the run
    frozen = fit_shared_vocabulary(prepared, min_frequency=MIN_FREQUENCY, max_literal_vocab_size=MAX_LITERAL_VOCAB_SIZE)
    inputs = build_arm_inputs(prepared, frozen, decision)  # fails closed on records the budget cannot represent
    return TrainConfiguration(prepared=prepared, frozen=frozen, decision=decision, inputs=inputs)


# -- D. label-free representability preflight ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RepresentabilityReport:
    """Per-split result of the frozen-budget check. Holds record ids, arm names, counts and categories only."""

    split: str
    max_length: int
    records: int
    arm_counts: Mapping[str, Mapping[str, int]]
    failing: Mapping[str, tuple[tuple[str, tuple[str, ...]], ...]]  # category -> ((record_id, arms), ...) first ids only
    failing_totals: Mapping[str, int]

    @property
    def passed(self) -> bool:
        return not any(self.failing_totals.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split, "max_length": self.max_length, "records": self.records,
            "arm_outcome_counts": {arm: dict(c) for arm, c in self.arm_counts.items()}, "arm_c": "not_budgeted_not_applicable",
            "failing_totals": dict(self.failing_totals),
            "failing_listed": {cat: [{"record_id": rid, "arms": list(arms)} for rid, arms in rows] for cat, rows in self.failing.items()},
            "listed_cap": _MAX_LISTED_IDS, "passed": self.passed,
            "action_if_failed": "stop_affected_stage_for_protocol_review", "labels_accessed": False, "scoring_inputs_encoded": False,
            "records_excluded": 0, "budget_changed": False,
        }

    def require_pass(self) -> None:
        if not self.passed:
            raise RepresentabilityError(
                f"split '{self.split}': {self.failing_totals.get('truncated', 0)} truncated and {self.failing_totals.get('rejected', 0)} "
                f"rejected record(s) under max_length={self.max_length}; stopping for protocol review (nothing excluded, truncated or re-budgeted)",
                self)


def _selected_length(decision: BudgetDecision) -> int:
    if not isinstance(decision, BudgetDecision):
        raise RunnerError("a BudgetDecision is required")
    return decision.require_selected()


def _budget_sequences(records: Sequence[EvalRecord], split: str) -> list[tuple[str, TokenSequence, TokenSequence]]:
    """Arm A and Arm B token sequences of each request, for length and tensorizer-outcome checks only.

    Uses the arm builders' own stage flow (``arms._front_end``: ``request_to_context-v1``, fusion, masking, normalization, tokenization; Arm A with an empty
    rule list, Arm B with the checked ``simpol-1`` rules) but stops there: no vocabulary, no ``ModelInput`` and no Arm C encoding is built, nothing is
    retained and no label is read. Relies on the private ``arms._front_end`` and ``arms._check_simulated``; a test pins the result to ``prepare_split``.
    """
    items = list(records)
    if not items or not all(isinstance(r, EvalRecord) for r in items):
        raise RunnerError("records must be a non-empty sequence of EvalRecord")
    if len({r.record_id for r in items}) != len(items):
        raise RunnerError("duplicate record ids")
    stages = (InputFusion(), PrivacyMasker(), Normalizer(), HybridTokenizer())
    out = []
    for rec in sorted(items, key=lambda r: r.record_id):
        try:
            _, seq_a = _arms._front_end(rec, VerifiedRuleInfo(event_id=rec.record_id, rules=[]), stages)
            rule_set = simulate_rules_for_record(rec)
            _arms._check_simulated(rule_set)
            _, seq_b = _arms._front_end(rec, rule_set.rule_info, stages)
        except Exception as exc:  # noqa: BLE001 - the message must not carry request content
            raise RunnerError(f"split '{split}': record {rec.record_id!r} could not be tokenized ({type(exc).__name__})") from None
        out.append((rec.record_id, seq_a, seq_b))
    return out


def preflight_prepared(prepared: PreparedSplit, decision: BudgetDecision) -> RepresentabilityReport:
    """Tensorizer outcome of every Arm A and Arm B request of an already prepared split (used for the training split)."""
    return _outcome_report([(r.record_id, r.sequence_a, r.sequence_b) for r in prepared.requests], prepared.split, decision)


def _outcome_report(sequences: Sequence[tuple[str, TokenSequence, TokenSequence]], split: str, decision: BudgetDecision) -> RepresentabilityReport:
    """Tensorizer outcome of every Arm A and Arm B request under the frozen budget. Reads no label, scores nothing, encodes no ids."""
    max_length = _selected_length(decision)
    counts = {arm: {"fits": 0, "truncated": 0, "rejected": 0} for arm in _BUDGETED_ARMS}
    failing: dict[str, dict[str, list[str]]] = {"truncated": {}, "rejected": {}}
    for record_id, seq_a, seq_b in sequences:
        for arm, seq in ((ARM_A, seq_a), (ARM_B, seq_b)):
            outcome = tensorizer_outcome(seq, max_length)
            counts[arm][outcome] += 1
            if outcome != "fits":
                failing[outcome].setdefault(record_id, []).append(arm)
    listed = {cat: tuple((rid, tuple(arms)) for rid, arms in sorted(rows.items()))[:_MAX_LISTED_IDS] for cat, rows in failing.items()}
    totals = {cat: len(rows) for cat, rows in failing.items()}
    return RepresentabilityReport(split=split, max_length=max_length, records=len(sequences), arm_counts=counts,
                                  failing=listed, failing_totals=totals)


def preflight_split(records: Sequence[EvalRecord], split: str, decision: BudgetDecision) -> RepresentabilityReport:
    """Preflight for validation records or a sealed test set's blinded records: lengths and tensorizer outcomes only (see ``_budget_sequences``)."""
    if split not in ("validation", "test"):
        raise RunnerError("the preflight covers the validation and test splits")
    return _outcome_report(_budget_sequences(records, split), split, decision)


def preflight_report(config: TrainConfiguration, validation: ValidationData, sealed_test: SealedTestSet) -> dict[str, Any]:
    """Preflight of train, validation and test against the frozen training budget; raises on the first failing split after building the report."""
    reports = {
        "train": preflight_prepared(config.prepared, config.decision),
        "validation": preflight_split(validation.records, "validation", config.decision),
        "test": preflight_split(sealed_test.blinded_records(), "test", config.decision),
    }
    result = {"runner": RUNNER_ID, "max_length": config.decision.selected_max_length, "splits": {k: v.to_dict() for k, v in reports.items()},
              "passed": all(r.passed for r in reports.values())}
    for report in reports.values():
        report.require_pass()
    return result


def encode_validation(config: TrainConfiguration, validation: ValidationData) -> ArmInputs:
    """Validation model inputs with the **frozen** training vocabulary and budget (no refit, no new budget); fails closed on any non-``fits``."""
    if not isinstance(config, TrainConfiguration) or not isinstance(validation, ValidationData):
        raise RunnerError("a TrainConfiguration and ValidationData are required")
    prepared = prepare_split(validation.records, "validation")
    preflight_prepared(prepared, config.decision).require_pass()
    return build_arm_inputs(prepared, config.frozen, config.decision)


# -- E. train-only numerical configuration ---------------------------------------------------------------------------------------------------


def canonical_candidate_id(arm: str, num_buckets: int, epochs: int, relative_step_c: float) -> str:
    return f"{arm}|nb={num_buckets:05d}|ep={epochs:04d}|c={relative_step_c:.2f}"


@dataclass(frozen=True)
class CandidateSpec:
    arm: str
    num_buckets: int
    epochs: int
    relative_step_c: float
    learning_rate: float
    candidate_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "arm": self.arm, "num_buckets": self.num_buckets, "epochs": self.epochs,
                "relative_step_c": self.relative_step_c, "learning_rate": self.learning_rate, "learning_rate_hex": self.learning_rate.hex()}


@dataclass(frozen=True)
class NumericalConfiguration:
    estimates: tuple[LambdaEstimate, ...]
    learning_rates: Mapping[str, Any]
    candidates: tuple[CandidateSpec, ...]

    def candidates_for(self, arm: str) -> tuple[CandidateSpec, ...]:
        return tuple(c for c in self.candidates if c.arm == arm)

    def to_dict(self) -> dict[str, Any]:
        return {"runner": RUNNER_ID, "derived_from": "train_split_only", "estimates": [e.to_dict() for e in self.estimates],
                "learning_rates": dict(self.learning_rates), "candidates_per_arm": len(self.candidates) // max(len(ARMS), 1),
                "candidates": [c.to_dict() for c in self.candidates]}


def build_candidates(lr_table: Mapping[str, Any]) -> tuple[CandidateSpec, ...]:
    """The 27 candidates per arm (``num_buckets`` x ``epochs`` x ``c``) from a ``learning_rate_table``; sorted by candidate id."""
    out = []
    for cell in lr_table["cells"]:
        for row in cell["rates"]:
            for epochs in EPOCHS_GRID:
                c = row["relative_step_c"]
                out.append(CandidateSpec(cell["arm"], cell["num_buckets"], epochs, c, row["learning_rate"],
                                         canonical_candidate_id(cell["arm"], cell["num_buckets"], epochs, c)))
    ids = [c.candidate_id for c in out]
    if len(set(ids)) != len(ids):
        raise RunnerError("duplicate candidate identifiers")
    return tuple(sorted(out, key=lambda c: c.candidate_id))


def prepare_numerical_configuration(config: TrainConfiguration) -> NumericalConfiguration:
    """Training-only eigenvalue estimates, learning rates and candidates for every arm and ``num_buckets``. Fits no model."""
    if not isinstance(config, TrainConfiguration):
        raise RunnerError("a TrainConfiguration is required")
    estimates = tuple(estimate_from_arm_inputs(config.inputs, arm=arm, num_buckets=nb) for arm in ARMS for nb in NUM_BUCKETS_GRID)
    table = learning_rate_table(estimates)
    return NumericalConfiguration(estimates=estimates, learning_rates=table, candidates=build_candidates(table))


def approved_candidate_ids(arm: str) -> tuple[str, ...]:
    """The 27 canonical candidate ids of one arm (no learning rates needed)."""
    if arm not in ARMS:
        raise RunnerError(f"arm must be one of {ARMS}")
    return tuple(sorted(canonical_candidate_id(arm, nb, ep, c) for nb in NUM_BUCKETS_GRID for ep in EPOCHS_GRID for c in RELATIVE_STEP_GRID))


# -- F. validation-only selection --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationCandidateResult:
    """One candidate's validation Average Precision, supplied by the (future) fitting stage."""

    candidate_id: str
    arm: str
    num_buckets: int
    epochs: int
    relative_step_c: float
    validation_ap: float | None
    split: str
    validation_ids_sha256: str
    validation_records: int


@dataclass(frozen=True)
class SelectionDecision:
    arm: str
    selected: ValidationCandidateResult
    rounded_ap: float
    tied_at_rounded_ap: int
    ranking: tuple[Mapping[str, Any], ...]
    rule: str = "max_validation_ap_rounded_10_then_fewer_epochs_smaller_buckets_smaller_c_then_candidate_id"

    def to_dict(self) -> dict[str, Any]:
        return {"runner": RUNNER_ID, "arm": self.arm, "rule": self.rule, "ap_round_decimals": AP_ROUND_DECIMALS,
                "selected_candidate_id": self.selected.candidate_id, "selected_validation_ap": self.selected.validation_ap,
                "selected_rounded_ap": self.rounded_ap, "candidates_tied_at_rounded_ap": self.tied_at_rounded_ap,
                "candidates": len(self.ranking), "ranking": [dict(r) for r in self.ranking],
                "basis": "validation_split_only", "test_data_used": False}


def _check_result(result: object, arm: str) -> ValidationCandidateResult:
    if not isinstance(result, ValidationCandidateResult):
        raise RunnerError("candidate results must be ValidationCandidateResult objects")
    if result.split != "validation":
        raise RunnerError("selection accepts validation-split results only")
    if result.arm != arm:
        raise RunnerError("a result belongs to a different arm")
    if (result.num_buckets not in NUM_BUCKETS_GRID or result.epochs not in EPOCHS_GRID or result.relative_step_c not in RELATIVE_STEP_GRID
            or isinstance(result.num_buckets, bool) or isinstance(result.epochs, bool)):
        raise RunnerError("a result has a hyperparameter value outside the approved grid")
    if result.candidate_id != canonical_candidate_id(result.arm, result.num_buckets, result.epochs, result.relative_step_c):
        raise RunnerError("a candidate id does not match its parameters")
    ap = result.validation_ap
    if ap is None or isinstance(ap, bool) or not isinstance(ap, numbers.Real) or not math.isfinite(ap) or not 0.0 <= ap <= 1.0:
        raise RunnerError("a validation AP is undefined, non-finite or outside [0, 1]")
    return result


def select_candidate(results: Sequence[ValidationCandidateResult], *, arm: str, validation_ids_sha256: str | None = None) -> SelectionDecision:
    """Highest validation AP (compared rounded to 10 decimals); ties: fewer epochs, smaller ``num_buckets``, smaller ``c``, candidate id.

    Pure. It accepts only validation-split results of one arm that cover exactly the approved 27 candidates, were computed on the same aligned
    validation records, and hold a defined, finite AP. Full-precision APs are preserved in the returned decision.
    """
    expected = set(approved_candidate_ids(arm))
    items = list(results)
    checked = [_check_result(r, arm) for r in items]
    ids = [r.candidate_id for r in checked]
    if len(set(ids)) != len(ids):
        raise RunnerError("duplicate candidate identifiers")
    if set(ids) != expected:
        raise RunnerError(f"candidates do not match the approved grid ({len(expected - set(ids))} missing, {len(set(ids) - expected)} unexpected)")
    fingerprints = {r.validation_ids_sha256 for r in checked}
    counts = {r.validation_records for r in checked}
    if len(fingerprints) != 1 or len(counts) != 1:
        raise RunnerError("candidate results were not computed on the same aligned validation records")
    if validation_ids_sha256 is not None and fingerprints != {validation_ids_sha256}:
        raise RunnerError("candidate results do not match the expected validation records")
    rounded = {r.candidate_id: round(float(r.validation_ap), AP_ROUND_DECIMALS) for r in checked}
    order = sorted(checked, key=lambda r: (-rounded[r.candidate_id], r.epochs, r.num_buckets, r.relative_step_c, r.candidate_id))
    best = order[0]
    ranking = tuple({"rank": k + 1, "candidate_id": r.candidate_id, "validation_ap": float(r.validation_ap),
                     "rounded_ap": rounded[r.candidate_id]} for k, r in enumerate(order))
    tied = sum(1 for r in checked if rounded[r.candidate_id] == rounded[best.candidate_id])
    return SelectionDecision(arm=arm, selected=best, rounded_ap=rounded[best.candidate_id], tied_at_rounded_ap=tied, ranking=ranking)


# -- G. unavailable gates -----------------------------------------------------------------------------------------------------------------------


def request_training_authorization() -> NoReturn:
    """Model fitting requires a verified pre-run manifest. The manifest writer and verifier do not exist yet, so this always raises."""
    raise AuthorizationUnavailable("training", [
        "the pre-run manifest writer and verifier are not implemented",
        "no verified manifest authorizes fitting any candidate model"])


def request_test_access() -> NoReturn:
    """Test scoring or label exposure requires a verified manifest, a verified selection record and an atomic attempt marker. None exist yet."""
    raise AuthorizationUnavailable("test access", [
        "the pre-run manifest verifier is not implemented",
        "the validation-selection record verifier is not implemented",
        "the persistent one-time attempt marker is not implemented",
        "the runner holds no test labels and builds no test scoring inputs"])


def source_sha256() -> str:
    """SHA-256 of this file with LF line endings, for the pre-run manifest."""
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
