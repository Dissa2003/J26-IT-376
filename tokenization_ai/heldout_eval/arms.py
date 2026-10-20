"""Paired A/B/C evaluation arm preparation (``arms-1``; draft protocol heldout-eval-1.2.0-draft, sections 12.8-12.12).

Evaluation infrastructure only. Nothing here fits a detection model, calls ``predict``, computes a metric, reads a label, or runs the
confirmatory study. It prepares, for the *same* requests, three representations:

* **Arm A** (``A_request_only``): ``request_to_context-v1`` -> fusion with an *empty* rule list -> privacy masking with provenance ->
  normalization with provenance -> structure-aware tokenization -> (later) shared vocabulary and tensorizer.
* **Arm B** (``B_simulated_rules``): the same, but the fusion receives the ``rule_info`` of the unchanged ``simpol-1`` simulator. The
  ``SimulatedRuleSet`` wrapper (``provenance="simulated"``, ``verification_status="not_verified_simulated"``, catalog version and SHA-256)
  is kept beside the sequence and re-checked; these rules are deterministic functions of the request, not Member 2 verified rules.
* **Arm C** (``C_char3``): the *actual* ``NormalizationResult`` of the rule-free Arm A front end is passed to ``armc-char3-v1``. Arm C never
  sees rules, never has a vocabulary, and the privacy boundary of ``arm_c`` is never bypassed (no result object is built by hand).

Controls
--------
* A split is named explicitly (``train`` / ``validation`` / ``test``); it is never inferred from record content. A split is built from a
  non-empty list of ``EvalRecord`` with unique ids and is sorted by record id, so input order cannot change any value.
* Model inputs are request content only. The labels, scenario families, template ids, sources and timestamps are never read here
  (``record.meta`` is not touched); the future runner aligns labels to ``record_ids`` itself.
* One shared vocabulary, fitted **once, on the training split only**, on the training **Arm B** sequences. Arm B contains every Arm A
  token (Arm A differs only by having no rule tokens), so one fit on Arm B counts each request once and serves both arms; fitting on A and
  B together would count every shared literal twice and defeat ``min_frequency``. Validation and test are encoded with the frozen
  vocabulary and never refit; unseen literals map to the unknown token and are counted by ``unknown_token_report``.
  ``min_frequency=2`` is the **proposed** experimental setting (``MIN_FREQUENCY_PROPOSED``) and is **not frozen**.
* One shared ``max_length`` from ``budget-rule-1`` (a ``BudgetDecision`` computed from the training Arm B lengths) is applied to Arms A and
  B with the real ``Tensorizer``. Truncated records are allowed and counted; a record the tensorizer would reject fails closed with a
  ``PairedExclusionRequired`` carrying an ``ExclusionProposal`` (ids and reasons only) for protocol review. Nothing is excluded,
  truncated or re-budgeted automatically.
* Errors and diagnostics never contain request content, only record ids, stage names and counts.

Known limitations (see PROTOCOL.md 12.13): Arm B adds simulated, request-derived information; Arm C sees query and value characters that
Arms A and B turn into atomic tokens or the unknown token; feature mass and collisions differ by arm; a shared model class and
``num_buckets`` do not give equal information.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import (
    fusion as _fusion_module,
    masking as _masking_module,
    model as _model_module,
    models as _models_module,
    normalization as _normalization_module,
    tensorizer as _tensorizer_module,
    tokenizer as _tokenizer_module,
    vocabulary as _vocabulary_module,
)
from ..fusion import InputFusion
from ..masking import PrivacyMasker
from ..model import LightweightAnomalyModel
from ..models import ModelInput, TokenSequence, VerifiedRuleInfo
from ..normalization import Normalizer
from ..tensorizer import Tensorizer
from ..tokenizer import TOKEN_ROLE_LITERAL_API, TOKEN_ROLE_LITERAL_RULE, HybridTokenizer
from ..vocabulary import Vocabulary
from . import arm_c as _arm_c_module
from . import policy_sim as _policy_sim_module
from . import request_context as _request_context_module
from . import token_budget as _token_budget_module
from .arm_c import ENCODER_ID, encode_request, encoder_parameters
from .generator import GENERATOR_VERSION
from .policy_sim import (
    CATALOG_SHA256,
    CATALOG_VERSION,
    NOT_VERIFIED_STATUS,
    SIMULATED_PROVENANCE,
    SimulatedRuleSet,
    simulate_rules_for_record,
)
from .request_context import MAPPING_ID, request_to_context
from .splits import EvalRecord
from .synthetic_catalog import CATALOG_VERSION as SYNTHETIC_CATALOG_VERSION
from .token_budget import BudgetDecision, STATUS_SELECTED, sequence_lengths, tensorizer_outcome

ARMS_VERSION = "arms-1"
ARM_A = "A_request_only"
ARM_B = "B_simulated_rules"
ARM_C = "C_char3"
ARMS = (ARM_A, ARM_B, ARM_C)
SPLITS = ("train", "validation", "test")
MIN_FREQUENCY_PROPOSED = 2  # proposed experimental setting; NOT frozen by the protocol


class ArmBuildError(RuntimeError):
    """A record could not be prepared. The message holds only the record id and stage, never request content."""


class PairingError(ValueError):
    """The three arms do not describe exactly the same requests."""


class PairedExclusionRequired(RuntimeError):
    """The frozen budget cannot represent some records. Nothing was excluded; ``proposal`` is for protocol review."""

    def __init__(self, proposal: "ExclusionProposal") -> None:
        super().__init__(f"{len(proposal.records)} record(s) cannot be represented under max_length={proposal.max_length} "
                         f"in split '{proposal.split}'; protocol review required, nothing was removed")
        self.proposal = proposal


# -- result types ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedRequest:
    """One request in the three arm representations before vocabulary encoding. Holds masked content only, no raw context, no label."""

    record_id: str
    sequence_a: TokenSequence
    sequence_b: TokenSequence
    model_input_c: ModelInput
    rule_set: SimulatedRuleSet


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    requests: tuple[PreparedRequest, ...]

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(r.record_id for r in self.requests)

    def summary(self) -> dict[str, Any]:
        """Counts only; safe to log or persist."""
        policies: dict[str, int] = {}
        for r in self.requests:
            for pid in r.rule_set.policy_ids:
                policies[pid] = policies.get(pid, 0) + 1
        return {
            "split": self.split, "records": len(self.requests), "arms": list(ARMS),
            "records_with_simulated_rules": sum(1 for r in self.requests if r.rule_set.policy_ids),
            "simulated_policy_counts": dict(sorted(policies.items())),
        }


@dataclass(frozen=True)
class FrozenVocabulary:
    """A vocabulary fitted once on the training split. Do not refit; use ``record`` for the manifest."""

    vocabulary: Vocabulary
    record: Mapping[str, Any]


@dataclass(frozen=True)
class RejectedRecord:
    record_id: str
    arms: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ExclusionProposal:
    """Records the frozen budget cannot represent. A proposal only: removal from all arms needs protocol approval."""

    split: str
    max_length: int
    records: tuple[RejectedRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "max_length": self.max_length, "action": "proposal_only_nothing_excluded",
                "records": [{"record_id": r.record_id, "arms": list(r.arms), "reason": r.reason} for r in self.records]}


@dataclass(frozen=True)
class ArmInputs:
    """Model inputs of the three arms for one split, aligned by ``record_ids``. Validated at construction."""

    split: str
    record_ids: tuple[str, ...]
    arm_a: tuple[ModelInput, ...]
    arm_b: tuple[ModelInput, ...]
    arm_c: tuple[ModelInput, ...]
    max_length: int
    outcomes_a: tuple[str, ...] = field(default=())
    outcomes_b: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        n = len(self.record_ids)
        if len(set(self.record_ids)) != n:
            raise PairingError("duplicate record ids")
        for name, items in (("A", self.arm_a), ("B", self.arm_b), ("C", self.arm_c)):
            if len(items) != n:
                raise PairingError(f"arm {name} has {len(items)} representations for {n} records")
            if tuple(m.event_id for m in items) != self.record_ids:
                raise PairingError(f"arm {name} does not describe exactly the same records in the same order")
        for arm, outcomes in (("A", self.outcomes_a), ("B", self.outcomes_b)):
            if outcomes and len(outcomes) != n:
                raise PairingError(f"arm {arm} outcomes do not cover every record")
        for m in self.arm_a + self.arm_b:
            if len(m.input_ids) != self.max_length or len(m.attention_mask) != self.max_length:
                raise PairingError("Arm A and Arm B inputs must both have the shared max_length")

    def summary(self) -> dict[str, Any]:
        def count(outcomes: tuple[str, ...], key: str) -> int:
            return sum(1 for o in outcomes if o == key)
        return {
            "split": self.split, "paired_records": len(self.record_ids), "max_length": self.max_length,
            "arm_a": {"fits": count(self.outcomes_a, "fits"), "truncated": count(self.outcomes_a, "truncated")},
            "arm_b": {"fits": count(self.outcomes_b, "fits"), "truncated": count(self.outcomes_b, "truncated")},
        }


# -- stage flow --------------------------------------------------------------------------------------------------------------------


def _front_end(record: EvalRecord, rule_info: VerifiedRuleInfo, stages: tuple[Any, Any, Any, Any]):
    fusion, masker, normalizer, tokenizer = stages
    context = request_to_context(record)
    fused = fusion.fuse(context, rule_info)
    masked = masker.mask_with_provenance(fused)
    normalized = normalizer.normalize_with_provenance(masked.fused, masked.masked_paths)
    return normalized, tokenizer.tokenize(normalized.fused, normalized.masked_paths)


def _check_simulated(rule_set: SimulatedRuleSet) -> None:
    if (rule_set.provenance != SIMULATED_PROVENANCE or rule_set.verification_status != NOT_VERIFIED_STATUS
            or rule_set.catalog_version != CATALOG_VERSION or rule_set.catalog_sha256 != CATALOG_SHA256):
        raise ArmBuildError("Arm B rules must come from the unchanged simulated catalog with explicit simulated provenance")


def _prepare_one(record: EvalRecord, stages: tuple[Any, Any, Any, Any]) -> PreparedRequest:
    rid = record.record_id
    try:
        stage = "arm A front end"
        normalized_a, sequence_a = _front_end(record, VerifiedRuleInfo(event_id=rid, rules=[]), stages)
        stage = "arm B simulated rules"
        rule_set = simulate_rules_for_record(record)
        _check_simulated(rule_set)
        _, sequence_b = _front_end(record, rule_set.rule_info, stages)
        stage = "arm C encoder"
        model_input_c = encode_request(normalized_a)  # the real NormalizationResult of the rule-free front end
    except ArmBuildError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately generic; the message must not carry request content
        raise ArmBuildError(f"record {rid!r}: {stage} failed ({type(exc).__name__})") from None
    return PreparedRequest(rid, sequence_a, sequence_b, model_input_c, rule_set)


def prepare_split(records: Iterable[EvalRecord], split: str) -> PreparedSplit:
    """Build the three representations of every record of one explicitly named split (sorted by record id; fail closed)."""
    if split not in SPLITS:
        raise ArmBuildError(f"split must be one of {SPLITS}")
    items = list(records)
    if not items:
        raise ArmBuildError("a split needs at least one record")
    if not all(isinstance(r, EvalRecord) for r in items):
        raise ArmBuildError("records must be EvalRecord objects")
    ids = [r.record_id for r in items]
    if len(set(ids)) != len(ids):
        raise ArmBuildError("duplicate record ids")
    stages = (InputFusion(), PrivacyMasker(), Normalizer(), HybridTokenizer())
    prepared = tuple(_prepare_one(r, stages) for r in sorted(items, key=lambda r: r.record_id))
    return PreparedSplit(split=split, requests=prepared)


def assert_disjoint(*splits: PreparedSplit) -> None:
    """Different splits must not share a record id, and each split may appear once."""
    names = [s.split for s in splits]
    if len(set(names)) != len(names):
        raise PairingError("a split appears more than once")
    seen: set[str] = set()
    for s in splits:
        overlap = seen & set(s.record_ids)
        if overlap:
            raise PairingError(f"{len(overlap)} record id(s) occur in more than one split")
        seen |= set(s.record_ids)


# -- shared vocabulary (train only) ------------------------------------------------------------------------------------------


def _vocabulary_fingerprint(vocabulary: Vocabulary) -> str:
    tokens = vocabulary.decode(list(range(len(vocabulary))))  # id order; reserved tokens first, then literals by (-count, token)
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def fit_shared_vocabulary(train: PreparedSplit, *, min_frequency: int = MIN_FREQUENCY_PROPOSED,
                          max_literal_vocab_size: int | None = None) -> FrozenVocabulary:
    """Fit the single vocabulary shared by Arms A and B on the **training** split's Arm B sequences (see module docstring)."""
    if not isinstance(train, PreparedSplit) or train.split != "train":
        raise ArmBuildError("the vocabulary can only be fitted on the training split")
    vocabulary = Vocabulary(max_literal_vocab_size=max_literal_vocab_size, min_frequency=min_frequency)
    vocabulary.fit([r.sequence_b for r in train.requests])
    record = {
        "fitted_on_split": "train", "fitted_on": "arm_b_training_sequences", "training_records": len(train.requests),
        "min_frequency": min_frequency, "min_frequency_status": ("proposed_not_frozen" if min_frequency == MIN_FREQUENCY_PROPOSED
                                                                  else "non_default_setting"),
        "max_literal_vocab_size": max_literal_vocab_size, "size": len(vocabulary), "sha256": _vocabulary_fingerprint(vocabulary),
        "unknown_token": "literals not in the frozen vocabulary map to the unknown id",
    }
    return FrozenVocabulary(vocabulary=vocabulary, record=record)


def unknown_token_report(prepared: PreparedSplit, frozen: FrozenVocabulary) -> dict[str, Any]:
    """Counts of literal tokens that fall back to the unknown id under the frozen vocabulary (no content)."""
    unk = frozen.vocabulary.unk_id
    out: dict[str, Any] = {"split": prepared.split, "vocabulary_sha256": frozen.record["sha256"]}
    for arm, attr in ((ARM_A, "sequence_a"), (ARM_B, "sequence_b")):
        literals = unknown = 0
        for r in prepared.requests:
            seq = getattr(r, attr)
            for token_id, role in zip(frozen.vocabulary.encode(seq), seq.roles):
                if role in (TOKEN_ROLE_LITERAL_API, TOKEN_ROLE_LITERAL_RULE):
                    literals += 1
                    unknown += token_id == unk
        out[arm] = {"literal_tokens": literals, "unknown_tokens": unknown,
                    "unknown_share": round(unknown / literals, 6) if literals else 0.0}
    return out


# -- shared budget -------------------------------------------------------------------------------------------------------------


def training_arm_b_lengths(train: PreparedSplit) -> dict[str, int]:
    """The input for ``select_shared_max_length``: training Arm B lengths by record id."""
    if train.split != "train":
        raise ArmBuildError("budget lengths for the selection come from the training split only")
    return sequence_lengths({r.record_id: r.sequence_b for r in train.requests})


def propose_exclusions(prepared: PreparedSplit, decision: BudgetDecision) -> ExclusionProposal:
    """Records whose framing plus rules cannot fit the frozen budget in Arm A or Arm B. Excludes nothing."""
    max_length = _selected(decision)
    rejected = []
    for r in prepared.requests:
        arms = tuple(arm for arm, seq in ((ARM_A, r.sequence_a), (ARM_B, r.sequence_b))
                     if tensorizer_outcome(seq, max_length) == "rejected")
        if arms:
            rejected.append(RejectedRecord(r.record_id, arms, "mandatory framing and rule segment exceed max_length"))
    return ExclusionProposal(prepared.split, max_length, tuple(rejected))


def _selected(decision: BudgetDecision) -> int:
    if not isinstance(decision, BudgetDecision) or decision.status != STATUS_SELECTED or decision.selected_max_length is None:
        raise ArmBuildError("a BudgetDecision with a selected max_length is required")
    return decision.selected_max_length


def build_arm_inputs(prepared: PreparedSplit, frozen: FrozenVocabulary, decision: BudgetDecision) -> ArmInputs:
    """Encode a split with the frozen vocabulary and the shared budget. Never refits and never truncates silently.

    Records the tensorizer would reject make this fail closed (:class:`PairedExclusionRequired`); payload truncation is allowed and counted.
    """
    max_length = _selected(decision)
    if not isinstance(frozen, FrozenVocabulary):
        raise ArmBuildError("a FrozenVocabulary is required")
    proposal = propose_exclusions(prepared, decision)
    if proposal.records:
        raise PairedExclusionRequired(proposal)
    tensorizer = Tensorizer(frozen.vocabulary, max_length=max_length)
    arm_a, arm_b, outcomes_a, outcomes_b = [], [], [], []
    for r in prepared.requests:
        arm_a.append(tensorizer.tensorize(r.sequence_a))
        arm_b.append(tensorizer.tensorize(r.sequence_b))
        outcomes_a.append(tensorizer_outcome(r.sequence_a, max_length))
        outcomes_b.append(tensorizer_outcome(r.sequence_b, max_length))
    return ArmInputs(
        split=prepared.split, record_ids=prepared.record_ids, arm_a=tuple(arm_a), arm_b=tuple(arm_b),
        arm_c=tuple(r.model_input_c for r in prepared.requests), max_length=max_length,
        outcomes_a=tuple(outcomes_a), outcomes_b=tuple(outcomes_b),
    )


# -- model preparation (no training) ----------------------------------------------------------------------------------------------


def new_untrained_model(num_buckets: int) -> LightweightAnomalyModel:
    """A fresh, untrained model. Every hyperparameter setting must use its own instance (``fit`` warm-starts from existing weights)."""
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int) or num_buckets < 1:
        raise ArmBuildError("num_buckets must be a positive integer")
    return LightweightAnomalyModel(num_buckets=num_buckets)


# -- provenance -----------------------------------------------------------------------------------------------------------------


def _source_sha256(module: Any) -> str:
    return hashlib.sha256(Path(module.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_provenance(prepared: Sequence[PreparedSplit], frozen: FrozenVocabulary | None = None,
                     decision: BudgetDecision | None = None) -> dict[str, Any]:
    """JSON-safe provenance for the future pre-run manifest. Holds ids of components, hashes and counts, never request content."""
    assert_disjoint(*prepared)
    modules = {
        "fusion.py": _fusion_module, "masking.py": _masking_module, "normalization.py": _normalization_module,
        "tokenizer.py": _tokenizer_module, "vocabulary.py": _vocabulary_module, "tensorizer.py": _tensorizer_module,
        "models.py": _models_module, "model.py": _model_module, "request_context.py": _request_context_module,
        "arm_c.py": _arm_c_module, "token_budget.py": _token_budget_module, "policy_sim.py": _policy_sim_module,
        "arms.py": sys.modules[__name__],
    }
    return {
        "arms_version": ARMS_VERSION, "arms": {"A": ARM_A, "B": ARM_B, "C": ARM_C},
        "request_to_context": MAPPING_ID, "arm_c_encoder": dict(encoder_parameters()) | {"encoder_id": ENCODER_ID},
        "generator_version": GENERATOR_VERSION, "synthetic_catalog_version": SYNTHETIC_CATALOG_VERSION,
        "arm_b_rules": {"catalog_version": CATALOG_VERSION, "catalog_sha256": CATALOG_SHA256, "provenance": SIMULATED_PROVENANCE,
                        "verification_status": NOT_VERIFIED_STATUS,
                        "note": "deterministic functions of the request; not Member 2 verified rules"},
        "budget": None if decision is None else decision.to_dict(),
        "vocabulary": None if frozen is None else dict(frozen.record),
        "paired_records": {p.split: len(p.requests) for p in prepared},
        "source_sha256": {name: _source_sha256(mod) for name, mod in modules.items()},
        "stages": ["request_to_context-v1", "InputFusion", "PrivacyMasker.mask_with_provenance",
                   "Normalizer.normalize_with_provenance", "HybridTokenizer.tokenize", "Vocabulary", "Tensorizer"],
        "no_model_trained": True,
    }
