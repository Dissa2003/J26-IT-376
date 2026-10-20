"""Descriptive diagnostics of the simulated policy catalog on a synthetic dataset (not a model).

This module is deliberately separate from ``policy_sim.py``: the simulator never sees labels, and only
these diagnostics do, to *describe* how the rules relate to the synthetic labels. Diagnostics were not
used to choose, add or drop a rule. They are descriptive attention flags, not proof that shortcuts are
absent, and they train nothing and compute no accuracy, PR-AUC, ROC-AUC, precision, recall or F1.

The token-budget section runs the existing, unmodified Member 4 stages (fusion, masking, normalization,
tokenizer, vocabulary, tensorizer) read-only on the synthetic requests, with and without the simulated
rules. A vocabulary is fitted on the diagnostic sequences solely so lengths can be encoded; it is not a
model and nothing is evaluated with it. The request-to-context mapping used here (JSON object body to
payload; form body to a string-valued dict; xml/plain body to ``{"text": body}``) is a diagnostic
convenience and is not a decision about the final evaluation runner.

Messages and outputs hold only counts, ids and hashes, never raw request content.
"""

from __future__ import annotations

import json
import random
import statistics
import urllib.parse
from collections import Counter
from typing import Any

from ..fusion import InputFusion
from ..masking import PrivacyMasker
from ..models import UnifiedApiContext, VerifiedRuleInfo
from ..normalization import Normalizer
from ..tensorizer import Tensorizer
from ..tokenizer import HybridTokenizer
from ..vocabulary import Vocabulary
from .audit import normalized_mutual_information
from .generator import SyntheticDataset
from .policy_sim import (
    ALLOWED_ATTRIBUTES,
    CATALOG_SHA256,
    CATALOG_VERSION,
    POLICIES,
    NOT_VERIFIED_STATUS,
    SIMULATED_PROVENANCE,
    simulate_rules_for_record,
)
from .splits import EvalRecord, group_id_for

DEFAULT_MIN_TEMPLATE_SUPPORT = 3
_SHUFFLES = 20


def rule_assignments(records: tuple[EvalRecord, ...] | list[EvalRecord]) -> dict[str, tuple[str, ...]]:
    """record_id -> ordered tuple of simulated policy ids (uses request features only)."""
    return {r.record_id: simulate_rules_for_record(r).policy_ids for r in records}


def to_context(record: EvalRecord) -> UnifiedApiContext:
    """Diagnostic mapping of a synthetic request to the pipeline's input context (no label, no metadata)."""
    f = record.features
    headers = dict(f.headers)
    body = f.body
    payload: dict[str, Any] = {}
    if body:
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
        elif "=" in body and " " not in body.strip():
            payload = dict(urllib.parse.parse_qsl(body, keep_blank_values=True))
        else:
            payload = {"text": body}
    return UnifiedApiContext(
        event_id=record.record_id, method=f.method, path=f.path, content_type=headers.get("content-type"),
        headers={k: v for k, v in headers.items() if k != "content-type"} | ({"content-type": headers["content-type"]}
                                                                              if "content-type" in headers else {}),
        payload=payload,
    )


def _sequences(records: list[EvalRecord], with_rules: bool, assignments: dict[str, tuple[str, ...]]):
    fusion, masker, normalizer, tokenizer = InputFusion(), PrivacyMasker(), Normalizer(), HybridTokenizer()
    out = []
    for r in records:
        context = to_context(r)
        rule_info = (simulate_rules_for_record(r).rule_info if with_rules
                     else VerifiedRuleInfo(event_id=r.record_id, rules=[]))
        fused = fusion.fuse(context, rule_info)
        masked = masker.mask_with_provenance(fused)
        normalized = normalizer.normalize_with_provenance(masked.fused, masked.masked_paths)
        out.append(tokenizer.tokenize(normalized.fused, normalized.masked_paths))
    return out


def _stats(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {"mean": round(sum(values) / len(values), 2), "median": float(statistics.median(values)),
            "p95": float(ordered[int(0.95 * (len(ordered) - 1))]), "max": float(ordered[-1])}


def token_budget(dataset: SyntheticDataset, max_lengths: tuple[int, ...] = (128, 256)) -> dict[str, Any]:
    """Token lengths and tensorizer outcomes for Arm A (no rules) versus Arm B (simulated rules)."""
    records = list(dataset.records)
    assignments = rule_assignments(records)
    seq_a = _sequences(records, False, assignments)
    seq_b = _sequences(records, True, assignments)
    len_a, len_b = [len(s.tokens) for s in seq_a], [len(s.tokens) for s in seq_b]
    rule_tokens = [b - a for a, b in zip(len_a, len_b)]
    vocabulary = Vocabulary()
    vocabulary.fit(seq_a + seq_b)  # for length diagnostics only; not a model
    per_length: dict[str, Any] = {}
    for max_length in max_lengths:
        tensorizer = Tensorizer(vocabulary, max_length=max_length)
        outcome: dict[str, dict[str, int]] = {}
        failed: dict[str, set[int]] = {"A_request_only": set(), "B_simulated_rules": set()}
        for arm, seqs in (("A_request_only", seq_a), ("B_simulated_rules", seq_b)):
            truncated = rejected = 0
            for i, s in enumerate(seqs):
                if len(s.tokens) > max_length:
                    truncated += 1
                try:
                    tensorizer.tensorize(s)
                except ValueError:
                    rejected += 1
                    failed[arm].add(i)
            outcome[arm] = {"longer_than_max_length": truncated, "rejected_framing_does_not_fit": rejected}
        outcome["records_rejected_in_either_arm"] = {"count": len(failed["A_request_only"] | failed["B_simulated_rules"])}
        per_length[str(max_length)] = outcome
    return {
        "tokens_arm_a": _stats(len_a), "tokens_arm_b": _stats(len_b),
        "rule_tokens_per_request": _stats(rule_tokens),
        "rule_tokens_per_rule": 8 + 4 * len(ALLOWED_ATTRIBUTES),
        "by_max_length": per_length,
        "note": "Tokens beyond max_length are truncated from the payload only; a request whose framing plus rules "
                "cannot fit is rejected. Rejected records must be removed from every arm to keep comparisons paired.",
    }


def rule_diagnostics(dataset: SyntheticDataset, *, min_template_support: int = DEFAULT_MIN_TEMPLATE_SUPPORT,
                     seed: int = 0, include_token_budget: bool = True) -> dict[str, Any]:
    """Coverage, per-request statistics, descriptive class associations and token budget."""
    records = list(dataset.records)
    n = len(records)
    labels = [r.meta.label for r in records]
    templates = [group_id_for(r.features) for r in records]
    assigned = rule_assignments(records)
    ids = [p.rule_id for p in POLICIES]

    per_policy: dict[str, Any] = {}
    flags: list[str] = []
    for pid in ids:
        applies = [pid in assigned[r.record_id] for r in records]
        n1 = sum(1 for a, y in zip(applies, labels) if a and y == 1)
        n0 = sum(1 for a, y in zip(applies, labels) if a and y == 0)
        tmpl = {t for t, a in zip(templates, applies) if a}
        nmi = normalized_mutual_information(applies, labels)
        total = n0 + n1
        entry = {
            "records": total, "share_of_records": round(total / n, 4), "templates": len(tmpl),
            "class_0_records": n0, "class_1_records": n1,
            "suspicious_share_among_applied": round(n1 / total, 4) if total else None,
            "normalized_mi_with_label": round(nmi, 4),
        }
        per_policy[pid] = entry
        if total == 0:
            flags.append(f"{pid}: applies to no request in this dataset (zero coverage)")
        elif (n0 == 0 or n1 == 0) and len(tmpl) >= min_template_support:
            flags.append(f"{pid}: applies to {len(tmpl)} templates and only one class")
    counts = [len(assigned[r.record_id]) for r in records]
    rule_sets = [assigned[r.record_id] for r in records]
    by_set: dict[tuple[str, ...], Counter] = {}
    tmpl_of_set: dict[tuple[str, ...], set[str]] = {}
    for rs, y, t in zip(rule_sets, labels, templates):
        by_set.setdefault(rs, Counter())[y] += 1
        tmpl_of_set.setdefault(rs, set()).add(t)
    exclusive_t = sum(sum(c.values()) for rs, c in by_set.items() if len(c) == 1 and len(tmpl_of_set[rs]) >= min_template_support)
    rng = random.Random(seed)
    ref = max(normalized_mutual_information(rule_sets, rng.sample(labels, n)) for _ in range(_SHUFFLES))
    set_nmi = normalized_mutual_information(rule_sets, labels)
    if exclusive_t / n >= 0.05:
        flags.append(f"rule sets: {exclusive_t / n:.1%} of records have a rule set that occurs in >= {min_template_support} "
                     "templates and in only one class (attention flag, not a criterion)")
    report: dict[str, Any] = {
        "catalog": {"version": CATALOG_VERSION, "sha256": CATALOG_SHA256, "policies": len(POLICIES),
                    "provenance": SIMULATED_PROVENANCE, "verification_status": NOT_VERIFIED_STATUS},
        "records": n,
        "rules_per_request": {"mean": round(sum(counts) / n, 3), "max": max(counts),
                              "histogram": dict(sorted(Counter(counts).items()))},
        "empty_rule_share": round(sum(1 for c in counts if c == 0) / n, 4),
        "records_with_duplicate_policy_ids": sum(1 for rs in rule_sets if len(set(rs)) != len(rs)),
        "policies_with_zero_coverage": [pid for pid, e in per_policy.items() if e["records"] == 0],
        "per_policy": per_policy,
        "rule_set_feature": {
            "distinct_rule_sets": len(by_set), "normalized_mi_with_label": round(set_nmi, 4),
            "shuffled_reference_max": round(ref, 4),
            "class_exclusive_template_share": round(exclusive_t / n, 4),
        },
        "attention_flags": flags,
        "note": ("Descriptive only. The rules are deterministic functions of the request, so an association between a "
                 "rule and a label can only mirror the request-label association; it is not independent information and "
                 "this report is not proof that shortcuts are absent."),
    }
    if include_token_budget:
        report["token_budget"] = token_budget(dataset)
    return report
