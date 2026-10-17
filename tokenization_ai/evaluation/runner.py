"""Deterministic KPI 1-3 evaluation runner (see PROTOCOL.md).

Scope: KPI 1 input alignment, KPI 2 token mapping, KPI 3 tensor correctness.
Not covered: latency (KPI 4), downstream usability (KPI 5), detection quality.
The anomaly model is never imported or exercised here.

The runner scores the *production* components against expectations in the
frozen fixtures. It never derives an expectation from the production
tokenizer; ``spec.py`` supplies the independent grammar acceptor and
reference encoder.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..adapters import InputContractError, to_unified_api_context, to_verified_rule_info
from ..fusion import InputFusion
from ..masking import PrivacyMasker
from ..models import TokenSequence
from ..normalization import Normalizer
from ..tensorizer import Tensorizer
from ..tokenizer import HybridTokenizer
from ..vocabulary import Vocabulary
from . import spec

PROTOCOL_VERSION = "1.0.0"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
PROTOCOL_FILE = Path(__file__).parent / "PROTOCOL.md"
DEFAULT_MAX_LENGTH = 128

TARGETS = {"kpi1_input_alignment": 90.0, "kpi2_token_mapping": 90.0, "kpi3_tensor_correctness": 95.0}

EVAL_FILES = ("curated_eval.json", "generated_eval.json")
DEV_FILES = ("dev_curated.json", "dev_generated.json")
INVALID_FILE = "invalid_cases.json"
_PRODUCTION_FILES = (
    "adapters.py", "fusion.py", "masking.py", "normalization.py",
    "tokenizer.py", "vocabulary.py", "tensorizer.py",
)

LIMITATIONS = [
    "Synthetic, hand-picked, small-N specification-coverage evaluation; not a random sample of traffic.",
    "Percentages are the share of curated scenarios handled as specified. They are not detection accuracy, "
    "precision, recall, F1, formal-rule verification correctness, or evidence of Formal Verification/3 integration.",
    "The oracle is independent in process (hand-written fixtures, separate grammar acceptor and reference "
    "encoder) but not in authorship; it detects regressions and deviations from the written design, not a flawed design.",
    "Rules are mock fixtures; Formal Verification's real verified-rule schema is unavailable.",
    "The vocabulary is fitted on a tiny synthetic dev corpus; the unknown-token rate characterises that corpus only.",
    "One tensor configuration (max_length=128). The anomaly model is not exercised by KPIs 1-3.",
    "No real dataset (SR-BH) was used.",
]


@dataclass
class Components:
    """Production collaborators under evaluation (injectable for runner self-tests)."""

    to_context: Callable[[Any], Any] = to_unified_api_context
    to_rules: Callable[[Any], Any] = to_verified_rule_info
    fusion: Any = field(default_factory=InputFusion)
    masker: Any = field(default_factory=PrivacyMasker)
    normalizer: Any = field(default_factory=Normalizer)
    tokenizer: Any = field(default_factory=HybridTokenizer)
    vocabulary_factory: Callable[[], Any] = Vocabulary
    tensorizer_factory: Callable[[Any, int], Any] = Tensorizer


# --- helpers ---------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, centre - half) * 100, 2), round(min(1.0, centre + half) * 100, 2)]


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "percent": round(100 * numerator / denominator, 2) if denominator else None,
        "wilson95_percent": wilson_interval(numerator, denominator),
    }


def _with_target(result: dict[str, Any], target: float) -> dict[str, Any]:
    pct = result["percent"]
    result["target_percent"] = target
    result["target_met"] = None if pct is None else pct >= target
    return result


def load_fixtures(fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, Any]]:
    docs = {}
    for name in (*EVAL_FILES, *DEV_FILES, INVALID_FILE):
        docs[name] = json.loads((fixture_dir / name).read_text(encoding="utf-8"))
    return docs


def fixture_hashes(fixture_dir: Path = FIXTURE_DIR) -> dict[str, str]:
    names = (*EVAL_FILES, *DEV_FILES, INVALID_FILE)
    hashes = {f"fixtures/{n}": sha256_file(fixture_dir / n) for n in names}
    hashes["PROTOCOL.md"] = sha256_file(PROTOCOL_FILE)
    return hashes


def _pipeline_context(c: Components, case_input: dict[str, Any]):
    context = c.to_context(case_input["context"])
    rule_source = case_input.get("rule_info")
    if rule_source is None:
        rule_source = {"event_id": context.event_id, "rules": []}
    return context, c.to_rules(rule_source)


def _fused_dict(fused: Any) -> dict[str, Any]:
    ctx, info = fused.context, fused.rule_info
    return {
        "event_id": ctx.event_id,
        "method": ctx.method,
        "path": ctx.path,
        "headers": dict(ctx.headers),
        "payload": ctx.payload,
        "content_type": ctx.content_type,
        "client_ip": ctx.client_ip,
        "threat_score": ctx.threat_score,
        "rules": [{"rule_id": r.rule_id, "attributes": r.attributes} for r in info.rules],
    }


def _front(c: Components, case_input: dict[str, Any]):
    """adapters -> fusion -> masking -> normalization -> tokenization (production)."""
    context, rules = _pipeline_context(c, case_input)
    fused = c.fusion.fuse(context, rules)
    masked = c.masker.mask_with_provenance(fused)
    normalized = c.normalizer.normalize_with_provenance(masked.fused, masked.masked_paths)
    sequence = c.tokenizer.tokenize(normalized.fused, normalized.masked_paths)
    return fused, normalized.masked_paths, sequence


def _norm_path(path: Any) -> tuple:
    return tuple(path)


def _valid_cases(docs: dict[str, dict]) -> list[tuple[str, dict]]:
    out = []
    for name in EVAL_FILES:
        for case in docs[name]["cases"]:
            out.append((docs[name]["group"], case))
    return out


def _by_group(rows: list[tuple[str, bool]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[bool]] = {}
    for group, ok in rows:
        groups.setdefault(group, []).append(ok)
    return {g: rate(sum(v), len(v)) for g, v in sorted(groups.items())}


def _case_group(file_group: str, case: dict) -> str:
    return case.get("group", file_group)


# --- KPI 1 -------------------------------------------------------------------


def evaluate_kpi1(c: Components, docs: dict[str, dict]) -> dict[str, Any]:
    rows: list[tuple[str, bool]] = []
    mismatches: list[dict[str, Any]] = []
    for file_group, case in _valid_cases(docs):
        group = _case_group(file_group, case)
        expected = case["expect"]["fused"]
        try:
            context, rules = _pipeline_context(c, case["input"])
            actual = _fused_dict(c.fusion.fuse(context, rules))
        except Exception as exc:  # noqa: BLE001 - every failure is a recorded miss
            rows.append((group, False))
            mismatches.append({"case_id": case["id"], "field": "<exception>", "actual": type(exc).__name__})
            continue
        diffs = [k for k in expected if actual.get(k) != expected[k]]
        rows.append((group, not diffs))
        for k in diffs:
            mismatches.append(
                {"case_id": case["id"], "field": k, "expected": expected[k], "actual": actual.get(k)}
            )

    rejected = 0
    details = []
    for case in docs[INVALID_FILE]["cases"]:
        try:
            _pipeline_context(c, case["input"])  # adapters
            c.fusion.fuse(*_pipeline_context(c, case["input"]))  # fusion
            outcome = "accepted"
        except InputContractError:
            outcome = "rejected"
            rejected += 1
        except Exception as exc:  # noqa: BLE001
            outcome = f"unexpected:{type(exc).__name__}"
        details.append({"case_id": case["id"], "contract_rule": case["contract_rule"], "outcome": outcome})

    valid = _with_target(rate(sum(ok for _, ok in rows), len(rows)), TARGETS["kpi1_input_alignment"])
    valid["by_group"] = _by_group(rows)
    valid["check_type"] = "independent expected-output"
    valid["mismatches"] = mismatches
    return {
        "valid_event_alignment": valid,
        "invalid_event_rejection": {
            **rate(rejected, len(details)),
            "note": "Reported separately; not part of the alignment percentage.",
            "details": details,
        },
    }


# --- KPI 2 -------------------------------------------------------------------


def evaluate_kpi2(c: Components, docs: dict[str, dict]) -> dict[str, Any]:
    seq_rows: list[tuple[str, bool]] = []
    prov_rows: list[tuple[str, bool]] = []
    exact_cases = 0
    expected_positions = matched_positions = text_matched = role_correct = 0
    tp = fp = fn = 0
    grammar_ok = grammar_total = 0
    errors: list[dict[str, Any]] = []
    prov_errors: list[dict[str, Any]] = []
    grammar_errors: list[dict[str, Any]] = []

    for file_group, case in _valid_cases(docs):
        group = _case_group(file_group, case)
        exp_tokens, exp_roles = spec.parse_compact(case["expect"]["tokens"])
        exp_paths = {_norm_path(p) for p in case["expect"]["masked_paths"]}
        expected_positions += len(exp_tokens)
        try:
            _fused, paths, sequence = _front(c, case["input"])
        except Exception as exc:  # noqa: BLE001
            seq_rows.append((group, False))
            prov_rows.append((group, False))
            fn += len(exp_paths)
            errors.append({"case_id": case["id"], "error": type(exc).__name__})
            continue

        act_tokens, act_roles = list(sequence.tokens), list(sequence.roles)
        ok = act_tokens == exp_tokens and act_roles == exp_roles
        seq_rows.append((group, ok))
        exact_cases += ok
        for i, (et, er) in enumerate(zip(exp_tokens, exp_roles)):
            if i < len(act_tokens) and act_tokens[i] == et:
                text_matched += 1
                if act_roles[i] == er:
                    role_correct += 1
                    matched_positions += 1
        if not ok:
            diff_index = next(
                (i for i in range(max(len(exp_tokens), len(act_tokens)))
                 if i >= len(exp_tokens) or i >= len(act_tokens)
                 or exp_tokens[i] != act_tokens[i] or exp_roles[i] != act_roles[i]),
                None,
            )
            errors.append(
                {
                    "case_id": case["id"],
                    "expected_length": len(exp_tokens),
                    "actual_length": len(act_tokens),
                    "first_difference_index": diff_index,
                    "expected_at": spec.to_compact(exp_tokens, exp_roles)[diff_index] if diff_index is not None and diff_index < len(exp_tokens) else None,
                    "actual_at": spec.to_compact(act_tokens, act_roles)[diff_index] if diff_index is not None and diff_index < len(act_tokens) else None,
                }
            )

        act_paths = {_norm_path(p) for p in paths}
        prov_ok = act_paths == exp_paths
        prov_rows.append((group, prov_ok))
        tp += len(act_paths & exp_paths)
        fp += len(act_paths - exp_paths)
        fn += len(exp_paths - act_paths)
        if not prov_ok:
            prov_errors.append(
                {
                    "case_id": case["id"],
                    "missing": sorted(map(list, exp_paths - act_paths), key=str),
                    "unexpected": sorted(map(list, act_paths - exp_paths), key=str),
                }
            )

        grammar_total += 1
        problems = spec.well_formed(act_tokens, act_roles)
        if not problems:
            grammar_ok += 1
        else:
            grammar_errors.append({"case_id": case["id"], "problems": problems[:3]})

    exact = _with_target(rate(exact_cases, len(seq_rows)), TARGETS["kpi2_token_mapping"])
    exact["by_group"] = _by_group(seq_rows)
    exact["check_type"] = "independent expected-output"
    exact["mapping_errors"] = errors
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "sequence_exact_match": exact,
        "token_level_match": {
            **rate(matched_positions, expected_positions),
            "definition": "positions where token text and role both equal expected; missing positions count as unmatched",
        },
        "role_correctness": {
            **rate(role_correct, text_matched),
            "definition": "among positions whose text matched, the share whose role matched",
        },
        "masking_provenance": {
            "event_exact_match": {**rate(sum(ok for _, ok in prov_rows), len(prov_rows)), "by_group": _by_group(prov_rows)},
            "path_true_positive": tp,
            "path_false_positive": fp,
            "path_false_negative": fn,
            "path_precision": None if precision is None else round(precision, 4),
            "path_recall": None if recall is None else round(recall, 4),
            "check_type": "independent expected-output",
            "errors": prov_errors,
            "note": "Measured against hand-curated sensitive locations; says nothing about unlisted sensitive data.",
        },
        "grammar_conformance_of_actual_output": {
            **rate(grammar_ok, grammar_total),
            "check_type": "implementation-invariant (independent grammar acceptor)",
            "errors": grammar_errors,
            "note": "Not part of the KPI 2 percentage.",
        },
    }


# --- KPI 3 -------------------------------------------------------------------

_INVARIANT = "invariant"
_INDEPENDENT = "independent"

_CHECKS = {
    "length": _INVARIANT,
    "mask_binary_prefix": _INVARIANT,
    "ids_in_range": _INVARIANT,
    "padding_ids": _INVARIANT,
    "event_id": _INVARIANT,
    "deterministic": _INVARIANT,
    "valid_count": _INDEPENDENT,
    "token_id_consistency": _INDEPENDENT,
    "reserved_never_unk": _INDEPENDENT,
    "framing_preserved": _INDEPENDENT,
    "rule_segment_preserved": _INDEPENDENT,
    "payload_balanced": _INDEPENDENT,
    "payload_subsequence": _INDEPENDENT,
}


def _expected_decoded(tokens, roles, api_known, rule_known):
    out = []
    for token, role in zip(tokens, roles):
        if role == "literal_api":
            out.append(token if token in api_known else spec.UNK)
        elif role == "literal_rule":
            out.append(token if token in rule_known else spec.UNK)
        else:
            out.append(token)
    return out


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(item in it for item in small)


def _balanced(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "[API:OBJ_START]" or tokens[-1] != "[API:OBJ_END]":
        return False
    depth = {"OBJ": 0, "LIST": 0}
    for t in tokens:
        if t == "[API:OBJ_START]":
            depth["OBJ"] += 1
        elif t == "[API:OBJ_END]":
            depth["OBJ"] -= 1
        elif t == "[API:LIST_START]":
            depth["LIST"] += 1
        elif t == "[API:LIST_END]":
            depth["LIST"] -= 1
        if depth["OBJ"] < 0 or depth["LIST"] < 0:
            return False
    return depth["OBJ"] == 0 and depth["LIST"] == 0


def _tensor_checks(tensorizer, vocab, seq: TokenSequence, exp_roles, api_known, rule_known, max_length):
    """Return ({check: bool}, tensor, truncated, decoded_expected)."""
    results: dict[str, bool] = {}
    tensor = tensorizer.tensorize(seq)
    ids, mask = tensor.input_ids, tensor.attention_mask
    n = sum(mask)
    truncated = len(seq.tokens) > max_length
    unk = vocab.unk_id
    pad = vocab.pad_id

    results["length"] = len(ids) == max_length and len(mask) == max_length
    results["mask_binary_prefix"] = set(mask) <= {0, 1} and mask == [1] * n + [0] * (len(mask) - n)
    results["ids_in_range"] = all(0 <= i < len(vocab) for i in ids)
    results["padding_ids"] = all(i == pad for i, m in zip(ids, mask) if m == 0)
    results["event_id"] = tensor.event_id == seq.event_id
    again = tensorizer.tensorize(seq)
    results["deterministic"] = again.input_ids == ids and again.attention_mask == mask

    expected_dec = _expected_decoded(seq.tokens, exp_roles, api_known, rule_known)
    decoded = vocab.decode(ids[:n]) if n <= len(ids) else []
    if not truncated:
        results["valid_count"] = n == len(seq.tokens)
        results["token_id_consistency"] = decoded == expected_dec
        results["reserved_never_unk"] = all(
            ids[i] != unk for i, role in enumerate(exp_roles) if role in ("control", "masked") and i < len(ids)
        )
    else:
        results["valid_count"] = 0 < n <= max_length

        def idx(tokens, tag, start=0):
            return tokens.index(tag, start) if tag in tokens[start:] else -1

        e_ps, e_pe = idx(expected_dec, "[API:PAYLOAD_START]"), idx(expected_dec, "[API:PAYLOAD_END]")
        d_ps, d_pe = idx(decoded, "[API:PAYLOAD_START]"), idx(decoded, "[API:PAYLOAD_END]")
        located = min(e_ps, e_pe, d_ps, d_pe) >= 0
        if located:
            results["framing_preserved"] = (
                decoded[: d_ps + 1] == expected_dec[: e_ps + 1] and decoded[d_pe:] == expected_dec[e_pe:]
            )
            e_payload, d_payload = expected_dec[e_ps + 1 : e_pe], decoded[d_ps + 1 : d_pe]
            results["payload_balanced"] = _balanced(d_payload)
            results["payload_subsequence"] = _is_subsequence(d_payload, e_payload)
            e_rs, e_re = idx(expected_dec, "[RULES_START]"), idx(expected_dec, "[RULES_END]")
            d_rs, d_re = idx(decoded, "[RULES_START]"), idx(decoded, "[RULES_END]")
            results["rule_segment_preserved"] = (
                min(e_rs, e_re, d_rs, d_re) >= 0 and decoded[d_rs : d_re + 1] == expected_dec[e_rs : e_re + 1]
            )
        else:
            for name in ("framing_preserved", "rule_segment_preserved", "payload_balanced", "payload_subsequence"):
                results[name] = False
    return results, tensor, truncated, expected_dec


def evaluate_kpi3(c: Components, docs: dict[str, dict], max_length: int) -> dict[str, Any]:
    # 1. Fit the vocabulary on the DEV groups only (production tokenizer).
    vocab = c.vocabulary_factory()
    dev_sequences: list[TokenSequence] = []
    dev_failures: list[str] = []
    api_known: set[str] = set()
    rule_known: set[str] = set()
    dev_cases = [case for name in DEV_FILES for case in docs[name]["cases"]]
    for case in dev_cases:
        tokens, roles = spec.parse_compact(case["expect"]["tokens"])
        for token, role in zip(tokens, roles):
            if role == "literal_api":
                api_known.add(token)
            elif role == "literal_rule":
                rule_known.add(token)
        try:
            dev_sequences.append(_front(c, case["input"])[2])
        except Exception as exc:  # noqa: BLE001
            dev_failures.append(f"{case['id']}: {type(exc).__name__}")
    vocab.fit(dev_sequences)
    tensorizer = c.tensorizer_factory(vocab, max_length)

    # 2. Score tensors built from the INDEPENDENT expected sequences.
    per_case: list[dict[str, Any]] = []
    group_rows: list[tuple[str, bool]] = []
    check_counts = {name: [0, 0] for name in _CHECKS}  # [applicable, passed]
    failures: list[dict[str, Any]] = []
    cap_total = cap_ok = 0
    cap_details = []
    consistency_total = consistency_ok = 0
    unk_expected = unk_observed = literal_positions = 0
    unk_ids_all = valid_positions_all = 0
    trunc_cases = padded_cases = exact_fit_cases = 0

    for file_group, case in _valid_cases(docs):
        group = _case_group(file_group, case)
        tokens, roles = spec.parse_compact(case["expect"]["tokens"])
        seq = TokenSequence(event_id=case["input"]["context"]["event_id"], tokens=tokens, roles=roles)
        outcome = case["expect"]["tensor"]

        if outcome == "capacity_error":
            cap_total += 1
            try:
                tensorizer.tensorize(seq)
                verdict = "tensor_produced_unexpectedly"
            except ValueError:
                verdict = "rejected_as_specified"
                cap_ok += 1
            except Exception as exc:  # noqa: BLE001
                verdict = f"unexpected:{type(exc).__name__}"
            cap_details.append({"case_id": case["id"], "outcome": verdict})
            continue

        try:
            results, tensor, truncated, decoded_exp = _tensor_checks(
                tensorizer, vocab, seq, roles, api_known, rule_known, max_length
            )
        except Exception as exc:  # noqa: BLE001
            group_rows.append((group, False))
            failures.append({"case_id": case["id"], "error": type(exc).__name__})
            continue

        for name, passed in results.items():
            check_counts[name][0] += 1
            check_counts[name][1] += bool(passed)
        case_ok = all(results.values())
        group_rows.append((group, case_ok))
        if not case_ok:
            failures.append({"case_id": case["id"], "failed_checks": [k for k, v in results.items() if not v]})

        n = sum(tensor.attention_mask)
        valid_positions_all += n
        unk_ids_all += sum(1 for i, m in zip(tensor.input_ids, tensor.attention_mask) if m and i == vocab.unk_id)
        if truncated:
            trunc_cases += 1
        elif len(tokens) == max_length:
            exact_fit_cases += 1
        else:
            padded_cases += 1
        if not truncated:
            for i, role in enumerate(roles):
                if role in ("literal_api", "literal_rule"):
                    literal_positions += 1
                    unk_expected += decoded_exp[i] == spec.UNK
                    unk_observed += tensor.input_ids[i] == vocab.unk_id

        # secondary: pipeline-consistency (production tokens -> tensor == expected-seq tensor)
        consistency_total += 1
        try:
            prod_seq = _front(c, case["input"])[2]
            prod_tensor = tensorizer.tensorize(prod_seq)
            consistency_ok += (
                prod_tensor.input_ids == tensor.input_ids and prod_tensor.attention_mask == tensor.attention_mask
            )
        except Exception:  # noqa: BLE001
            pass

    correct = _with_target(
        rate(sum(ok for _, ok in group_rows), len(group_rows)), TARGETS["kpi3_tensor_correctness"]
    )
    correct["by_group"] = _by_group(group_rows)
    correct["failures"] = failures
    checks = {
        name: {
            "type": _CHECKS[name],
            "applicable": a,
            "passed": p,
            "percent": round(100 * p / a, 2) if a else None,
        }
        for name, (a, p) in check_counts.items()
    }
    return {
        "configuration": {
            "max_length": max_length,
            "vocabulary_size": len(vocab),
            "vocabulary_fitted_on": "dev_curated + dev_generated only (never scored)",
            "dev_sequences_fitted": len(dev_sequences),
            "dev_failures": dev_failures,
            "tensor_input": "independent expected token sequences",
        },
        "tensor_correctness": correct,
        "checks": checks,
        "sequence_length_mix": {
            "padded": padded_cases,
            "exact_fit": exact_fit_cases,
            "truncated": trunc_cases,
        },
        "unknown_token_rate": {
            "scope": "non-truncated sequences, literal positions only",
            "literal_positions": literal_positions,
            "expected_unknown": unk_expected,
            "observed_unknown": unk_observed,
            "expected_rate_percent": round(100 * unk_expected / literal_positions, 2) if literal_positions else None,
            "observed_rate_percent": round(100 * unk_observed / literal_positions, 2) if literal_positions else None,
            "agrees_with_independent_expectation": unk_expected == unk_observed,
            "unk_id_share_of_all_valid_positions_percent": (
                round(100 * unk_ids_all / valid_positions_all, 2) if valid_positions_all else None
            ),
            "note": "Reported separately; an [UNK] for an unseen literal is the specified behaviour, not a failure.",
        },
        "capacity_rejection": {**rate(cap_ok, cap_total), "details": cap_details},
        "pipeline_consistency_secondary": {
            **rate(consistency_ok, consistency_total),
            "note": "Tensor from the production token sequence equals the tensor from the expected sequence.",
        },
    }


# --- orchestration -----------------------------------------------------------


def run_evaluation(
    components: Components | None = None,
    fixture_dir: Path = FIXTURE_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    c = components or Components()
    docs = load_fixtures(fixture_dir)
    results = {
        "kpi1_input_alignment": evaluate_kpi1(c, docs),
        "kpi2_token_mapping": evaluate_kpi2(c, docs),
        "kpi3_aligned_tensor_correctness": evaluate_kpi3(c, docs, max_length),
    }
    body = json.dumps(results, sort_keys=True, default=str)
    package_dir = Path(__file__).parent.parent
    return {
        "protocol_version": PROTOCOL_VERSION,
        "scope_statement": (
            "Synthetic specification-coverage evaluation of KPIs 1-3. Not real-world detection accuracy, "
            "not formal verification, not Formal Verification/3 integration."
        ),
        "targets_percent": TARGETS,
        "data": {
            "valid_eval_cases": sum(len(docs[n]["cases"]) for n in EVAL_FILES),
            "invalid_cases": len(docs[INVALID_FILE]["cases"]),
            "dev_cases": sum(len(docs[n]["cases"]) for n in DEV_FILES),
            "seeds": {"dev": 1101, "eval": 2202},
            "fixture_sha256": fixture_hashes(fixture_dir),
        },
        "results": results,
        "results_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "limitations": LIMITATIONS,
        "run_metadata": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "production_source_sha256": {
                name: sha256_file(package_dir / name)
                for name in _PRODUCTION_FILES
                if (package_dir / name).exists()
            },
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    r = report["results"]
    k1, k2, k3 = r["kpi1_input_alignment"], r["kpi2_token_mapping"], r["kpi3_aligned_tensor_correctness"]

    def row(label, m, target=None):
        pct = "n/a" if m["percent"] is None else f"{m['percent']:.2f}%"
        ci = m.get("wilson95_percent")
        ci_text = "n/a" if not ci else f"{ci[0]:.1f}-{ci[1]:.1f}%"
        met = "" if target is None else ("met" if m.get("target_met") else "NOT met")
        tgt = "" if target is None else f"≥{target:g}%"
        return f"| {label} | {m['numerator']} / {m['denominator']} | {pct} | {ci_text} | {tgt} | {met} |"

    lines = [
        "# LogiShield KPI 1-3 evaluation report",
        "",
        f"Protocol {report['protocol_version']} · results hash `{report['results_sha256'][:16]}…`",
        "",
        "> " + report["scope_statement"],
        "> Targets are proposal targets, not achieved real-world results.",
        "",
        "| Measure | Numerator / denominator | Rate | Wilson 95% | Target | Synthetic-set outcome |",
        "|---|---|---|---|---|---|",
        row("KPI 1 valid-event input alignment", k1["valid_event_alignment"], 90),
        row("KPI 2 sequence exact match", k2["sequence_exact_match"], 90),
        row("KPI 3 tensor correctness", k3["tensor_correctness"], 95),
        "",
        "## Reported separately (not part of the KPI percentages)",
        "",
        "| Measure | Numerator / denominator | Rate |",
        "|---|---|---|",
    ]
    extras = [
        ("KPI 1 invalid-event rejection", k1["invalid_event_rejection"]),
        ("KPI 2 token-level match (text+role)", k2["token_level_match"]),
        ("KPI 2 role correctness (text matched)", k2["role_correctness"]),
        ("KPI 2 masking provenance, exact per event", k2["masking_provenance"]["event_exact_match"]),
        ("KPI 2 grammar conformance of actual output (invariant)", k2["grammar_conformance_of_actual_output"]),
        ("KPI 3 capacity rejection as specified", k3["capacity_rejection"]),
        ("KPI 3 pipeline consistency (secondary)", k3["pipeline_consistency_secondary"]),
    ]
    for label, m in extras:
        pct = "n/a" if m["percent"] is None else f"{m['percent']:.2f}%"
        lines.append(f"| {label} | {m['numerator']} / {m['denominator']} | {pct} |")

    u = k3["unknown_token_rate"]
    lines += [
        "",
        "## KPI 3 detail",
        "",
        f"- max_length {k3['configuration']['max_length']}, vocabulary size {k3['configuration']['vocabulary_size']}, "
        f"fitted on dev only ({k3['configuration']['dev_sequences_fitted']} sequences).",
        f"- Length mix: {k3['sequence_length_mix']}.",
        f"- Unknown literal rate (non-truncated): observed {u['observed_rate_percent']}% "
        f"({u['observed_unknown']}/{u['literal_positions']}), independently expected "
        f"{u['expected_rate_percent']}% ({u['expected_unknown']}/{u['literal_positions']}); "
        f"agree: {u['agrees_with_independent_expectation']}.",
        "",
        "| Check | Type | Passed / applicable |",
        "|---|---|---|",
    ]
    for name, m in k3["checks"].items():
        lines.append(f"| {name} | {m['type']} | {m['passed']} / {m['applicable']} |")

    lines += ["", "## Preserved failures", ""]
    failures_found = False
    for label, items in (
        ("KPI 1 mismatches", k1["valid_event_alignment"]["mismatches"]),
        ("KPI 1 invalid events not rejected", [d for d in k1["invalid_event_rejection"]["details"] if d["outcome"] != "rejected"]),
        ("KPI 2 mapping errors", k2["sequence_exact_match"]["mapping_errors"]),
        ("KPI 2 provenance errors", k2["masking_provenance"]["errors"]),
        ("KPI 3 failures", k3["tensor_correctness"]["failures"]),
    ):
        if items:
            failures_found = True
            lines.append(f"**{label}**")
            for item in items:
                lines.append(f"- `{json.dumps(item, ensure_ascii=False, default=str)}`")
            lines.append("")
    if not failures_found:
        lines.append("None recorded in this run.")
        lines.append("")

    lines += ["## Limitations", ""] + [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"
