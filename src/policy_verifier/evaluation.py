from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .conflict_detector import ConflictDetector
from .logic import to_fol
from .models import PolicyAttributes, StructuredPolicy
from .parser import PolicyParser, UnsupportedPolicyError
from .pipeline import VerificationPipeline
from .verifier import PolicyVerificationEngine


"""Evaluation schema:

Supported records require parsed, attributes, FOL, inference, query, solver, and
conflict expectations. Unsupported records require only an explicit rejection
outcome; formal fields must be listed in ``not_applicable`` and are not
pretended to have been produced.
"""


class EvaluationDatasetError(ValueError):
    """Raised when a dataset record does not follow the evaluation schema."""


SUPPORTED_EXPECTED_FIELDS = {
    "parsed",
    "attributes",
    "fol_formula",
    "inference_conclusions",
    "query_type",
    "solver_outcome",
    "conflict",
}
SUPPORTED_OPTIONAL_EXPECTED_FIELDS = {"unsupported"}
UNSUPPORTED_EXPECTED_FIELDS = {"parsed", "unsupported", "rejection", "not_applicable"}
NOT_APPLICABLE_FIELDS = {
    "attributes",
    "fol_formula",
    "inference_conclusions",
    "query_type",
    "solver_outcome",
    "conflict",
}
SOLVER_OUTCOMES = {"SAT", "UNSAT", "UNKNOWN"}
QUERY_TYPES = {"consistency", "entailment"}
VERIFICATION_MEANINGS = {
    "ENTAILED",
    "NOT_ENTAILED",
    "CONSISTENT",
    "CONTRADICTORY",
    "UNKNOWN",
    "UNSUPPORTED",
}


def _verification_meaning(query_type: str | None, solver_outcome: str | None) -> str:
    if solver_outcome == "UNKNOWN":
        return "UNKNOWN"
    if query_type == "entailment":
        return {"UNSAT": "ENTAILED", "SAT": "NOT_ENTAILED"}.get(
            solver_outcome, "UNSUPPORTED"
        )
    if query_type == "consistency":
        return {"SAT": "CONSISTENT", "UNSAT": "CONTRADICTORY"}.get(
            solver_outcome, "UNSUPPORTED"
        )
    return "UNSUPPORTED"


def _is_v2_record(record: dict[str, Any]) -> bool:
    return "policy_texts" in record


def _inputs(record: dict[str, Any]) -> list[str]:
    if _is_v2_record(record):
        return record["policy_texts"]
    value = record["input"]
    return value if isinstance(value, list) else [value]


def _expected_fields(record: dict[str, Any]) -> dict[str, Any]:
    if not _is_v2_record(record):
        return record["expected"]
    query_type = record["query_type"]
    outcome = record["expected_solver_outcome"]
    return {
        "parsed": True,
        "query_type": query_type,
        "solver_outcome": outcome,
        "verification_meaning": _verification_meaning(query_type, outcome),
    }


def _validate_dataset(dataset: Any) -> None:
    errors: list[str] = []
    if not isinstance(dataset, dict):
        raise EvaluationDatasetError("Dataset must be an object.")
    if not isinstance(dataset.get("version"), str):
        errors.append("dataset.version: expected a string")
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        errors.append("dataset.cases: expected a list")
    else:
        seen_ids: set[str] = set()
        for index, record in enumerate(cases):
            prefix = f"cases[{index}]"
            if not isinstance(record, dict):
                errors.append(f"{prefix}: expected an object")
                continue
            case_id = record.get("id")
            label = case_id if isinstance(case_id, str) else prefix
            if not isinstance(case_id, str) or not case_id:
                errors.append(f"{label}.id: expected a non-empty string")
            elif case_id in seen_ids:
                errors.append(f"{case_id}.id: duplicate case ID")
            else:
                seen_ids.add(case_id)
            if _is_v2_record(record):
                policy_texts = record.get("policy_texts")
                if (
                    not isinstance(policy_texts, list)
                    or not policy_texts
                    or not all(isinstance(item, str) and item for item in policy_texts)
                ):
                    errors.append(
                        f"{label}.policy_texts: expected a non-empty list of strings"
                    )
                query_type = record.get("query_type")
                if query_type not in QUERY_TYPES:
                    errors.append(f"{label}.query_type: invalid query type")
                if not isinstance(record.get("premises"), list) or not all(
                    isinstance(item, str) and item for item in record.get("premises", [])
                ):
                    errors.append(
                        f"{label}.premises: expected a list of non-empty strings"
                    )
                if query_type == "entailment" and not isinstance(
                    record.get("conclusion"), str
                ):
                    errors.append(
                        f"{label}.conclusion: required for entailment cases"
                    )
                if record.get("expected_solver_outcome") not in SOLVER_OUTCOMES:
                    errors.append(
                        f"{label}.expected_solver_outcome: invalid solver outcome"
                    )
                if (
                    not isinstance(record.get("justification"), str)
                    or "\n" in record.get("justification", "")
                ):
                    errors.append(
                        f"{label}.justification: expected a one-line string"
                    )
                continue
            if "input" not in record:
                errors.append(f"{label}.input: required field is missing")
            elif not isinstance(record["input"], (str, list)):
                errors.append(f"{label}.input: expected a string or list of strings")
            elif isinstance(record["input"], list) and not all(
                isinstance(item, str) for item in record["input"]
            ):
                errors.append(f"{label}.input: every list item must be a string")
            expected = record.get("expected")
            if not isinstance(expected, dict):
                errors.append(f"{label}.expected: expected an object")
                continue
            parsed = expected.get("parsed")
            unsupported = expected.get("unsupported")
            if not isinstance(parsed, bool):
                errors.append(f"{label}.expected.parsed: expected a boolean")
            if unsupported is not None and not isinstance(unsupported, bool):
                errors.append(f"{label}.expected.unsupported: expected a boolean")
            if parsed is False or unsupported is True:
                if parsed is not False or unsupported is not True:
                    errors.append(
                        f"{label}.expected: unsupported records must set "
                        "parsed=false and unsupported=true"
                    )
                unknown = set(expected) - UNSUPPORTED_EXPECTED_FIELDS
                if unknown:
                    errors.append(
                        f"{label}.expected: unsupported record has invalid fields "
                        f"{sorted(unknown)}; use not_applicable"
                    )
                if expected.get("rejection") != "unsupported_policy":
                    errors.append(
                        f"{label}.expected.rejection: must be 'unsupported_policy'"
                    )
                not_applicable = expected.get("not_applicable")
                if not isinstance(not_applicable, list) or set(not_applicable) != NOT_APPLICABLE_FIELDS:
                    errors.append(
                        f"{label}.expected.not_applicable: must list exactly "
                        f"{sorted(NOT_APPLICABLE_FIELDS)}"
                    )
                continue
            missing = SUPPORTED_EXPECTED_FIELDS - set(expected)
            if missing:
                errors.append(
                    f"{label}.expected: missing fields {sorted(missing)}"
                )
            unknown = set(expected) - (
                SUPPORTED_EXPECTED_FIELDS | SUPPORTED_OPTIONAL_EXPECTED_FIELDS
            )
            if unknown:
                errors.append(f"{label}.expected: unknown fields {sorted(unknown)}")
            if expected.get("query_type") not in QUERY_TYPES:
                errors.append(f"{label}.expected.query_type: invalid query type")
            if expected.get("solver_outcome") not in SOLVER_OUTCOMES:
                errors.append(f"{label}.expected.solver_outcome: invalid solver outcome")
            if not isinstance(expected.get("conflict"), bool):
                errors.append(f"{label}.expected.conflict: expected a boolean")
            if not isinstance(expected.get("attributes"), dict):
                errors.append(f"{label}.expected.attributes: expected an object")
            if not isinstance(expected.get("fol_formula"), str):
                errors.append(f"{label}.expected.fol_formula: expected a string")
            if not isinstance(expected.get("inference_conclusions"), list):
                errors.append(
                    f"{label}.expected.inference_conclusions: expected a list"
                )
    if errors:
        raise EvaluationDatasetError(
            "Invalid evaluation dataset:\n" + "\n".join(f"- {error}" for error in errors)
        )


def _structured(policy: Any) -> StructuredPolicy:
    attrs = PolicyAttributes(
        subject=policy.subject,
        action=policy.action,
        resource=policy.resource,
        effect=policy.effect,
        conditions=policy.conditions,
        interpretation=policy.interpretation,
    )
    return StructuredPolicy(
        policy_id=policy.policy_id,
        raw_text=policy.raw_text,
        attributes=attrs,
        normalized_key=f"{policy.subject}:{policy.action}:{policy.resource}",
    )


def _actual(record: dict[str, Any], pipeline: VerificationPipeline) -> dict[str, Any]:
    parser = PolicyParser()
    inputs = _inputs(record)
    try:
        policies = [parser.parse_text(text, f"{record['id']}-{index}") for index, text in enumerate(inputs)]
    except UnsupportedPolicyError as exc:
        return {
            "parsed": False,
            "unsupported": True,
            "rejection": "unsupported_policy",
            "error": str(exc),
        }
    report = pipeline.verify(policies[0])
    conflicts = ConflictDetector().find_conflicts(policies)
    fol = to_fol(_structured(policies[0]))
    if record.get("query_type") == "consistency":
        solver_result = PolicyVerificationEngine().check_consistency(policies)
    elif record.get("query") or _is_v2_record(record):
        query = record.get("query", record)
        solver_result = PolicyVerificationEngine().check_entailment(
            query["premises"], query["conclusion"]
        )
    else:
        solver_result = report.solver_result
    return {
        "parsed": True,
        "unsupported": False,
        "attributes": policies[0].model_dump(mode="json"),
        "fol_formula": fol.formula,
        "inference_conclusions": [step.conclusion for step in report.inference_trace],
        "query_type": solver_result.query_type if solver_result else None,
        "solver_outcome": solver_result.status.value if solver_result else None,
        "verification_meaning": _verification_meaning(
            solver_result.query_type if solver_result else None,
            solver_result.status.value if solver_result else None,
        ),
        "rule_status": report.rule_status,
        "conflict": bool(conflicts),
        "conflict_count": len(conflicts),
    }


def _baseline(record: dict[str, Any]) -> dict[str, Any]:
    """Comparable verifier that bypasses the explicit calculus trace."""
    parser = PolicyParser()
    inputs = _inputs(record)
    try:
        policies = [parser.parse_text(text, f"baseline-{record['id']}-{index}") for index, text in enumerate(inputs)]
    except UnsupportedPolicyError:
        return {"parsed": False, "unsupported": True, "rejection": "unsupported_policy"}
    structured = _structured(policies[0])
    formula = to_fol(structured)
    individual = "Alice"
    antecedent = formula.antecedent.model_copy(update={"arguments": [individual]}).render()
    consequent = formula.consequent.model_copy(update={"arguments": [individual]}).render()
    rule = f"{antecedent} → {consequent}"
    if record.get("query_type") == "consistency":
        result = PolicyVerificationEngine().check_consistency(policies)
    elif record.get("query") or _is_v2_record(record):
        query = record.get("query", record)
        result = PolicyVerificationEngine().check_entailment(
            query["premises"], query["conclusion"]
        )
    else:
        result = PolicyVerificationEngine().check_entailment([antecedent, rule], consequent)
    conflicts = ConflictDetector().find_conflicts(policies)
    return {
        "parsed": True,
        "unsupported": False,
        "fol_formula": formula.formula,
        "query_type": result.query_type,
        "solver_outcome": result.status.value,
        "verification_meaning": _verification_meaning(
            result.query_type, result.status.value
        ),
        "conflict": bool(conflicts),
    }


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return all(key in actual and _matches(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return actual == expected
    return actual == expected


def _checks_for_record(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    """Compare every applicable expected field; missing actual fields fail."""
    return {
        name: _matches(actual[name], value) if name in actual else False
        for name, value in expected.items()
        if name != "not_applicable"
    }


def _load_dataset(path: str | Path) -> dict[str, Any]:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_dataset(dataset)
    return dataset


def evaluate_dataset(path: str | Path | Iterable[str | Path]) -> dict[str, Any]:
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    datasets = [_load_dataset(item) for item in paths]
    records = [
        (dataset, record)
        for dataset in datasets
        for record in dataset["cases"]
    ]
    pipeline = VerificationPipeline()
    rows: list[dict[str, Any]] = []
    for dataset, record in records:
        actual = _actual(record, pipeline)
        expected = _expected_fields(record)
        checks = _checks_for_record(actual, expected)
        rows.append({
            "id": record["id"],
            "source": record["source"],
            "dataset_version": dataset["version"],
            "justification": record.get("justification"),
            "checks": checks,
            "passed": bool(checks) and all(checks.values()),
            "actual": actual,
            "ground_truth": expected,
        })

    conflict_tp = sum(bool(row["actual"].get("conflict")) and bool(row["ground_truth"].get("conflict")) for row in rows)
    conflict_fp = sum(bool(row["actual"].get("conflict")) and not bool(row["ground_truth"].get("conflict")) for row in rows)
    conflict_fn = sum(not bool(row["actual"].get("conflict")) and bool(row["ground_truth"].get("conflict")) for row in rows)
    precision = conflict_tp / (conflict_tp + conflict_fp) if conflict_tp + conflict_fp else 0.0
    recall = conflict_tp / (conflict_tp + conflict_fn) if conflict_tp + conflict_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    def metric(key: str) -> float:
        applicable = [row for row in rows if key in row["checks"]]
        return (
            sum(row["checks"][key] for row in applicable) / len(applicable)
            if applicable
            else 0.0
        )
    baseline = [_baseline(record) for _, record in records]
    by_query_type: dict[str, dict[str, Any]] = {}
    for query_type in sorted(QUERY_TYPES):
        typed_rows = [
            row for row in rows if row["actual"].get("query_type") == query_type
        ]
        by_query_type[query_type] = {
            "case_count": len(typed_rows),
            "passed_cases": sum(row["passed"] for row in typed_rows),
            "verification_outcome_accuracy": (
                sum(row["checks"].get("solver_outcome", False) for row in typed_rows)
                / len(typed_rows)
                if typed_rows
                else 0.0
            ),
            "verification_meaning_accuracy": (
                sum(
                    row["checks"].get("verification_meaning", False)
                    for row in typed_rows
                )
                / len(typed_rows)
                if typed_rows
                else 0.0
            ),
        }
    return {
        "dataset_version": (
            datasets[0]["version"]
            if len(datasets) == 1
            else [dataset["version"] for dataset in datasets]
        ),
        "case_count": len(rows),
        "passed_cases": sum(row["passed"] for row in rows),
        "metrics": {
            "parsing_accuracy": metric("parsed"),
            "fol_correctness": metric("fol_formula"),
            "inference_correctness": metric("inference_conclusions"),
            "verification_outcome_accuracy": metric("solver_outcome"),
            "conflict_precision": precision,
            "conflict_recall": recall,
            "conflict_f1": f1,
            "by_query_type": by_query_type,
        },
        "baseline": {
            "cases": len(baseline),
            "parsed_cases": sum(item.get("parsed", False) for item in baseline),
            "solver_outcomes": Counter(item.get("solver_outcome") for item in baseline if item.get("parsed")),
        },
        "results": rows,
    }
