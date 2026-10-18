from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parser import PolicyParser, _canonical
from .pipeline import VerificationPipeline


def evaluate_request_dataset(path: str | Path) -> dict[str, Any]:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = dataset["cases"]
    parser = PolicyParser()
    pipeline = VerificationPipeline()
    results: list[dict[str, Any]] = []

    for case in cases:
        policies = []
        for index, text in enumerate(case["policy_texts"]):
            parsed = parser.parse_text(text, policy_id=f"{case['id']}-{index}")
            policies.append(
                parsed.model_copy(
                    update={
                        "subject": _canonical(parsed.subject),
                        "action": _canonical(parsed.action),
                        "resource": _canonical(parsed.resource),
                    }
                )
            )
        report = pipeline.engine.check_request(
            policies,
            subject_id=_canonical(case["id"]),
            roles=[_canonical(role) for role in case["subject_roles"]],
            action=_canonical(case["action"]),
            resource=_canonical(case["resource"]),
        )
        actual = {
            "decision": report.decision.value,
            "reason_code": report.reason_code,
        }
        expected = {
            "decision": case["expected_decision"],
            "reason_code": case["expected_reason_code"],
        }
        results.append({
            "id": case["id"],
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
        })

    decision_counts: dict[str, dict[str, int]] = {}
    for result in results:
        decision = result["expected"]["decision"]
        metrics = decision_counts.setdefault(decision, {"case_count": 0, "correct": 0})
        metrics["case_count"] += 1
        metrics["correct"] += int(result["passed"])

    passed_cases = sum(result["passed"] for result in results)
    return {
        "dataset_version": dataset["version"],
        "case_count": len(results),
        "passed_cases": passed_cases,
        "accuracy": passed_cases / len(results) if results else 1.0,
        "accuracy_by_decision": {
            decision: {
                **metrics,
                "accuracy": metrics["correct"] / metrics["case_count"],
            }
            for decision, metrics in decision_counts.items()
        },
        "results": results,
        "mismatches": [result for result in results if not result["passed"]],
    }
