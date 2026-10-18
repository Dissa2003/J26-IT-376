import json
from pathlib import Path

import pytest

import src.policy_verifier.evaluation as evaluation
from src.policy_verifier.evaluation import EvaluationDatasetError, evaluate_dataset


def test_reviewed_dataset_has_expected_size_and_all_cases_pass():
    path = Path(__file__).parents[1] / "src" / "policy_verifier" / "evaluation_dataset.json"
    result = evaluate_dataset(path)
    assert result["case_count"] == 37
    assert result["passed_cases"] == result["case_count"]
    assert result["metrics"]["conflict_f1"] == 1.0
    assert result["metrics"]["fol_correctness"] == 1.0
    assert result["metrics"]["inference_correctness"] == 1.0
    assert result["metrics"]["verification_outcome_accuracy"] == 1.0
    outcomes = {
        row["actual"]["solver_outcome"]
        for row in result["results"]
        if row["actual"].get("solver_outcome")
    }
    assert {"SAT", "UNSAT"} <= outcomes


def test_evaluation_baseline_uses_same_cases():
    path = Path(__file__).parents[1] / "src" / "policy_verifier" / "evaluation_dataset.json"
    result = evaluate_dataset(path)
    assert result["baseline"]["cases"] == result["case_count"]


def test_missing_actual_fields_fail_instead_of_being_skipped(monkeypatch):
    path = Path(__file__).parents[1] / "src" / "policy_verifier" / "evaluation_dataset.json"
    monkeypatch.setattr(evaluation, "_actual", lambda record, pipeline: {"parsed": True})

    result = evaluate_dataset(path)

    assert result["passed_cases"] == 0
    assert result["results"][0]["passed"] is False
    assert result["results"][0]["checks"]["solver_outcome"] is False


def test_malformed_expected_fields_report_case_and_field(tmp_path):
    dataset = {
        "version": "test",
        "cases": [{
            "id": "bad-01",
            "input": "Administrators can read user profiles.",
            "expected": {
                "parsed": True,
                "unsupported": False,
                "query_type": "not-a-query",
                "solver_outcome": "UNSAT",
                "conflict": False,
            },
        }],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match=r"bad-01\.expected"):
        evaluate_dataset(path)
