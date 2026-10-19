import json
from pathlib import Path

import pytest

import src.policy_verifier.evaluation as evaluation
from src.policy_verifier.evaluation import EvaluationDatasetError, evaluate_dataset
from src.policy_verifier.request_evaluation import evaluate_request_dataset


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


def test_v2_dataset_reports_independent_cases_and_meanings():
    path = Path(__file__).parents[1] / "src" / "policy_verifier" / "evaluation_dataset_v2.json"
    result = evaluate_dataset(path)

    assert result["case_count"] == 17
    assert result["metrics"]["by_query_type"]["consistency"]["case_count"] == 8
    assert result["metrics"]["by_query_type"]["entailment"]["case_count"] == 9
    assert result["results"][0]["actual"]["verification_meaning"] == "CONSISTENT"
    assert result["results"][2]["actual"]["verification_meaning"] == "CONTRADICTORY"
    assert result["results"][8]["actual"]["verification_meaning"] == "ENTAILED"
    assert result["results"][12]["actual"]["verification_meaning"] == "NOT_ENTAILED"


def test_evaluation_can_load_baseline_and_v2_together():
    root = Path(__file__).parents[1] / "src" / "policy_verifier"
    result = evaluate_dataset([
        root / "evaluation_dataset.json",
        root / "evaluation_dataset_v2.json",
    ])

    assert result["case_count"] == 54
    assert result["dataset_version"] == [
        "policy-verification-v1",
        "policy-verification-v2",
    ]


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


def test_v3_request_evaluation_has_13_cases_and_runs(capsys):
    path = Path(__file__).parents[1] / "src" / "policy_verifier" / "evaluation_dataset_v3.json"
    result = evaluate_request_dataset(path)

    assert result["case_count"] == 13
    assert len(result["results"]) == 13
    assert set(result["accuracy_by_decision"]) == {
        "ALLOWED",
        "DENIED",
        "NOT_DETERMINED",
    }
    for mismatch in result["mismatches"]:
        print(mismatch)
    capsys.readouterr()
