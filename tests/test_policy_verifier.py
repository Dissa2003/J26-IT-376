from fastapi.testclient import TestClient

from api.routes import app
from src.policy_verifier.cache import LocalRuleCache
from src.policy_verifier.calculus import infer
from src.policy_verifier.conflict_detector import ConflictDetector
from src.policy_verifier.logic import to_fol
from src.policy_verifier.models import (
    PolicyAttributes,
    PolicyEffect,
    PolicyInterpretation,
    SecurityPolicy,
    StructuredPolicy,
    VerificationStatus,
)
from src.policy_verifier.parser import PolicyParser, UnsupportedPolicyError
from src.policy_verifier.pipeline import VerificationPipeline
from src.policy_verifier.verifier import PolicyVerificationEngine


def test_parser_supports_explicit_grammar_and_rejects_guessing():
    only = PolicyParser().parse_text("Only administrators can delete user accounts.", "p1")
    permitted = PolicyParser().parse_text("Administrators can delete user accounts.", "p2")
    assert only.interpretation is PolicyInterpretation.ONLY_ROLE
    assert permitted.interpretation is PolicyInterpretation.ROLE_PERMISSION
    assert only.subject == permitted.subject == "Administrators"
    try:
        PolicyParser().parse_text("Maybe administrators should probably delete accounts.")
    except UnsupportedPolicyError:
        pass
    else:
        raise AssertionError("unsupported text was accepted")


def test_only_and_permission_have_distinct_fol_direction():
    parser = PolicyParser()
    formulas = []
    for text in ("Only administrators can delete user accounts.", "Administrators can delete user accounts."):
        policy = parser.parse_text(text, "p")
        attrs = PolicyAttributes(
            subject=policy.subject,
            action=policy.action,
            resource=policy.resource,
            effect=policy.effect,
            interpretation=policy.interpretation,
        )
        formulas.append(to_fol(StructuredPolicy(
            policy_id="p", raw_text=text, attributes=attrs,
            normalized_key="Administrators:Delete:UserAccounts",
        )))
    assert formulas[0].formula != formulas[1].formula
    assert "CanDelete(x, UserAccounts) → Administrators(x)" in formulas[0].formula
    assert "Administrators(x) → CanDelete(x, UserAccounts)" in formulas[1].formula


def test_inference_trace_is_auditable():
    policy = PolicyParser().parse_text("Administrators can delete user accounts.", "p1")
    report = VerificationPipeline().verify(policy)
    assert [step.rule for step in report.inference_trace] == [
        "Universal Instantiation",
        "Modus Ponens",
    ]
    assert report.inference_trace[-1].premises
    assert report.inference_trace[-1].conclusion


def test_z3_distinguishes_consistency_and_entailment():
    parser = PolicyParser()
    allow = parser.parse_text("Administrators can delete user accounts.", "allow")
    deny = parser.parse_text("Administrators cannot delete user accounts.", "deny")
    engine = PolicyVerificationEngine()
    assert engine.check_consistency([allow, deny]).status is VerificationStatus.UNSAT
    entailment = engine.check_entailment("Administrators(Alice)", "CanDelete(Alice, UserAccounts)")
    assert entailment.status is VerificationStatus.UNSAT
    assert entailment.is_entailment is True


def test_conflicts_are_scoped_by_resource_and_conditions():
    parser = PolicyParser()
    detector = ConflictDetector()
    policies = [
        parser.parse_text("Administrators can delete user accounts.", "allow"),
        parser.parse_text("Administrators cannot delete invoices.", "different-resource"),
    ]
    assert detector.find_conflicts(policies) == []


def test_cache_accepts_only_eligible_pipeline_reports():
    report = VerificationPipeline().verify(
        PolicyParser().parse_text("Administrators can delete user accounts.", "p1")
    )
    cache = LocalRuleCache()
    assert cache.put_if_eligible(report) is True
    assert cache.get("p1") is report


def test_api_returns_complete_pipeline_and_rejects_conflict():
    client = TestClient(app)
    response = client.post("/api/v1/verify-policies", json=[{
        "policy_id": "p1",
        "subject": "Administrators",
        "action": "Delete",
        "resource": "UserAccounts",
        "effect": "Allow",
        "raw_text": "Administrators can delete user accounts.",
    }])
    assert response.status_code == 200
    body = response.json()
    assert body["fol_formula"]["formula"]
    assert len(body["inference_trace"]) == 2
    assert body["solver_result"]["query_type"] == "entailment"

    conflict = client.post("/api/v1/verify-policies", json=[
        {
            "policy_id": "p1", "subject": "Administrators", "action": "Delete",
            "resource": "UserAccounts", "effect": "Allow", "raw_text": "allow",
        },
        {
            "policy_id": "p2", "subject": "Administrators", "action": "Delete",
            "resource": "UserAccounts", "effect": "Deny", "raw_text": "deny",
        },
    ])
    assert conflict.status_code == 400
    assert conflict.json()["detail"]["report"]["conflicting_policies"] == ["p1", "p2"]


def test_text_api_parses_supported_input_and_rejects_unsupported_input():
    client = TestClient(app)
    supported = client.post("/api/v1/verify-text", json={
        "policy_text": "Only administrators can delete user accounts.",
        "policy_id": "text-1",
    })
    assert supported.status_code == 200
    assert supported.json()["policy_id"] == "text-1"
    unsupported = client.post("/api/v1/verify-text", json={
        "policy_text": "Maybe administrators should delete accounts.",
    })
    assert unsupported.status_code == 422
    assert unsupported.json()["detail"]["code"] == "unsupported_policy"
