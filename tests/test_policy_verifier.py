from fastapi.testclient import TestClient
import z3

from api import routes
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
    SolverResult,
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
    assert [constraint.kind for constraint in report.constraints] == [
        "rule", "fact", "query", "negated_query"
    ]
    assert report.rule_status == "eligible"


def test_z3_distinguishes_consistency_and_entailment():
    parser = PolicyParser()
    allow = parser.parse_text("Administrators can delete user accounts.", "allow")
    deny = parser.parse_text("Administrators cannot delete user accounts.", "deny")
    engine = PolicyVerificationEngine()
    assert engine.check_consistency([allow, deny]).status is VerificationStatus.UNSAT
    entailment = engine.check_entailment("Administrators(Alice)", "CanDelete(Alice, UserAccounts)")
    assert entailment.status is VerificationStatus.UNSAT
    assert entailment.is_entailment is True
    assert engine.check_entailment(["Administrators(Alice)"], "CanRead(Alice, UserProfiles)").status is VerificationStatus.SAT


def test_consistency_uses_grammar_direction_for_only_role_rules():
    parser = PolicyParser()
    policies = [
        parser.parse_text("Only auditors can read audit logs.", "only"),
        parser.parse_text("Auditors cannot read audit logs.", "deny"),
    ]

    result = PolicyVerificationEngine().check_consistency(policies)

    assert result.status is VerificationStatus.SAT


def test_consistency_role_assumptions_can_be_toggled():
    parser = PolicyParser()
    policies = [
        parser.parse_text("Only administrators can delete user accounts.", "only"),
        parser.parse_text("Guests can delete user accounts.", "grant"),
    ]
    engine = PolicyVerificationEngine()

    with_assumptions = engine.check_consistency(policies)
    without_role_witness = engine.check_consistency(
        policies, assume_role_witness=False
    )
    without_exclusivity = engine.check_consistency(
        policies, assume_mutually_exclusive_roles=False
    )

    assert with_assumptions.status is VerificationStatus.UNSAT
    assert without_role_witness.status is VerificationStatus.SAT
    assert without_exclusivity.status is VerificationStatus.SAT


def test_unknown_is_preserved_from_an_isolated_solver_response():
    class UnknownSolver:
        def add(self, *constraints):
            pass

        def check(self):
            return z3.unknown

    result = PolicyVerificationEngine(solver_factory=UnknownSolver).check_entailment(
        ["Administrators(Alice)"],
        "CanDelete(Alice, UserAccounts)",
    )
    assert result.status is VerificationStatus.UNKNOWN
    assert result.is_entailment is False


def test_conflicts_are_scoped_by_resource_and_conditions():
    parser = PolicyParser()
    detector = ConflictDetector()
    policies = [
        parser.parse_text("Administrators can delete user accounts.", "allow"),
        parser.parse_text("Administrators cannot delete invoices.", "different-resource"),
    ]
    assert detector.find_conflicts(policies) == []


def test_conditions_are_part_of_conflict_scope():
    parser = PolicyParser()
    detector = ConflictDetector()
    allow = parser.parse_text("Administrators can read user profiles.", "allow")
    deny = parser.parse_text("Administrators cannot read user profiles.", "deny")
    deny.conditions = [{"attribute": "environment", "value": "test"}]
    assert detector.find_conflicts([allow, deny]) == []


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
    assert body["overall_valid"] is True
    assert len(body["reports"]) == 1
    report = body["reports"][0]
    assert report["fol_formula"]["formula"]
    assert len(report["inference_trace"]) == 2
    assert report["solver_result"]["query_type"] == "entailment"
    assert report["submission"] is None

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
    detail = conflict.json()["detail"]["report"]
    assert detail["overall_valid"] is False
    assert [item["conflict_id"] for item in detail["conflicts"]] == ["p1:p2"]
    assert [report["conflicting_policies"] for report in detail["reports"]] == [
        ["p1", "p2"], ["p1", "p2"]
    ]


def test_api_verifies_every_submitted_policy():
    client = TestClient(app)
    response = client.post("/api/v1/verify-policies", json=[
        {
            "policy_id": "multi-only", "subject": "Administrators", "action": "Delete",
            "resource": "UserAccounts", "effect": "Allow", "interpretation": "only_role",
            "raw_text": "Only administrators can delete user accounts.",
        },
        {
            "policy_id": "multi-guest", "subject": "Guests", "action": "Delete",
            "resource": "UserAccounts", "effect": "Allow",
            "raw_text": "Guests can delete user accounts.",
        },
    ])
    assert response.status_code == 200
    body = response.json()
    assert body["overall_valid"] is False
    assert body["consistency_result"]["status"] == "UNSAT"
    assert [report["policy_id"] for report in body["reports"]] == ["multi-only", "multi-guest"]
    assert all(report["rule_status"] == "contradictory" for report in body["reports"])
    assert [report["submission"]["index"] for report in body["reports"]] == [1, 2]
    assert len({report["submission"]["submission_id"] for report in body["reports"]}) == 1
    records = client.get("/api/v1/verification-records").json()
    assert [record["policy_id"] for record in records[:2]] == ["multi-only", "multi-guest"]
    cached = [report["policy_id"] for report in client.get("/api/v1/rule-cache").json()]
    assert "multi-only" not in cached and "multi-guest" not in cached


def test_api_treats_unknown_consistency_as_not_valid(monkeypatch):
    monkeypatch.setattr(
        routes.pipeline.engine,
        "check_consistency",
        lambda *args, **kwargs: SolverResult(
            status=VerificationStatus.UNKNOWN,
            query_type="consistency",
            explanation="Z3 could not determine satisfiability.",
        ),
    )
    client = TestClient(app)
    response = client.post("/api/v1/verify-policies", json=[{
        "policy_id": "unknown-consistency",
        "subject": "Administrators",
        "action": "Delete",
        "resource": "UserAccounts",
        "effect": "Allow",
        "raw_text": "Administrators can delete user accounts.",
    }])
    assert response.status_code == 200
    body = response.json()
    assert body["overall_valid"] is False
    assert body["reports"][0]["is_valid"] is False
    assert body["reports"][0]["rule_status"] == "contradictory"
    assert body["consistency_result"]["status"] == "UNKNOWN"
    assert "unknown-consistency" not in [
        report["policy_id"] for report in client.get("/api/v1/rule-cache").json()
    ]


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
