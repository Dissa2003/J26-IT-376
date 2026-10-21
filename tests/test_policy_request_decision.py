from fastapi.testclient import TestClient
import z3

from api.routes import app, pipeline
from src.policy_verifier.verifier import PolicyVerificationEngine


client = TestClient(app)


def _request(policy_texts, roles, action="Delete", resource="UserAccounts", subject_id="subject-1"):
    policies = []
    for index, text in enumerate(policy_texts):
        if text.startswith("Only "):
            subject, action_text, resource_text = "Administrators", "Delete", "UserAccounts"
            effect = "Allow"
            interpretation = "only_role"
        elif " cannot " in text:
            subject = "Auditors" if text.startswith("Auditors") else "Administrators"
            action_text, resource_text = "Delete", "UserAccounts"
            effect = "Deny"
            interpretation = "role_prohibition"
        else:
            subject, action_text, resource_text = "Administrators", "Delete", "UserAccounts"
            effect = "Allow"
            interpretation = "role_permission"
        policies.append({
            "policy_id": f"request-{index}",
            "subject": subject,
            "action": action_text,
            "resource": resource_text,
            "effect": effect,
            "raw_text": text,
            "interpretation": interpretation,
        })
    return client.post(
        "/api/v1/verify-request",
        json={
            "policies": policies,
            "request": {
                "subject_id": subject_id,
                "roles": roles,
                "action": action,
                "resource": resource,
            },
        },
    )


def test_independent_request_decisions():
    assert _request(["Administrators can delete user accounts."], ["Administrators"]).json()["decision"] == "ALLOWED"
    denied_by_missing_role = _request(
        ["Administrators can delete user accounts."], ["Guests"]
    ).json()
    assert denied_by_missing_role["decision"] == "NOT_DETERMINED"
    assert denied_by_missing_role["reason_code"] == "not_covered_by_policies"

    assert _request(
        ["Only administrators can delete user accounts."], ["Guests"]
    ).json()["decision"] == "DENIED"
    assert _request(
        ["Only administrators can delete user accounts."], ["Administrators"]
    ).json()["decision"] == "NOT_DETERMINED"
    assert _request(
        ["Administrators cannot delete user accounts."], ["Administrators"]
    ).json()["decision"] == "DENIED"


def test_all_policies_are_checked_for_request_consistency():
    response = _request(
        [
            "Administrators can delete user accounts.",
            "Auditors cannot delete user accounts.",
        ],
        ["Administrators", "Auditors"],
    )
    body = response.json()
    assert body["decision"] == "NOT_DETERMINED"
    assert body["reason_code"] == "policy_set_contradictory"
    assert body["consistency_query"]["status"] == "unsat"


def test_uncovered_action_and_request_mutations():
    assert _request(
        ["Administrators can delete user accounts."],
        ["Administrators"],
        action="Read",
        resource="UserProfiles",
    ).json()["decision"] == "NOT_DETERMINED"

    admin = _request(["Administrators can delete user accounts."], ["Administrators"]).json()
    guest = _request(["Administrators can delete user accounts."], ["Guests"]).json()
    other_subject = _request(
        ["Administrators can delete user accounts."],
        ["Administrators"],
        subject_id="different-subject",
    ).json()
    assert admin["decision"] == "ALLOWED"
    assert guest["decision"] == "NOT_DETERMINED"
    assert other_subject["decision"] == admin["decision"]


def test_unmatched_roles_are_reported_only_when_not_determined():
    standard = [
        "The request roles are a complete role assignment for the subject.",
        "Policy conditions are outside the supported policy grammar.",
    ]
    near_miss = _request(["Administrators can delete user accounts."], ["Administrator"]).json()
    assert near_miss["decision"] == "NOT_DETERMINED"
    assert near_miss["limitations"] == standard + [
        "Role 'Administrator' matched no policy role. Did you mean 'Administrators'?"
    ]

    unrelated = _request(["Administrators can delete user accounts."], ["Guests"]).json()
    assert unrelated["limitations"] == standard + ["Role 'Guests' matched no policy role."]

    allowed = _request(
        ["Administrators can delete user accounts."], ["Administrators", "Contractors"]
    ).json()
    assert allowed["decision"] == "ALLOWED"
    assert allowed["limitations"] == standard


def test_request_names_match_policy_regardless_of_spacing_and_case():
    response = _request(
        ["Administrators can delete user accounts."],
        ["Administrators"],
        resource="user accounts",
    ).json()
    assert response["decision"] == "ALLOWED"
    assert response["reason_code"] == "request_entailed"

    for resource in ("UserAccounts", "Useraccounts", "useraccounts", "USER ACCOUNTS"):
        assert _request(
            ["Administrators can delete user accounts."],
            ["Administrators"],
            resource=resource,
        ).json()["decision"] == "ALLOWED", resource
    assert _request(
        ["Administrators can delete user accounts."],
        ["administrators"],
        action="delete",
        resource="user accounts",
    ).json()["decision"] == "ALLOWED"


def test_unknown_solver_is_not_determined(monkeypatch):
    class UnknownSolver:
        def add(self, *constraints):
            pass

        def check(self):
            return z3.unknown

        def reason_unknown(self):
            return "mocked unknown"

    monkeypatch.setattr(
        pipeline,
        "engine",
        PolicyVerificationEngine(solver_factory=UnknownSolver),
    )
    response = _request(["Administrators can delete user accounts."], ["Administrators"])
    assert response.status_code == 200
    assert response.json()["decision"] == "NOT_DETERMINED"
    assert response.json()["reason_code"] == "solver_unknown"
    assert "mocked unknown" in response.json()["explanation"]


def test_invalid_request_body_returns_422():
    base = {
        "policies": [{
            "policy_id": "p1",
            "subject": "Administrators",
            "action": "Delete",
            "resource": "UserAccounts",
            "effect": "Allow",
            "raw_text": "Administrators can delete user accounts.",
        }],
        "request": {
            "subject_id": "subject-1",
            "roles": ["Administrators"],
            "action": "Delete",
            "resource": "UserAccounts",
        },
    }
    empty_subject = {**base, "request": {**base["request"], "subject_id": ""}}
    empty_action = {**base, "request": {**base["request"], "action": ""}}
    assert client.post("/api/v1/verify-request", json=empty_subject).status_code == 422
    assert client.post("/api/v1/verify-request", json=empty_action).status_code == 422
    assert client.post("/api/v1/verify-request", json={"policies": []}).status_code == 422


def test_raw_text_only_role_must_match_submitted_interpretation():
    payload = {
        "policies": [{
            "policy_id": "only-mismatch",
            "subject": "Administrators",
            "action": "Delete",
            "resource": "UserAccounts",
            "effect": "Allow",
            "raw_text": "Only administrators can delete user accounts.",
        }],
        "request": {
            "subject_id": "subject-1",
            "roles": ["Administrators"],
            "action": "Delete",
            "resource": "UserAccounts",
        },
    }
    mismatch = client.post("/api/v1/verify-request", json=payload)
    assert mismatch.status_code == 422
    assert mismatch.json()["detail"]["code"] == "policy_field_mismatch"
    assert mismatch.json()["detail"]["policies"] == [{
        "policy_id": "only-mismatch",
        "fields": ["interpretation"],
    }]


def test_matching_raw_text_policy_passes():
    payload = {
        "policies": [{
            "policy_id": "only-match",
            "subject": "Administrators",
            "action": "Delete",
            "resource": "UserAccounts",
            "effect": "Allow",
            "interpretation": "only_role",
            "raw_text": "Only administrators can delete user accounts.",
        }],
        "request": {
            "subject_id": "subject-1",
            "roles": ["Guests"],
            "action": "Delete",
            "resource": "UserAccounts",
        },
    }
    response = client.post("/api/v1/verify-request", json=payload)
    assert response.status_code == 200
    assert response.json()["decision"] == "DENIED"


def test_legacy_endpoints_keep_their_existing_shapes():
    text_response = client.post(
        "/api/v1/verify-text",
        json={
            "policy_text": "Administrators can delete user accounts.",
            "policy_id": "legacy-text",
        },
    )
    assert text_response.status_code == 200
    assert text_response.json()["solver_result"]["query_type"] == "entailment"

    policies_response = client.post(
        "/api/v1/verify-policies",
        json=[{
            "policy_id": "legacy-policies",
            "subject": "Administrators",
            "action": "Delete",
            "resource": "UserAccounts",
            "effect": "Allow",
            "raw_text": "Administrators can delete user accounts.",
        }],
    )
    assert policies_response.status_code == 200
    assert policies_response.json()["reports"][0]["solver_result"]["query_type"] == "entailment"
