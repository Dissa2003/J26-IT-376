"""Tests for request_decision_adapter (Formal Verification ``verify-request`` -> ``VerifiedRuleInfo``), D4-7G3.

The genuine-output tests call Formal Verification's real ``POST /api/v1/verify-request`` in process (the same pattern as
its own tests) and feed the unmodified JSON body to the adapter. They need ``z3-solver`` (in the root requirements).
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

import tokenization_ai.request_decision_adapter as rda
from tokenization_ai.adapters import InputContractError
from tokenization_ai.models import UnifiedApiContext, VerifiedRuleInfo
from tokenization_ai.request_decision_adapter import (
    DECISION_RULE_ID,
    RequestDecisionError,
    request_decision_to_verified_rule_info,
)
from tokenization_ai.service import TokenizationAIService

CONTEXT = UnifiedApiContext(event_id="evt-fv-1", method="DELETE", path="/api/v1/users/42", client_ip="10.0.0.5",
                            headers={"authorization": "Bearer synthetic-token"}, payload={"email": "person@example.invalid"})


def _report(decision="ALLOWED", reason="request_entailed", consistency="sat", allow="unsat", deny="sat", **extra):
    body = {"decision": decision, "reason_code": reason, "explanation": "Subject Alice-01 may delete UserAccounts.",
            "consistency_query": {"status": consistency, "reason_unknown": None}, "allow_query": {"status": allow, "reason_unknown": None},
            "deny_query": {"status": deny, "reason_unknown": None},
            "constraints": [{"constraint_id": "p1:rule", "expression": "∀x (Administrators(x) → CanDelete(x, UserAccounts))",
                             "kind": "rule", "source_policy_ids": ["p1"]}],
            "assumptions": ["Role-witness axiom enabled."], "execution_time_ms": 1.25, "limitations": ["process-local"]}
    body.update(extra)
    return body


# -- genuine Formal Verification output -----------------------------------------------------------------------------------


def _fv_client():
    pytest.importorskip("z3")
    from fastapi.testclient import TestClient

    from api.routes import app
    return TestClient(app)


def _fv_request(client, policy_text, effect, interpretation, roles, subject="Administrators"):
    policy = {"policy_id": "p1", "subject": subject, "action": "Delete", "resource": "UserAccounts", "effect": effect,
              "raw_text": policy_text, "interpretation": interpretation}
    response = client.post("/api/v1/verify-request", json={"policies": [policy], "request": {
        "subject_id": "subject-1", "roles": roles, "action": "Delete", "resource": "UserAccounts"}})
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("policy_text,effect,interpretation,roles,decision,reason", [
    ("Administrators can delete user accounts.", "Allow", "role_permission", ["Administrators"], "ALLOWED", "request_entailed"),
    ("Administrators cannot delete user accounts.", "Deny", "role_prohibition", ["Administrators"], "DENIED", "request_prohibited"),
    ("Only administrators can delete user accounts.", "Allow", "only_role", ["Guests"], "DENIED", "request_prohibited"),
    ("Administrators can delete user accounts.", "Allow", "role_permission", ["Guests"], "NOT_DETERMINED", "not_covered_by_policies"),
])
def test_genuine_verify_request_output_is_converted_without_invented_data(policy_text, effect, interpretation, roles, decision, reason):
    body = _fv_request(_fv_client(), policy_text, effect, interpretation, roles)
    assert "event_id" not in body  # the real contract has none
    info = request_decision_to_verified_rule_info(CONTEXT, body)
    assert isinstance(info, VerifiedRuleInfo) and info.event_id == CONTEXT.event_id and len(info.rules) == 1
    rule = info.rules[0]
    assert rule.rule_id == DECISION_RULE_ID and rule.attributes == {
        "decision": decision, "reason_code": reason, "evidence_level": "request_level_decision", "per_policy_outcomes": "unavailable",
        "event_association": "caller_asserted", "source_contract": "verify-request-response-v1"}


def test_a_genuine_contradictory_policy_set_is_carried_as_not_determined():
    client = _fv_client()
    policies = [{"policy_id": "p1", "subject": "Administrators", "action": "Delete", "resource": "UserAccounts", "effect": "Allow",
                 "raw_text": "Administrators can delete user accounts.", "interpretation": "role_permission"},
                {"policy_id": "p2", "subject": "Auditors", "action": "Delete", "resource": "UserAccounts", "effect": "Deny",
                 "raw_text": "Auditors cannot delete user accounts.", "interpretation": "role_prohibition"}]
    body = client.post("/api/v1/verify-request", json={"policies": policies, "request": {
        "subject_id": "s", "roles": ["Administrators", "Auditors"], "action": "Delete", "resource": "UserAccounts"}}).json()
    rule = request_decision_to_verified_rule_info(CONTEXT, body).rules[0]
    assert (rule.attributes["decision"], rule.attributes["reason_code"]) == ("NOT_DETERMINED", "policy_set_contradictory")


def test_the_converted_decision_runs_through_the_pipeline_with_masking_active():
    info = request_decision_to_verified_rule_info(CONTEXT, _report())
    analysis = TokenizationAIService.create_default().analyze(CONTEXT, info)
    tokens = " ".join(analysis.sequence.tokens)
    assert "ALLOWED" in tokens and "request_entailed" in tokens  # the decision reaches the rule segment
    assert "person@example.invalid" not in tokens and "synthetic-token" not in tokens  # request masking is unchanged
    assert 0.0 <= analysis.prediction.score <= 1.0


# -- what is never copied or invented -------------------------------------------------------------------------------------


def test_free_text_formulas_and_subject_identifiers_are_never_copied():
    rule = request_decision_to_verified_rule_info(CONTEXT, _report()).rules[0]
    dumped = json.dumps(rule.model_dump())
    for leaked in ("Alice-01", "∀x", "Administrators", "p1", "Role-witness", "process-local", "1.25"):
        assert leaked not in dumped


def test_the_event_id_comes_only_from_the_context_and_a_report_event_id_is_refused():
    assert request_decision_to_verified_rule_info(CONTEXT, _report()).event_id == "evt-fv-1"
    with pytest.raises(RequestDecisionError, match="event_id"):
        request_decision_to_verified_rule_info(CONTEXT, _report(event_id="evt-other"))


def test_additional_unknown_fields_are_tolerated_but_not_copied():
    rule = request_decision_to_verified_rule_info(CONTEXT, _report(future_field={"x": 1})).rules[0]
    assert "future_field" not in rule.attributes  # the schema allows additional properties; only the allowlist is carried


# -- fail-closed validation -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mutate,field", [
    (lambda r: r.pop("decision"), "decision"),
    (lambda r: r.pop("allow_query"), "allow_query"),
    (lambda r: r.pop("execution_time_ms"), "execution_time_ms"),
    (lambda r: r.update(decision="ALLOW"), "decision"),
    (lambda r: r.update(decision=None), "decision"),
    (lambda r: r.update(reason_code="request_prohibited"), "reason_code"),  # wrong pairing for ALLOWED
    (lambda r: r.update(allow_query={"status": "UNSAT"}), "allow_query"),
    (lambda r: r.update(deny_query="sat"), "deny_query"),
    (lambda r: r.update(execution_time_ms=-1), "execution_time_ms"),
    (lambda r: r.update(execution_time_ms=float("nan")), "execution_time_ms"),
    (lambda r: r.update(execution_time_ms=True), "execution_time_ms"),
    (lambda r: r.update(explanation=None), "explanation"),
    (lambda r: r.update(allow_query={"status": "sat"}), "reason_code"),  # ALLOWED claimed but the request is not entailed
    (lambda r: r.update(consistency_query={"status": "unsat"}), "reason_code"),  # a contradictory set reported as ALLOWED
    (lambda r: r.update(deny_query={"status": "unknown"}), "reason_code"),  # an unknown query reported as ALLOWED
])
def test_malformed_or_inconsistent_reports_are_rejected_naming_a_field(mutate, field):
    report = _report()
    mutate(report)
    with pytest.raises(RequestDecisionError) as caught:
        request_decision_to_verified_rule_info(CONTEXT, report)
    assert field in str(caught.value) and "Alice-01" not in str(caught.value)


@pytest.mark.parametrize("statuses,decision,reason", [
    (("unsat", "sat", "sat"), "NOT_DETERMINED", "policy_set_contradictory"),
    (("unsat", "unknown", "unknown"), "NOT_DETERMINED", "policy_set_contradictory"),
    (("sat", "unknown", "sat"), "NOT_DETERMINED", "solver_unknown"),
    (("sat", "unsat", "sat"), "ALLOWED", "request_entailed"),
    (("sat", "sat", "unsat"), "DENIED", "request_prohibited"),
    (("sat", "sat", "sat"), "NOT_DETERMINED", "not_covered_by_policies"),
])
def test_every_documented_decision_is_accepted_when_consistent(statuses, decision, reason):
    c, a, d = statuses
    rule = request_decision_to_verified_rule_info(CONTEXT, _report(decision, reason, c, a, d)).rules[0]
    assert (rule.attributes["decision"], rule.attributes["reason_code"]) == (decision, reason)


@pytest.mark.parametrize("context,report", [(None, _report()), ({"event_id": "x"}, _report()), (CONTEXT, None), (CONTEXT, [_report()]),
                                            (CONTEXT, json.dumps(_report()))])
def test_wrong_input_types_are_rejected(context, report):
    with pytest.raises(RequestDecisionError):
        request_decision_to_verified_rule_info(context, report)  # type: ignore[arg-type]
    assert issubclass(RequestDecisionError, InputContractError)


def test_the_input_report_is_not_modified():
    report = _report()
    before = copy.deepcopy(report)
    request_decision_to_verified_rule_info(CONTEXT, report)
    assert report == before


def test_the_module_does_no_io_and_imports_no_other_member_code():
    tree = ast.parse(Path(rda.__file__).read_text(encoding="utf-8"))
    imported = {(("." * n.level) + (n.module or "")) if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert imported <= {"__future__", "math", "collections.abc", "typing", ".adapters", ".models"}, imported
