"""Formal Verification ``verify-request`` decision -> ``VerifiedRuleInfo`` (Member 4-side compatibility adapter).

Converts the documented HTTP 200 body of Formal Verification's ``POST /api/v1/verify-request``
(contract ``verify-request-response-v1``, ``docs/output_contracts/``, produced by
``src/policy_verifier/verifier.py::PolicyVerificationEngine.check_request``) into the rule input
of this pipeline. It is a pure function over a caller-supplied mapping: it performs no I/O, does
not import or call Formal Verification, and is not wired into the API, the dashboard or the
runtime path.

What the result carries
-----------------------
One rule, ``formal-verification:request-decision``, holding only the enumerated ``decision``
(``ALLOWED``, ``DENIED``, ``NOT_DETERMINED``) and ``reason_code``, plus fixed provenance
attributes stating that this is a **request-level** decision, that **per-policy outcomes are
unavailable**, and that the association with the API event is **caller-asserted**.

What it never does
------------------
- It never reads, invents or infers an ``event_id``: the contract has none (its README says the
  caller must associate the response with the API event), so ``event_id`` comes only from the
  supplied ``UnifiedApiContext``. A report that carries its own ``event_id`` is rejected.
- It never creates per-policy outcomes or rule matches: the response states one decision for
  the whole request, and the ``constraints`` list says which formulas were encoded, not which
  policy decided.
- It never copies ``explanation``, ``constraints``, ``assumptions``, ``limitations`` or Z3 reason
  strings: they hold free text, formulas and subject identifiers (possible personal data).
- It never checks that the decision is about *this* HTTP request: the contract has no mapping
  from HTTP requests (the caller supplies subject, roles, action and resource to Formal
  Verification). That association is the caller's responsibility.

Fail-closed validation
----------------------
Required fields, the enumerations and the fixed decision/reason pairing of the schema are
checked, and the stated ``reason_code`` must equal the one recomputed from the three reported
solver statuses with Formal Verification's own branch order (contradictory, then unknown, then
entailed, then prohibited, otherwise not covered). Anything else raises a content-safe
:class:`RequestDecisionError` naming a field, never a value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .adapters import InputContractError
from .models import UnifiedApiContext, VerifiedRule, VerifiedRuleInfo

SOURCE_CONTRACT = "verify-request-response-v1"
DECISION_RULE_ID = "formal-verification:request-decision"
_REQUIRED_FIELDS = ("decision", "reason_code", "explanation", "consistency_query", "allow_query", "deny_query", "execution_time_ms")
_REASONS_BY_DECISION = {
    "ALLOWED": frozenset({"request_entailed"}),
    "DENIED": frozenset({"request_prohibited"}),
    "NOT_DETERMINED": frozenset({"policy_set_contradictory", "solver_unknown", "not_covered_by_policies"}),
}
_QUERY_FIELDS = ("consistency_query", "allow_query", "deny_query")
_SOLVER_STATUSES = frozenset({"sat", "unsat", "unknown"})


class RequestDecisionError(InputContractError):
    """The supplied report is not a valid ``verify-request-response-v1`` body. Messages name fields only."""


def request_decision_to_verified_rule_info(context: UnifiedApiContext, report: Mapping[str, Any]) -> VerifiedRuleInfo:
    """Convert one ``verify-request`` response into rule input for the event described by ``context``."""
    if not isinstance(context, UnifiedApiContext):
        raise RequestDecisionError("context must be a UnifiedApiContext")
    if not isinstance(report, Mapping):
        raise RequestDecisionError("report must be a mapping (the JSON body of verify-request)")
    if "event_id" in report:
        raise RequestDecisionError("report.event_id: the verify-request contract carries no event id; refusing an unrecognised contract")
    missing = [name for name in _REQUIRED_FIELDS if name not in report]
    if missing:
        raise RequestDecisionError(f"report is missing required field(s): {', '.join(missing)}")

    decision, reason_code = report["decision"], report["reason_code"]
    if not isinstance(decision, str) or decision not in _REASONS_BY_DECISION:
        raise RequestDecisionError("report.decision: must be ALLOWED, DENIED or NOT_DETERMINED")
    if not isinstance(reason_code, str) or reason_code not in _REASONS_BY_DECISION[decision]:
        raise RequestDecisionError("report.reason_code: does not pair with report.decision")
    statuses = {}
    for name in _QUERY_FIELDS:
        query = report[name]
        if not isinstance(query, Mapping) or not isinstance(query.get("status"), str) or query["status"] not in _SOLVER_STATUSES:
            raise RequestDecisionError(f"report.{name}.status: must be sat, unsat or unknown")
        statuses[name] = query["status"]
    elapsed = report["execution_time_ms"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise RequestDecisionError("report.execution_time_ms: must be a finite, non-negative number")
    if not isinstance(report["explanation"], str):
        raise RequestDecisionError("report.explanation: must be a string")
    if _reason_from_statuses(statuses) != reason_code:
        raise RequestDecisionError("report.reason_code: inconsistent with the reported solver statuses")

    rule = VerifiedRule(rule_id=DECISION_RULE_ID, attributes={
        "decision": decision,
        "reason_code": reason_code,
        "evidence_level": "request_level_decision",
        "per_policy_outcomes": "unavailable",
        "event_association": "caller_asserted",
        "source_contract": SOURCE_CONTRACT,
    })
    return VerifiedRuleInfo(event_id=context.event_id, rules=[rule])


def _reason_from_statuses(statuses: Mapping[str, str]) -> str:
    """Formal Verification's own decision order (``PolicyVerificationEngine.check_request``)."""
    if statuses["consistency_query"] == "unsat":
        return "policy_set_contradictory"
    if "unknown" in statuses.values():
        return "solver_unknown"
    if statuses["allow_query"] == "unsat":
        return "request_entailed"
    if statuses["deny_query"] == "unsat":
        return "request_prohibited"
    return "not_covered_by_policies"
