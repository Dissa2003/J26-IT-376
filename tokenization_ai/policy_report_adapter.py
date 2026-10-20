"""Isolated compatibility adapter: Member 2-style policy reports -> ``VerifiedRuleInfo``.

THIS IS NOT A COMPLETED MEMBER 2 INTEGRATION. Nothing here calls Member 2, imports
Member 2's code, or is wired into the pipeline, the API or the dashboard. It is a
strict conversion function over plain mappings that the *caller* supplies.

What an accepted report means -- and does not mean
--------------------------------------------------
Member 2's ``eligible`` status is a **policy-level** gate: the policy parsed, has no
recorded conflict, and its entailment check (a negated query that is UNSAT, so the
entailment holds) succeeded on a fixed demonstration individual. It is **not**
verification of a runtime request. ``SAT``/``UNSAT`` alone, parsed text alone, or any
report that is not eligible is never accepted as a verified rule here.

Event identity and applicability
--------------------------------
- ``event_id`` comes only from the supplied ``UnifiedApiContext``. It is never read from,
  or inferred from, a report; a report that carries its own ``event_id`` is rejected as
  an unrecognised contract.
- Which policies apply to the request is **asserted by the caller** through
  ``applicable_policy_ids`` and must match the supplied reports exactly. This module does
  not decide applicability and never maps HTTP fields onto policy terms.

Safety
------
- Only a small allowlist of structured attributes is copied. Raw policy text, formulas,
  traces, constraints, explanations and any other field are never copied.
- Reports with conditions are rejected: dropping them would silently turn a conditional
  policy into an unconditional rule.
- Invalid input raises a content-safe error that names a report index and a field, never
  a supplied value. There is no fallback to mock rules.
- Rule segments are mandatory in the tensor; the tensorizer raises ``ValueError`` rather
  than truncating rules. That behaviour is unchanged and applies to converted rules too.

The field names and enum values below were read from Member 2's published models and
are restated here as plain data; they can drift, and the adapter then rejects the input.
Any evaluation that uses this module needs updated source provenance under a new
protocol version; the archived KPI evidence does not cover it.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from .adapters import InputContractError
from .models import UnifiedApiContext, VerifiedRule, VerifiedRuleInfo

MAX_POLICY_REPORTS = 32

# Structured attributes copied into ``VerifiedRule.attributes``, in this order.
ALLOWED_ATTRIBUTES: tuple[str, ...] = ("subject", "action", "resource", "effect", "interpretation")

_ALLOWED_EFFECTS = frozenset({"Allow", "Deny"})
_ALLOWED_INTERPRETATIONS = frozenset({"role_permission", "only_role", "role_prohibition"})
# Printable text, no control characters, 1-64 chars, no leading/trailing whitespace.
_SAFE_TEXT = re.compile(r"[^\x00-\x20\x7f](?:[^\x00-\x1f\x7f]{0,62}[^\x00-\x20\x7f])?")


class PolicyReportError(InputContractError):
    """A policy report (or the call itself) is malformed, ambiguous or inconsistent."""


class PolicyReportNotEligibleError(PolicyReportError):
    """A well-formed report does not meet the policy-level eligibility requirements."""


def policy_reports_to_verified_rule_info(
    context: UnifiedApiContext,
    reports: Sequence[Mapping[str, Any]],
    *,
    applicable_policy_ids: Collection[str],
) -> VerifiedRuleInfo:
    """Convert eligible policy reports into rule information for ``context``.

    ``applicable_policy_ids`` is the caller's explicit assertion of which policies apply to
    this request. It must equal the set of ``policy_id`` values in ``reports`` exactly.
    The order of ``reports`` is preserved. Raises :class:`PolicyReportError` (or its
    :class:`PolicyReportNotEligibleError` subclass) on any problem; never returns mock rules.
    """
    if not isinstance(context, UnifiedApiContext):
        raise PolicyReportError("context must be a UnifiedApiContext")
    reports = _check_reports_container(reports)
    asserted = _check_applicable_ids(applicable_policy_ids)

    rules: list[VerifiedRule] = []
    seen: set[str] = set()
    for index, report in enumerate(reports):
        rule = _convert_report(report, index)
        if rule.rule_id in seen:
            raise PolicyReportError(f"reports[{index}]: duplicate policy_id")
        seen.add(rule.rule_id)
        rules.append(rule)

    if seen != asserted:
        missing = len(asserted - seen)
        extra = len(seen - asserted)
        raise PolicyReportError(
            "applicable_policy_ids do not match the supplied reports exactly "
            f"({missing} asserted id(s) without a report, {extra} report(s) not asserted)"
        )
    return VerifiedRuleInfo(event_id=context.event_id, rules=rules)


def _check_reports_container(reports: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(reports, (str, bytes, bytearray, Mapping)) or not isinstance(reports, Sequence):
        raise PolicyReportError("reports must be a list or tuple of mappings")
    if not reports:
        raise PolicyReportError(
            "reports is empty; an empty rule list is a different input and is never "
            "produced from missing reports"
        )
    if len(reports) > MAX_POLICY_REPORTS:
        raise PolicyReportError(f"too many reports (maximum {MAX_POLICY_REPORTS})")
    return reports


def _check_applicable_ids(ids: Any) -> set[str]:
    if isinstance(ids, (str, bytes, bytearray, Mapping)) or not isinstance(ids, Collection):
        raise PolicyReportError("applicable_policy_ids must be a collection of policy ids")
    if not ids:
        raise PolicyReportError("applicable_policy_ids is empty; applicability must be explicit")
    checked: list[str] = []
    for position, value in enumerate(ids):
        if not _is_safe_text(value):
            raise PolicyReportError(f"applicable_policy_ids[{position}] is not a valid policy id")
        checked.append(value)
    if len(set(checked)) != len(checked):
        raise PolicyReportError("applicable_policy_ids contains duplicates")
    return set(checked)


def _convert_report(report: Any, index: int) -> VerifiedRule:
    where = f"reports[{index}]"
    if not isinstance(report, Mapping):
        raise PolicyReportError(f"{where} must be a mapping")
    if "event_id" in report:
        raise PolicyReportError(
            f"{where} carries its own event_id; event identity is taken only from the "
            "supplied context and this report shape is not recognised"
        )

    policy_id = report.get("policy_id")
    if not _is_safe_text(policy_id):
        raise PolicyReportError(f"{where}: policy_id is missing or not a valid id")

    # Eligibility, mirroring Member 2's documented gate. UNSAT is the expected solver
    # status here because the entailment is checked through its negated query.
    if report.get("is_valid") is not True:
        raise PolicyReportNotEligibleError(f"{where}: is_valid is not true")
    if report.get("rule_status") != "eligible":
        raise PolicyReportNotEligibleError(f"{where}: rule_status is not 'eligible'")
    if report.get("status") != "UNSAT":
        raise PolicyReportNotEligibleError(f"{where}: top-level status is not the expected UNSAT")

    solver = report.get("solver_result")
    if not isinstance(solver, Mapping):
        raise PolicyReportError(f"{where}: solver_result is missing or not a mapping")
    if solver.get("query_type") != "entailment":
        raise PolicyReportNotEligibleError(f"{where}: solver_result.query_type is not 'entailment'")
    if solver.get("status") != "UNSAT":
        raise PolicyReportNotEligibleError(f"{where}: solver_result.status is not the expected UNSAT")
    if solver.get("is_entailment") is not True:
        raise PolicyReportNotEligibleError(f"{where}: solver_result.is_entailment is not true")

    conflicts = report.get("conflicts")
    if not isinstance(conflicts, list):
        raise PolicyReportError(f"{where}: conflicts is missing or not a list")
    if conflicts:
        raise PolicyReportNotEligibleError(f"{where}: the report records conflicts")
    if report.get("conflicting_policies"):
        raise PolicyReportNotEligibleError(f"{where}: the report lists conflicting policies")

    attributes = report.get("attributes")
    if not isinstance(attributes, Mapping):
        raise PolicyReportError(f"{where}: attributes is missing or not a mapping")
    conditions = attributes.get("conditions")
    if conditions not in (None, []):
        raise PolicyReportNotEligibleError(
            f"{where}: policies with conditions cannot be represented safely and are rejected"
        )

    copied: dict[str, str] = {}
    for name in ALLOWED_ATTRIBUTES:
        value = attributes.get(name)
        if not _is_safe_text(value):
            raise PolicyReportError(f"{where}: attributes.{name} is missing or not a valid value")
        copied[name] = value
    if copied["effect"] not in _ALLOWED_EFFECTS:
        raise PolicyReportError(f"{where}: attributes.effect is not a recognised effect")
    if copied["interpretation"] not in _ALLOWED_INTERPRETATIONS:
        raise PolicyReportError(f"{where}: attributes.interpretation is not recognised")

    return VerifiedRule(rule_id=policy_id, attributes=copied)


def _is_safe_text(value: Any) -> bool:
    return type(value) is str and _SAFE_TEXT.fullmatch(value) is not None
