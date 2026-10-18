from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import z3

from .models import (
    FormalConstraint,
    RequestDecision,
    RequestDecisionReport,
    RequestQueryResult,
    SecurityPolicy,
    SolverResult,
    VerificationReport,
    VerificationStatus,
)


class PolicyVerificationEngine:
    """Z3 adapter; parsing and the explicit calculus remain separate."""

    def __init__(self, solver_factory: Callable[[], z3.Solver] = z3.Solver) -> None:
        self._solver_factory = solver_factory

    def check_consistency(
        self,
        policies: list[SecurityPolicy],
        *,
        assume_mutually_exclusive_roles: bool = True,
        assume_role_witness: bool = True,
    ) -> SolverResult:
        solver = self._solver_factory()
        individual_sort = z3.DeclareSort("PolicyIndividual")
        individual = z3.Const("individual", individual_sort)
        role_predicates: dict[str, z3.FuncDeclRef] = {}
        action_predicates: dict[tuple[str, str], z3.FuncDeclRef] = {}

        def role_predicate(subject: str) -> z3.FuncDeclRef:
            return role_predicates.setdefault(
                subject,
                z3.Function(
                    f"Role_{_z3_name(subject)}",
                    individual_sort,
                    z3.BoolSort(),
                ),
            )

        def action_predicate(action: str, resource: str) -> z3.FuncDeclRef:
            key = (action, resource)
            return action_predicates.setdefault(
                key,
                z3.Function(
                    f"Can_{_z3_name(action)}_{_z3_name(resource)}",
                    individual_sort,
                    z3.BoolSort(),
                ),
            )

        for policy in policies:
            role = role_predicate(policy.subject)
            permission = action_predicate(policy.action, policy.resource)
            if policy.interpretation.value == "only_role":
                formula = z3.Implies(permission(individual), role(individual))
            elif policy.effect.value == "Deny":
                formula = z3.Implies(role(individual), z3.Not(permission(individual)))
            else:
                formula = z3.Implies(role(individual), permission(individual))
            solver.add(z3.ForAll([individual], formula))

        subjects = sorted(role_predicates)
        if assume_role_witness:
            for subject in subjects:
                witness = z3.Const(f"Witness_{_z3_name(subject)}", individual_sort)
                solver.add(role_predicates[subject](witness))
        if assume_mutually_exclusive_roles and len(subjects) > 1:
            for index, left_subject in enumerate(subjects):
                for right_subject in subjects[index + 1 :]:
                    solver.add(
                        z3.ForAll(
                            [individual],
                            z3.Not(
                                z3.And(
                                    role_predicates[left_subject](individual),
                                    role_predicates[right_subject](individual),
                                )
                            ),
                        )
                    )
        result = solver.check()
        status = _status(result)
        message = {
            VerificationStatus.SAT: "The supplied constraints have a model.",
            VerificationStatus.UNSAT: "The supplied constraints are inconsistent.",
            VerificationStatus.UNKNOWN: "Z3 could not determine satisfiability.",
        }[status]
        return SolverResult(status=status, query_type="consistency", explanation=message)

    def check_entailment(
        self,
        premises: Sequence[str] | str,
        conclusion_key: str,
        rule_key: str | None = None,
    ) -> SolverResult:
        """Check whether K entails C by solving K ∧ ¬C.

        The two-string form is retained for legacy callers: it represents the
        common K = {antecedent, antecedent -> conclusion} case.
        """
        if isinstance(premises, str):
            premise_values = [premises]
            if rule_key is None:
                rule_key = f"{premises} → {conclusion_key}"
        else:
            premise_values = list(premises)
        solver = self._solver_factory()
        for premise in premise_values:
            solver.add(_parse_expression(premise))
        if rule_key is not None and rule_key not in premise_values:
            solver.add(_parse_expression(rule_key))
        solver.add(z3.Not(_parse_expression(conclusion_key)))
        result = solver.check()
        if result == z3.unsat:
            return SolverResult(
                status=VerificationStatus.UNSAT,
                query_type="entailment",
                is_entailment=True,
                explanation="Premises together with the negated conclusion are unsatisfiable; the conclusion is entailed.",
            )
        if result == z3.unknown:
            return SolverResult(
                status=VerificationStatus.UNKNOWN,
                query_type="entailment",
                explanation="Z3 could not determine whether the conclusion is entailed.",
            )
        return SolverResult(
            status=VerificationStatus.SAT,
            query_type="entailment",
            model={"counterexample": "premises do not entail the conclusion"},
            explanation="A model exists with the negated conclusion; entailment was not established.",
        )

    def check_request(
        self,
        policies: list[SecurityPolicy],
        *,
        subject_id: str,
        roles: list[str],
        action: str,
        resource: str,
        assume_role_witness: bool = True,
    ) -> RequestDecisionReport:
        import time

        start = time.perf_counter()
        individual_sort = z3.DeclareSort("PolicyIndividual")
        subject = z3.Const(f"Subject_{_z3_name(subject_id)}", individual_sort)
        role_names = sorted({policy.subject for policy in policies} | set(roles))
        role_predicates = {
            name: z3.Function(
                f"Role_{_z3_name(name)}",
                individual_sort,
                z3.BoolSort(),
            )
            for name in role_names
        }
        action_predicates: dict[tuple[str, str], z3.FuncDeclRef] = {}

        def action_predicate(
            action_name: str, resource_name: str
        ) -> z3.FuncDeclRef:
            key = (action_name, resource_name)
            return action_predicates.setdefault(
                key,
                z3.Function(
                    f"Can_{_z3_name(action_name)}_{_z3_name(resource_name)}",
                    individual_sort,
                    z3.BoolSort(),
                ),
            )

        policy_formulas: list[z3.BoolRef] = []
        constraints: list[FormalConstraint] = []
        for policy in policies:
            role = role_predicates[policy.subject]
            permission = action_predicate(policy.action, policy.resource)
            if policy.interpretation.value == "only_role":
                formula = z3.ForAll(
                    [subject],
                    z3.Implies(permission(subject), role(subject)),
                )
                expression = (
                    f"∀x (Can{policy.action}(x, {policy.resource}) → "
                    f"{policy.subject}(x))"
                )
            elif policy.effect.value == "Deny":
                formula = z3.ForAll(
                    [subject],
                    z3.Implies(role(subject), z3.Not(permission(subject))),
                )
                expression = (
                    f"∀x ({policy.subject}(x) → ¬Can{policy.action}"
                    f"(x, {policy.resource}))"
                )
            else:
                formula = z3.ForAll(
                    [subject],
                    z3.Implies(role(subject), permission(subject)),
                )
                expression = (
                    f"∀x ({policy.subject}(x) → Can{policy.action}"
                    f"(x, {policy.resource}))"
                )
            policy_formulas.append(formula)
            constraints.append(
                FormalConstraint(
                    constraint_id=f"{policy.policy_id}:rule",
                    expression=expression,
                    kind="rule",
                    source_policy_ids=[policy.policy_id],
                )
            )

        request_roles = set(roles)
        role_facts: list[z3.BoolRef] = []
        for role_name in role_names:
            listed = role_name in request_roles
            role_facts.append(
                role_predicates[role_name](subject)
                if listed
                else z3.Not(role_predicates[role_name](subject))
            )
            constraints.append(
                FormalConstraint(
                    constraint_id=f"request:role:{_z3_name(role_name)}",
                    expression=(
                        f"{role_name}({subject_id})"
                        if listed
                        else f"¬{role_name}({subject_id})"
                    ),
                    kind="fact",
                )
            )

        if assume_role_witness:
            for role_name in role_names:
                witness = z3.Const(
                    f"Witness_{_z3_name(role_name)}",
                    individual_sort,
                )
                policy_formulas.append(role_predicates[role_name](witness))

        requested_permission = action_predicate(action, resource)(subject)
        q_expression = f"Can{action}({subject_id}, {resource})"
        constraints.extend(
            [
                FormalConstraint(
                    constraint_id="request:query",
                    expression=q_expression,
                    kind="query",
                ),
                FormalConstraint(
                    constraint_id="request:negated-query",
                    expression=f"¬{q_expression}",
                    kind="negated_query",
                ),
            ]
        )

        def run_query(extra: z3.BoolRef | None = None) -> RequestQueryResult:
            solver = self._solver_factory()
            solver.add(*policy_formulas, *role_facts)
            if extra is not None:
                solver.add(extra)
            result = solver.check()
            raw_status = str(result)
            if result == z3.unknown:
                reason = (
                    solver.reason_unknown()
                    if hasattr(solver, "reason_unknown")
                    else "solver returned unknown"
                )
                return RequestQueryResult(
                    status="unknown",
                    reason_unknown=reason,
                )
            return RequestQueryResult(status=raw_status)

        consistency_query = run_query()
        if consistency_query.status == "unsat":
            decision = RequestDecision.NOT_DETERMINED
            reason_code = "policy_set_contradictory"
            explanation = "The submitted policies and complete request role assignment are inconsistent."
            allow_query = run_query(z3.Not(requested_permission))
            deny_query = run_query(requested_permission)
        else:
            allow_query = run_query(z3.Not(requested_permission))
            deny_query = run_query(requested_permission)
            queries = (consistency_query, allow_query, deny_query)
            if any(query.status == "unknown" for query in queries):
                decision = RequestDecision.NOT_DETERMINED
                reason_code = "solver_unknown"
                unknown_reasons = [
                    query.reason_unknown
                    for query in queries
                    if query.status == "unknown" and query.reason_unknown
                ]
                explanation = (
                    "Z3 returned unknown for at least one request decision query."
                    + (f" Reasons: {'; '.join(unknown_reasons)}" if unknown_reasons else "")
                )
            elif allow_query.status == "unsat":
                decision = RequestDecision.ALLOWED
                reason_code = "request_entailed"
                explanation = "The request is entailed by the submitted policies and role assignment."
            elif deny_query.status == "unsat":
                decision = RequestDecision.DENIED
                reason_code = "request_prohibited"
                explanation = "The negation of the request is entailed by the submitted policies and role assignment."
            else:
                decision = RequestDecision.NOT_DETERMINED
                reason_code = "not_covered_by_policies"
                explanation = "Neither the request nor its negation is entailed by the submitted policies."

        return RequestDecisionReport(
            decision=decision,
            reason_code=reason_code,
            explanation=explanation,
            consistency_query=consistency_query,
            allow_query=allow_query,
            deny_query=deny_query,
            constraints=constraints,
            assumptions=[
                "Role-witness axiom enabled."
                if assume_role_witness
                else "Role-witness axiom disabled.",
                "Role-exclusivity axiom disabled for independent request checking.",
            ],
            execution_time_ms=(time.perf_counter() - start) * 1000,
            limitations=[
                "The request roles are a complete role assignment for the subject.",
                "Policy conditions are outside the supported policy grammar.",
            ],
        )

    def process_and_verify(self, policies: list[dict[str, Any] | SecurityPolicy]) -> VerificationReport:
        """Legacy compatibility method for callers using the old report shape."""
        typed = [p if isinstance(p, SecurityPolicy) else SecurityPolicy.model_validate(p) for p in policies]
        result = self.check_consistency(typed)
        return VerificationReport(
            is_valid=result.status is VerificationStatus.SAT,
            status=result.status.value,
            fol_expressions=[],
            conflicting_policies=None,
            execution_time_ms=0.0,
        )


def _status(result: z3.CheckSatResult) -> VerificationStatus:
    if result == z3.sat:
        return VerificationStatus.SAT
    if result == z3.unsat:
        return VerificationStatus.UNSAT
    return VerificationStatus.UNKNOWN


def _symbol(value: str) -> z3.BoolRef:
    return z3.Bool(value.replace(":", "_").replace(" ", "_"))


def _z3_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _parse_expression(value: str) -> z3.BoolRef:
    expression = value.strip()
    if "→" in expression:
        left, right = expression.split("→", 1)
        return z3.Implies(_parse_expression(left), _parse_expression(right))
    if expression.startswith("¬"):
        return z3.Not(_parse_expression(expression[1:]))
    return _symbol(expression)
