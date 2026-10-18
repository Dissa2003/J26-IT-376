from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import z3

from .models import SecurityPolicy, SolverResult, VerificationReport, VerificationStatus


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
