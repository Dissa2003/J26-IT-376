from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import z3

from .models import SecurityPolicy, SolverResult, VerificationReport, VerificationStatus


class PolicyVerificationEngine:
    """Z3 adapter; parsing and the explicit calculus remain separate."""

    def __init__(self, solver_factory: Callable[[], z3.Solver] = z3.Solver) -> None:
        self._solver_factory = solver_factory

    def check_consistency(self, policies: list[SecurityPolicy]) -> SolverResult:
        solver = self._solver_factory()
        symbols: dict[str, z3.BoolRef] = {}
        for policy in policies:
            key = f"{policy.subject}:{policy.action}:{policy.resource}"
            symbol = symbols.setdefault(key, z3.Bool(key.replace(":", "_")))
            solver.add(symbol if policy.effect.value == "Allow" else z3.Not(symbol))
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


def _parse_expression(value: str) -> z3.BoolRef:
    expression = value.strip()
    if "→" in expression:
        left, right = expression.split("→", 1)
        return z3.Implies(_parse_expression(left), _parse_expression(right))
    if expression.startswith("¬"):
        return z3.Not(_parse_expression(expression[1:]))
    return _symbol(expression)
