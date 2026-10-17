from __future__ import annotations

from typing import Any

import z3

from .models import SecurityPolicy, SolverResult, VerificationReport, VerificationStatus


class PolicyVerificationEngine:
    """Z3 adapter; parsing and the explicit calculus remain separate."""

    def check_consistency(self, policies: list[SecurityPolicy]) -> SolverResult:
        solver = z3.Solver()
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

    def check_entailment(self, antecedent_key: str, conclusion_key: str) -> SolverResult:
        antecedent = z3.Bool(antecedent_key.replace(":", "_"))
        conclusion = z3.Bool(conclusion_key.replace(":", "_"))
        solver = z3.Solver()
        solver.add(antecedent)
        solver.add(z3.Implies(antecedent, conclusion))
        solver.add(z3.Not(conclusion))
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
