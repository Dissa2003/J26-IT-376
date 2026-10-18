from __future__ import annotations

import time

from .calculus import infer
from .conflict_detector import ConflictDetector
from .constraints import generate_constraints
from .logic import to_fol
from .models import (
    PolicyAttributes,
    SecurityPolicy,
    StructuredPolicy,
    VerificationExplanation,
    VerificationReport,
    VerificationStatus,
)
from .verifier import PolicyVerificationEngine


class VerificationPipeline:
    def __init__(self, engine: PolicyVerificationEngine | None = None) -> None:
        self.engine = engine or PolicyVerificationEngine()
        self.conflict_detector = ConflictDetector()

    def verify(self, policy: SecurityPolicy, individual: str = "Alice") -> VerificationReport:
        start = time.perf_counter()
        attrs = PolicyAttributes(
            subject=policy.subject,
            action=policy.action,
            resource=policy.resource,
            effect=policy.effect,
            conditions=policy.conditions,
            interpretation=policy.interpretation,
        )
        structured = StructuredPolicy(
            policy_id=policy.policy_id,
            raw_text=policy.raw_text,
            attributes=attrs,
            normalized_key=f"{policy.subject}:{policy.action}:{policy.resource}",
        )
        formula = to_fol(structured)
        trace = infer(formula, individual)
        constraints = generate_constraints(structured, formula)
        instantiated_rule = trace[0].conclusion
        request_fact = trace[1].premises[0]
        conclusion_key = trace[-1].conclusion
        constraints.extend([
            constraints[0].model_copy(update={
                "constraint_id": f"{policy.policy_id}:fact",
                "expression": request_fact,
                "kind": "fact",
            }),
            constraints[0].model_copy(update={
                "constraint_id": f"{policy.policy_id}:query",
                "expression": conclusion_key,
                "kind": "query",
            }),
            constraints[0].model_copy(update={
                "constraint_id": f"{policy.policy_id}:negated-query",
                "expression": f"¬{conclusion_key}",
                "kind": "negated_query",
            }),
        ])
        solver_result = self.engine.check_entailment(
            [request_fact, instantiated_rule],
            conclusion_key,
        )
        conflicts = self.conflict_detector.find_conflicts([policy])
        eligible = (
            solver_result.is_entailment
            and not conflicts
            and solver_result.status is VerificationStatus.UNSAT
        )
        rule_status = "eligible" if eligible else ("conflicted" if conflicts else "rejected")
        return VerificationReport(
            is_valid=eligible,
            status=solver_result.status.value,
            fol_expressions=[formula.formula],
            conflicting_policies=None,
            execution_time_ms=(time.perf_counter() - start) * 1000,
            policy_id=policy.policy_id,
            original_policy=policy.raw_text,
            attributes=attrs,
            structured_policy=structured,
            fol_formula=formula,
            inference_trace=trace,
            constraints=constraints,
            solver_result=solver_result,
            conflicts=conflicts,
            explanation=VerificationExplanation(
                summary=solver_result.explanation,
                formally_established=solver_result.is_entailment,
                evidence=[formula.formula, trace[-1].conclusion],
                limitations=["The request fact is represented by the deterministic demonstration individual."],
            ),
            rule_status=rule_status,
        )
