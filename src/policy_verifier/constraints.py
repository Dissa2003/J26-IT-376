from __future__ import annotations

from .models import FormalConstraint, FOLFormula, StructuredPolicy


def generate_constraints(policy: StructuredPolicy, formula: FOLFormula) -> list[FormalConstraint]:
    return [
        FormalConstraint(
            constraint_id=f"{policy.policy_id}:rule",
            expression=formula.formula,
            kind="rule",
            source_policy_ids=[policy.policy_id],
        )
    ]

