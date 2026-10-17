from __future__ import annotations

from .models import FOLFormula, PolicyInterpretation, Predicate, StructuredPolicy


def to_fol(policy: StructuredPolicy) -> FOLFormula:
    attrs = policy.attributes
    permission = Predicate(name="Can" + attrs.action, arguments=["x", attrs.resource])
    role = Predicate(name=attrs.subject, arguments=["x"])
    if attrs.interpretation is PolicyInterpretation.ONLY_ROLE:
        antecedent, consequent = permission, role
        explanation = "Anyone who can perform the action must have the named role."
    elif attrs.effect.value == "Deny":
        antecedent, consequent = role, permission.model_copy(update={"negated": True})
        explanation = "Members of the named role are prohibited from performing the action."
    else:
        antecedent, consequent = role, permission
        explanation = "Members of the named role are permitted to perform the action."
    return FOLFormula(
        formula=f"∀x ({antecedent.render()} → {consequent.render()})",
        antecedent=antecedent,
        consequent=consequent,
        interpretation=attrs.interpretation,
        explanation=explanation,
    )
