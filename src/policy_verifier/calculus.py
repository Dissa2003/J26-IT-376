from __future__ import annotations

from .models import FOLFormula, InferenceStep


def infer(formula: FOLFormula, individual: str) -> list[InferenceStep]:
    instantiated_antecedent = formula.antecedent.model_copy(
        update={"arguments": [individual if arg == "x" else arg for arg in formula.antecedent.arguments]}
    ).render()
    instantiated_consequent = formula.consequent.model_copy(
        update={"arguments": [individual if arg == "x" else arg for arg in formula.consequent.arguments]}
    ).render()
    return [
        InferenceStep(
            step_number=1,
            rule="Universal Instantiation",
            premises=[formula.formula],
            conclusion=f"{instantiated_antecedent} → {instantiated_consequent}",
            explanation=f"Replace the universally quantified variable x with {individual}.",
        ),
        InferenceStep(
            step_number=2,
            rule="Modus Ponens",
            premises=[instantiated_antecedent, f"{instantiated_antecedent} → {instantiated_consequent}"],
            conclusion=instantiated_consequent,
            explanation="The antecedent is supplied as the request fact, so the consequent follows.",
        ),
    ]

