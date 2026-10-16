"""Low-latency runtime safety verification for engine requests."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from typing import Any

import z3
from pydantic import BaseModel, ConfigDict, Field

from shared_contracts.models import FeatureVector


class RuntimeVerificationResult(BaseModel):
    """The result of checking one request against runtime safety invariants."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    z3_status: str = Field(pattern="^(SAT|UNSAT|UNKNOWN)$")
    decision: str = Field(pattern="^(ALLOW|BLOCK|FALLBACK)$")
    violated_rules: list[str]
    proof_summary: str
    execution_time_ms: float


class RuntimeLTLVerifier:
    """Check ground request context against a small set of safety invariants.

    The verifier deliberately models each supplied value as a Z3 constant. This
    makes the result a proof about the complete request context, rather than a
    satisfiable assignment for missing request values.
    """

    def __init__(self, timeout_ms: int = 5) -> None:
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        self.timeout_ms = timeout_ms
        # Pay Z3's one-time native initialization cost outside the request path.
        warm_solver = z3.Solver()
        warm_solver.set("timeout", timeout_ms)
        warm_solver.check()

    def verify(
        self,
        event_id: str | FeatureVector,
        context: Mapping[str, Any] | None = None,
        custom_constraints: Iterable[str | z3.BoolRef] | None = None,
        **context_values: Any,
    ) -> RuntimeVerificationResult:
        """Verify a request context and return an ALLOW/BLOCK/FALLBACK result.

        ``context`` may be a mapping or a ``FeatureVector``. Keyword values are
        merged into it and are useful for metadata such as ``role`` or
        ``payload_size``. Custom constraints use Python/Z3 expression syntax,
        for example ``"threat_score <= 0.8"`` or ``"role == 'admin'"``.
        """
        started = time.perf_counter()
        if isinstance(event_id, FeatureVector):
            feature_vector = event_id
            event_id = feature_vector.event_id
            values: dict[str, Any] = dict(feature_vector.features)
        else:
            values = {}

        if context is not None:
            if isinstance(context, FeatureVector):
                values.update(context.features)
                event_id = context.event_id
            else:
                values.update(context)
        values.update(context_values)

        if "unified_context_vector" in values:
            vector = values["unified_context_vector"]
            if isinstance(vector, (list, tuple)):
                values.update(
                    {f"context_{index}": value for index, value in enumerate(vector)}
                )

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        variables = self._ground_variables(values)
        rules = self._core_rules(values, variables)
        rules.extend(self._custom_rules(custom_constraints, variables))
        solver.add(
            *(
                variable == variables[f"_{name}"]
                for name, variable in variables.items()
                if not name.startswith("_") and f"_{name}" in variables
            )
        )
        solver.add(*(expression for _, expression in rules))

        checked = solver.check()
        if checked == z3.sat:
            status, decision = "SAT", "ALLOW"
            violated_rules: list[str] = []
            summary = f"{len(rules)} runtime safety rules satisfied"
        elif checked == z3.unsat:
            status, decision = "UNSAT", "BLOCK"
            violated_rules = self._violated_rules(rules, variables)
            summary = (
                f"{len(violated_rules)} runtime safety rule(s) violated"
                if violated_rules
                else "Runtime safety constraints are inconsistent"
            )
        else:
            status, decision = "UNKNOWN", "FALLBACK"
            violated_rules = []
            summary = "Z3 could not decide within the configured runtime timeout"

        elapsed = (time.perf_counter() - started) * 1000
        return RuntimeVerificationResult(
            event_id=str(event_id),
            z3_status=status,
            decision=decision,
            violated_rules=violated_rules,
            proof_summary=summary,
            execution_time_ms=elapsed,
        )

    def verify_context(
        self,
        event_id: str,
        context: Mapping[str, Any],
        custom_constraints: Iterable[str | z3.BoolRef] | None = None,
    ) -> RuntimeVerificationResult:
        """Explicitly named alias for callers that verify request contexts."""
        return self.verify(event_id, context, custom_constraints)

    @staticmethod
    def _ground_variables(values: Mapping[str, Any]) -> dict[str, z3.ExprRef]:
        variables: dict[str, z3.ExprRef] = {}
        for name, value in values.items():
            if not isinstance(name, str) or not name.isidentifier():
                continue
            if isinstance(value, bool):
                variable: z3.ExprRef = z3.Bool(name)
                bound = z3.BoolVal(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                variable = z3.Int(name)
                bound = z3.IntVal(value)
            elif isinstance(value, float):
                if not math.isfinite(value):
                    continue
                variable = z3.Real(name)
                bound = z3.RealVal(str(value))
            elif isinstance(value, str):
                variable = z3.String(name)
                bound = z3.StringVal(value)
            else:
                continue
            variables[name] = variable
            variables[f"_{name}"] = bound
        return variables

    @staticmethod
    def _core_rules(
        values: Mapping[str, Any], variables: Mapping[str, z3.ExprRef]
    ) -> list[tuple[str, z3.BoolRef]]:
        rules: list[tuple[str, z3.BoolRef]] = []

        def add(name: str, expression: z3.BoolRef) -> None:
            rules.append((name, expression))

        if "threat_score" in variables:
            threat_score = variables["threat_score"]
            add("threat_score_bounds", z3.And(threat_score >= 0, threat_score <= 1))

        for name in ("rate_limit", "payload_size", "request_rate", "frequency"):
            if name in variables:
                add(f"{name}_non_negative", variables[name] >= 0)

        if "request_rate" in variables and "rate_limit" in variables:
            add(
                "rate_within_limit",
                variables["request_rate"] <= variables["rate_limit"],
            )
        if "frequency" in variables and "rate_limit" in variables:
            add("frequency_within_limit", variables["frequency"] <= variables["rate_limit"])

        privileged_actions = {"admin", "delete", "manage", "write"}
        action = values.get("action")
        if action in privileged_actions and "role" in variables:
            add("privileged_role_required", variables["role"] == z3.StringVal("admin"))

        return rules

    @staticmethod
    def _custom_rules(
        constraints: Iterable[str | z3.BoolRef] | None,
        variables: Mapping[str, z3.ExprRef],
    ) -> list[tuple[str, z3.BoolRef]]:
        rules: list[tuple[str, z3.BoolRef]] = []
        if constraints is None:
            return rules
        if isinstance(constraints, str):
            constraints = [constraints]
        expression_names = {
            name: value
            for name, value in variables.items()
            if not name.startswith("_")
        }
        for index, constraint in enumerate(constraints, start=1):
            if isinstance(constraint, z3.BoolRef):
                expression = constraint
            elif isinstance(constraint, str):
                try:
                    expression = eval(  # noqa: S307 - builtins are disabled below
                        constraint,
                        {"__builtins__": {}, "z3": z3},
                        expression_names,
                    )
                except (SyntaxError, NameError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid custom constraint: {constraint!r}") from exc
                if isinstance(expression, bool):
                    expression = z3.BoolVal(expression)
                if not isinstance(expression, z3.BoolRef):
                    raise ValueError(f"Custom constraint is not boolean: {constraint!r}")
            else:
                raise TypeError("custom constraints must be strings or Z3 boolean expressions")
            rules.append((f"custom_constraint_{index}", expression))
        return rules

    @staticmethod
    def _violated_rules(
        rules: list[tuple[str, z3.BoolRef]],
        variables: Mapping[str, z3.ExprRef],
    ) -> list[str]:
        violated: list[str] = []
        for name, expression in rules:
            checker = z3.Solver()
            checker.set("timeout", 1)
            bindings = [
                variable == value
                for key, variable in variables.items()
                if not key.startswith("_")
                and (value := variables.get(f"_{key}")) is not None
            ]
            checker.add(*bindings, z3.Not(expression))
            if checker.check() == z3.sat:
                violated.append(name)
        return violated
