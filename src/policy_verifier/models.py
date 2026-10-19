from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class PolicyEffect(str, Enum):
    ALLOW = "Allow"
    DENY = "Deny"


class PolicyInterpretation(str, Enum):
    ROLE_PERMISSION = "role_permission"
    ONLY_ROLE = "only_role"
    ROLE_PROHIBITION = "role_prohibition"


class VerificationStatus(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID = "INVALID"


class RequestDecision(str, Enum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    NOT_DETERMINED = "NOT_DETERMINED"


class PolicyCondition(BaseModel):
    attribute: str
    operator: Literal["equals"] = "equals"
    value: str


class SecurityPolicy(BaseModel):
    policy_id: str
    subject: str = Field(description="Role or user executing the action")
    action: str = Field(description="API action being attempted")
    resource: str = Field(description="Target endpoint or resource")
    effect: PolicyEffect
    raw_text: str
    conditions: list[PolicyCondition] = Field(default_factory=list)
    interpretation: PolicyInterpretation = PolicyInterpretation.ROLE_PERMISSION
    source: str = "api"


class PolicyAttributes(BaseModel):
    subject: str
    action: str
    resource: str
    effect: PolicyEffect
    conditions: list[PolicyCondition] = Field(default_factory=list)
    interpretation: PolicyInterpretation


class StructuredPolicy(BaseModel):
    policy_id: str
    raw_text: str
    attributes: PolicyAttributes
    normalized_key: str
    limitations: list[str] = Field(default_factory=list)

class Predicate(BaseModel):
    name: str
    arguments: list[str]
    negated: bool = False

    def render(self) -> str:
        prefix = "¬" if self.negated else ""
        return f"{prefix}{self.name}({', '.join(self.arguments)})"


class FOLFormula(BaseModel):
    formula: str
    quantifier: str = "∀"
    antecedent: Predicate
    consequent: Predicate
    interpretation: PolicyInterpretation
    explanation: str


class InferenceStep(BaseModel):
    step_number: int
    rule: Literal["Universal Instantiation", "Modus Ponens"]
    premises: list[str]
    conclusion: str
    explanation: str


class FormalConstraint(BaseModel):
    constraint_id: str
    expression: str
    kind: Literal["rule", "fact", "query", "negated_query"]
    source_policy_ids: list[str] = Field(default_factory=list)


class SolverResult(BaseModel):
    status: VerificationStatus
    query_type: Literal["consistency", "entailment"]
    is_entailment: bool = False
    model: dict[str, Any] | None = None
    unsat_core: list[str] = Field(default_factory=list)
    explanation: str


class ConflictFinding(BaseModel):
    conflict_id: str
    category: Literal["direct_permission_prohibition", "contradictory_constraints"]
    policy_ids: list[str]
    status: VerificationStatus
    evidence: list[str]
    explanation: str
    confirmed: bool = True


class VerificationExplanation(BaseModel):
    summary: str
    formally_established: bool
    evidence: list[str]
    limitations: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    # Legacy fields retained for existing callers.
    is_valid: bool
    status: str
    fol_expressions: list[str] = Field(default_factory=list)
    conflicting_policies: list[str] | None = None
    execution_time_ms: float

    policy_id: str | None = None
    original_policy: str | None = None
    attributes: PolicyAttributes | None = None
    structured_policy: StructuredPolicy | None = None
    fol_formula: FOLFormula | None = None
    inference_trace: list[InferenceStep] = Field(default_factory=list)
    constraints: list[FormalConstraint] = Field(default_factory=list)
    solver_result: SolverResult | None = None
    conflicts: list[ConflictFinding] = Field(default_factory=list)
    explanation: VerificationExplanation | None = None
    rule_status: Literal["eligible", "rejected", "unsupported", "conflicted"] = "rejected"
    limitations: list[str] = Field(default_factory=list)


class RequestQueryResult(BaseModel):
    status: Literal["sat", "unsat", "unknown"]
    reason_unknown: str | None = None


class RequestDecisionReport(BaseModel):
    decision: RequestDecision
    reason_code: Literal[
        "policy_set_contradictory",
        "solver_unknown",
        "not_covered_by_policies",
        "request_entailed",
        "request_prohibited",
    ]
    explanation: str
    consistency_query: RequestQueryResult
    allow_query: RequestQueryResult
    deny_query: RequestQueryResult
    constraints: list[FormalConstraint] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    execution_time_ms: float
    limitations: list[str] = Field(default_factory=list)
