from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class PolicyEffect(str, Enum):
    ALLOW = "Allow"
    DENY = "Deny"

class SecurityPolicy(BaseModel):
    policy_id: str
    subject: str = Field(description="Role or user executing the action")
    action: str = Field(description="API action being attempted")
    resource: str = Field(description="Target endpoint or resource")
    effect: PolicyEffect
    raw_text: str

class VerificationReport(BaseModel):
    is_valid: bool
    status: str  # "SAT" (No Conflicts) or "UNSAT" (Conflicts Found)
    fol_expressions: List[str]
    conflicting_policies: Optional[List[str]] = None
    execution_time_ms: float