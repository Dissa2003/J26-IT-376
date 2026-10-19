from fastapi import APIRouter, FastAPI, HTTPException, status
from typing import List
from pydantic import BaseModel, Field, field_validator
from src.policy_verifier.models import (
    RequestDecisionReport,
    SecurityPolicy,
    VerificationReport,
    VerificationStatus,
)
from src.policy_verifier.conflict_detector import ConflictDetector
from src.policy_verifier.cache import LocalRuleCache
from src.policy_verifier.parser import PolicyParser, UnsupportedPolicyError, _canonical
from src.policy_verifier.pipeline import VerificationPipeline

app = FastAPI(title="Policy Verification Engine API")
router = APIRouter()
pipeline = VerificationPipeline()
conflict_detector = ConflictDetector()
parser = PolicyParser()
rule_cache = LocalRuleCache()
verification_records: list[VerificationReport] = []


class PolicyTextRequest(BaseModel):
    policy_text: str = Field(min_length=1, max_length=500)
    policy_id: str | None = Field(default=None, max_length=64)


class IndependentRequest(BaseModel):
    subject_id: str = Field(min_length=1, max_length=64)
    roles: list[str] = Field(default_factory=list, max_length=32)
    action: str = Field(min_length=1, max_length=64)
    resource: str = Field(min_length=1, max_length=128)

    @field_validator("subject_id", "action", "resource", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return _canonical(value)

    @field_validator("roles", mode="before")
    @classmethod
    def normalize_roles(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("must be a list")
        normalized: list[str] = []
        for role in value:
            if not isinstance(role, str) or not role.strip():
                raise ValueError("roles must contain non-empty strings")
            normalized.append(_canonical(role))
        return normalized


class IndependentRequestBody(BaseModel):
    policies: list[SecurityPolicy] = Field(min_length=1, max_length=100)
    request: IndependentRequest

@router.get("/")
def root():
    return {"status": "active", "message": "Policy Verification API is running"}

@router.post("/api/v1/verify-policies", response_model=VerificationReport)
def verify_policies(policies: List[SecurityPolicy]):
    if not policies:
        raise HTTPException(status_code=422, detail="At least one policy is required.")
    report = pipeline.verify(policies[0])
    report.conflicts = conflict_detector.find_conflicts(policies)
    if report.conflicts:
        report.is_valid = False
        report.status = "UNSAT"
        report.rule_status = "conflicted"
        report.conflicting_policies = [
            policy_id
            for conflict in report.conflicts
            for policy_id in conflict.policy_ids
        ]
    consistency = pipeline.engine.check_consistency(
        policies,
        assume_role_witness=True,
        assume_mutually_exclusive_roles=True,
    )
    if consistency.status is VerificationStatus.UNSAT:
        consistency = consistency.model_copy(update={
            "explanation": "The submitted policies are logically contradictory under "
            "the role-witness and exclusive-roles assumptions.",
        })
        report.is_valid = False
        if not report.conflicts:
            report.rule_status = "contradictory"
    elif consistency.status is VerificationStatus.SAT:
        consistency = consistency.model_copy(update={
            "explanation": "The submitted policies are mutually consistent.",
        })
    else:
        consistency = consistency.model_copy(update={
            "explanation": "Z3 returned an indeterminate result for consistency "
            "— treating as not valid for safety.",
        })
        report.is_valid = False
        if not report.conflicts:
            report.rule_status = "contradictory"
    report.consistency_result = consistency
    verification_records.insert(0, report)
    del verification_records[100:]
    rule_cache.put_if_eligible(report)
    
    if report.conflicts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Policy conflict detected (UNSAT)", "report": report.model_dump()}
        )
    
    return report


@router.post("/api/v1/verify-text", response_model=VerificationReport)
def verify_text(request: PolicyTextRequest):
    try:
        policy = parser.parse_text(request.policy_text, request.policy_id)
    except UnsupportedPolicyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "unsupported_policy", "message": str(exc)},
        ) from exc
    report = pipeline.verify(policy)
    verification_records.insert(0, report)
    del verification_records[100:]
    rule_cache.put_if_eligible(report)
    return report


@router.post("/api/v1/verify-request", response_model=RequestDecisionReport)
def verify_request(body: IndependentRequestBody):
    mismatches: list[dict[str, object]] = []
    for policy in body.policies:
        if not policy.raw_text.strip():
            continue
        try:
            parsed = parser.parse_text(policy.raw_text, policy.policy_id)
        except UnsupportedPolicyError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "policy_raw_text_unsupported",
                    "policy_id": policy.policy_id,
                    "message": str(exc),
                },
            ) from exc
        differing_fields = [
            field
            for field in ("subject", "action", "resource", "effect", "interpretation")
            if getattr(policy, field) != getattr(parsed, field)
        ]
        if differing_fields:
            mismatches.append(
                {
                    "policy_id": policy.policy_id,
                    "fields": differing_fields,
                }
            )
    if mismatches:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "policy_field_mismatch",
                "message": "Submitted policy fields do not match raw_text.",
                "policies": mismatches,
            },
        )
    policies = [
        policy.model_copy(
            update={
                "subject": _canonical(policy.subject),
                "action": _canonical(policy.action),
                "resource": _canonical(policy.resource),
            }
        )
        for policy in body.policies
    ]
    return pipeline.engine.check_request(
        policies,
        subject_id=body.request.subject_id,
        roles=body.request.roles,
        action=body.request.action,
        resource=body.request.resource,
    )


@router.get("/api/v1/verification-records", response_model=list[VerificationReport])
def list_verification_records():
    """Return process-local records; this is not durable persistence."""
    return verification_records


@router.get("/api/v1/rule-cache", response_model=list[VerificationReport])
def list_cached_rules():
    """Return rules accepted by the local in-memory eligibility gate."""
    return rule_cache.list()


@router.get("/api/v1/conflicts")
def list_conflicts():
    return [conflict for report in verification_records for conflict in report.conflicts]


app.include_router(router)