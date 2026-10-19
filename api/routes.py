import uuid

from fastapi import APIRouter, FastAPI, HTTPException, status
from typing import List
from pydantic import BaseModel, Field, field_validator
from src.policy_verifier.models import (
    MultiPolicyVerificationReport,
    RequestDecisionReport,
    SecurityPolicy,
    SubmissionInfo,
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

@router.post("/api/v1/verify-policies", response_model=MultiPolicyVerificationReport)
def verify_policies(policies: List[SecurityPolicy]):
    if not policies:
        raise HTTPException(status_code=422, detail="At least one policy is required.")
    reports = [pipeline.verify(policy) for policy in policies]
    if len(policies) > 1:
        submission_id = uuid.uuid4().hex[:8]
        for index, report in enumerate(reports, 1):
            report.submission = SubmissionInfo(
                submission_id=submission_id, index=index, size=len(reports)
            )
    conflicts = conflict_detector.find_conflicts(policies)
    for report in reports:
        report.conflicts = [
            conflict for conflict in conflicts
            if report.policy_id in conflict.policy_ids
        ]
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
    elif consistency.status is VerificationStatus.SAT:
        consistency = consistency.model_copy(update={
            "explanation": "The submitted policies are mutually consistent.",
        })
    else:
        consistency = consistency.model_copy(update={
            "explanation": "Z3 returned an indeterminate result for consistency "
            "— treating as not valid for safety.",
        })
    for report in reports:
        report.consistency_result = consistency
        # Consistency is a property of the whole set, so a contradictory or
        # indeterminate set makes every policy in it ineligible.
        if consistency.status is not VerificationStatus.SAT:
            report.is_valid = False
            if not report.conflicts:
                report.rule_status = "contradictory"
    verification_records[0:0] = reports
    del verification_records[100:]
    for report in reports:
        rule_cache.put_if_eligible(report)

    result = MultiPolicyVerificationReport(
        reports=reports,
        conflicts=conflicts,
        consistency_result=consistency,
        overall_valid=(
            all(report.is_valid for report in reports)
            and not conflicts
            and consistency.status is VerificationStatus.SAT
        ),
    )
    if conflicts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Policy conflict detected (UNSAT)", "report": result.model_dump()}
        )

    return result


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
    # A conflict is attached to every report it involves; list it once per submission.
    seen: set[tuple[str, str]] = set()
    conflicts = []
    for report in verification_records:
        submission = report.submission.submission_id if report.submission else str(id(report))
        for conflict in report.conflicts:
            key = (submission, conflict.conflict_id)
            if key not in seen:
                seen.add(key)
                conflicts.append(conflict)
    return conflicts


app.include_router(router)