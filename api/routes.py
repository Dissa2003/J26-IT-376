import difflib
import logging
import os
import uuid

from fastapi import APIRouter, FastAPI, HTTPException, status
from typing import List
from pydantic import BaseModel, Field, field_validator
from src.policy_verifier.models import (
    MultiPolicyVerificationReport,
    RequestDecision,
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
from src.policy_verifier.request_cache import RequestDecisionCache, cache_key

log = logging.getLogger(__name__)
app = FastAPI(title="Policy Verification Engine API")
router = APIRouter()
pipeline = VerificationPipeline()
conflict_detector = ConflictDetector()
parser = PolicyParser()
rule_cache = LocalRuleCache()
# Redis cache for verify-request. POLICY_REDIS_URL picks the server (the shared
# REDIS_URL points at the docker-compose host name); POLICY_REQUEST_CACHE=0 turns it off.
request_cache = RequestDecisionCache.from_url(
    None if os.environ.get("POLICY_REQUEST_CACHE", "1") == "0"
    else os.environ.get("POLICY_REDIS_URL", "redis://localhost:6379/0")
)
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
    # _canonical is not idempotent ("UserAccounts" -> "Useraccounts"), so names are
    # matched on a case- and space-insensitive key and resolved to the policy's own
    # spelling instead of being canonicalised a second time.
    known_names: dict[str, str] = {}
    for policy in body.policies:
        for name in (policy.subject, policy.action, policy.resource):
            known_names.setdefault(_name_key(name), name)

    def resolve(name: str) -> str:
        return known_names.get(_name_key(name), name)

    policies = [
        policy.model_copy(
            update={
                "subject": resolve(policy.subject),
                "action": resolve(policy.action),
                "resource": resolve(policy.resource),
            }
        )
        for policy in body.policies
    ]
    roles = [resolve(role) for role in body.request.roles]
    action = resolve(body.request.action)
    resource = resolve(body.request.resource)
    policy_roles = sorted({policy.subject for policy in policies})
    unmatched_roles = [role for role in dict.fromkeys(roles) if role not in policy_roles]
    log.debug(
        "verify-request names: policy roles=%s actions=%s resources=%s | "
        "request roles=%s action=%r resource=%r | unmatched request roles=%s",
        policy_roles,
        sorted({policy.action for policy in policies}),
        sorted({policy.resource for policy in policies}),
        roles,
        action,
        resource,
        unmatched_roles,
    )
    key = cache_key(policies, body.request.subject_id, roles, action, resource)
    report = request_cache.get(key)
    if report is None:
        report = pipeline.engine.check_request(
            policies,
            subject_id=body.request.subject_id,
            roles=roles,
            action=action,
            resource=resource,
        )
        request_cache.set(key, report)
    if report.decision is RequestDecision.NOT_DETERMINED:
        report.limitations.extend(
            _unmatched_role_note(role, policy_roles) for role in unmatched_roles
        )
    return report


def _unmatched_role_note(role: str, policy_roles: list[str]) -> str:
    keys = {_name_key(name): name for name in policy_roles}
    close = difflib.get_close_matches(_name_key(role), list(keys), n=1, cutoff=0.8)
    note = f"Role '{role}' matched no policy role."
    return f"{note} Did you mean '{keys[close[0]]}'?" if close else note


def _name_key(value: str) -> str:
    return "".join(value.split()).casefold()


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