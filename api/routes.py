from fastapi import APIRouter, FastAPI, HTTPException, status
from typing import List
from pydantic import BaseModel, Field
from src.policy_verifier.models import SecurityPolicy, VerificationReport
from src.policy_verifier.conflict_detector import ConflictDetector
from src.policy_verifier.cache import LocalRuleCache
from src.policy_verifier.parser import PolicyParser, UnsupportedPolicyError
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