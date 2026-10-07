from fastapi import FastAPI, HTTPException, status
from typing import List
from src.policy_verifier.models import SecurityPolicy, VerificationReport
from src.policy_verifier.verifier import PolicyVerificationEngine

app = FastAPI(title="Policy Verification Engine API")
engine = PolicyVerificationEngine()

@app.get("/")
def root():
    return {"status": "active", "message": "Policy Verification API is running"}

@app.post("/api/v1/verify-policies", response_model=VerificationReport)
def verify_policies(policies: List[SecurityPolicy]):
    raw_policies = [p.dict() for p in policies]
    report = engine.process_and_verify(raw_policies)
    
    if report.status == "UNSAT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Policy conflict detected (UNSAT)",
                "report": report.dict()
            }
        )
    
    return report