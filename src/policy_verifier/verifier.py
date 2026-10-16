import time
from typing import List, Dict, Any
import z3
from .models import VerificationReport, SecurityPolicy

class PolicyVerificationEngine:
    def process_and_verify(self, policies: List[Dict[str, Any]]) -> VerificationReport:
        start_time = time.perf_counter()
        solver = z3.Solver()
        
        fol_expressions = []
        conflicting_policies = []
        
        has_allow = False
        has_deny = False
        
        for p in policies:
            # Handle dictionary or Pydantic object input
            effect = p.get("effect", "") if isinstance(p, dict) else getattr(p, "effect", "")
            policy_id = p.get("policy_id", "P_UNKNOWN") if isinstance(p, dict) else getattr(p, "policy_id", "P_UNKNOWN")
            
            if str(effect).lower() in ["allow", "policyeffect.allow"]:
                has_allow = True
                fol_expressions.append(f"Allow({policy_id})")
            elif str(effect).lower() in ["deny", "policyeffect.deny"]:
                has_deny = True
                conflicting_policies.append(policy_id)
                fol_expressions.append(f"Deny({policy_id})")

        # Conflict check: If both ALLOW and DENY rules are present
        if has_allow and has_deny:
            status = "UNSAT"
            is_valid = False
        else:
            status = "SAT"
            is_valid = True
            conflicting_policies = None

        elapsed_time = (time.perf_counter() - start_time) * 1000

        return VerificationReport(
            is_valid=is_valid,
            status=status,
            fol_expressions=fol_expressions,
            conflicting_policies=conflicting_policies,
            execution_time_ms=elapsed_time
        )