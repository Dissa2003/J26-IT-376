import uuid
from typing import Optional
from .models import SecurityPolicy, PolicyEffect

class PolicyParser:
    def parse_text(self, text: str, policy_id: Optional[str] = None) -> SecurityPolicy:
        pid = policy_id or str(uuid.uuid4())[:8]
        text_lower = text.lower()

        effect = PolicyEffect.DENY if any(w in text_lower for w in ["cannot", "deny", "no one", "forbidden"]) else PolicyEffect.ALLOW

        if "admin" in text_lower:
            subject = "Admin"
        elif "manager" in text_lower:
            subject = "Manager"
        elif "user" in text_lower:
            subject = "User"
        else:
            subject = "Everyone"

        if "delete" in text_lower:
            action, resource = "Delete", "UserAccount"
        elif "read" in text_lower or "view" in text_lower:
            action, resource = "Read", "UserProfile"
        elif "approve" in text_lower:
            action, resource = "Approve", "Invoice"
        else:
            action, resource = "Access", "Resource"

        return SecurityPolicy(
            policy_id=pid,
            subject=subject,
            action=action,
            resource=resource,
            effect=effect,
            raw_text=text
        )