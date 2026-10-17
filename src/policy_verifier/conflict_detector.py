from __future__ import annotations

from .models import ConflictFinding, SecurityPolicy, VerificationStatus


class ConflictDetector:
    def find_conflicts(self, policies: list[SecurityPolicy]) -> list[ConflictFinding]:
        findings: list[ConflictFinding] = []
        for index, left in enumerate(policies):
            for right in policies[index + 1 :]:
                same_scope = (
                    left.subject == right.subject
                    and left.action == right.action
                    and left.resource == right.resource
                    and left.conditions == right.conditions
                )
                if same_scope and left.effect != right.effect:
                    findings.append(
                        ConflictFinding(
                            conflict_id=f"{left.policy_id}:{right.policy_id}",
                            category="direct_permission_prohibition",
                            policy_ids=[left.policy_id, right.policy_id],
                            status=VerificationStatus.UNSAT,
                            evidence=[left.raw_text, right.raw_text],
                            explanation="The same subject, action, resource, and conditions contain both permission and prohibition.",
                        )
                    )
        return findings

    def check_conflicts(self, fol_expressions: list[str]) -> tuple[str, list[str]]:
        """Compatibility adapter for the original public method."""
        has_allow = any("Allow(" in expression for expression in fol_expressions)
        has_deny = any("Deny(" in expression for expression in fol_expressions)
        return ("UNSAT", fol_expressions) if has_allow and has_deny else ("SAT", [])
