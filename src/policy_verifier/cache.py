from __future__ import annotations

from .models import VerificationReport


class LocalRuleCache:
    """In-memory cache; persistence and runtime integration are intentionally deferred."""

    def __init__(self) -> None:
        self._rules: dict[str, VerificationReport] = {}

    def put_if_eligible(self, report: VerificationReport) -> bool:
        eligible = (
            report.rule_status == "eligible"
            and report.solver_result is not None
            and report.solver_result.is_entailment
            and not report.conflicts
        )
        if eligible and report.policy_id:
            self._rules[report.policy_id] = report
        return eligible

    def get(self, policy_id: str) -> VerificationReport | None:
        return self._rules.get(policy_id)

    def list(self) -> list[VerificationReport]:
        return list(self._rules.values())
