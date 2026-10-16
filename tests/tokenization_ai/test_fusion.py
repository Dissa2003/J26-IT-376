"""Unit tests for input fusion."""

import pytest

from tokenization_ai.adapters import InputContractError
from tokenization_ai.fusion import InputFusion
from tokenization_ai.models import (
    UnifiedApiContext,
    VerifiedRule,
    VerifiedRuleInfo,
)


def test_fuse_preserves_all_context_and_rule_fields(
    api_context: UnifiedApiContext, rule_info: VerifiedRuleInfo
) -> None:
    fused = InputFusion().fuse(api_context, rule_info)

    assert fused.event_id == api_context.event_id
    assert fused.context == api_context
    assert fused.rule_info == rule_info


def test_fuse_rejects_mismatched_event_ids(api_context: UnifiedApiContext) -> None:
    other_rule_info = VerifiedRuleInfo(event_id="some-other-event")

    with pytest.raises(InputContractError, match="different events"):
        InputFusion().fuse(api_context, other_rule_info)


def test_fuse_succeeds_when_rule_info_has_no_rules(
    api_context: UnifiedApiContext,
) -> None:
    empty_rule_info = VerifiedRuleInfo(event_id=api_context.event_id)

    fused = InputFusion().fuse(api_context, empty_rule_info)

    assert fused.rule_info.rules == []


def test_fuse_preserves_multiple_verified_rules_in_order(
    api_context: UnifiedApiContext,
) -> None:
    rules = [
        VerifiedRule(rule_id="R-1", attributes={"field": "amount", "operator": "<="}),
        VerifiedRule(rule_id="R-2", attributes={"field": "method", "value": "POST"}),
        VerifiedRule(rule_id="R-3"),
    ]
    rule_info = VerifiedRuleInfo(event_id=api_context.event_id, rules=rules)

    fused = InputFusion().fuse(api_context, rule_info)

    assert fused.rule_info.rules == rules
    assert [rule.rule_id for rule in fused.rule_info.rules] == ["R-1", "R-2", "R-3"]
