"""Unit tests for the Member 4 input contracts."""

import pytest
from pydantic import ValidationError

from tokenization_ai.models import FusedInput, UnifiedApiContext, VerifiedRuleInfo


def test_fused_input_exposes_the_shared_event_id(
    api_context: UnifiedApiContext, rule_info: VerifiedRuleInfo
) -> None:
    fused = FusedInput(context=api_context, rule_info=rule_info)

    assert fused.event_id == "e1"
    assert fused.context is api_context
    assert fused.rule_info is rule_info


def test_fused_input_rejects_mismatched_event_ids(api_context: UnifiedApiContext) -> None:
    other = VerifiedRuleInfo(event_id="e2")

    with pytest.raises(ValidationError, match="different events"):
        FusedInput(context=api_context, rule_info=other)


def test_contracts_are_immutable(api_context: UnifiedApiContext) -> None:
    with pytest.raises(ValidationError):
        api_context.method = "GET"  # type: ignore[misc]
