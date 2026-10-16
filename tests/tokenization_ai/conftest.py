"""Shared fixtures for Member 4 tests."""

import pytest

from tokenization_ai.models import FusedInput, UnifiedApiContext, VerifiedRuleInfo


@pytest.fixture
def api_context() -> UnifiedApiContext:
    return UnifiedApiContext(event_id="e1", method="POST", path="/orders")


@pytest.fixture
def rule_info() -> VerifiedRuleInfo:
    return VerifiedRuleInfo(event_id="e1")


@pytest.fixture
def fused_input(api_context: UnifiedApiContext, rule_info: VerifiedRuleInfo) -> FusedInput:
    return FusedInput(context=api_context, rule_info=rule_info)
