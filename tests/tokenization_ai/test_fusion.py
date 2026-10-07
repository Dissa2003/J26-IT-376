"""Unit tests for input fusion."""

import pytest

from tokenization_ai.fusion import InputFusion
from tokenization_ai.models import UnifiedApiContext, VerifiedRuleInfo


def test_fuse_is_not_implemented_yet(
    api_context: UnifiedApiContext, rule_info: VerifiedRuleInfo
) -> None:
    """Placeholder until fusion is implemented."""
    with pytest.raises(NotImplementedError):
        InputFusion().fuse(api_context, rule_info)
