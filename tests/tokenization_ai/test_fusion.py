"""Unit tests for input fusion."""

import pytest

from tokenization_ai.fusion import InputFusion
from tokenization_ai.models import RuleInfo, UnifiedContext


def test_fuse_is_not_implemented_yet() -> None:
    """Placeholder until fusion is implemented."""
    context = UnifiedContext(event_id="e1")
    rule_info = RuleInfo(event_id="e1")
    with pytest.raises(NotImplementedError):
        InputFusion().fuse(context, rule_info)
