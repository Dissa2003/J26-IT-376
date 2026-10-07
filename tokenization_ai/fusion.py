"""Input fusion of unified API context and verified rule information."""

from __future__ import annotations

from .models import FusedInput, RuleInfo, UnifiedContext


class InputFusion:
    """Merge a unified context and its rule information into one record."""

    def fuse(self, context: UnifiedContext, rule_info: RuleInfo) -> FusedInput:
        """Return the fused record for a single event."""
        raise NotImplementedError
