"""Input fusion of unified API context and verified rule information."""

from __future__ import annotations

from pydantic import ValidationError

from .adapters import InputContractError
from .models import FusedInput, UnifiedApiContext, VerifiedRuleInfo


class InputFusion:
    """Merge a unified context and its verified rule information by event id."""

    def fuse(self, context: UnifiedApiContext, rule_info: VerifiedRuleInfo) -> FusedInput:
        """Return the fused record for a single event.

        Both inputs are carried through unchanged; fusion performs no masking,
        normalization, or other transformation. Raises ``InputContractError``
        if ``context`` and ``rule_info`` reference different events.
        """
        try:
            return FusedInput(context=context, rule_info=rule_info)
        except ValidationError as exc:
            raise InputContractError(str(exc)) from exc
