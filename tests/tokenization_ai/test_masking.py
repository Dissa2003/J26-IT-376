"""Unit tests for privacy-aware masking."""

import pytest

from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.models import FusedInput


def test_mask_is_not_implemented_yet() -> None:
    """Placeholder until masking is implemented."""
    with pytest.raises(NotImplementedError):
        PrivacyMasker().mask(FusedInput(event_id="e1"))
