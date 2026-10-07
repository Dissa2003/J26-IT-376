"""Unit tests for normalization."""

import pytest

from tokenization_ai.models import FusedInput
from tokenization_ai.normalization import Normalizer


def test_normalize_is_not_implemented_yet() -> None:
    """Placeholder until normalization is implemented."""
    with pytest.raises(NotImplementedError):
        Normalizer().normalize(FusedInput(event_id="e1"))
