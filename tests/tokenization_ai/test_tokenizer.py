"""Unit tests for hybrid structure-aware tokenization."""

import pytest

from tokenization_ai.models import FusedInput
from tokenization_ai.tokenizer import HybridTokenizer


def test_tokenize_is_not_implemented_yet() -> None:
    """Placeholder until tokenization is implemented."""
    with pytest.raises(NotImplementedError):
        HybridTokenizer().tokenize(FusedInput(event_id="e1"))
