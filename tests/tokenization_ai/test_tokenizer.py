"""Unit tests for hybrid structure-aware tokenization."""

import pytest

from tokenization_ai.models import FusedInput
from tokenization_ai.tokenizer import HybridTokenizer


def test_tokenize_is_not_implemented_yet(fused_input: FusedInput) -> None:
    """Placeholder until tokenization is implemented."""
    with pytest.raises(NotImplementedError):
        HybridTokenizer().tokenize(fused_input)
