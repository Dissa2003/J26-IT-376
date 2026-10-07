"""Unit tests for tensor preparation."""

import pytest

from tokenization_ai.models import TokenSequence
from tokenization_ai.tensorizer import Tensorizer
from tokenization_ai.vocabulary import Vocabulary


def test_tensorize_is_not_implemented_yet() -> None:
    """Placeholder until tensor preparation is implemented."""
    tensorizer = Tensorizer(Vocabulary(), max_length=8)
    with pytest.raises(NotImplementedError):
        tensorizer.tensorize(TokenSequence(event_id="e1"))
