"""Tensor preparation: token ids padded or truncated to model input shape."""

from __future__ import annotations

from .models import ModelInput, TokenSequence
from .vocabulary import Vocabulary


class Tensorizer:
    """Convert token sequences into fixed-length model inputs."""

    def __init__(self, vocabulary: Vocabulary, max_length: int) -> None:
        self.vocabulary = vocabulary
        self.max_length = max_length

    def tensorize(self, sequence: TokenSequence) -> ModelInput:
        """Return token ids and attention mask of length ``max_length``."""
        raise NotImplementedError
