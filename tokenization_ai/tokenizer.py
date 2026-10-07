"""Hybrid structure-aware tokenization of normalized records."""

from __future__ import annotations

from .models import FusedInput, TokenSequence


class HybridTokenizer:
    """Tokenize a record while preserving its structural boundaries."""

    def tokenize(self, fused: FusedInput) -> TokenSequence:
        """Return the ordered token sequence for a normalized record."""
        raise NotImplementedError
