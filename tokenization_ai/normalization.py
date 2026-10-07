"""Normalization of masked records into a canonical form for tokenization."""

from __future__ import annotations

from .models import FusedInput


class Normalizer:
    """Canonicalize keys and values so equivalent inputs tokenize identically."""

    def normalize(self, fused: FusedInput) -> FusedInput:
        """Return a new normalized record."""
        raise NotImplementedError
