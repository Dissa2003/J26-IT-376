"""Privacy-aware masking of sensitive values before tokenization."""

from __future__ import annotations

from .models import FusedInput


class PrivacyMasker:
    """Replace sensitive values with typed placeholders that preserve structure."""

    def mask(self, fused: FusedInput) -> FusedInput:
        """Return a new masked record without mutating the caller's object."""
        raise NotImplementedError
