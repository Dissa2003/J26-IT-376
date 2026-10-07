"""Token-to-id vocabulary used for embedding lookup."""

from __future__ import annotations

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


class Vocabulary:
    """Bidirectional mapping between tokens and integer ids."""

    def encode(self, tokens: list[str]) -> list[int]:
        """Map tokens to ids, using the unknown id for unseen tokens."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> list[str]:
        """Map ids back to tokens."""
        raise NotImplementedError
