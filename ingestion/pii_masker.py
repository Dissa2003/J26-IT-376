"""Synchronous recursive masking of personally identifiable information."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED_PII]"

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CARD = re.compile(r"^(?:\d[ -]?){13,19}$")
_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "auth",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}


class PIIMasker:
    """Mask sensitive values in JSON-compatible mappings and sequences."""

    def mask(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a new masked payload without mutating the caller's object."""
        masked = self._mask_value(payload, key_hint=None)
        if not isinstance(masked, dict):
            raise TypeError("payload must be a mapping")
        return masked

    def _mask_value(self, value: Any, key_hint: str | None) -> Any:
        """Recursively mask mappings, lists, and sensitive scalar values."""
        if key_hint and self._is_credential_key(key_hint):
            return REDACTED
        if isinstance(value, Mapping):
            return {
                str(key): self._mask_value(item, str(key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._mask_value(item, key_hint) for item in value]
        if isinstance(value, tuple):
            return tuple(self._mask_value(item, key_hint) for item in value)
        if isinstance(value, str) and self._contains_pii(value):
            return REDACTED
        return value

    @staticmethod
    def _is_credential_key(key: str) -> bool:
        """Match common credential names, including hyphenated variants."""
        normalized = key.casefold().replace("-", "_").replace(" ", "_")
        return normalized in _CREDENTIAL_KEYS or normalized.endswith("_token")

    @staticmethod
    def _contains_pii(value: str) -> bool:
        """Detect email, payment-card, and IP-address string values."""
        if _EMAIL.fullmatch(value) or _CARD.fullmatch(value):
            return True
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return False
        return True
