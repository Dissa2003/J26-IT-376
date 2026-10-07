"""Privacy-aware masking of sensitive values before tokenization."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Any

from .models import FusedInput

MASKED_AUTH = "[MASKED_AUTH]"
MASKED_SECRET = "[MASKED_SECRET]"
MASKED_PII = "[MASKED_PII]"
MASKED_IP = "[MASKED_IP]"

# Header names (normalized) that carry authentication material outright.
_AUTH_HEADER_KEYS = {
    "authorization",
    "proxy_authorization",
    "cookie",
    "set_cookie",
    "x_api_key",
    "x_auth_token",
    "x_csrf_token",
}

# Payload/header field names (normalized) that hold credentials or secrets.
_SECRET_KEYS = {
    "access_token",
    "api_key",
    "auth",
    "credential",
    "credentials",
    "password",
    "passwd",
    "pin",
    "private_key",
    "refresh_token",
    "secret",
    "token",
    "otp",
}

# Payload/header field names (normalized) that hold PII rather than secrets.
_PII_KEYS = {
    "email",
    "phone",
    "phone_number",
    "mobile",
    "mobile_number",
    "ssn",
    "card",
    "card_number",
    "credit_card",
    "cvv",
    "cvc",
    "iban",
    "account_number",
    "client_ip",
    "ip_address",
}

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CARD = re.compile(r"^(?:\d[ -]?){13,19}$")


class PrivacyMasker:
    """Replace sensitive values with typed placeholders that preserve structure."""

    def mask(self, fused: FusedInput) -> FusedInput:
        """Return a new fused record with the API context's sensitive values masked.

        The verified rule information is carried through unchanged; the original
        ``fused`` object (and its nested context) is never mutated.
        """
        context = fused.context
        masked_context = context.model_copy(
            update={
                "headers": self._mask_headers(context.headers),
                "payload": self._mask_value(context.payload, key_hint=None),
                "client_ip": MASKED_IP if context.client_ip else context.client_ip,
            }
        )
        return FusedInput(context=masked_context, rule_info=fused.rule_info)

    @classmethod
    def _mask_headers(cls, headers: Mapping[str, str]) -> dict[str, str]:
        """Mask authentication, credential, and PII header values by name."""
        return {key: cls._mask_header_value(key, value) for key, value in headers.items()}

    @classmethod
    def _mask_header_value(cls, key: str, value: str) -> str:
        normalized = cls._normalize_key(key)
        if normalized in _AUTH_HEADER_KEYS or normalized.endswith("_token") or normalized.endswith("_key"):
            return MASKED_AUTH
        placeholder = cls._classify_key(key)
        if placeholder:
            return placeholder
        if isinstance(value, str):
            placeholder = cls._classify_value(value)
            if placeholder:
                return placeholder
        return value

    @classmethod
    def _mask_value(cls, value: Any, key_hint: str | None) -> Any:
        """Recursively mask mappings, lists, and sensitive scalar values."""
        if key_hint:
            placeholder = cls._classify_key(key_hint)
            if placeholder:
                return placeholder
        if isinstance(value, Mapping):
            return {
                str(key): cls._mask_value(item, str(key)) for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._mask_value(item, key_hint) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._mask_value(item, key_hint) for item in value)
        if isinstance(value, str):
            placeholder = cls._classify_value(value)
            if placeholder:
                return placeholder
        return value

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize a field name for case- and separator-insensitive matching."""
        return key.casefold().replace("-", "_").replace(" ", "_")

    @classmethod
    def _classify_key(cls, key: str) -> str | None:
        """Return the placeholder for a field name known to be sensitive, if any."""
        normalized = cls._normalize_key(key)
        if normalized in _SECRET_KEYS or normalized.endswith("_token") or normalized.endswith("_secret"):
            return MASKED_SECRET
        if normalized in _PII_KEYS or normalized.endswith("_email") or normalized.endswith("_phone"):
            return MASKED_PII
        return None

    @staticmethod
    def _classify_value(value: str) -> str | None:
        """Detect email, payment-card, and IP-address string values.

        Phone numbers are masked by field name only (see ``_PII_KEYS``): their
        digit formatting is too close to dates, ids, and other benign fields
        to classify reliably from the value alone.
        """
        if _EMAIL.fullmatch(value) or _CARD.fullmatch(value):
            return MASKED_PII
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return None
        return MASKED_IP
