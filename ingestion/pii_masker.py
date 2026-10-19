"""Synchronous recursive AST-level masking of sensitive attributes & transient memory cleanup.

Includes Syntax-Preserving AST PII Masking (FR3 / NFR4) which protects downstream
formal SMT/Z3 verification and neuro-symbolic tokenizers by preserving syntactic attack
metacharacters (SQLi, XSS, parameter tampering) intact while anonymizing alphanumeric PII.
"""

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

# Regex to detect attack metacharacters, SQLi clauses, and XSS syntactic markers
_SYNTACTIC_ATTACK_PATTERN = re.compile(
    r"""(?:<script.*?</script>|alert\([^)]*\)|\bDROP\s+TABLE\s+\w+\b|--|/\*|\*/|;|'|"|<script|</script>|javascript:|onerror=|onload=|<img|<svg|\b(?:OR|AND)\s+['"]?\w+['"]?\s*=\s*['"]?\w+['"]?|\bUNION\s+SELECT\b|\bDROP\s+TABLE\b|\bSELECT\s+.+\s+FROM\b)""",
    re.IGNORECASE,
)


class ASTPIIMasker:
    """Mask sensitive values in JSON-compatible AST structures with syntax preservation and memory hygiene."""

    def __init__(self) -> None:
        self.masked_count: int = 0
        self.retained_count: int = 0
        self.preserved_attack_tokens: int = 0

    def mask(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a new masked payload without mutating the caller's object."""
        masked = self._mask_value(payload, key_hint=None)
        if not isinstance(masked, dict):
            raise TypeError("payload must be a mapping")
        return masked

    def mask_with_retention_metrics(
        self, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Anonymize payload and return (masked_payload, semantic_attribute_retention_ratio)."""
        self.masked_count = 0
        self.retained_count = 0
        self.preserved_attack_tokens = 0
        masked = self.mask(payload)
        total = self.masked_count + self.retained_count
        retention = (self.retained_count / total) if total > 0 else 1.0
        return masked, retention

    def _mask_value(self, value: Any, key_hint: str | None) -> Any:
        """Recursively traverse AST mappings, lists, and sensitive scalar values."""
        if key_hint and self._is_credential_key(key_hint):
            if isinstance(value, str) and self._has_syntactic_attack_tokens(value):
                self.masked_count += 1
                self.preserved_attack_tokens += 1
                return self._mask_preserving_syntax(value)
            self.masked_count += 1
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
            if self._has_syntactic_attack_tokens(value):
                self.masked_count += 1
                self.preserved_attack_tokens += 1
                return self._mask_preserving_syntax(value)
            self.masked_count += 1
            return REDACTED

        self.retained_count += 1
        return value

    def _mask_preserving_syntax(self, value: str) -> str:
        """Mask alphanumeric PII literals in-place while strictly preserving syntactic attack tokens.
        
        Example: "secretpass' OR 1=1; --" -> "[REDACTED_PII]' OR 1=1; --"
        Example: "<script>alert('xss')</script>user@test.org" -> "<script>alert('xss')</script>[REDACTED_PII]"
        """
        attack_spans: list[tuple[int, int]] = []
        for match in _SYNTACTIC_ATTACK_PATTERN.finditer(value):
            attack_spans.append(match.span())

        def is_attack_or_syntax_char(idx: int) -> bool:
            for start, end in attack_spans:
                if start <= idx < end:
                    return True
            return value[idx] in {"'", '"', ';', '<', '>', '=', '(', ')', '-', '/', '*', ' '}

        result = []
        i = 0
        while i < len(value):
            if is_attack_or_syntax_char(i):
                result.append(value[i])
                i += 1
            else:
                token_chars = []
                while i < len(value) and not is_attack_or_syntax_char(i):
                    token_chars.append(value[i])
                    i += 1
                token_str = "".join(token_chars)
                if token_str.upper() in {"OR", "AND", "UNION", "SELECT", "FROM", "WHERE", "DROP", "TABLE", "NULL", "TRUE", "FALSE"}:
                    result.append(token_str)
                else:
                    result.append(REDACTED)

        return "".join(result)

    @staticmethod
    def _has_syntactic_attack_tokens(value: str) -> bool:
        """Check if string contains SQLi/XSS/Command-Injection metacharacters."""
        return bool(_SYNTACTIC_ATTACK_PATTERN.search(value))

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

    @staticmethod
    def secure_wipe(target: Any) -> None:
        """Securely wipe transient in-memory objects to prevent data leakage (NFR4)."""
        if isinstance(target, bytearray):
            for i in range(len(target)):
                target[i] = 0
        elif isinstance(target, dict):
            for k in list(target.keys()):
                val = target[k]
                if isinstance(val, (dict, list, bytearray)):
                    ASTPIIMasker.secure_wipe(val)
                del target[k]
            target.clear()
        elif isinstance(target, list):
            for item in target:
                if isinstance(item, (dict, list, bytearray)):
                    ASTPIIMasker.secure_wipe(item)
            target.clear()


# Backward-compatible alias
PIIMasker = ASTPIIMasker
