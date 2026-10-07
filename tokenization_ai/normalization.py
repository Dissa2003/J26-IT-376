"""Normalization of masked records into a canonical form for tokenization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .models import FusedInput

_REPEATED_SLASHES = re.compile(r"/{2,}")


class Normalizer:
    """Canonicalize a masked ``FusedInput`` so equivalent inputs normalize identically.

    Normalization only reshapes formatting (method casing, path separators,
    header-name casing, numeric type, dict key order). It never renames a
    payload field, drops a value, or changes a masking placeholder.
    """

    def normalize(self, fused: FusedInput) -> FusedInput:
        """Return a new fused record in canonical form.

        The original ``fused`` object (and its nested context) is never
        mutated, and verified rule information is carried through unchanged.
        """
        context = fused.context
        normalized_context = context.model_copy(
            update={
                "method": self._normalize_method(context.method),
                "path": self._normalize_path(context.path),
                "headers": self._normalize_headers(context.headers),
                "payload": self._normalize_value(context.payload),
            }
        )
        return FusedInput(context=normalized_context, rule_info=fused.rule_info)

    @staticmethod
    def _normalize_method(method: str) -> str:
        """Upper-case the HTTP method and drop surrounding whitespace."""
        return method.strip().upper()

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Ensure a single leading slash, collapse repeats, and drop a trailing slash."""
        trimmed = path.strip()
        if not trimmed.startswith("/"):
            trimmed = "/" + trimmed
        collapsed = _REPEATED_SLASHES.sub("/", trimmed)
        if len(collapsed) > 1 and collapsed.endswith("/"):
            collapsed = collapsed.rstrip("/") or "/"
        return collapsed

    @staticmethod
    def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
        """Lower-case and trim header names; header values are left exactly as given."""
        return {key.strip().lower(): value for key, value in headers.items()}

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """Recursively canonicalize payload values without renaming any field.

        Mapping keys are sorted so payloads that differ only in field order
        normalize identically. Primitive types are preserved exactly: ``int``
        stays ``int``, ``float`` stays ``float``, ``bool`` and ``None`` pass
        through unchanged, and masking placeholders (plain strings) survive
        untouched; lists keep their order.
        """
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, Mapping):
            return {
                str(key): cls._normalize_value(value[key])
                for key in sorted(value.keys(), key=str)
            }
        if isinstance(value, (list, tuple)):
            return [cls._normalize_value(item) for item in value]
        return value
