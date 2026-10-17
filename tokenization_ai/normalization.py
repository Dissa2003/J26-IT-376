"""Normalization of masked records into a canonical form for tokenization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .masking import FieldPath
from .models import FusedInput

_REPEATED_SLASHES = re.compile(r"/{2,}")


@dataclass(frozen=True)
class NormalizationResult:
    """A normalized ``FusedInput`` together with its re-keyed masking provenance.

    ``masked_paths`` is the same provenance produced by
    ``PrivacyMasker.mask_with_provenance``, carried through
    :meth:`Normalizer.normalize_with_provenance` so it still points at the
    right fields after normalization -- see that method's docstring for why
    only header paths ever need re-keying.
    """

    fused: FusedInput
    masked_paths: frozenset[FieldPath]


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

    def normalize_with_provenance(
        self, fused: FusedInput, masked_paths: frozenset[FieldPath] = frozenset()
    ) -> NormalizationResult:
        """Normalize ``fused`` and re-key ``masked_paths`` to survive normalization.

        Of the transforms normalization applies, only header-name casing can
        invalidate a field path: a path recorded as
        ``("headers", "Authorization")`` before normalization must become
        ``("headers", "authorization")`` after it, to still point at the
        header's post-normalization key. Payload keys are never renamed by
        normalization (dict iteration order changes, but a field's name
        does not), and list order/indices are preserved, so payload- and
        list-based paths need no re-keying and are passed through as-is.

        Delegates to :meth:`normalize` for the actual normalization, so
        there is exactly one normalization implementation.
        """
        normalized = self.normalize(fused)
        remapped = {self._remap_path(path) for path in masked_paths}
        return NormalizationResult(fused=normalized, masked_paths=frozenset(remapped))

    @classmethod
    def _remap_path(cls, path: FieldPath) -> FieldPath:
        if len(path) >= 2 and path[0] == "headers":
            return ("headers", cls._normalize_header_key(str(path[1]))) + tuple(path[2:])
        return path

    @staticmethod
    def _normalize_header_key(key: str) -> str:
        """The exact header-name canonicalization :meth:`_normalize_headers` applies."""
        return key.strip().lower()

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

    @classmethod
    def _normalize_headers(cls, headers: Mapping[str, str]) -> dict[str, str]:
        """Lower-case and trim header names; header values are left exactly as given."""
        return {cls._normalize_header_key(key): value for key, value in headers.items()}

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
