"""Validation for raw events before they enter the ingestion stream."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from shared_contracts.models import RawEvent

_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class IngestionValidator:
    """Validate and normalize untrusted event payloads."""

    def validate(self, payload: Mapping[str, Any]) -> RawEvent:
        """Return a validated event or raise ``ValueError`` with field details."""
        try:
            event = RawEvent.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("invalid event schema", exc.errors()) from exc

        if not event.source.strip():
            raise ValueError("source must be non-empty")
        if event.timestamp.tzinfo is None or event.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        self._validate_mapping(event.payload, "payload")
        return event

    def _validate_mapping(self, values: Mapping[str, Any], path: str) -> None:
        """Validate field names and recursively validate JSON-compatible values."""
        for key, value in values.items():
            if not isinstance(key, str) or not _KEY_PATTERN.fullmatch(key):
                raise ValueError(f"{path} contains an invalid key: {key!r}")
            self._validate_value(value, f"{path}.{key}")

    def _validate_value(self, value: Any, path: str) -> None:
        """Reject non-JSON values and non-finite numeric values recursively."""
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{path} must be finite")
            return
        if isinstance(value, Mapping):
            self._validate_mapping(value, path)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                self._validate_value(item, f"{path}[{index}]")
            return
        raise ValueError(f"{path} has an unsupported value type")
