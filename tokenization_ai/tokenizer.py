"""Hybrid structure-aware tokenization of normalized records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import FusedInput, TokenSequence

SEQ_START = "[SEQ_START]"
SEQ_END = "[SEQ_END]"

API_METHOD = "[API:METHOD]"
API_PATH_START = "[API:PATH_START]"
API_PATH_SEG = "[API:PATH_SEG]"
API_PATH_END = "[API:PATH_END]"
API_HEADERS_START = "[API:HEADERS_START]"
API_HEADER_NAME = "[API:HEADER_NAME]"
API_HEADER_VALUE = "[API:HEADER_VALUE]"
API_HEADERS_END = "[API:HEADERS_END]"
API_PAYLOAD_START = "[API:PAYLOAD_START]"
API_PAYLOAD_END = "[API:PAYLOAD_END]"
API_KEY = "[API:KEY]"
API_OBJ_START = "[API:OBJ_START]"
API_OBJ_END = "[API:OBJ_END]"
API_LIST_START = "[API:LIST_START]"
API_LIST_END = "[API:LIST_END]"
API_VAL_STR = "[API:VAL_STR]"
API_VAL_INT = "[API:VAL_INT]"
API_VAL_FLOAT = "[API:VAL_FLOAT]"
API_VAL_BOOL = "[API:VAL_BOOL]"
API_VAL_NULL = "[API:VAL_NULL]"

RULES_START = "[RULES_START]"
RULES_END = "[RULES_END]"
RULE_START = "[RULE_START]"
RULE_END = "[RULE_END]"
RULE_ID = "[RULE:ID]"
RULE_ATTRS_START = "[RULE:ATTRS_START]"
RULE_ATTRS_END = "[RULE:ATTRS_END]"
RULE_KEY = "[RULE:KEY]"
RULE_OBJ_START = "[RULE:OBJ_START]"
RULE_OBJ_END = "[RULE:OBJ_END]"
RULE_LIST_START = "[RULE:LIST_START]"
RULE_LIST_END = "[RULE:LIST_END]"
RULE_VAL_STR = "[RULE:VAL_STR]"
RULE_VAL_INT = "[RULE:VAL_INT]"
RULE_VAL_FLOAT = "[RULE:VAL_FLOAT]"
RULE_VAL_BOOL = "[RULE:VAL_BOOL]"
RULE_VAL_NULL = "[RULE:VAL_NULL]"


@dataclass(frozen=True)
class _ValueTags:
    """Tag set used to encode one namespace's (API or rule) nested values."""

    key: str
    obj_start: str
    obj_end: str
    list_start: str
    list_end: str
    val_str: str
    val_int: str
    val_float: str
    val_bool: str
    val_null: str


_API_TAGS = _ValueTags(
    key=API_KEY,
    obj_start=API_OBJ_START,
    obj_end=API_OBJ_END,
    list_start=API_LIST_START,
    list_end=API_LIST_END,
    val_str=API_VAL_STR,
    val_int=API_VAL_INT,
    val_float=API_VAL_FLOAT,
    val_bool=API_VAL_BOOL,
    val_null=API_VAL_NULL,
)

_RULE_TAGS = _ValueTags(
    key=RULE_KEY,
    obj_start=RULE_OBJ_START,
    obj_end=RULE_OBJ_END,
    list_start=RULE_LIST_START,
    list_end=RULE_LIST_END,
    val_str=RULE_VAL_STR,
    val_int=RULE_VAL_INT,
    val_float=RULE_VAL_FLOAT,
    val_bool=RULE_VAL_BOOL,
    val_null=RULE_VAL_NULL,
)


class HybridTokenizer:
    """Tokenize a record while preserving its structural boundaries.

    Every value is preceded by a namespaced tag such as ``[API:VAL_STR]`` or
    ``[RULE:VAL_STR]``, so API-derived and rule-derived tokens are never
    confusable even when a field name happens to match. Dict keys are
    visited in sorted order so key ordering in the source object cannot
    change the resulting sequence; list items are visited in their given
    order since position in a list is itself meaningful.
    """

    def tokenize(self, fused: FusedInput) -> TokenSequence:
        """Return the ordered token sequence for a normalized record."""
        context = fused.context
        tokens: list[str] = [SEQ_START]

        tokens.append(API_METHOD)
        tokens.append(context.method)

        tokens.append(API_PATH_START)
        for segment in self._path_segments(context.path):
            tokens.append(API_PATH_SEG)
            tokens.append(segment)
        tokens.append(API_PATH_END)

        tokens.append(API_HEADERS_START)
        for name in sorted(context.headers.keys(), key=str):
            tokens.append(API_HEADER_NAME)
            tokens.append(name)
            tokens.append(API_HEADER_VALUE)
            tokens.append(context.headers[name])
        tokens.append(API_HEADERS_END)

        tokens.append(API_PAYLOAD_START)
        self._encode_value(context.payload, _API_TAGS, tokens)
        tokens.append(API_PAYLOAD_END)

        tokens.append(RULES_START)
        for rule in fused.rule_info.rules:
            tokens.append(RULE_START)
            tokens.append(RULE_ID)
            tokens.append(rule.rule_id)
            tokens.append(RULE_ATTRS_START)
            self._encode_value(rule.attributes, _RULE_TAGS, tokens)
            tokens.append(RULE_ATTRS_END)
            tokens.append(RULE_END)
        tokens.append(RULES_END)

        tokens.append(SEQ_END)
        return TokenSequence(event_id=fused.event_id, tokens=tokens)

    @staticmethod
    def _path_segments(path: str) -> list[str]:
        """Split a path into its non-empty segments, in order."""
        return [segment for segment in path.split("/") if segment]

    @classmethod
    def _encode_value(cls, value: Any, tags: _ValueTags, tokens: list[str]) -> None:
        """Recursively append tokens for ``value``, tagged within ``tags``'s namespace.

        ``bool`` is checked ahead of ``int`` because ``bool`` is an ``int``
        subclass in Python. Masking placeholders are plain strings and are
        therefore encoded atomically as a single ``val_str`` token, never
        split or inspected.
        """
        if isinstance(value, bool):
            tokens.append(tags.val_bool)
            tokens.append("true" if value else "false")
        elif isinstance(value, int):
            tokens.append(tags.val_int)
            tokens.append(str(value))
        elif isinstance(value, float):
            tokens.append(tags.val_float)
            tokens.append(str(value))
        elif value is None:
            tokens.append(tags.val_null)
        elif isinstance(value, str):
            tokens.append(tags.val_str)
            tokens.append(value)
        elif isinstance(value, Mapping):
            tokens.append(tags.obj_start)
            for key in sorted(value.keys(), key=str):
                tokens.append(tags.key)
                tokens.append(str(key))
                cls._encode_value(value[key], tags, tokens)
            tokens.append(tags.obj_end)
        elif isinstance(value, (list, tuple)):
            tokens.append(tags.list_start)
            for item in value:
                cls._encode_value(item, tags, tokens)
            tokens.append(tags.list_end)
        else:
            tokens.append(tags.val_str)
            tokens.append(str(value))
