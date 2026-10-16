"""Hybrid structure-aware tokenization of normalized records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .masking import FieldPath
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
API_HEADER_VALUE_MASKED = "[API:HEADER_VALUE_MASKED]"
API_HEADERS_END = "[API:HEADERS_END]"
API_PAYLOAD_START = "[API:PAYLOAD_START]"
API_PAYLOAD_END = "[API:PAYLOAD_END]"
API_KEY = "[API:KEY]"
API_OBJ_START = "[API:OBJ_START]"
API_OBJ_END = "[API:OBJ_END]"
API_LIST_START = "[API:LIST_START]"
API_LIST_END = "[API:LIST_END]"
API_VAL_STR = "[API:VAL_STR]"
API_VAL_MASKED_STR = "[API:VAL_MASKED_STR]"
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

# Authoritative token roles (see ``TokenSequence.roles``). Every token this
# tokenizer emits gets exactly one of these, decided by the same code path
# that produces the token's text -- never inferred afterward from
# neighboring tokens. This is what makes the role information trustworthy
# even when an attacker controls arbitrary literal text (a payload key,
# a header value, ...): the role of a *different* token can never be
# changed by what text appears elsewhere in the sequence.
TOKEN_ROLE_CONTROL = "control"
TOKEN_ROLE_LITERAL_API = "literal_api"
TOKEN_ROLE_LITERAL_RULE = "literal_rule"
TOKEN_ROLE_MASKED = "masked"


class _TokenEmitter:
    """Appends tokens and their authoritative roles in lockstep.

    Every public method appends exactly one token and exactly one role, so
    ``tokens`` and ``roles`` can never drift out of length-sync with each
    other, and every call site states the role it intends rather than
    leaving it to be inferred later from context.
    """

    __slots__ = ("tokens", "roles")

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.roles: list[str] = []

    def control(self, tag: str) -> None:
        """Emit one of the tokenizer's own fixed structural/control tags."""
        self.tokens.append(tag)
        self.roles.append(TOKEN_ROLE_CONTROL)

    def literal(self, text: str, role: str) -> None:
        """Emit an ordinary open-vocabulary literal in the given namespace role."""
        self.tokens.append(text)
        self.roles.append(role)

    def masked(self, text: str) -> None:
        """Emit a value the trusted masking stage genuinely substituted."""
        self.tokens.append(text)
        self.roles.append(TOKEN_ROLE_MASKED)


@dataclass(frozen=True)
class _ValueTags:
    """Tag set used to encode one namespace's (API or rule) nested values.

    ``literal_role`` is the ``TOKEN_ROLE_*`` this namespace's ordinary
    literals carry. ``val_masked_str`` is the trusted-provenance variant of
    ``val_str``, emitted instead of it when the caller's ``masked_paths``
    says this exact position was genuinely substituted by
    ``PrivacyMasker`` -- never when a value merely happens to read like a
    masking placeholder. It is ``None`` for the rule namespace: masking
    never touches verified rule information today, so there is no genuine
    "masked rule value" case to represent.
    """

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
    literal_role: str
    val_masked_str: str | None = None


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
    literal_role=TOKEN_ROLE_LITERAL_API,
    val_masked_str=API_VAL_MASKED_STR,
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
    literal_role=TOKEN_ROLE_LITERAL_RULE,
    val_masked_str=None,
)


class HybridTokenizer:
    """Tokenize a record while preserving its structural boundaries.

    Every value is preceded by a namespaced tag such as ``[API:VAL_STR]`` or
    ``[RULE:VAL_STR]``, so API-derived and rule-derived tokens are never
    confusable even when a field name happens to match. Dict keys are
    visited in sorted order so key ordering in the source object cannot
    change the resulting sequence; list items are visited in their given
    order since position in a list is itself meaningful.

    Each token is also tagged with an authoritative ``TOKEN_ROLE_*`` (see
    ``TokenSequence.roles``), decided by this same code as it emits the
    token -- so a downstream consumer such as ``Vocabulary`` never has to
    guess a token's role from its neighbors, which is what previously let
    an attacker-controlled literal corrupt the classification of a
    different, genuine control tag elsewhere in the sequence.
    """

    def tokenize(
        self, fused: FusedInput, masked_paths: frozenset[FieldPath] = frozenset()
    ) -> TokenSequence:
        """Return the ordered token sequence for a normalized record.

        ``masked_paths`` is the trusted provenance produced by
        ``PrivacyMasker.mask_with_provenance`` (optionally re-keyed by
        ``Normalizer.normalize_with_provenance``): the set of field
        locations that were genuinely substituted by masking. A value whose
        path is in this set is both tagged with the namespace's
        ``val_masked_str``/``API_HEADER_VALUE_MASKED`` tag and given the
        ``TOKEN_ROLE_MASKED`` role, instead of the ordinary string tag and
        literal role -- this is what lets the vocabulary tell a genuinely
        masked value apart from attacker-controlled text that merely reads
        like a placeholder. The default (no provenance) keeps this
        method's behavior unchanged for callers that don't supply it.
        """
        context = fused.context
        emitter = _TokenEmitter()

        emitter.control(SEQ_START)

        emitter.control(API_METHOD)
        emitter.literal(context.method, TOKEN_ROLE_LITERAL_API)

        emitter.control(API_PATH_START)
        for segment in self._path_segments(context.path):
            emitter.control(API_PATH_SEG)
            emitter.literal(segment, TOKEN_ROLE_LITERAL_API)
        emitter.control(API_PATH_END)

        emitter.control(API_HEADERS_START)
        for name in sorted(context.headers.keys(), key=str):
            emitter.control(API_HEADER_NAME)
            emitter.literal(name, TOKEN_ROLE_LITERAL_API)
            if ("headers", name) in masked_paths:
                emitter.control(API_HEADER_VALUE_MASKED)
                emitter.masked(context.headers[name])
            else:
                emitter.control(API_HEADER_VALUE)
                emitter.literal(context.headers[name], TOKEN_ROLE_LITERAL_API)
        emitter.control(API_HEADERS_END)

        emitter.control(API_PAYLOAD_START)
        self._encode_value(context.payload, _API_TAGS, emitter, ("payload",), masked_paths)
        emitter.control(API_PAYLOAD_END)

        emitter.control(RULES_START)
        for rule in fused.rule_info.rules:
            emitter.control(RULE_START)
            emitter.control(RULE_ID)
            emitter.literal(rule.rule_id, TOKEN_ROLE_LITERAL_RULE)
            emitter.control(RULE_ATTRS_START)
            self._encode_value(rule.attributes, _RULE_TAGS, emitter, (), frozenset())
            emitter.control(RULE_ATTRS_END)
            emitter.control(RULE_END)
        emitter.control(RULES_END)

        emitter.control(SEQ_END)
        return TokenSequence(event_id=fused.event_id, tokens=emitter.tokens, roles=emitter.roles)

    @staticmethod
    def _path_segments(path: str) -> list[str]:
        """Split a path into its non-empty segments, in order."""
        return [segment for segment in path.split("/") if segment]

    @classmethod
    def _encode_value(
        cls,
        value: Any,
        tags: _ValueTags,
        emitter: _TokenEmitter,
        path: FieldPath,
        masked_paths: frozenset[FieldPath],
    ) -> None:
        """Recursively emit tokens for ``value``, tagged within ``tags``'s namespace.

        ``bool`` is checked ahead of ``int`` because ``bool`` is an ``int``
        subclass in Python. A string value whose ``path`` is in
        ``masked_paths`` is emitted via ``emitter.masked`` instead of
        ``emitter.literal`` -- this is a provenance decision made from
        ``path``, never from the value's own text, so it cannot be spoofed
        by attacker-controlled content that merely reads like a masking
        placeholder. Either way the value is still encoded atomically as a
        single token, never split or inspected.
        """
        if isinstance(value, bool):
            emitter.control(tags.val_bool)
            emitter.literal("true" if value else "false", tags.literal_role)
        elif isinstance(value, int):
            emitter.control(tags.val_int)
            emitter.literal(str(value), tags.literal_role)
        elif isinstance(value, float):
            emitter.control(tags.val_float)
            emitter.literal(str(value), tags.literal_role)
        elif value is None:
            emitter.control(tags.val_null)
        elif isinstance(value, str):
            if tags.val_masked_str is not None and path in masked_paths:
                emitter.control(tags.val_masked_str)
                emitter.masked(value)
            else:
                emitter.control(tags.val_str)
                emitter.literal(value, tags.literal_role)
        elif isinstance(value, Mapping):
            emitter.control(tags.obj_start)
            for key in sorted(value.keys(), key=str):
                emitter.control(tags.key)
                emitter.literal(str(key), tags.literal_role)
                cls._encode_value(value[key], tags, emitter, path + (str(key),), masked_paths)
            emitter.control(tags.obj_end)
        elif isinstance(value, (list, tuple)):
            emitter.control(tags.list_start)
            for index, item in enumerate(value):
                cls._encode_value(item, tags, emitter, path + (index,), masked_paths)
            emitter.control(tags.list_end)
        else:
            emitter.control(tags.val_str)
            emitter.literal(str(value), tags.literal_role)
