"""Tensor preparation: token ids padded or truncated to model input shape.

Truncation policy
------------------
When a sequence's encoded length exceeds ``max_length``, only the payload's
own content is ever shrunk. Everything else -- ``SEQ_START``/``SEQ_END``,
the method, path, headers, and the *entire* verified-rule segment
(``RULES_START`` .. ``RULES_END``) -- is treated as mandatory and is never
cut, never reordered, and never partially dropped: per-project decision,
silently discarding verified-rule information is not an acceptable
truncation strategy, so if the mandatory framing plus the complete rule
segment cannot fit in ``max_length`` even with an empty payload, this
raises ``ValueError`` instead.

Payload shrinking is structure-aware: it only ever drops *complete*
top-level fields (and, recursively, complete list elements), never cuts in
the middle of a value, and the result is always a grammatically balanced
token sequence (every ``OBJ_START``/``LIST_START`` has a matching
``OBJ_END``/``LIST_END``). Fields/elements are kept in their original
(tokenizer-emitted, already-deterministic) order, greedily: each unit that
fits whole is kept whole; a unit that doesn't fit whole is recursively
shrunk (if it is itself an object/list) or dropped entirely (if it's an
atomic scalar, which can never be partially kept without corrupting the
literal). If nothing at all fits, the payload degrades to the grammar's own
minimal valid representation, an empty object (``OBJ_START``, ``OBJ_END``).

Segment boundaries are located by *(role, token)* pairs -- a position must
have ``TOKEN_ROLE_CONTROL`` **and** match one of the tokenizer's own
imported tag constants -- never by matching token text alone. This
continues the Phase 7 invariant that identity is never inferred from text:
an attacker-controlled payload value or key could otherwise be textually
identical to a boundary tag such as ``"[API:PAYLOAD_START]"`` without this
code mistaking it for the real boundary.
"""

from __future__ import annotations

from .models import ModelInput, TokenSequence
from .tokenizer import (
    API_LIST_END,
    API_LIST_START,
    API_OBJ_END,
    API_OBJ_START,
    API_PAYLOAD_END,
    API_PAYLOAD_START,
    API_VAL_BOOL,
    API_VAL_FLOAT,
    API_VAL_INT,
    API_VAL_MASKED_STR,
    API_VAL_NULL,
    API_VAL_STR,
    TOKEN_ROLE_CONTROL,
)
from .vocabulary import Vocabulary

# The API-namespace container open/close tags (payload is always encoded in
# the API namespace -- it's a plain ``dict[str, Any]``, never rule data).
_OPEN_TAGS = frozenset({API_OBJ_START, API_LIST_START})
_CLOSE_TAGS = frozenset({API_OBJ_END, API_LIST_END})
_MATCHING_CLOSE = {API_OBJ_START: API_OBJ_END, API_LIST_START: API_LIST_END}

# API-namespace scalar type tags: each is followed by exactly one literal
# token (the value itself). ``API_VAL_NULL`` is followed by nothing.
_SCALAR_VALUE_TAGS = frozenset(
    {API_VAL_STR, API_VAL_MASKED_STR, API_VAL_INT, API_VAL_FLOAT, API_VAL_BOOL}
)


class Tensorizer:
    """Convert token sequences into fixed-length model inputs."""

    def __init__(self, vocabulary: Vocabulary, max_length: int) -> None:
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self.vocabulary = vocabulary
        self.max_length = max_length

    def tensorize(self, sequence: TokenSequence) -> ModelInput:
        """Return token ids and attention mask, both exactly ``max_length`` long.

        Raises whatever :meth:`Vocabulary.encode` raises for a malformed
        ``sequence`` (role/length mismatches, unrecognized roles) or an
        unfitted vocabulary -- this method does not duplicate that
        validation. Raises ``ValueError`` if the sequence is too long to
        fit even after maximal structure-aware payload truncation (see the
        module docstring). Never mutates ``sequence`` or ``self.vocabulary``.
        """
        ids = self.vocabulary.encode(sequence)

        if len(ids) > self.max_length:
            shrunk = self._truncate_payload(sequence)
            ids = self.vocabulary.encode(shrunk)
            if len(ids) > self.max_length:
                # Should be unreachable given the budget arithmetic below;
                # fail loudly rather than silently emit an oversized input.
                raise ValueError(
                    f"event {sequence.event_id!r}: sequence could not be reduced to "
                    f"max_length={self.max_length} even after payload truncation "
                    f"({len(ids)} tokens remain)"
                )
            event_id = shrunk.event_id
        else:
            event_id = sequence.event_id

        pad_id = self.vocabulary.pad_id
        padding = self.max_length - len(ids)
        input_ids = ids + [pad_id] * padding
        attention_mask = [1] * len(ids) + [0] * padding
        return ModelInput(event_id=event_id, input_ids=input_ids, attention_mask=attention_mask)

    def _truncate_payload(self, sequence: TokenSequence) -> TokenSequence:
        """Return a copy of ``sequence`` with only its payload content shrunk.

        Raises ``ValueError`` if the mandatory framing (everything outside
        the payload's own value span, including the full rule segment)
        does not fit in ``max_length`` even with an empty payload.
        """
        tokens, roles = sequence.tokens, sequence.roles
        payload_start = self._control_index(tokens, roles, API_PAYLOAD_START)
        payload_end = self._control_index(tokens, roles, API_PAYLOAD_END)

        value_start = payload_start + 1  # the payload's own OBJ_START
        value_end = payload_end  # one past the payload's own OBJ_END
        if not (roles[value_start] == TOKEN_ROLE_CONTROL and tokens[value_start] == API_OBJ_START):
            raise ValueError(
                f"event {sequence.event_id!r}: payload value does not start with "
                f"{API_OBJ_START!r} as the tokenizer's grammar requires"
            )

        fixed_outside_payload = len(tokens) - (value_end - value_start)
        minimum_required = fixed_outside_payload + 2  # empty payload: OBJ_START, OBJ_END
        if minimum_required > self.max_length:
            raise ValueError(
                f"event {sequence.event_id!r}: mandatory framing and verified-rule "
                f"segment alone need {minimum_required} tokens, which exceeds "
                f"max_length={self.max_length}; refusing to drop rule or context "
                "information to make it fit"
            )

        payload_budget = self.max_length - fixed_outside_payload
        kept_value_indices = self._fit_value(tokens, roles, value_start, value_end, payload_budget)

        kept_indices = list(range(0, value_start)) + kept_value_indices + list(range(value_end, len(tokens)))
        trunc_tokens = [tokens[i] for i in kept_indices]
        trunc_roles = [roles[i] for i in kept_indices]
        return TokenSequence(event_id=sequence.event_id, tokens=trunc_tokens, roles=trunc_roles)

    @classmethod
    def _fit_value(
        cls, tokens: list[str], roles: list[str], start: int, end: int, budget: int
    ) -> list[int]:
        """Return original indices for a structurally valid value spanning
        ``[start, end)``, shrunk (dropping whole fields/elements, recursively)
        to fit within ``budget`` tokens. ``budget`` is guaranteed sufficient
        by the caller (at least 2, for an empty container) whenever this is
        called on a container; for an atomic scalar there is no way to
        shrink it, so it is simply dropped (returns ``[]``) if it doesn't
        fit whole.
        """
        length = end - start
        if length <= budget:
            return list(range(start, end))

        if not (roles[start] == TOKEN_ROLE_CONTROL and tokens[start] in _OPEN_TAGS):
            # An atomic scalar (or null) that doesn't fit whole can't be
            # partially kept without corrupting the literal -- drop it.
            return []

        open_tag = tokens[start]
        close_tag = _MATCHING_CLOSE[open_tag]
        overhead = 2  # the open and close tag themselves
        if overhead > budget:
            return []

        content_start = start + 1
        content_end = end - 1  # index of the matching close tag
        inner_budget = budget - overhead

        if open_tag == API_OBJ_START:
            kept_inner = cls._fit_object_fields(tokens, roles, content_start, content_end, inner_budget)
        else:
            kept_inner = cls._fit_list_elements(tokens, roles, content_start, content_end, inner_budget)

        return [start, *kept_inner, content_end]

    @classmethod
    def _fit_object_fields(
        cls, tokens: list[str], roles: list[str], start: int, end: int, budget: int
    ) -> list[int]:
        kept: list[int] = []
        remaining = budget
        for key_start, value_start, value_end in cls._split_object_fields(tokens, roles, start, end):
            whole_length = value_end - key_start
            if whole_length <= remaining:
                kept.extend(range(key_start, value_end))
                remaining -= whole_length
                continue
            key_cost = value_start - key_start  # the API_KEY tag plus its key literal
            if key_cost < remaining:
                shrunk_value = cls._fit_value(tokens, roles, value_start, value_end, remaining - key_cost)
                if shrunk_value:
                    kept.extend(range(key_start, value_start))
                    kept.extend(shrunk_value)
                    remaining -= key_cost + len(shrunk_value)
            # Otherwise this field is dropped entirely; later, possibly
            # smaller, fields are still given a chance to fit.
        return kept

    @classmethod
    def _fit_list_elements(
        cls, tokens: list[str], roles: list[str], start: int, end: int, budget: int
    ) -> list[int]:
        kept: list[int] = []
        remaining = budget
        for elem_start, elem_end in cls._split_list_elements(tokens, roles, start, end):
            whole_length = elem_end - elem_start
            if whole_length <= remaining:
                kept.extend(range(elem_start, elem_end))
                remaining -= whole_length
                continue
            shrunk = cls._fit_value(tokens, roles, elem_start, elem_end, remaining)
            if shrunk:
                kept.extend(shrunk)
                remaining -= len(shrunk)
        return kept

    @classmethod
    def _split_object_fields(
        cls, tokens: list[str], roles: list[str], start: int, end: int
    ) -> list[tuple[int, int, int]]:
        """Split an object's content into ``(key_start, value_start, value_end)`` fields."""
        fields = []
        i = start
        while i < end:
            key_start = i
            value_start = i + 2  # API_KEY tag, key literal
            value_end = cls._value_end(tokens, roles, value_start)
            fields.append((key_start, value_start, value_end))
            i = value_end
        return fields

    @classmethod
    def _split_list_elements(
        cls, tokens: list[str], roles: list[str], start: int, end: int
    ) -> list[tuple[int, int]]:
        """Split a list's content into ``(element_start, element_end)`` elements."""
        elements = []
        i = start
        while i < end:
            elem_end = cls._value_end(tokens, roles, i)
            elements.append((i, elem_end))
            i = elem_end
        return elements

    @staticmethod
    def _value_end(tokens: list[str], roles: list[str], start: int) -> int:
        """Return the index one past the value encoding starting at ``start``.

        A value is either a container (balanced via depth-counting, counting
        only role-confirmed control tags -- never a literal's text alone,
        even if it happens to read like a boundary tag) or a scalar (a
        control type-tag followed by exactly one literal/masked token, or
        ``API_VAL_NULL`` alone).
        """
        if roles[start] == TOKEN_ROLE_CONTROL and tokens[start] in _OPEN_TAGS:
            depth = 1
            i = start + 1
            while depth > 0:
                if roles[i] == TOKEN_ROLE_CONTROL:
                    if tokens[i] in _OPEN_TAGS:
                        depth += 1
                    elif tokens[i] in _CLOSE_TAGS:
                        depth -= 1
                i += 1
            return i
        if roles[start] == TOKEN_ROLE_CONTROL and tokens[start] == API_VAL_NULL:
            return start + 1
        if roles[start] == TOKEN_ROLE_CONTROL and tokens[start] in _SCALAR_VALUE_TAGS:
            return start + 2
        raise ValueError(f"index {start}: expected a value-starting control tag, found {tokens[start]!r}")

    @staticmethod
    def _control_index(tokens: list[str], roles: list[str], tag: str) -> int:
        """Index of the first token with role CONTROL and text ``tag``.

        Deliberately checks role *and* text together -- never text alone --
        so an attacker-controlled literal that merely reads like ``tag``
        can never be mistaken for the real structural boundary.
        """
        for index, (token, role) in enumerate(zip(tokens, roles)):
            if role == TOKEN_ROLE_CONTROL and token == tag:
                return index
        raise ValueError(f"expected control token {tag!r} not found in sequence")
