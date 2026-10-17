"""Unit tests for tensor preparation."""

import pytest

from tokenization_ai.models import FusedInput, TokenSequence, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.tensorizer import Tensorizer
from tokenization_ai.tokenizer import (
    API_OBJ_END,
    API_OBJ_START,
    HybridTokenizer,
    TOKEN_ROLE_CONTROL,
)
from tokenization_ai.vocabulary import Vocabulary


def _fused(rules: list[VerifiedRule] | None = None, **context_overrides: object) -> FusedInput:
    fields: dict[str, object] = {"event_id": "e1", "method": "POST", "path": "/orders"}
    fields.update(context_overrides)
    context = UnifiedApiContext(**fields)
    rule_info = VerifiedRuleInfo(event_id="e1", rules=rules or [])
    return FusedInput(context=context, rule_info=rule_info)


def _sequence(**kwargs: object) -> TokenSequence:
    return HybridTokenizer().tokenize(_fused(**kwargs))


def _fitted_vocab(*sequences: TokenSequence) -> Vocabulary:
    vocabulary = Vocabulary()
    vocabulary.fit(sequences)
    return vocabulary


def _decoded_non_pad(tensorizer: Tensorizer, model_input) -> list[str]:
    pad_id = tensorizer.vocabulary.pad_id
    real_ids = [i for i in model_input.input_ids if i != pad_id]
    return tensorizer.vocabulary.decode(real_ids)


def _assert_balanced(tokens: list[str], roles: list[str]) -> None:
    """Every OBJ_START/LIST_START (by role+identity) has a matching close, in order."""
    from tokenization_ai.tensorizer import _CLOSE_TAGS, _OPEN_TAGS

    depth = 0
    for token, role in zip(tokens, roles):
        if role == TOKEN_ROLE_CONTROL and token in _OPEN_TAGS:
            depth += 1
        elif role == TOKEN_ROLE_CONTROL and token in _CLOSE_TAGS:
            depth -= 1
            assert depth >= 0, "closing tag with no matching open tag"
    assert depth == 0, "unbalanced open tags at end of sequence"


# -- Constructor validation -------------------------------------------------


def test_constructor_rejects_zero_max_length() -> None:
    with pytest.raises(ValueError):
        Tensorizer(Vocabulary(), max_length=0)


def test_constructor_rejects_negative_max_length() -> None:
    with pytest.raises(ValueError):
        Tensorizer(Vocabulary(), max_length=-1)


# -- Basic shape, padding, and unfitted/malformed-input propagation --------


def test_tensorize_raises_for_unfitted_vocabulary() -> None:
    tensorizer = Tensorizer(Vocabulary(), max_length=16)

    with pytest.raises(RuntimeError):
        tensorizer.tensorize(_sequence())


def test_tensorize_propagates_vocabulary_error_for_mismatched_tokens_and_roles() -> None:
    vocabulary = _fitted_vocab(_sequence())
    tensorizer = Tensorizer(vocabulary, max_length=16)
    bad = TokenSequence(event_id="e1", tokens=["a", "b"], roles=["control"])

    with pytest.raises(ValueError):
        tensorizer.tensorize(bad)


def test_tensorize_propagates_vocabulary_error_for_unrecognized_role() -> None:
    vocabulary = _fitted_vocab(_sequence())
    tensorizer = Tensorizer(vocabulary, max_length=16)
    bad = TokenSequence(event_id="e1", tokens=["a"], roles=["not-a-real-role"])

    with pytest.raises(ValueError):
        tensorizer.tensorize(bad)


def test_tensorize_empty_sequence_is_all_padding() -> None:
    empty = TokenSequence(event_id="e1", tokens=[], roles=[])
    vocabulary = _fitted_vocab(empty)
    tensorizer = Tensorizer(vocabulary, max_length=8)

    result = tensorizer.tensorize(empty)

    assert result.event_id == "e1"
    assert result.input_ids == [vocabulary.pad_id] * 8
    assert result.attention_mask == [0] * 8


def test_tensorize_short_sequence_matches_encode_exactly_apart_from_padding() -> None:
    sequence = _sequence(payload={"amount": 5})
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) + 10
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    result = tensorizer.tensorize(sequence)

    expected_ids = vocabulary.encode(sequence)
    assert result.input_ids[: len(expected_ids)] == expected_ids
    assert result.input_ids[len(expected_ids) :] == [vocabulary.pad_id] * 10
    assert result.attention_mask == [1] * len(expected_ids) + [0] * 10
    assert len(result.input_ids) == max_length
    assert len(result.attention_mask) == max_length


def test_tensorize_exact_length_sequence_has_no_padding() -> None:
    sequence = _sequence(payload={"amount": 5})
    vocabulary = _fitted_vocab(sequence)
    exact_length = len(sequence.tokens)
    tensorizer = Tensorizer(vocabulary, max_length=exact_length)

    result = tensorizer.tensorize(sequence)

    assert result.input_ids == vocabulary.encode(sequence)
    assert result.attention_mask == [1] * exact_length


def test_tensorize_preserves_event_id() -> None:
    sequence = _sequence(payload={"amount": 5})
    vocabulary = _fitted_vocab(sequence)
    tensorizer = Tensorizer(vocabulary, max_length=len(sequence.tokens))

    result = tensorizer.tensorize(sequence)

    assert result.event_id == "e1"


def test_tensorize_does_not_mutate_the_sequence_or_vocabulary() -> None:
    sequence = _sequence(payload={"amount": 5, "items": ["a", "b", "c"]})
    vocabulary = _fitted_vocab(sequence)
    before_tokens = list(sequence.tokens)
    before_roles = list(sequence.roles)
    before_len = len(vocabulary)
    tensorizer = Tensorizer(vocabulary, max_length=len(sequence.tokens) - 4)

    tensorizer.tensorize(sequence)

    assert sequence.tokens == before_tokens
    assert sequence.roles == before_roles
    assert len(vocabulary) == before_len


def test_tensorize_is_deterministic() -> None:
    sequence = _sequence(payload={f"field_{i}": f"value_{i}" for i in range(20)})
    vocabulary = _fitted_vocab(sequence)
    tensorizer = Tensorizer(vocabulary, max_length=len(sequence.tokens) - 15)

    first = tensorizer.tensorize(sequence)
    second = tensorizer.tensorize(sequence)

    assert first == second


# -- Oversized sequences: payload truncation --------------------------------


def test_tensorize_truncates_oversized_payload_by_dropping_whole_fields() -> None:
    sequence = _sequence(payload={f"field_{i}": f"value_{i}" for i in range(20)})
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) - 20  # forces dropping some fields
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    result = tensorizer.tensorize(sequence)

    assert len(result.input_ids) == max_length
    assert len(result.attention_mask) == max_length
    decoded = _decoded_non_pad(tensorizer, result)
    kept_fields = sum(1 for i in range(20) if f"field_{i}" in decoded)
    assert 0 < kept_fields < 20  # some kept whole, some dropped whole


def test_tensorize_truncated_payload_is_structurally_balanced() -> None:
    sequence = _sequence(
        payload={
            "a": {"nested": {"deep": "value"}},
            **{f"field_{i}": f"value_{i}" for i in range(20)},
            "items": ["x", "y", "z", "w"],
        }
    )
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) - 30
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    shrunk = tensorizer._truncate_payload(sequence)

    _assert_balanced(shrunk.tokens, shrunk.roles)
    assert len(shrunk.tokens) <= max_length


def test_tensorize_shrinks_list_elements_when_a_single_field_dominates() -> None:
    sequence = _sequence(payload={"items": [f"item_{i}" for i in range(30)]})
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) - 40
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    result = tensorizer.tensorize(sequence)
    decoded = _decoded_non_pad(tensorizer, result)

    assert "items" in decoded  # the field itself survives
    kept_items = sum(1 for i in range(30) if f"item_{i}" in decoded)
    assert 0 < kept_items < 30  # a partial, but non-empty, list survived

    shrunk = tensorizer._truncate_payload(sequence)
    _assert_balanced(shrunk.tokens, shrunk.roles)


def test_tensorize_degrades_to_a_minimal_empty_payload_when_nothing_fits() -> None:
    sequence = _sequence(payload={f"field_{i}": f"value_{i}" for i in range(20)})
    vocabulary = _fitted_vocab(sequence)
    # Budget: mandatory framing + empty payload (2 tokens), nothing more.
    minimal_sequence = _sequence(payload={})
    vocabulary = _fitted_vocab(sequence, minimal_sequence)
    max_length = len(minimal_sequence.tokens)
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    result = tensorizer.tensorize(sequence)

    assert len(result.input_ids) == max_length
    shrunk = tensorizer._truncate_payload(sequence)
    payload_start = shrunk.tokens.index("[API:PAYLOAD_START]")
    payload_end = shrunk.tokens.index("[API:PAYLOAD_END]")
    assert shrunk.tokens[payload_start + 1 : payload_end] == [API_OBJ_START, API_OBJ_END]


# -- Verified-rule preservation (the core Phase 8 requirement) -------------


def test_tensorize_preserves_the_entire_rule_segment_when_payload_is_oversized() -> None:
    rules = [VerifiedRule(rule_id="R-1", attributes={"severity": "high", "field": "amount"})]
    sequence = _sequence(payload={f"field_{i}": f"value_{i}" for i in range(50)}, rules=rules)
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) - 100  # forces heavy payload truncation
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    result = tensorizer.tensorize(sequence)
    decoded = _decoded_non_pad(tensorizer, result)

    assert "R-1" in decoded
    assert "high" in decoded
    assert "amount" in decoded
    # The payload was in fact cut (not all 50 fields survived).
    assert sum(1 for i in range(50) if f"field_{i}" in decoded) < 50


def test_tensorize_preserves_multiple_rules_with_no_rule_silently_discarded() -> None:
    rules = [
        VerifiedRule(rule_id=f"R-{i}", attributes={"note": f"rule-{i}"}) for i in range(5)
    ]
    sequence = _sequence(payload={f"field_{i}": f"value_{i}" for i in range(50)}, rules=rules)
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) - 150
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    result = tensorizer.tensorize(sequence)
    decoded = _decoded_non_pad(tensorizer, result)

    for i in range(5):
        assert f"R-{i}" in decoded, f"rule R-{i} was discarded"
        assert f"rule-{i}" in decoded, f"attributes of rule R-{i} were discarded"


def test_tensorize_raises_when_mandatory_framing_and_rules_cannot_fit() -> None:
    huge_rule_attrs = {f"k{i}": f"v{i}" for i in range(50)}
    sequence = _sequence(payload={}, rules=[VerifiedRule(rule_id="R-1", attributes=huge_rule_attrs)])
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) - 1  # payload is already minimal; nothing can be dropped
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    with pytest.raises(ValueError):
        tensorizer.tensorize(sequence)


# -- Adversarial inputs: literals that look like structural tags -----------


def test_tensorize_truncation_is_not_confused_by_a_payload_key_matching_a_boundary_tag() -> None:
    payload = {API_OBJ_START: "x", "[API:PAYLOAD_END]": "y"}
    payload.update({f"field_{i}": f"value_{i}" for i in range(20)})
    rules = [VerifiedRule(rule_id="R-1", attributes={"field": "amount"})]
    sequence = _sequence(payload=payload, rules=rules)
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) - 20
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    result = tensorizer.tensorize(sequence)
    decoded = _decoded_non_pad(tensorizer, result)

    assert "R-1" in decoded  # rule segment still correctly located and preserved
    shrunk = tensorizer._truncate_payload(sequence)
    _assert_balanced(shrunk.tokens, shrunk.roles)


def test_tensorize_truncation_is_not_confused_by_a_rule_attribute_matching_rules_end() -> None:
    sequence = _sequence(
        payload={f"field_{i}": f"value_{i}" for i in range(20)},
        rules=[VerifiedRule(rule_id="R-1", attributes={"note": "[RULES_END]"})],
    )
    vocabulary = _fitted_vocab(sequence)
    max_length = len(sequence.tokens) - 20
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    result = tensorizer.tensorize(sequence)
    decoded = _decoded_non_pad(tensorizer, result)

    assert "R-1" in decoded
    assert "[RULES_END]" in decoded  # the lookalike attribute value itself survives, as data


def test_control_index_is_not_fooled_by_a_literal_with_matching_text() -> None:
    from tokenization_ai.tensorizer import Tensorizer as T

    sequence = _sequence(payload={"comment": "[API:PAYLOAD_START]"})
    index = T._control_index(sequence.tokens, sequence.roles, "[API:PAYLOAD_START]")

    assert sequence.roles[index] == TOKEN_ROLE_CONTROL
    # The genuine control token, not the look-alike literal value.
    comment_value_index = sequence.tokens.index("comment") + 2
    assert index != comment_value_index
