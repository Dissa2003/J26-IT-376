"""Unit tests for hybrid structure-aware tokenization."""

from tokenization_ai.models import FusedInput, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.tokenizer import (
    API_HEADER_NAME,
    API_HEADER_VALUE,
    API_HEADER_VALUE_MASKED,
    API_KEY,
    API_METHOD,
    API_OBJ_END,
    API_OBJ_START,
    API_PATH_SEG,
    API_PAYLOAD_START,
    API_VAL_BOOL,
    API_VAL_FLOAT,
    API_VAL_INT,
    API_VAL_MASKED_STR,
    API_VAL_NULL,
    API_VAL_STR,
    HybridTokenizer,
    RULE_ID,
    RULE_KEY,
    RULE_VAL_STR,
    RULES_END,
    RULES_START,
)


def _fused(rules: list[VerifiedRule] | None = None, **context_overrides: object) -> FusedInput:
    fields: dict[str, object] = {"event_id": "e1", "method": "POST", "path": "/orders"}
    fields.update(context_overrides)
    context = UnifiedApiContext(**fields)
    rule_info = VerifiedRuleInfo(event_id="e1", rules=rules or [])
    return FusedInput(context=context, rule_info=rule_info)


def _pair_after_named_key(tokens: list[str], key_tag: str, key: str) -> tuple[str, str]:
    """Return the (type-tag, value) pair following the first ``key_tag``/``key`` match."""
    for index, token in enumerate(tokens):
        if token == key_tag and tokens[index + 1] == key:
            return tokens[index + 2], tokens[index + 3]
    raise AssertionError(f"key {key!r} not found under tag {key_tag!r}")


def _pair_after_key(tokens: list[str], key: str) -> tuple[str, str]:
    return _pair_after_named_key(tokens, API_KEY, key)


def _tag_after_key(tokens: list[str], key: str) -> str:
    """Return the type-tag following the first ``API_KEY``/``key`` match (no value token)."""
    for index, token in enumerate(tokens):
        if token == API_KEY and tokens[index + 1] == key:
            return tokens[index + 2]
    raise AssertionError(f"key {key!r} not found")


def test_tokenize_includes_method_and_ordered_path_segments() -> None:
    fused = _fused(method="POST", path="/orders/items")

    result = HybridTokenizer().tokenize(fused)

    assert result.event_id == "e1"
    method_index = result.tokens.index(API_METHOD)
    assert result.tokens[method_index + 1] == "POST"
    seg_indices = [i for i, token in enumerate(result.tokens) if token == API_PATH_SEG]
    assert [result.tokens[i + 1] for i in seg_indices] == ["orders", "items"]


def test_tokenize_includes_header_names_and_values_sorted_by_name() -> None:
    fused = _fused(headers={"x-request-id": "abc", "accept": "*/*"})

    tokens = HybridTokenizer().tokenize(fused).tokens

    name_indices = [i for i, token in enumerate(tokens) if token == API_HEADER_NAME]
    names = [tokens[i + 1] for i in name_indices]
    assert names == ["accept", "x-request-id"]
    value_indices = [i for i, token in enumerate(tokens) if token == API_HEADER_VALUE]
    values = [tokens[i + 1] for i in value_indices]
    assert values == ["*/*", "abc"]


def test_tokenize_preserves_nested_object_structure() -> None:
    fused = _fused(payload={"customer": {"name": "Jane"}})

    tokens = HybridTokenizer().tokenize(fused).tokens

    assert tokens.count(API_OBJ_START) == 2  # payload root + nested "customer" object
    assert tokens.count(API_OBJ_END) == 2
    assert _pair_after_named_key(tokens, API_KEY, "name") == (API_VAL_STR, "Jane")


def test_tokenize_preserves_list_order() -> None:
    fused_a = _fused(payload={"tags": ["vip", "returning"]})
    fused_b = _fused(payload={"tags": ["returning", "vip"]})

    tokens_a = HybridTokenizer().tokenize(fused_a).tokens
    tokens_b = HybridTokenizer().tokenize(fused_b).tokens

    assert tokens_a != tokens_b
    str_value_indices = [i for i, token in enumerate(tokens_a) if token == API_VAL_STR]
    assert [tokens_a[i + 1] for i in str_value_indices] == ["vip", "returning"]


def test_tokenize_tags_primitive_types_distinctly() -> None:
    fused = _fused(
        payload={
            "amount": 5,
            "ratio": 5.5,
            "active": True,
            "deleted": False,
            "middle_name": None,
            "note": "5",
        }
    )

    tokens = HybridTokenizer().tokenize(fused).tokens

    assert _pair_after_key(tokens, "amount") == (API_VAL_INT, "5")
    assert _pair_after_key(tokens, "ratio") == (API_VAL_FLOAT, "5.5")
    assert _pair_after_key(tokens, "active") == (API_VAL_BOOL, "true")
    assert _pair_after_key(tokens, "deleted") == (API_VAL_BOOL, "false")
    assert _pair_after_key(tokens, "note") == (API_VAL_STR, "5")
    assert _tag_after_key(tokens, "middle_name") == API_VAL_NULL


def test_tokenize_keeps_masking_placeholders_as_single_atomic_values() -> None:
    fused = _fused(
        payload={"email": "[MASKED_PII]", "token": "[MASKED_SECRET]"},
        headers={"authorization": "[MASKED_AUTH]"},
    )

    tokens = HybridTokenizer().tokenize(fused).tokens

    assert _pair_after_key(tokens, "email") == (API_VAL_STR, "[MASKED_PII]")
    assert _pair_after_key(tokens, "token") == (API_VAL_STR, "[MASKED_SECRET]")
    header_value_index = tokens.index(API_HEADER_VALUE)
    assert tokens[header_value_index + 1] == "[MASKED_AUTH]"


def test_tokenize_includes_verified_rule_identifiers_and_attributes() -> None:
    fused = _fused(
        rules=[VerifiedRule(rule_id="R-42", attributes={"field": "amount", "threshold": 100})]
    )

    tokens = HybridTokenizer().tokenize(fused).tokens

    rule_id_index = tokens.index(RULE_ID)
    assert tokens[rule_id_index + 1] == "R-42"
    assert _pair_after_named_key(tokens, RULE_KEY, "field") == (RULE_VAL_STR, "amount")


def test_tokenize_distinguishes_rule_tokens_from_api_tokens_for_the_same_field_name() -> None:
    fused = _fused(
        payload={"field": "payload-value"},
        rules=[VerifiedRule(rule_id="R-1", attributes={"field": "rule-value"})],
    )

    tokens = HybridTokenizer().tokenize(fused).tokens

    assert _pair_after_named_key(tokens, API_KEY, "field") == (API_VAL_STR, "payload-value")
    assert _pair_after_named_key(tokens, RULE_KEY, "field") == (RULE_VAL_STR, "rule-value")


def test_tokenize_is_deterministic_for_the_same_input() -> None:
    fused = _fused(
        payload={"b": 2, "a": {"z": 1, "y": [3, 2, 1]}},
        rules=[VerifiedRule(rule_id="R-1", attributes={"y": 1, "x": 2})],
    )

    first = HybridTokenizer().tokenize(fused)
    second = HybridTokenizer().tokenize(fused)

    assert first.tokens == second.tokens


def test_tokenize_is_unaffected_by_payload_key_order() -> None:
    fused_a = _fused(payload={"amount": 5, "currency": "USD"})
    fused_b = _fused(payload={"currency": "USD", "amount": 5})

    tokens_a = HybridTokenizer().tokenize(fused_a).tokens
    tokens_b = HybridTokenizer().tokenize(fused_b).tokens

    assert tokens_a == tokens_b


def test_tokenize_is_unaffected_by_header_key_order() -> None:
    fused_a = _fused(headers={"accept": "*/*", "x-request-id": "abc"})
    fused_b = _fused(headers={"x-request-id": "abc", "accept": "*/*"})

    tokens_a = HybridTokenizer().tokenize(fused_a).tokens
    tokens_b = HybridTokenizer().tokenize(fused_b).tokens

    assert tokens_a == tokens_b


def test_tokenize_produces_different_sequences_for_structurally_different_payloads() -> None:
    fused_a = _fused(payload={"tags": ["vip"]})
    fused_b = _fused(payload={"tags": "vip"})

    tokens_a = HybridTokenizer().tokenize(fused_a).tokens
    tokens_b = HybridTokenizer().tokenize(fused_b).tokens

    assert tokens_a != tokens_b


def test_tokenize_distinguishes_the_same_value_at_different_nesting_levels() -> None:
    fused_a = _fused(payload={"a": {"x": 1}, "b": 1})
    fused_b = _fused(payload={"a": 1, "b": {"x": 1}})

    tokens_a = HybridTokenizer().tokenize(fused_a).tokens
    tokens_b = HybridTokenizer().tokenize(fused_b).tokens

    assert tokens_a != tokens_b


def test_tokenize_does_not_mutate_the_original_fused_input() -> None:
    fused = _fused(
        payload={"amount": 5, "nested": {"qty": 2}, "tags": ["a", "b"]},
        headers={"accept": "*/*"},
        rules=[VerifiedRule(rule_id="R-1", attributes={"field": "amount"})],
    )
    before = fused.model_dump()

    HybridTokenizer().tokenize(fused)

    assert fused.model_dump() == before


def test_tokenize_handles_empty_payload_and_no_rules(fused_input: FusedInput) -> None:
    result = HybridTokenizer().tokenize(fused_input)

    assert result.event_id == "e1"
    assert API_PAYLOAD_START in result.tokens
    assert API_OBJ_START in result.tokens and API_OBJ_END in result.tokens
    assert RULES_START in result.tokens and RULES_END in result.tokens


# -- masked_paths provenance: trusted masking tags (security) ------------


def test_tokenize_without_masked_paths_is_unaffected_backward_compatible() -> None:
    fused = _fused(payload={"email": "[MASKED_PII]"})

    tokens = HybridTokenizer().tokenize(fused).tokens

    value_index = tokens.index("email") + 2
    assert tokens[value_index - 1] == API_VAL_STR
    assert tokens[value_index] == "[MASKED_PII]"


def test_tokenize_tags_a_genuinely_masked_payload_value_as_trusted() -> None:
    fused = _fused(payload={"email": "[MASKED_PII]"})

    tokens = HybridTokenizer().tokenize(fused, masked_paths=frozenset({("payload", "email")})).tokens

    value_index = tokens.index("email") + 2
    assert tokens[value_index - 1] == API_VAL_MASKED_STR
    assert tokens[value_index] == "[MASKED_PII]"


def test_tokenize_leaves_an_attacker_lookalike_payload_value_as_ordinary() -> None:
    fused = _fused(payload={"nickname": "[MASKED_PII]"})

    # No path reported as masked -- this value was never actually masked.
    tokens = HybridTokenizer().tokenize(fused, masked_paths=frozenset()).tokens

    value_index = tokens.index("nickname") + 2
    assert tokens[value_index - 1] == API_VAL_STR
    assert tokens[value_index] == "[MASKED_PII]"


def test_tokenize_tags_a_genuinely_masked_header_value_as_trusted() -> None:
    fused = _fused(headers={"authorization": "[MASKED_AUTH]"})

    tokens = HybridTokenizer().tokenize(
        fused, masked_paths=frozenset({("headers", "authorization")})
    ).tokens

    header_value_index = tokens.index(API_HEADER_VALUE_MASKED) + 1
    assert tokens[header_value_index] == "[MASKED_AUTH]"
    assert API_HEADER_VALUE not in tokens


def test_tokenize_leaves_an_attacker_lookalike_header_value_as_ordinary() -> None:
    fused = _fused(headers={"x-note": "[MASKED_AUTH]"})

    tokens = HybridTokenizer().tokenize(fused, masked_paths=frozenset()).tokens

    assert API_HEADER_VALUE_MASKED not in tokens
    header_value_index = tokens.index(API_HEADER_VALUE) + 1
    assert tokens[header_value_index] == "[MASKED_AUTH]"


def test_tokenize_tags_masked_values_correctly_inside_nested_objects_and_lists() -> None:
    fused = _fused(
        payload={
            "customer": {"email": "[MASKED_PII]"},
            "cards": [{"card_number": "[MASKED_PII]"}, {"card_number": "not-masked"}],
        }
    )
    masked_paths = frozenset(
        {
            ("payload", "customer", "email"),
            ("payload", "cards", 0, "card_number"),
        }
    )

    tokens = HybridTokenizer().tokenize(fused, masked_paths=masked_paths).tokens

    email_value_index = tokens.index("email") + 2
    assert tokens[email_value_index - 1] == API_VAL_MASKED_STR

    card_number_indices = [i for i, t in enumerate(tokens) if t == "card_number"]
    first_card_value_index = card_number_indices[0] + 2
    second_card_value_index = card_number_indices[1] + 2
    assert tokens[first_card_value_index - 1] == API_VAL_MASKED_STR
    assert tokens[second_card_value_index - 1] == API_VAL_STR
