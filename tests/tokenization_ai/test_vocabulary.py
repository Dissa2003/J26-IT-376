"""Unit tests for the token-to-id vocabulary."""

import pytest

from tokenization_ai.masking import MASKED_AUTH, MASKED_IP, MASKED_PII, MASKED_SECRET, PrivacyMasker
from tokenization_ai.models import FusedInput, TokenSequence, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.normalization import Normalizer
from tokenization_ai.tokenizer import (
    API_HEADER_VALUE,
    API_HEADER_VALUE_MASKED,
    API_KEY,
    API_OBJ_END,
    API_OBJ_START,
    API_PAYLOAD_END,
    API_PAYLOAD_START,
    API_VAL_INT,
    API_VAL_MASKED_STR,
    API_VAL_STR,
    HybridTokenizer,
    RULE_ID,
    RULE_KEY,
    RULE_VAL_STR,
    SEQ_END,
    SEQ_START,
    TOKEN_ROLE_CONTROL,
    TOKEN_ROLE_LITERAL_API,
    TOKEN_ROLE_LITERAL_RULE,
    TOKEN_ROLE_MASKED,
)
from tokenization_ai.vocabulary import PAD_TOKEN, UNK_TOKEN, Vocabulary


def _fused(rules: list[VerifiedRule] | None = None, **context_overrides: object) -> FusedInput:
    fields: dict[str, object] = {"event_id": "e1", "method": "POST", "path": "/orders"}
    fields.update(context_overrides)
    context = UnifiedApiContext(**fields)
    rule_info = VerifiedRuleInfo(event_id="e1", rules=rules or [])
    return FusedInput(context=context, rule_info=rule_info)


def _sequence(**kwargs: object) -> TokenSequence:
    return HybridTokenizer().tokenize(_fused(**kwargs))


def _tokens(**kwargs: object) -> list[str]:
    return _sequence(**kwargs).tokens


def _pipeline_sequence(**kwargs: object) -> TokenSequence:
    """Run the full mask -> normalize -> tokenize pipeline with real provenance.

    Used for the masking-provenance security tests below, where what
    matters is the *trusted* path end-to-end (mask_with_provenance ->
    normalize_with_provenance -> tokenize(..., masked_paths=...)), not just
    the tokenizer in isolation.
    """
    mask_result = PrivacyMasker().mask_with_provenance(_fused(**kwargs))
    norm_result = Normalizer().normalize_with_provenance(mask_result.fused, mask_result.masked_paths)
    return HybridTokenizer().tokenize(norm_result.fused, norm_result.masked_paths)


def _minimal_sequence(key: str, value: str) -> TokenSequence:
    """A minimal, fully-controlled, well-formed API-namespace sequence.

    Used where a test needs total control over exactly which literals
    exist, free of the unrelated literals (method, path, ...) a real
    ``_fused``/``_sequence`` call would also produce.
    """
    return TokenSequence(
        event_id="e1",
        tokens=[SEQ_START, API_KEY, key, API_VAL_STR, value, SEQ_END],
        roles=[
            TOKEN_ROLE_CONTROL,
            TOKEN_ROLE_CONTROL,
            TOKEN_ROLE_LITERAL_API,
            TOKEN_ROLE_CONTROL,
            TOKEN_ROLE_LITERAL_API,
            TOKEN_ROLE_CONTROL,
        ],
    )


def _value_index_after_key(tokens: list[str], key_tag: str, key: str) -> int:
    """Index of the value token immediately following the first ``key_tag``/``key`` match."""
    for index, token in enumerate(tokens):
        if token == key_tag and tokens[index + 1] == key:
            return index + 3  # key_tag, key, type-tag, value
    raise AssertionError(f"key {key!r} not found under tag {key_tag!r}")


# -- Basic reserved/fit/encode/decode behavior ----------------------------


def test_reserved_tokens_get_fixed_ids_before_fit() -> None:
    vocabulary = Vocabulary()

    assert vocabulary.pad_id == 0
    assert vocabulary.unk_id == 1
    assert vocabulary.decode([vocabulary.pad_id]) == [PAD_TOKEN]
    assert vocabulary.decode([vocabulary.unk_id]) == [UNK_TOKEN]
    assert len({vocabulary._token_to_id[(None, API_KEY)], vocabulary._token_to_id[(None, RULE_KEY)]}) == 2


def test_encode_before_fit_raises() -> None:
    vocabulary = Vocabulary()

    with pytest.raises(RuntimeError):
        vocabulary.encode(_sequence())


def test_fit_assigns_stable_non_unk_ids_for_seen_literal_values() -> None:
    sequence = _sequence(payload={"amount": 5})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)

    assert vocabulary.unk_id not in ids
    assert vocabulary.decode(ids) == sequence.tokens


def test_fit_is_deterministic_regardless_of_corpus_presentation_order() -> None:
    seq_a = _sequence(payload={"alpha": "x"})
    seq_b = _sequence(payload={"beta": "y"})
    seq_c = _sequence(payload={"gamma": "z"})

    vocab_forward = Vocabulary()
    vocab_forward.fit([seq_a, seq_b, seq_c])

    vocab_reversed = Vocabulary()
    vocab_reversed.fit([seq_c, seq_b, seq_a])

    assert vocab_forward._token_to_id == vocab_reversed._token_to_id


def test_encode_unseen_literal_value_maps_to_unk_id() -> None:
    vocabulary = Vocabulary()
    vocabulary.fit([_sequence(payload={"amount": 5})])

    sequence = _sequence(payload={"amount": 999})
    result = vocabulary.encode(sequence)

    value_index = _value_index_after_key(sequence.tokens, API_KEY, "amount")
    assert result[value_index] == vocabulary.unk_id


def test_namespace_separation_gives_distinct_ids_for_the_same_literal() -> None:
    sequence = _sequence(
        payload={"field": "payload-value"},
        rules=[VerifiedRule(rule_id="R-1", attributes={"field": "rule-value"})],
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    tokens = sequence.tokens
    api_key_index = tokens.index(API_KEY)
    rule_key_index = tokens.index(RULE_KEY)
    assert tokens[api_key_index + 1] == "field"
    assert tokens[rule_key_index + 1] == "field"

    ids = vocabulary.encode(sequence)
    api_field_id = ids[api_key_index + 1]
    rule_field_id = ids[rule_key_index + 1]

    assert api_field_id != rule_field_id
    assert vocabulary.unk_id not in (api_field_id, rule_field_id)


def test_decode_round_trips_encode_for_a_realistic_sequence() -> None:
    sequence = _sequence(
        payload={"amount": 5, "customer": {"name": "Jane", "tags": ["vip", "returning"]}},
        headers={"accept": "*/*"},
        rules=[VerifiedRule(rule_id="R-1", attributes={"field": "amount"})],
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    assert vocabulary.decode(vocabulary.encode(sequence)) == sequence.tokens


def test_decode_unknown_id_raises_key_error() -> None:
    vocabulary = Vocabulary()
    vocabulary.fit([_sequence()])

    with pytest.raises(KeyError):
        vocabulary.decode([len(vocabulary) + 1000])


def test_min_frequency_filters_rare_literals_to_unk() -> None:
    frequent = _sequence(payload={"amount": "common"})
    rare = _sequence(payload={"amount": "rare-value"})
    vocabulary = Vocabulary(min_frequency=2)

    vocabulary.fit([frequent, frequent, rare])

    frequent_ids = vocabulary.encode(frequent)
    rare_ids = vocabulary.encode(rare)
    rare_value_index = _value_index_after_key(rare.tokens, API_KEY, "amount")
    assert vocabulary.unk_id not in frequent_ids
    assert rare_ids[rare_value_index] == vocabulary.unk_id


def test_max_literal_vocab_size_caps_growth_keeping_most_frequent() -> None:
    common = _minimal_sequence("field", "common-value")
    uncommon = _minimal_sequence("field", "rare-value")
    # Candidates: "field" (freq 4), "common-value" (freq 3), "rare-value" (freq 1).
    vocabulary = Vocabulary(max_literal_vocab_size=2)

    vocabulary.fit([common, common, common, uncommon])

    common_ids = vocabulary.encode(common)
    uncommon_ids = vocabulary.encode(uncommon)
    assert vocabulary.unk_id not in common_ids
    assert uncommon_ids[4] == vocabulary.unk_id  # position of "rare-value"


def test_max_literal_vocab_size_applies_independently_per_namespace() -> None:
    sequence = _sequence(
        payload={"a": "x"},
        rules=[VerifiedRule(rule_id="R-1", attributes={"b": "y"})],
    )
    vocabulary = Vocabulary(max_literal_vocab_size=1)

    vocabulary.fit([sequence])

    # Each namespace gets its own slot budget: one API literal and one rule
    # literal should survive, not a single literal shared across both.
    api_literal_count = sum(1 for key in vocabulary._token_to_id if key[0] == "api")
    rule_literal_count = sum(1 for key in vocabulary._token_to_id if key[0] == "rule")
    assert api_literal_count == 1
    assert rule_literal_count == 1


def test_refitting_discards_the_previous_vocabulary() -> None:
    first = _sequence(payload={"amount": "seen-only-in-first-fit"})
    second = _sequence(payload={"amount": "different-value"})
    vocabulary = Vocabulary()

    vocabulary.fit([first])
    vocabulary.fit([second])

    first_value_index = _value_index_after_key(first.tokens, API_KEY, "amount")
    second_value_index = _value_index_after_key(second.tokens, API_KEY, "amount")
    assert vocabulary.encode(first)[first_value_index] == vocabulary.unk_id
    assert vocabulary.encode(second)[second_value_index] != vocabulary.unk_id


def test_control_tokens_are_never_unk_even_with_an_empty_fitting_corpus() -> None:
    vocabulary = Vocabulary()
    vocabulary.fit([])

    sequence = TokenSequence(
        event_id="e1",
        tokens=[SEQ_START, API_PAYLOAD_START, API_OBJ_START, API_OBJ_END, API_PAYLOAD_END, SEQ_END],
        roles=[TOKEN_ROLE_CONTROL] * 6,
    )
    ids = vocabulary.encode(sequence)

    assert vocabulary.unk_id not in ids


def test_masking_placeholders_are_reserved_and_never_unk() -> None:
    vocabulary = Vocabulary()
    vocabulary.fit([])  # no training data mentions the placeholder at all

    sequence = TokenSequence(event_id="e1", tokens=[MASKED_PII], roles=[TOKEN_ROLE_MASKED])
    ids = vocabulary.encode(sequence)

    assert ids == [vocabulary._token_to_id[(None, MASKED_PII)]]
    assert vocabulary.unk_id not in ids


def test_fit_raises_when_tokens_and_roles_differ_in_length() -> None:
    vocabulary = Vocabulary()
    bad = TokenSequence(event_id="e1", tokens=[SEQ_START, SEQ_END], roles=[TOKEN_ROLE_CONTROL])

    with pytest.raises(ValueError):
        vocabulary.fit([bad])


def test_fit_raises_on_an_unrecognized_role() -> None:
    vocabulary = Vocabulary()
    bad = TokenSequence(event_id="e1", tokens=["x"], roles=["not-a-real-role"])

    with pytest.raises(ValueError):
        vocabulary.fit([bad])


def test_fit_raises_when_a_control_labeled_token_is_not_actually_reserved() -> None:
    vocabulary = Vocabulary()
    bad = TokenSequence(event_id="e1", tokens=["not-a-real-tag"], roles=[TOKEN_ROLE_CONTROL])

    with pytest.raises(ValueError):
        vocabulary.fit([bad])


def test_encode_raises_when_tokens_and_roles_differ_in_length() -> None:
    vocabulary = Vocabulary()
    vocabulary.fit([_sequence()])
    bad = TokenSequence(event_id="e1", tokens=[SEQ_START, SEQ_END], roles=[TOKEN_ROLE_CONTROL])

    with pytest.raises(ValueError):
        vocabulary.encode(bad)


def test_encode_raises_on_an_unrecognized_role() -> None:
    vocabulary = Vocabulary()
    vocabulary.fit([_sequence()])
    bad = TokenSequence(event_id="e1", tokens=["x"], roles=["not-a-real-role"])

    with pytest.raises(ValueError):
        vocabulary.encode(bad)


def test_fit_raises_on_empty_roles_with_nonempty_tokens() -> None:
    vocabulary = Vocabulary()
    bad = TokenSequence(event_id="e1", tokens=[SEQ_START], roles=[])

    with pytest.raises(ValueError):
        vocabulary.fit([bad])


def test_encode_raises_on_empty_roles_with_nonempty_tokens() -> None:
    vocabulary = Vocabulary()
    vocabulary.fit([_sequence()])
    bad = TokenSequence(event_id="e1", tokens=[SEQ_START], roles=[])

    with pytest.raises(ValueError):
        vocabulary.encode(bad)


def test_encode_still_maps_an_unseen_literal_with_a_valid_role_to_unk_not_an_error() -> None:
    # An invalid *role* and an unseen *literal value* must remain distinct
    # failure modes: the former is a malformed-input error, the latter is
    # an ordinary, expected runtime event.
    vocabulary = Vocabulary()
    vocabulary.fit([_sequence(payload={"amount": "seen-value"})])
    unseen = TokenSequence(event_id="e1", tokens=["never-seen-before"], roles=[TOKEN_ROLE_LITERAL_API])

    ids = vocabulary.encode(unseen)

    assert ids == [vocabulary.unk_id]


def test_fit_accepts_an_empty_corpus() -> None:
    vocabulary = Vocabulary()

    vocabulary.fit([])

    assert vocabulary.is_fitted
    sequence = TokenSequence(event_id="e1", tokens=[SEQ_START], roles=[TOKEN_ROLE_CONTROL])
    assert vocabulary.encode(sequence)[0] != vocabulary.unk_id


def test_vocabulary_does_not_mutate_the_tokens_passed_in() -> None:
    sequence = _sequence(payload={"amount": 5})
    before_tokens = list(sequence.tokens)
    before_roles = list(sequence.roles)
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    vocabulary.encode(sequence)

    assert sequence.tokens == before_tokens
    assert sequence.roles == before_roles


# -- Security regression tests: Variant 1 (structural-tag collision),    --
# -- value position -- the original, narrower scenario --------------------


def test_structural_tag_lookalike_used_as_an_api_value_is_not_the_control_tags_id() -> None:
    sequence = _sequence(payload={"comment": RULE_ID, "other": "x"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    comment_value_index = _value_index_after_key(sequence.tokens, API_KEY, "comment")
    real_rule_id_tag_id = vocabulary._token_to_id[(None, RULE_ID)]

    assert ids[comment_value_index] != real_rule_id_tag_id
    assert vocabulary.decode([ids[comment_value_index]]) == [RULE_ID]


def test_structural_tag_lookalike_used_as_a_rule_value_is_not_the_control_tags_id() -> None:
    sequence = _sequence(rules=[VerifiedRule(rule_id="R-1", attributes={"note": API_KEY})])
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    note_value_index = _value_index_after_key(sequence.tokens, RULE_KEY, "note")
    real_api_key_tag_id = vocabulary._token_to_id[(None, API_KEY)]

    assert ids[note_value_index] != real_api_key_tag_id
    assert vocabulary.decode([ids[note_value_index]]) == [API_KEY]


def test_namespace_separation_holds_for_a_structural_tag_lookalike_literal() -> None:
    sequence = _sequence(
        payload={"a": RULE_VAL_STR},
        rules=[VerifiedRule(rule_id="R-1", attributes={"b": RULE_VAL_STR})],
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    api_value_index = _value_index_after_key(sequence.tokens, API_KEY, "a")
    rule_value_index = _value_index_after_key(sequence.tokens, RULE_KEY, "b")

    assert ids[api_value_index] != ids[rule_value_index]
    assert vocabulary.unk_id not in (ids[api_value_index], ids[rule_value_index])


def test_encoding_a_structural_tag_lookalike_is_deterministic() -> None:
    sequence = _sequence(payload={"comment": RULE_ID})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    assert vocabulary.encode(sequence) == vocabulary.encode(sequence)


def test_unseen_structural_tag_lookalike_value_falls_back_to_unk_like_any_other_literal() -> None:
    sequence = _sequence(payload={"comment": RULE_ID})
    vocabulary = Vocabulary()
    vocabulary.fit([_sequence(payload={"comment": "unrelated-value"})])

    ids = vocabulary.encode(sequence)
    comment_value_index = _value_index_after_key(sequence.tokens, API_KEY, "comment")

    assert ids[comment_value_index] == vocabulary.unk_id


def test_fit_also_treats_a_structural_tag_lookalike_value_as_an_ordinary_literal() -> None:
    sequence = _sequence(payload={"comment": RULE_ID})
    vocabulary = Vocabulary()

    vocabulary.fit([sequence, sequence])

    comment_value_index = _value_index_after_key(sequence.tokens, API_KEY, "comment")
    ids = vocabulary.encode(sequence)
    real_rule_id_tag_id = vocabulary._token_to_id[(None, RULE_ID)]
    assert ids[comment_value_index] != real_rule_id_tag_id
    assert ("api", RULE_ID) in vocabulary._token_to_id


# -- Security regression tests: the newly-fixed grammar-state collision  --
# -- (adversarial KEYS, not just values) -----------------------------------
#
# The previous (position-inferring) implementation decided a token's role
# from its *preceding* token's text. An attacker-controlled payload key
# (JSON object keys have no character restrictions) set to a tag string
# such as "[API:VAL_STR]" caused the *genuine* control tag immediately
# following it -- marking that same field's own value type -- to be
# misclassified as an ordinary literal, corrupting a legitimate tag's
# identity rather than just an attacker value's. Reading the role directly
# from TokenSequence.roles (authored by the tokenizer itself, never
# inferred) closes this: these tests construct exactly that adversarial
# input through the real tokenizer and verify every genuine control tag
# still gets its one, stable, reserved id.


def test_payload_key_equal_to_a_structural_tag_does_not_corrupt_the_following_genuine_tag() -> None:
    sequence = _sequence(payload={API_VAL_STR: 5, "normal_amount": 7})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    ids = vocabulary.encode(sequence)

    tokens = sequence.tokens
    crafted_key_index = tokens.index(API_VAL_STR)
    # The token right after the crafted key is the GENUINE API_VAL_INT tag
    # for that field's own value (5 is an int) -- it must keep its one
    # reserved id regardless of what the preceding key's text says.
    genuine_tag_index = crafted_key_index + 1
    assert tokens[genuine_tag_index] == API_VAL_INT
    assert sequence.roles[genuine_tag_index] == TOKEN_ROLE_CONTROL

    reserved_val_int_id = vocabulary._token_to_id[(None, API_VAL_INT)]
    assert ids[genuine_tag_index] == reserved_val_int_id

    # And the crafted key itself is still an ordinary API literal, not the
    # real API_VAL_STR control tag's id.
    real_val_str_id = vocabulary._token_to_id[(None, API_VAL_STR)]
    assert ids[crafted_key_index] != real_val_str_id


def test_payload_value_equal_to_a_structural_tag_does_not_corrupt_the_following_genuine_tag() -> None:
    # A value can be followed by another genuine tag too (e.g. the next
    # sibling key, or a closing boundary) -- verify that chain holds.
    sequence = _sequence(payload={"a": API_KEY, "b": 1})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    ids = vocabulary.encode(sequence)

    tokens = sequence.tokens
    crafted_value_index = tokens.index(API_KEY, tokens.index("a") + 1)
    genuine_next_tag_index = crafted_value_index + 1
    assert sequence.roles[genuine_next_tag_index] == TOKEN_ROLE_CONTROL

    genuine_next_tag = tokens[genuine_next_tag_index]
    reserved_id = vocabulary._token_to_id[(None, genuine_next_tag)]
    assert ids[genuine_next_tag_index] == reserved_id


def test_nested_objects_and_lists_with_adversarial_keys_do_not_corrupt_genuine_tags() -> None:
    sequence = _sequence(
        payload={
            "customer": {API_OBJ_START: "x", RULE_ID: [API_VAL_STR, 1, API_KEY]},
        }
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    ids = vocabulary.encode(sequence)

    tokens, roles = sequence.tokens, sequence.roles
    for index, (token, role) in enumerate(zip(tokens, roles)):
        if role == TOKEN_ROLE_CONTROL:
            assert (None, token) in vocabulary._token_to_id, (
                f"control token {token!r} at {index} should be pre-registered"
            )
            assert ids[index] == vocabulary._token_to_id[(None, token)], (
                f"control token {token!r} at {index} did not get its reserved id"
            )


def test_genuine_structural_tags_still_keep_their_reserved_ids_alongside_a_lookalike_value() -> None:
    sequence = _sequence(payload={"comment": RULE_ID})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens
    rule_id_text_positions = [i for i, token in enumerate(tokens) if token == RULE_ID]

    # The only occurrence of the literal text "[RULE:ID]" in this
    # payload-only sequence is the lookalike value, confirmed by its role.
    assert len(rule_id_text_positions) == 1
    assert sequence.roles[rule_id_text_positions[0]] == TOKEN_ROLE_LITERAL_API

    real_rule_id_tag_id = vocabulary._token_to_id[(None, RULE_ID)]
    assert vocabulary.decode([real_rule_id_tag_id]) == [RULE_ID]
    comment_value_index = _value_index_after_key(tokens, API_KEY, "comment")
    assert ids[comment_value_index] != real_rule_id_tag_id


def test_every_genuine_control_tag_keeps_its_reserved_id_in_a_fully_adversarial_sequence() -> None:
    """Requirement: every genuine structural control tag maps to its one
    designated reserved id, regardless of surrounding attacker-controlled
    literals -- exercised with every literal slot set to a tag-lookalike."""
    sequence = _sequence(
        method=API_VAL_STR,
        headers={API_KEY: API_HEADER_VALUE},
        payload={
            RULE_KEY: API_VAL_INT,
            "nested": {SEQ_START: [RULE_VAL_STR, API_OBJ_END]},
        },
        rules=[VerifiedRule(rule_id=API_HEADER_VALUE_MASKED, attributes={RULE_ID: API_KEY})],
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    ids = vocabulary.encode(sequence)

    assert vocabulary.decode(ids) == sequence.tokens  # text always round-trips

    for index, (token, role) in enumerate(zip(sequence.tokens, sequence.roles)):
        if role == TOKEN_ROLE_CONTROL:
            assert ids[index] == vocabulary._token_to_id[(None, token)]
        elif role == TOKEN_ROLE_LITERAL_API:
            assert ids[index] == vocabulary._token_to_id.get(("api", token), vocabulary.unk_id)
        elif role == TOKEN_ROLE_LITERAL_RULE:
            assert ids[index] == vocabulary._token_to_id.get(("rule", token), vocabulary.unk_id)


def test_literal_values_never_impersonate_a_reserved_tags_id_across_many_lookalikes() -> None:
    lookalikes = [API_KEY, API_VAL_STR, RULE_ID, RULE_KEY, SEQ_START, API_OBJ_END]
    sequence = _sequence(payload={f"field_{i}": text for i, text in enumerate(lookalikes)})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    ids = vocabulary.encode(sequence)

    for i, text in enumerate(lookalikes):
        value_index = _value_index_after_key(sequence.tokens, API_KEY, f"field_{i}")
        reserved_id = vocabulary._token_to_id[(None, text)]
        assert ids[value_index] != reserved_id, f"literal {text!r} must not get the reserved tag id"


# -- Security regression tests: Variant 2 (masking-placeholder            --
# -- impersonation), fixed via trusted masking provenance -----------------


def test_genuine_masked_email_value_gets_the_reserved_placeholder_id() -> None:
    sequence = _pipeline_sequence(payload={"email": "customer@example.com"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens
    value_index = _value_index_after_key(tokens, API_KEY, "email")

    assert tokens[value_index - 1] == API_VAL_MASKED_STR
    assert sequence.roles[value_index] == TOKEN_ROLE_MASKED
    assert ids[value_index] == vocabulary._token_to_id[(None, MASKED_PII)]


def test_genuine_masked_authorization_header_gets_the_reserved_placeholder_id() -> None:
    sequence = _pipeline_sequence(headers={"Authorization": "Bearer secret-value"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens
    header_value_index = tokens.index(API_HEADER_VALUE_MASKED) + 1

    assert sequence.roles[header_value_index] == TOKEN_ROLE_MASKED
    assert ids[header_value_index] == vocabulary._token_to_id[(None, MASKED_AUTH)]


def test_genuine_masked_secret_field_gets_the_reserved_placeholder_id() -> None:
    sequence = _pipeline_sequence(payload={"password": "do-not-send-me"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens
    value_index = _value_index_after_key(tokens, API_KEY, "password")

    assert tokens[value_index - 1] == API_VAL_MASKED_STR
    assert ids[value_index] == vocabulary._token_to_id[(None, MASKED_SECRET)]


def test_genuine_masked_client_facing_ip_field_gets_the_reserved_placeholder_id() -> None:
    # No sensitive key name here -- "note" is classified as an IP address
    # purely by its value pattern, exercising _classify_value's MASKED_IP path
    # (as opposed to "client_ip"/"ip_address", which are masked by key name
    # to MASKED_PII instead).
    sequence = _pipeline_sequence(payload={"note": "203.0.113.5"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens
    value_index = _value_index_after_key(tokens, API_KEY, "note")

    assert tokens[value_index - 1] == API_VAL_MASKED_STR
    assert ids[value_index] == vocabulary._token_to_id[(None, MASKED_IP)]


def test_attacker_masking_lookalike_is_an_ordinary_literal_not_the_reserved_id() -> None:
    # "nickname" is never classified as sensitive, so this field is never
    # masked -- its text merely happens to read like a placeholder.
    sequence = _pipeline_sequence(payload={"nickname": "[MASKED_PII]"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens
    value_index = _value_index_after_key(tokens, API_KEY, "nickname")
    real_placeholder_id = vocabulary._token_to_id[(None, MASKED_PII)]

    assert tokens[value_index - 1] == API_VAL_STR
    assert sequence.roles[value_index] == TOKEN_ROLE_LITERAL_API
    assert ids[value_index] != real_placeholder_id
    assert vocabulary.decode([ids[value_index]]) == ["[MASKED_PII]"]


def test_nested_payload_objects_and_lists_get_correct_masked_classification() -> None:
    sequence = _pipeline_sequence(
        payload={
            "customer": {"profile": {"phone": "+1 415 555 0100"}},
            "cards": [{"card_number": "4111 1111 1111 1111"}, {"nickname_for_card": "[MASKED_PII]"}],
        }
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens

    phone_index = _value_index_after_key(tokens, API_KEY, "phone")
    assert tokens[phone_index - 1] == API_VAL_MASKED_STR
    assert ids[phone_index] == vocabulary._token_to_id[(None, MASKED_PII)]

    card_number_index = _value_index_after_key(tokens, API_KEY, "card_number")
    assert tokens[card_number_index - 1] == API_VAL_MASKED_STR

    lookalike_index = _value_index_after_key(tokens, API_KEY, "nickname_for_card")
    assert tokens[lookalike_index - 1] == API_VAL_STR
    assert ids[lookalike_index] != vocabulary._token_to_id[(None, MASKED_PII)]


def test_header_case_normalization_preserves_masked_classification() -> None:
    # Masking sees "Authorization" (mixed case); normalization lower-cases
    # it to "authorization" before tokenization -- provenance must survive
    # that remapping, or this header would wrongly fall back to ordinary.
    sequence = _pipeline_sequence(headers={"Authorization": "Bearer secret-value"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    tokens = sequence.tokens
    header_name_index = tokens.index("authorization")
    assert tokens[header_name_index + 1] == API_HEADER_VALUE_MASKED
    ids = vocabulary.encode(sequence)
    assert ids[header_name_index + 2] == vocabulary._token_to_id[(None, MASKED_AUTH)]


def test_structural_tag_collision_from_variant_1_is_still_fixed_alongside_provenance() -> None:
    # Regression guard: the Variant-1 fix (now role-based) and the
    # Variant-2 provenance fix must coexist without either weakening the
    # other, and so must the newly-fixed adversarial-key scenario.
    sequence = _pipeline_sequence(payload={"comment": RULE_ID, API_VAL_STR: 5, "email": "a@b.com"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens

    comment_index = _value_index_after_key(tokens, API_KEY, "comment")
    real_rule_id_tag_id = vocabulary._token_to_id[(None, RULE_ID)]
    assert ids[comment_index] != real_rule_id_tag_id  # Variant 1 (value) still fixed

    crafted_key_index = tokens.index(API_VAL_STR)
    following_tag_index = crafted_key_index + 1
    assert sequence.roles[following_tag_index] == TOKEN_ROLE_CONTROL
    assert ids[following_tag_index] == vocabulary._token_to_id[(None, tokens[following_tag_index])]

    email_index = _value_index_after_key(tokens, API_KEY, "email")
    assert tokens[email_index - 1] == API_VAL_MASKED_STR  # Variant 2 still fixed too


def test_mixed_genuine_and_fake_placeholders_are_distinguishable() -> None:
    sequence = _pipeline_sequence(
        payload={
            "email": "alice@example.com",  # genuine
            "nickname": "[MASKED_PII]",  # fake
            "bio": "[MASKED_SECRET]",  # fake, different placeholder text
        }
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens

    email_index = _value_index_after_key(tokens, API_KEY, "email")
    nickname_index = _value_index_after_key(tokens, API_KEY, "nickname")
    bio_index = _value_index_after_key(tokens, API_KEY, "bio")

    assert tokens[email_index - 1] == API_VAL_MASKED_STR
    assert tokens[nickname_index - 1] == API_VAL_STR
    assert tokens[bio_index - 1] == API_VAL_STR

    assert ids[email_index] == vocabulary._token_to_id[(None, MASKED_PII)]
    assert ids[nickname_index] != vocabulary._token_to_id[(None, MASKED_PII)]
    assert ids[bio_index] != vocabulary._token_to_id[(None, MASKED_SECRET)]
    assert ids[nickname_index] != ids[bio_index]


def test_full_pipeline_encoding_with_provenance_is_deterministic() -> None:
    kwargs: dict[str, object] = {
        "payload": {"email": "a@b.com", "nickname": "[MASKED_PII]"},
        "headers": {"Authorization": "Bearer secret"},
    }
    seq_a = _pipeline_sequence(**kwargs)
    seq_b = _pipeline_sequence(**kwargs)
    assert seq_a.tokens == seq_b.tokens
    assert seq_a.roles == seq_b.roles

    vocabulary = Vocabulary()
    vocabulary.fit([seq_a])

    assert vocabulary.encode(seq_a) == vocabulary.encode(seq_b)


def test_vocabulary_encoding_without_any_provenance_remains_backward_compatible() -> None:
    # tokenize() with no masked_paths at all (the pre-security-hardening
    # call pattern). Every placeholder-looking string is an ordinary
    # literal -- safe (no Variant 2 protection, but no regression either).
    sequence = _sequence(payload={"email": "[MASKED_PII]"})
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])

    ids = vocabulary.encode(sequence)
    tokens = sequence.tokens
    value_index = _value_index_after_key(tokens, API_KEY, "email")

    assert tokens[value_index - 1] == API_VAL_STR
    assert vocabulary.decode(ids) == tokens
