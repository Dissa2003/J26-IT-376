"""Unit tests for tokenization_ai.dataset_adapters (the SR-BH 2020 adapter).

All row fixtures are small, hand-written, synthetic dicts shaped like the
raw SR-BH 2020 CSV schema confirmed in Phase 13A/13B -- never real dataset
rows, never real cookies/IPs/PII. Sentinel strings are used in place of real
sensitive content specifically so leakage tests can assert on their absence
without ever handling (or risking printing) anything actually sensitive.
"""

import pytest

from tokenization_ai.dataset_adapters import (
    CAPEC_COLUMNS,
    DENYLISTED_COLUMNS,
    LABEL_COLUMNS,
    NORMAL_COLUMN,
    BodyParseOutcome,
    LabelExtractionError,
    SrBh2020AdapterError,
    extract_binary_label,
    parse_request_body,
    to_empty_verified_rule_info,
    to_unified_api_context,
)
from tokenization_ai.fusion import InputFusion
from tokenization_ai.masking import MASKED_AUTH, MASKED_SECRET, PrivacyMasker
from tokenization_ai.normalization import Normalizer
from tokenization_ai.tokenizer import HybridTokenizer


def _full_row(**overrides: object) -> dict[str, object]:
    """A complete, synthetic 38-column SR-BH 2020 row, all benign/normal by default."""
    row: dict[str, object] = {
        "timestamp": "17/Jul/2020:12:23:34 +0100",
        "src_ip": "10.0.0.1",
        "src_port": "51234",
        "dst_ip": "10.0.0.2",
        "dst_port": "80",
        "request_http_method": "GET",
        "request_http_request": "/index.php",
        "request_http_protocol": "HTTP/1.1",
        "request_user_agent": "SyntheticAgent/1.0",
        "request_referer": "",
        "request_host": "example.test",
        "request_origin": "",
        "request_cookie": "",
        "request_content_type": "",
        "request_accept": "*/*",
        "request_accept_language": "en",
        "request_accept_encoding": "gzip",
        "request_do_not_track": "",
        "request_connection": "keep-alive",
        "request_body": "",
        "response_http_protocol": "HTTP/1.1",
        "response_http_status_code": "200",
        "response_http_status_message": "OK",
        "response_content_length": "512",
        NORMAL_COLUMN: "1",
    }
    for column in CAPEC_COLUMNS:
        row[column] = "0"
    row.update(overrides)
    return row


def _anomalous_row(**overrides: object) -> dict[str, object]:
    row = _full_row(**{NORMAL_COLUMN: "0", "66 - SQL Injection": "1"})
    row.update(overrides)
    return row


# -- Field allowlisting ---------------------------------------------------------


def test_to_unified_api_context_maps_method_and_path() -> None:
    context, outcome = to_unified_api_context(_full_row(), "row-1")

    assert context.method == "GET"
    assert context.path == "/index.php"
    assert outcome == BodyParseOutcome.EMPTY


def test_to_unified_api_context_maps_allowlisted_headers_only() -> None:
    row = _full_row(
        request_user_agent="SyntheticAgent/2.0",
        request_host="host.test",
        request_accept="text/html",
    )

    context, _outcome = to_unified_api_context(row, "row-1")

    assert context.headers["User-Agent"] == "SyntheticAgent/2.0"
    assert context.headers["Host"] == "host.test"
    assert context.headers["Accept"] == "text/html"


def test_to_unified_api_context_omits_empty_header_columns() -> None:
    context, _outcome = to_unified_api_context(_full_row(), "row-1")

    # referer, origin, cookie, content_type, do_not_track are "" in the fixture.
    assert "Referer" not in context.headers
    assert "Origin" not in context.headers
    assert "Cookie" not in context.headers


def test_to_unified_api_context_never_populates_client_ip() -> None:
    context, _outcome = to_unified_api_context(_full_row(), "row-1")

    # src_ip is denylisted -- this adapter never reads it into the context.
    assert context.client_ip is None


def test_to_unified_api_context_tolerates_unknown_extra_columns() -> None:
    row = _full_row()
    row["some_future_column_not_in_the_schema"] = "unexpected-value"

    context, _outcome = to_unified_api_context(row, "row-1")

    serialized = context.model_dump_json()
    assert "unexpected-value" not in serialized


# -- No response-side / label leakage -------------------------------------------


def test_to_unified_api_context_never_leaks_response_side_values() -> None:
    row = _full_row(
        response_http_status_code="RESPONSE-SENTINEL-CODE",
        response_http_status_message="RESPONSE-SENTINEL-MESSAGE",
        response_content_length="RESPONSE-SENTINEL-LENGTH",
        response_http_protocol="RESPONSE-SENTINEL-PROTOCOL",
    )

    context, _outcome = to_unified_api_context(row, "row-1")

    serialized = context.model_dump_json()
    assert "RESPONSE-SENTINEL" not in serialized


def test_to_unified_api_context_never_leaks_label_values() -> None:
    row = _anomalous_row()
    row[NORMAL_COLUMN] = "LABEL-SENTINEL-NORMAL"  # deliberately non-binary/sentinel-shaped

    context, _outcome = to_unified_api_context(row, "row-1")

    serialized = context.model_dump_json()
    assert "LABEL-SENTINEL" not in serialized


def test_denylisted_columns_covers_every_response_and_label_and_network_column() -> None:
    assert DENYLISTED_COLUMNS >= set(LABEL_COLUMNS)
    assert "response_http_status_code" in DENYLISTED_COLUMNS
    assert "src_ip" in DENYLISTED_COLUMNS
    assert "timestamp" in DENYLISTED_COLUMNS


# -- Missing / invalid required fields -------------------------------------------


def test_to_unified_api_context_rejects_a_missing_method() -> None:
    row = _full_row(request_http_method="")

    with pytest.raises(SrBh2020AdapterError):
        to_unified_api_context(row, "row-1")


def test_to_unified_api_context_rejects_a_row_missing_the_path_column() -> None:
    row = _full_row()
    del row["request_http_request"]

    with pytest.raises(SrBh2020AdapterError):
        to_unified_api_context(row, "row-1")


def test_to_unified_api_context_error_message_never_contains_other_row_values() -> None:
    row = _full_row(request_http_method="", request_cookie="SECRET-COOKIE-VALUE")

    with pytest.raises(SrBh2020AdapterError) as excinfo:
        to_unified_api_context(row, "row-1")

    assert "SECRET-COOKIE-VALUE" not in str(excinfo.value)


# -- Binary label extraction -----------------------------------------------------


def test_extract_binary_label_normal_row_is_zero() -> None:
    assert extract_binary_label(_full_row()) == 0


def test_extract_binary_label_anomalous_row_is_one() -> None:
    assert extract_binary_label(_anomalous_row()) == 1


def test_extract_binary_label_is_not_inverted_for_every_capec_column() -> None:
    for column in CAPEC_COLUMNS:
        row = _full_row(**{NORMAL_COLUMN: "0", column: "1"})
        assert extract_binary_label(row) == 1


def test_extract_binary_label_rejects_missing_label_column() -> None:
    row = _full_row()
    del row[NORMAL_COLUMN]

    with pytest.raises(LabelExtractionError):
        extract_binary_label(row)


def test_extract_binary_label_rejects_a_non_binary_value() -> None:
    row = _full_row(**{NORMAL_COLUMN: "maybe"})

    with pytest.raises(LabelExtractionError):
        extract_binary_label(row)


def test_extract_binary_label_error_never_echoes_the_bad_value() -> None:
    row = _full_row(**{NORMAL_COLUMN: "SENSITIVE-LOOKING-VALUE"})

    with pytest.raises(LabelExtractionError) as excinfo:
        extract_binary_label(row)

    assert "SENSITIVE-LOOKING-VALUE" not in str(excinfo.value)


def test_extract_binary_label_rejects_normal_and_capec_both_set() -> None:
    row = _full_row(**{NORMAL_COLUMN: "1", "66 - SQL Injection": "1"})

    with pytest.raises(LabelExtractionError):
        extract_binary_label(row)


def test_extract_binary_label_rejects_anomalous_with_no_capec_category() -> None:
    row = _full_row(**{NORMAL_COLUMN: "0"})  # all CAPEC columns remain "0"

    with pytest.raises(LabelExtractionError):
        extract_binary_label(row)


def test_extract_binary_label_does_not_silently_repair_a_contradiction() -> None:
    # A caller must receive an error, never a guessed/repaired label.
    row = _full_row(**{NORMAL_COLUMN: "1", "126 - Path Traversal": "1"})

    with pytest.raises(LabelExtractionError):
        extract_binary_label(row)


# -- event_id alignment -----------------------------------------------------------


def test_context_and_empty_rule_info_share_the_given_event_id() -> None:
    context, _outcome = to_unified_api_context(_full_row(), "row-42")
    rule_info = to_empty_verified_rule_info("row-42")

    assert context.event_id == "row-42"
    assert rule_info.event_id == "row-42"


def test_empty_verified_rule_info_has_no_rules() -> None:
    rule_info = to_empty_verified_rule_info("row-1")

    assert rule_info.rules == []


# -- Body parsing: JSON / form / empty / unsupported -----------------------------


def test_parse_request_body_empty_string_is_empty_outcome() -> None:
    payload, outcome = parse_request_body("")

    assert payload == {}
    assert outcome == BodyParseOutcome.EMPTY


def test_parse_request_body_json_object() -> None:
    payload, outcome = parse_request_body('{"username": "a", "remember": true}')

    assert payload == {"username": "a", "remember": True}
    assert outcome == BodyParseOutcome.JSON


def test_parse_request_body_form_urlencoded_single_values() -> None:
    payload, outcome = parse_request_body("log=admin&pwd=test&wp-submit=Log+In")

    assert payload == {"log": "admin", "pwd": "test", "wp-submit": "Log In"}
    assert outcome == BodyParseOutcome.FORM_URLENCODED


def test_parse_request_body_form_urlencoded_repeated_key_becomes_a_list() -> None:
    payload, outcome = parse_request_body("tag=a&tag=b")

    assert payload == {"tag": ["a", "b"]}
    assert outcome == BodyParseOutcome.FORM_URLENCODED


def test_parse_request_body_json_array_is_unsupported_not_fabricated() -> None:
    # A JSON array has no dict-shaped representation to offer without
    # inventing one -- must be reported as unsupported, not wrapped.
    payload, outcome = parse_request_body("[1, 2, 3]")

    assert payload == {}
    assert outcome == BodyParseOutcome.UNSUPPORTED


def test_parse_request_body_unparseable_free_text_is_unsupported() -> None:
    payload, outcome = parse_request_body("this is not json and not form data")

    assert payload == {}
    assert outcome == BodyParseOutcome.UNSUPPORTED


def test_to_unified_api_context_reports_unsupported_body_without_fabricating_payload() -> None:
    row = _full_row(request_body="this is not json and not form data")

    context, outcome = to_unified_api_context(row, "row-1")

    assert outcome == BodyParseOutcome.UNSUPPORTED
    assert context.payload == {}


# -- Sensitive header masking through the existing pipeline ----------------------


def test_cookie_header_from_the_adapter_is_masked_by_the_existing_pipeline() -> None:
    row = _full_row(request_cookie="session=SENSITIVE-SESSION-TOKEN")
    context, _outcome = to_unified_api_context(row, "row-1")
    rule_info = to_empty_verified_rule_info("row-1")
    fused = InputFusion().fuse(context, rule_info)

    mask_result = PrivacyMasker().mask_with_provenance(fused)

    assert mask_result.fused.context.headers["Cookie"] == MASKED_AUTH
    assert ("headers", "Cookie") in mask_result.masked_paths
    assert "SENSITIVE-SESSION-TOKEN" not in mask_result.fused.context.model_dump_json()


def test_adapter_does_not_bypass_masking_for_a_form_body_with_a_password_field() -> None:
    # NOTE: uses the key name "password", which masking.py's _SECRET_KEYS
    # already covers -- this test proves the adapter's output still flows
    # through real masking, not that masking covers every real-world key
    # name. See the Phase 13B report: WordPress's actual wp-login.php form
    # field is "pwd", which _SECRET_KEYS does NOT currently cover -- a real
    # gap this adapter work surfaced, flagged separately for approval rather
    # than silently patched here (masking.py is out of this phase's scope).
    row = _full_row(request_http_method="POST", request_body="log=admin&password=SENSITIVE-PASSWORD")
    context, _outcome = to_unified_api_context(row, "row-1")
    rule_info = to_empty_verified_rule_info("row-1")
    fused = InputFusion().fuse(context, rule_info)

    mask_result = PrivacyMasker().mask_with_provenance(fused)

    assert "SENSITIVE-PASSWORD" not in mask_result.fused.context.model_dump_json()


def test_wordpress_pwd_field_is_now_masked() -> None:
    """Regression test for the Phase 13C privacy fix.

    SR-BH 2020 comes from a WordPress honeypot; WordPress's own login form
    (wp-login.php) posts the password under the field name "pwd", not
    "password"/"passwd". Phase 13B's inspection found this field passed
    through masking UNMASKED; Phase 13C added "pwd" as an exact entry to
    masking.py's _SECRET_KEYS. This test proves the fix through the real,
    unmodified masking pipeline -- not by re-checking the key list directly.
    """
    row = _full_row(request_http_method="POST", request_body="log=admin&pwd=SENSITIVE-PASSWORD")
    context, _outcome = to_unified_api_context(row, "row-1")
    rule_info = to_empty_verified_rule_info("row-1")
    fused = InputFusion().fuse(context, rule_info)

    mask_result = PrivacyMasker().mask_with_provenance(fused)

    assert "SENSITIVE-PASSWORD" not in mask_result.fused.context.model_dump_json()
    assert mask_result.fused.context.payload["pwd"] == MASKED_SECRET
    assert ("payload", "pwd") in mask_result.masked_paths


# -- Non-mutation and determinism -------------------------------------------------


def test_to_unified_api_context_does_not_mutate_the_input_row() -> None:
    row = _full_row()
    before = dict(row)

    to_unified_api_context(row, "row-1")

    assert row == before


def test_to_unified_api_context_is_deterministic() -> None:
    row = _full_row()

    first, first_outcome = to_unified_api_context(row, "row-1")
    second, second_outcome = to_unified_api_context(row, "row-1")

    assert first == second
    assert first_outcome == second_outcome


def test_extract_binary_label_does_not_mutate_the_input_row() -> None:
    row = _full_row()
    before = dict(row)

    extract_binary_label(row)

    assert row == before


# -- Compatibility with the existing LogiShield pipeline ----------------------------


def test_adapter_output_flows_through_the_full_tokenization_chain() -> None:
    row = _full_row(
        request_http_method="POST",
        request_host="example.test",
        request_body="q=1&flag=true",
    )
    context, _outcome = to_unified_api_context(row, "row-7")
    rule_info = to_empty_verified_rule_info("row-7")

    fused = InputFusion().fuse(context, rule_info)
    mask_result = PrivacyMasker().mask_with_provenance(fused)
    norm_result = Normalizer().normalize_with_provenance(mask_result.fused, mask_result.masked_paths)
    sequence = HybridTokenizer().tokenize(norm_result.fused, norm_result.masked_paths)

    assert sequence.event_id == "row-7"
    assert len(sequence.tokens) == len(sequence.roles)
    assert len(sequence.tokens) > 0


# -- Phase 13C: pwd masking carries authoritative TOKEN_ROLE_MASKED provenance ------


def test_wordpress_pwd_value_gets_the_authoritative_masked_role() -> None:
    from tokenization_ai.tokenizer import TOKEN_ROLE_MASKED

    row = _full_row(request_http_method="POST", request_body="log=admin&pwd=SENSITIVE-PASSWORD")
    context, _outcome = to_unified_api_context(row, "row-1")
    rule_info = to_empty_verified_rule_info("row-1")

    fused = InputFusion().fuse(context, rule_info)
    mask_result = PrivacyMasker().mask_with_provenance(fused)
    norm_result = Normalizer().normalize_with_provenance(mask_result.fused, mask_result.masked_paths)
    sequence = HybridTokenizer().tokenize(norm_result.fused, norm_result.masked_paths)

    pwd_key_index = sequence.tokens.index("pwd")
    pwd_value_role = sequence.roles[pwd_key_index + 2]
    assert pwd_value_role == TOKEN_ROLE_MASKED
    assert "SENSITIVE-PASSWORD" not in sequence.tokens


def test_pwd_lookalike_placeholder_remains_an_ordinary_literal_token() -> None:
    from tokenization_ai.tokenizer import TOKEN_ROLE_LITERAL_API, TOKEN_ROLE_MASKED

    # "nickname" is never classified as sensitive, so a value that merely
    # reads like this module's masking placeholder is NOT genuinely masked.
    row = _full_row(request_http_method="POST", request_body="log=admin&nickname=%5BMASKED_SECRET%5D")
    context, _outcome = to_unified_api_context(row, "row-1")
    rule_info = to_empty_verified_rule_info("row-1")

    fused = InputFusion().fuse(context, rule_info)
    mask_result = PrivacyMasker().mask_with_provenance(fused)
    norm_result = Normalizer().normalize_with_provenance(mask_result.fused, mask_result.masked_paths)
    sequence = HybridTokenizer().tokenize(norm_result.fused, norm_result.masked_paths)

    nickname_key_index = sequence.tokens.index("nickname")
    nickname_value_role = sequence.roles[nickname_key_index + 2]
    assert nickname_value_role == TOKEN_ROLE_LITERAL_API
    assert nickname_value_role != TOKEN_ROLE_MASKED


def test_raw_pwd_value_never_appears_in_prediction_tokens_or_trace() -> None:
    """End-to-end: a real SR-BH-2020-shaped pwd field must not leak into
    tokens, the traced pipeline's stage summaries, or the final prediction."""
    from tokenization_ai.model import LightweightAnomalyModel
    from tokenization_ai.service import TokenizationAIService
    from tokenization_ai.tensorizer import Tensorizer
    from tokenization_ai.vocabulary import Vocabulary

    fusion, masker, normalizer, tokenizer = InputFusion(), PrivacyMasker(), Normalizer(), HybridTokenizer()

    def _tokenize_row(row: dict[str, object], event_id: str):
        context, _outcome = to_unified_api_context(row, event_id)
        rule_info = to_empty_verified_rule_info(event_id)
        fused = fusion.fuse(context, rule_info)
        mask_result = masker.mask_with_provenance(fused)
        norm_result = normalizer.normalize_with_provenance(mask_result.fused, mask_result.masked_paths)
        return tokenizer.tokenize(norm_result.fused, norm_result.masked_paths)

    benign_seq = _tokenize_row(_full_row(), "fit-benign")
    attack_row = _full_row(
        request_http_method="POST",
        request_body="log=admin&pwd=SENSITIVE-PASSWORD",
        **{NORMAL_COLUMN: "0", "66 - SQL Injection": "1"},
    )
    attack_seq = _tokenize_row(attack_row, "fit-attack")

    vocabulary = Vocabulary()
    vocabulary.fit([benign_seq, attack_seq])
    tensorizer = Tensorizer(vocabulary, max_length=256)
    model = LightweightAnomalyModel(num_buckets=128)
    model.fit([tensorizer.tensorize(benign_seq), tensorizer.tensorize(attack_seq)], [0, 1], epochs=20)

    service = TokenizationAIService(fusion, masker, normalizer, tokenizer, tensorizer, model)
    row = _full_row(request_http_method="POST", request_body="log=admin&pwd=SENSITIVE-PASSWORD")
    context, _outcome = to_unified_api_context(row, "row-live")
    rule_info = to_empty_verified_rule_info("row-live")
    trace = service.tracer.start("row-live")

    prediction = service.predict(context, rule_info, trace=trace)

    assert "SENSITIVE-PASSWORD" not in repr(trace.stages)
    assert 0.0 <= prediction.score <= 1.0
    sequence = service._tokenize(context, rule_info)
    assert "SENSITIVE-PASSWORD" not in sequence.tokens
