"""Regression tests for the PP1 audit finding D3 (masking recall gaps).

All values are synthetic. Each parametrized case below was *confirmed
unmasked* against the pre-fix implementation before the fix was written.
"""

import json

import pytest
from fastapi.testclient import TestClient

from tokenization_ai.app import create_app
from tokenization_ai.demo import _build_demo_service
from tokenization_ai.fusion import InputFusion
from tokenization_ai.masking import (
    MASKED_AUTH,
    MASKED_IP,
    MASKED_PII,
    MASKED_SECRET,
    PrivacyMasker,
)
from tokenization_ai.models import UnifiedApiContext, VerifiedRuleInfo
from tokenization_ai.normalization import Normalizer
from tokenization_ai.tokenizer import HybridTokenizer
from tokenization_ai.vocabulary import Vocabulary

SECRET = "synthetic-d3-secret-value-4711"


def _mask(payload=None, headers=None, path="/x"):
    context = UnifiedApiContext(
        event_id="d3", method="POST", path=path, payload=payload or {}, headers=headers or {}
    )
    fused = InputFusion().fuse(context, VerifiedRuleInfo(event_id="d3"))
    return PrivacyMasker().mask_with_provenance(fused)


def _paths(result):
    return {tuple(p) for p in result.masked_paths}


def _tokenize(payload=None, headers=None, path="/x"):
    masked = _mask(payload, headers, path)
    normalized = Normalizer().normalize_with_provenance(masked.fused, masked.masked_paths)
    return HybridTokenizer().tokenize(normalized.fused, normalized.masked_paths)


# --- previously unmasked payload keys -----------------------------------------

SECRET_KEYS = [
    "secret_key", "secret-key", "SECRET_KEY", "secretKey", "SecretKey",
    "api-key", "apiKey",  # camelCase splits to the pre-existing exact name "api_key"
    "session", "session_id", "sessionId", "Session-Id",
    "jwt", "JWT", "bearer", "Bearer",
    "accessToken", "refreshToken", "idToken",
    "userPassword", "new_password", "confirm-password", "oldPasswd",
    "passphrase", "passcode", "clientSecret", "secret_access_key", "csrf", "xsrf",
    "user_api_key", "stripe_secret_key", "aws_access_key",
]

AUTH_KEYS = [
    "authorization", "Authorization", "AUTHORIZATION", "proxy-authorization",
    "cookie", "Set-Cookie", "x-api-key", "X_API_KEY", "xApiKey", "X-Session-Id",
    "apikey", "APIKey", "X-ApiKey",
]

PII_KEYS = [
    "national_id", "nationalId", "National-ID", "nic", "passport_number", "tax_id",
    "date_of_birth", "dateOfBirth", "dob", "birth_date", "birthdate",
    "address", "Address", "home_address", "billingAddress", "email_address",
    "first_name", "firstName", "last_name", "full_name", "surname",
]


@pytest.mark.parametrize("key", SECRET_KEYS)
def test_secret_key_variants_are_masked_as_secret(key):
    result = _mask(payload={key: SECRET})
    assert result.fused.context.payload[key] == MASKED_SECRET
    assert _paths(result) == {("payload", key)}


@pytest.mark.parametrize("key", AUTH_KEYS)
def test_authentication_key_variants_are_masked_as_auth(key):
    result = _mask(payload={key: SECRET})
    assert result.fused.context.payload[key] == MASKED_AUTH
    assert _paths(result) == {("payload", key)}


@pytest.mark.parametrize("key", PII_KEYS)
def test_pii_key_variants_are_masked_as_pii(key):
    result = _mask(payload={key: "synthetic personal value"})
    assert result.fused.context.payload[key] == MASKED_PII
    assert _paths(result) == {("payload", key)}


def test_a_structured_value_under_a_sensitive_key_is_masked_whole():
    result = _mask(payload={"address": {"street": "1 Synthetic Rd", "city": "Colombo"}})
    assert result.fused.context.payload == {"address": MASKED_PII}
    assert "Synthetic Rd" not in json.dumps(result.fused.context.model_dump(mode="json"))


# --- headers ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "X-Session-Id", "x-session-id", "X_SESSION_ID", "X-Csrf", "x-auth",
        "apikey", "ApiKey", "X-ApiKey", "AUTHORIZATION", "Authorization",
        "authorization", "Proxy-Authorization", "Cookie", "X-Amz-Security-Token",
    ],
)
def test_header_variants_are_masked_whatever_their_capitalization(header):
    result = _mask(headers={header: SECRET})
    assert list(result.fused.context.headers.values()) == [MASKED_AUTH]
    assert _paths(result) == {("headers", header)}


@pytest.mark.parametrize("header", [" Authorization ", "\tAuthorization", "X-Session-Id ", " cookie"])
def test_whitespace_padded_header_names_do_not_bypass_masking(header):
    """Masking runs before normalization trims names; the old code missed these."""
    result = _mask(headers={header: SECRET})
    assert list(result.fused.context.headers.values()) == [MASKED_AUTH]

    sequence = _tokenize(headers={header: SECRET})
    assert SECRET not in sequence.tokens
    name = header.strip().lower()
    assert name in sequence.tokens
    assert (MASKED_AUTH, "masked") in zip(sequence.tokens, sequence.roles)


# --- value-shaped credentials ----------------------------------------------------


@pytest.mark.parametrize(
    "value,placeholder",
    [
        ("Bearer abcdef1234567890", MASKED_AUTH),
        ("bearer ABCDEF1234567890xyz", MASKED_AUTH),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r", MASKED_AUTH),
        ("alice@example.org", MASKED_PII),
        ("203.0.113.9", MASKED_IP),
    ],
)
def test_value_shaped_credentials_are_masked_under_unremarkable_keys(value, placeholder):
    result = _mask(payload={"note": value})
    assert result.fused.context.payload["note"] == placeholder
    assert _paths(result) == {("payload", "note")}


@pytest.mark.parametrize(
    "value",
    ["Bearer of bad news", "Basic information about orders", "bearer", "eyJ not a jwt", "Bearer short"],
)
def test_prose_that_resembles_credentials_is_not_masked(value):
    result = _mask(payload={"note": value})
    assert result.fused.context.payload["note"] == value
    assert _paths(result) == set()


# --- nested payloads and collections ---------------------------------------------


def test_nested_objects_and_lists_are_masked_with_exact_provenance_paths():
    payload = {
        "user": {"profile": {"dateOfBirth": "2000-01-01", "city": "Colombo"}, "jwt": SECRET},
        "history": [{"sessionId": SECRET, "page": "/home"}, {"page": "/cart", "Authorization": SECRET}],
        "tags": ("x", "y"),
    }
    result = _mask(payload=payload)
    assert _paths(result) == {
        ("payload", "user", "profile", "dateOfBirth"),
        ("payload", "user", "jwt"),
        ("payload", "history", 0, "sessionId"),
        ("payload", "history", 1, "Authorization"),
    }
    masked = result.fused.context.payload
    assert masked["user"]["profile"] == {"dateOfBirth": MASKED_PII, "city": "Colombo"}
    assert masked["history"][0] == {"sessionId": MASKED_SECRET, "page": "/home"}
    assert masked["history"][1] == {"page": "/cart", "Authorization": MASKED_AUTH}
    assert SECRET not in json.dumps(result.fused.context.model_dump(mode="json"))


def test_sensitive_query_parameters_use_the_extended_key_rules():
    result = _mask(path=f"/cb?sessionId={SECRET}&page=2&apiKey=k-123456789")
    assert _paths(result) == {("path",)}
    path = result.fused.context.path
    assert SECRET not in path and "k-123456789" not in path
    assert "page=2" in path


# --- non-sensitive data must be untouched ----------------------------------------


_ORDINARY_KEYS = [
    "name", "username", "description", "session_timeout", "sessionCount", "sort_key",
    "primary_key", "cache_key", "monkey", "token_count", "tokenizer", "address_count",
    "passport_control", "dobby", "bearer_of_news", "secret_santa", "csrf_enabled",
    "X-Request-Id", "User-Agent", "Content-Type", "Accept",
]


@pytest.mark.parametrize("key", _ORDINARY_KEYS)
def test_non_sensitive_payload_keys_are_left_unchanged(key):
    result = _mask(payload={key: "ordinary-value"})
    assert result.fused.context.payload == {key: "ordinary-value"}
    assert _paths(result) == set()


@pytest.mark.parametrize(
    "header",
    [k for k in _ORDINARY_KEYS if not k.endswith("_key")],  # pre-existing rule: header names ending "_key" are masked
)
def test_non_sensitive_header_names_are_left_unchanged(header):
    result = _mask(headers={header: "ordinary-value"})
    assert result.fused.context.headers == {header: "ordinary-value"}
    assert _paths(result) == set()


def test_existing_masking_behaviour_is_preserved():
    result = _mask(
        payload={"password": SECRET, "email": "a@b.co", "card": "4111 1111 1111 1111", "ip": "198.51.100.7"},
        headers={"Authorization": "Bearer x", "Cookie": "sid=1", "User-Agent": "demo"},
    )
    payload = result.fused.context.payload
    assert payload == {
        "password": MASKED_SECRET, "email": MASKED_PII, "card": MASKED_PII, "ip": MASKED_IP,
    }
    assert result.fused.context.headers == {
        "Authorization": MASKED_AUTH, "Cookie": MASKED_AUTH, "User-Agent": "demo",
    }


# --- nothing raw reaches tokens, vocabulary, API or traces ------------------------

_SENSITIVE_REQUEST = {
    "context": {
        "event_id": "d3-api",
        "metadata": {
            "method": "POST",
            "path": f"/session?sessionId={SECRET}",
            "headers": {" Authorization ": f"Bearer {SECRET}", "X-Session-Id": SECRET},
        },
        "payload": {
            "secretKey": SECRET,
            "history": [{"jwt": SECRET}],
            "address": "SYNTHETIC-HOME-ADDRESS-9",
            "note": f"Bearer {SECRET}",
            "dateOfBirth": "SYNTHETIC-DOB-1999",
        },
    },
    "rule_source": "none",
}
_RAW = [SECRET, "SYNTHETIC-HOME-ADDRESS-9", "SYNTHETIC-DOB-1999"]


def test_raw_values_never_enter_token_sequences_or_the_fitted_vocabulary():
    request = _SENSITIVE_REQUEST["context"]
    sequence = _tokenize(
        payload=request["payload"], headers=request["metadata"]["headers"], path=request["metadata"]["path"]
    )
    for raw in _RAW:
        assert not any(raw in token for token in sequence.tokens)
    assert sequence.roles.count("masked") >= 5

    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    stored = vocabulary.decode(list(range(len(vocabulary))))
    for raw in _RAW:
        assert not any(raw in token for token in stored)


def test_api_response_and_error_bodies_do_not_expose_raw_values():
    client = TestClient(create_app(_build_demo_service()))
    response = client.post("/v1/analyze", json=_SENSITIVE_REQUEST)
    assert response.status_code == 200
    for raw in _RAW:
        assert raw not in response.text
    body = response.json()
    assert body["masking"]["masked_field_count"] >= 6

    broken = json.loads(json.dumps(_SENSITIVE_REQUEST))
    broken["context"]["metadata"].pop("method")
    failed = client.post("/v1/analyze", json=broken)
    assert failed.status_code == 422
    for raw in _RAW:
        assert raw not in failed.text


def test_service_trace_records_counts_only():
    service = _build_demo_service()
    from tokenization_ai.adapters import to_unified_api_context, to_verified_rule_info

    context = to_unified_api_context(_SENSITIVE_REQUEST["context"])
    trace = service.tracer.start(context.event_id)
    service.predict(context, to_verified_rule_info({"event_id": context.event_id, "rules": []}), trace=trace)
    dumped = json.dumps([(s.stage, s.summary, s.error_type) for s in trace.stages])
    for raw in _RAW:
        assert raw not in dumped
    assert all(isinstance(v, (int, bool)) for s in trace.stages for v in s.summary.values())


# --- demo flows and tensor structure unchanged --------------------------------------


def test_existing_demo_scenarios_keep_their_masking_and_roles():
    client = TestClient(create_app(_build_demo_service()))
    scenarios = {
        s["id"]: s["request"]
        for s in json.loads(
            (__import__("pathlib").Path("tokenization_ai/static/scenarios.json")).read_text(encoding="utf-8")
        )["scenarios"]
    }
    benign = client.post("/v1/analyze", json=scenarios["benign"]).json()
    suspicious = client.post("/v1/analyze", json=scenarios["suspicious"]).json()
    assert benign["masking"]["masked_field_count"] == 0
    assert suspicious["masking"]["masked_field_count"] == 2
    assert suspicious["masked_representation"]["headers"]["authorization"] == MASKED_AUTH
    assert suspicious["masked_representation"]["payload"]["password"] == MASKED_SECRET
    assert benign["tensor"]["token_count_before_truncation"] == 26
    assert suspicious["tensor"]["token_count_before_truncation"] == 44


def test_masking_a_new_key_changes_only_the_value_token_not_the_structure():
    plain = _tokenize(payload={"note": "ordinary-value"})
    masked = _tokenize(payload={"jwt": "ordinary-value"})
    assert len(plain.tokens) == len(masked.tokens)
    assert masked.tokens[-5:-4] == ["[MASKED_SECRET]"] or "[MASKED_SECRET]" in masked.tokens
    index = masked.tokens.index("[MASKED_SECRET]")
    assert masked.roles[index] == "masked"
    assert masked.tokens[index - 1] == "[API:VAL_MASKED_STR]"
