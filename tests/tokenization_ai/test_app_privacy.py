"""Fail-closed privacy tests for ``POST /v1/analyze``.

Each sentinel is a distinctive string that the masking heuristic does NOT
recognize (unknown key names, free text, odd header names, query params,
path segments, rule attributes). Every one must be absent from successful
responses, validation errors and application logs. All data is synthetic.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from tokenization_ai.app import create_app
from tokenization_ai.demo import _build_demo_service
from tokenization_ai.models import UnifiedApiContext, VerifiedRuleInfo

S_KEY = "SENTINEL_unknown_key_k1"
S_VAL = "SENTINEL_free_text_v1"
S_NESTED_KEY = "SENTINEL_nested_key_k2"
S_NESTED_VAL = "SENTINEL_nested_val_v2"
S_LIST = "SENTINEL_list_item_v3"
S_HDR_NAME = "x-SENTINEL-header-name"
S_HDR_VAL = "SENTINEL_header_value_v4"
S_QPARAM = "SENTINEL_qparam_k5"
S_QVAL = "SENTINEL_qvalue_v5"
S_SEGMENT = "SENTINEL_segment_v6"
S_RULE_ID = "SENTINEL-RULE-ID-v7"
S_RULE_KEY = "SENTINEL_rule_attr_key_k8"
S_RULE_VAL = "SENTINEL_rule_attr_val_v8"
S_NUMBER = 918273645
S_EVENT = "SENTINEL event id with spaces"
S_RECOGNIZED = "SENTINEL_recognized_password_v9"

ALL_STRING_SENTINELS = [
    S_KEY,
    S_VAL,
    S_NESTED_KEY,
    S_NESTED_VAL,
    S_LIST,
    S_HDR_NAME,
    S_HDR_VAL,
    S_QPARAM,
    S_QVAL,
    S_SEGMENT,
    S_RULE_ID,
    S_RULE_KEY,
    S_RULE_VAL,
    str(S_NUMBER),
    S_RECOGNIZED,
]


def _context(event_id="evt-priv"):
    return {
        "event_id": event_id,
        "metadata": {
            "method": "POST",
            "path": f"/{S_SEGMENT}/orders?{S_QPARAM}={S_QVAL}&token={S_RECOGNIZED}",
            "headers": {S_HDR_NAME: S_HDR_VAL, "Authorization": f"Bearer {S_RECOGNIZED}"},
        },
        "payload": {
            S_KEY: S_VAL,
            "password": S_RECOGNIZED,
            "amount": S_NUMBER,
            "nested": {S_NESTED_KEY: {"deeper": [S_LIST, {"leaf": S_NESTED_VAL}]}},
            "items": [S_LIST, S_NUMBER],
        },
        "threat_score": 0.3,
    }


def _rules(event_id="evt-priv"):
    return {
        "event_id": event_id,
        "rules": [{"rule_id": S_RULE_ID, S_RULE_KEY: S_RULE_VAL, "nested": [S_RULE_VAL]}],
    }


@pytest.fixture
def service():
    return _build_demo_service()


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def _assert_clean(text: str):
    for sentinel in ALL_STRING_SENTINELS:
        assert sentinel not in text, f"leaked {sentinel!r}"


def test_successful_response_hides_every_unrecognized_value(client, caplog):
    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/v1/analyze",
            json={
                "context": _context(),
                "rule_source": "mock_fixture",
                "rule_info": _rules(),
            },
        )
    assert response.status_code == 200
    _assert_clean(response.text)
    _assert_clean(caplog.text)

    body = response.json()
    rep = body["masked_representation"]
    # Structure survives: keys are numbered placeholders, nesting/list shape is kept.
    assert any(k.startswith("[UNRECOGNIZED_KEY]") for k in rep["payload"])
    assert rep["payload"]["password"] == "[MASKED_SECRET]"
    assert rep["headers"]["authorization"] == "[MASKED_AUTH]"
    assert "[UNRECOGNIZED_SEGMENT]" in rep["path"]
    assert "[UNRECOGNIZED_PARAM]=[UNRECOGNIZED_VALUE]" in rep["path"]
    # "token" is not a literal the demo vocabulary knows, so the name is hidden
    # even though the trusted masker recognized (and masked) its value.
    assert "[UNRECOGNIZED_PARAM]=[MASKED_SECRET]" in rep["path"]
    assert rep["method"] == "POST"
    assert body["privacy"]["hidden_in_representation"] > 0
    assert body["privacy"]["hidden_in_tokens"] > 0
    assert body["privacy"]["output_kind"] == "privacy_safe_projection"

    # Roles and masking indicators survive for the token view.
    roles = {item["role"] for item in body["tokens"]["items"]}
    assert {"control", "masked", "literal_api", "literal_rule"} <= roles
    tokens = [item["token"] for item in body["tokens"]["items"]]
    assert "[UNRECOGNIZED_LITERAL]" in tokens
    assert "[MASKED_SECRET]" in tokens

    # Rules are labelled as mock and the sentinel id is hidden.
    status = body["provenance"]["rule_status"]
    assert status["rule_ids"] == ["[UNRECOGNIZED_RULE_ID]"]
    assert status["verification_performed_by_this_service"] is False
    assert "mock" in status["label"]


def test_masked_field_names_are_projected_too(client):
    context = _context()
    context["payload"] = {S_NESTED_KEY: {"password": S_RECOGNIZED}}
    response = client.post("/v1/analyze", json={"context": context, "rule_source": "none"})
    assert response.status_code == 200
    _assert_clean(response.text)
    assert "payload.[UNRECOGNIZED_KEY].password" in response.json()["masking"]["masked_fields"]


def test_projection_does_not_change_model_input(service):
    """The model must score the same internal data whatever is displayed."""
    client = TestClient(create_app(service))
    context_dict = _context()
    response = client.post(
        "/v1/analyze",
        json={"context": context_dict, "rule_source": "mock_fixture", "rule_info": _rules()},
    )
    assert response.status_code == 200

    from tokenization_ai.adapters import to_unified_api_context, to_verified_rule_info

    direct = service.predict(
        to_unified_api_context(context_dict), to_verified_rule_info(_rules())
    )
    assert response.json()["prediction"]["score"] == direct.score
    # Unrecognized literals really did reach the model as unknown ids.
    assert response.json()["tensor"]["unknown_token_count"] > 0
    assert isinstance(UnifiedApiContext, type) and isinstance(VerifiedRuleInfo, type)


def test_known_demo_literals_are_still_shown(client):
    body = client.post(
        "/v1/analyze",
        json={
            "context": {
                "event_id": "evt-known",
                "metadata": {"method": "GET", "path": "/orders/123"},
                "payload": {"amount": 25, "currency": "USD"},
            },
            "rule_source": "none",
        },
    ).json()
    assert body["masked_representation"]["path"] == "/orders/123"
    assert body["masked_representation"]["payload"] == {"amount": 25, "currency": "USD"}
    assert body["privacy"]["hidden_in_representation"] == 0


def test_unrecognized_number_and_method_are_hidden(client):
    context = {
        "event_id": "evt-num",
        "metadata": {"method": S_VAL, "path": "/orders"},
        "payload": {"amount": S_NUMBER},
    }
    response = client.post("/v1/analyze", json={"context": context, "rule_source": "none"})
    assert response.status_code == 200
    _assert_clean(response.text)
    assert response.json()["masked_representation"]["method"] == "[UNRECOGNIZED_METHOD]"
    assert response.json()["masked_representation"]["payload"]["amount"] == "[UNRECOGNIZED_NUMBER]"


@pytest.mark.parametrize(
    "body",
    [
        # Unknown request field names and values.
        {"context": _context(), "rule_source": "none", S_KEY: S_VAL},
        # Unknown rule_source value.
        {"context": _context(), "rule_source": S_VAL},
        # Wrong type for context, sentinel nested inside.
        {"context": [S_VAL], "rule_source": "none"},
        {"context": {S_KEY: S_VAL}, "rule_source": "none"},
        # Contract violations: wrong header value type under an unknown header name.
        {
            "context": {
                "event_id": "evt-bad",
                "metadata": {"method": "GET", "path": "/x", "headers": {S_HDR_NAME: 12345}},
                "payload": {S_KEY: S_VAL},
            },
            "rule_source": "none",
        },
        # Bad threat score alongside secrets.
        {"context": {**_context(), "threat_score": 9}, "rule_source": "none"},
        # Rule contract violation.
        {
            "context": _context(),
            "rule_source": "mock_fixture",
            "rule_info": {"event_id": "evt-priv", "rules": [{S_RULE_KEY: S_RULE_VAL}]},
        },
        # Event id mismatch.
        {
            "context": _context("evt-a"),
            "rule_source": "mock_fixture",
            "rule_info": _rules("evt-b"),
        },
        # Rule source conflict.
        {"context": _context(), "rule_source": "none", "rule_info": _rules()},
    ],
)
def test_error_responses_never_echo_sentinels(client, caplog, body):
    with caplog.at_level(logging.DEBUG):
        response = client.post("/v1/analyze", json=body)
    assert response.status_code in (422,)
    _assert_clean(response.text)
    _assert_clean(caplog.text)
    assert "input" not in response.json()


def test_invalid_event_id_is_rejected_without_echo(client):
    response = client.post(
        "/v1/analyze", json={"context": _context(S_EVENT), "rule_source": "none"}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_event_id"
    assert "SENTINEL" not in response.text


def test_malformed_json_and_unexpected_errors_do_not_echo(service, caplog):
    client = TestClient(create_app(service), raise_server_exceptions=False)
    with caplog.at_level(logging.DEBUG):
        bad_json = client.post(
            "/v1/analyze",
            content=('{"context": {"%s": "%s"' % (S_KEY, S_VAL)).encode(),
            headers={"content-type": "application/json"},
        )
        assert bad_json.status_code == 422

        def boom(*_a, **_k):
            raise RuntimeError  # handled as readiness, no message

        service.analyze = boom
        not_ready = client.post(
            "/v1/analyze", json={"context": _context(), "rule_source": "none"}
        )
        assert not_ready.status_code == 503

        def leak(*_a, **_k):
            raise Exception(S_VAL)

        service.analyze = leak
        crashed = client.post(
            "/v1/analyze", json={"context": _context(), "rule_source": "none"}
        )
        assert crashed.status_code == 500
    for text in (bad_json.text, not_ready.text, crashed.text, caplog.text):
        _assert_clean(text)


def test_body_limits_still_enforced(client):
    big = {"context": _context(), "rule_source": "none", "pad": "x" * (64 * 1024 + 1)}
    assert client.post("/v1/analyze", json=big).status_code == 413
    chunked = client.post(
        "/v1/analyze",
        content=iter([json.dumps({"context": _context(), "rule_source": "none"}).encode()]),
        headers={"content-type": "application/json"},
    )
    assert chunked.status_code == 411


def test_verified_source_is_never_described_as_verified_by_service(client):
    body = client.post(
        "/v1/analyze",
        json={
            "context": {
                "event_id": "evt-v",
                "metadata": {"method": "GET", "path": "/orders"},
                "payload": {},
            },
            "rule_source": "verified",
            "rule_info": {"event_id": "evt-v", "rules": [{"rule_id": "MOCK-R-SQLI"}]},
        },
    ).json()
    status = body["provenance"]["rule_status"]
    assert status["source"] == "verified"
    assert "not verified by this service" in status["label"]
    assert status["verification_performed_by_this_service"] is False
    assert "asserted by the caller" in status["note"]
