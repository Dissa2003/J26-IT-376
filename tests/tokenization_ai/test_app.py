"""TestClient tests for the Member 4 ``POST /v1/analyze`` service.

All inputs are SYNTHETIC. Scores here show the pipeline is wired end to
end through HTTP; they are not evidence of detection accuracy.
"""

import json

import pytest
from fastapi.testclient import TestClient

from tokenization_ai.app import (
    CONTRACT_VERSION,
    MAX_BODY_BYTES,
    MAX_MASKED_FIELDS_RETURNED,
    MAX_TOKENS_RETURNED,
    create_app,
)
from tokenization_ai.demo import _build_demo_service
from tokenization_ai.model import LightweightAnomalyModel

SECRET_TOKEN = "synthetic-bearer-SECRET-123"
SECRET_PASSWORD = "synthetic-password-SECRET-456"
SECRET_EMAIL = "synthetic.person@example.com"
SECRET_IP = "203.0.113.77"
SECRET_QUERY = "synthetic-query-SECRET-789"
SECRET_RULE_ATTR = "synthetic-rule-attr-SECRET-000"

BENIGN = {
    "event_id": "evt-benign",
    "metadata": {"method": "GET", "path": "/orders/123"},
    "payload": {"amount": 25, "currency": "USD"},
    "threat_score": 0.1,
}


def _suspicious(event_id="evt-susp"):
    return {
        "event_id": event_id,
        "metadata": {
            "method": "POST",
            "path": f"/login?token={SECRET_QUERY}",
            "client_ip": SECRET_IP,
            "headers": {"Authorization": f"Bearer {SECRET_TOKEN}"},
        },
        "payload": {
            "username": "admin",
            "password": SECRET_PASSWORD,
            "contact": SECRET_EMAIL,
            "q": "' OR 1=1 --",
        },
        "threat_score": 0.5,
    }


def _rules(event_id, **attrs):
    return {"event_id": event_id, "rules": [{"rule_id": "MOCK-R-SQLI", **attrs}]}


@pytest.fixture
def service():
    return _build_demo_service()


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def _post(client, body):
    return client.post("/v1/analyze", json=body)


def test_benign_request_returns_full_analysis(client):
    response = _post(client, {"context": BENIGN, "rule_source": "none"})
    assert response.status_code == 200
    body = response.json()
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["event_id"] == "evt-benign"
    assert 0.0 <= body["prediction"]["score"] <= 1.0
    assert body["masked_representation"]["method"] == "GET"
    assert body["masked_representation"]["path"] == "/orders/123"
    assert body["masked_representation"]["payload"] == {"amount": 25, "currency": "USD"}
    assert body["masking"]["masked_field_count"] == 0
    stages = [s["stage"] for s in body["timings"]["stages"]]
    assert stages == [
        "fusion",
        "masking",
        "normalization",
        "tokenization",
        "tensorization",
        "model_inference",
    ]
    assert body["timings"]["total_elapsed_ms"] >= 0


def test_suspicious_request_scores_higher_than_benign_on_demo_fixtures(client):
    benign = _post(client, {"context": BENIGN, "rule_source": "none"}).json()
    suspicious = _post(
        client,
        {
            "context": _suspicious(),
            "rule_source": "mock_fixture",
            "rule_info": _rules("evt-susp", severity="high"),
        },
    )
    assert suspicious.status_code == 200
    body = suspicious.json()
    assert body["event_id"] == "evt-susp"
    assert body["masking"]["masked_field_count"] >= 4
    # Wiring check on the model's own synthetic training fixtures only.
    assert body["prediction"]["score"] > benign["prediction"]["score"]
    roles = {item["role"] for item in body["tokens"]["items"]}
    assert {"control", "masked", "literal_api"} <= roles


def test_response_never_contains_raw_secrets(client):
    response = _post(
        client,
        {
            "context": _suspicious(),
            "rule_source": "mock_fixture",
            "rule_info": _rules("evt-susp", note=SECRET_RULE_ATTR),
        },
    )
    assert response.status_code == 200
    text = response.text
    for secret in (
        SECRET_TOKEN,
        SECRET_PASSWORD,
        SECRET_EMAIL,
        SECRET_IP,
        SECRET_QUERY,
        SECRET_RULE_ATTR,
    ):
        assert secret not in text
    body = response.json()
    assert body["masked_representation"]["headers"]["authorization"] == "[MASKED_AUTH]"
    assert body["masked_representation"]["payload"]["password"] == "[MASKED_SECRET]"
    assert "headers.authorization" in body["masking"]["masked_fields"]
    assert "payload.password" in body["masking"]["masked_fields"]
    # The raw request is not echoed under any key.
    assert "context" not in body and "rule_info" not in body


def test_unrecognized_rule_literals_are_hidden_in_token_view(client):
    # Previously every rule literal was blanket-redacted. Now a rule literal is
    # shown only if the fitted vocabulary knows it (the demo corpus's own
    # "MOCK-R-SQLI"); caller-supplied attributes the vocabulary has never seen
    # are hidden. The rule id the vocabulary knows stays visible.
    body = _post(
        client,
        {
            "context": BENIGN,
            "rule_source": "mock_fixture",
            "rule_info": _rules("evt-benign", note=SECRET_RULE_ATTR),
        },
    ).json()
    rule_tokens = [i["token"] for i in body["tokens"]["items"] if i["role"] == "literal_rule"]
    assert "MOCK-R-SQLI" in rule_tokens
    assert "[UNRECOGNIZED_RULE_LITERAL]" in rule_tokens
    assert SECRET_RULE_ATTR not in json.dumps(body)
    assert body["provenance"]["rule_status"]["rule_ids"] == ["MOCK-R-SQLI"]


@pytest.mark.parametrize(
    "rule_source,rule_info,expected_source",
    [
        ("none", None, "none"),
        ("mock_fixture", _rules("evt-benign"), "mock_fixture"),
        ("verified", _rules("evt-benign"), "verified"),
    ],
)
def test_rule_status_is_always_labelled(client, rule_source, rule_info, expected_source):
    payload = {"context": BENIGN, "rule_source": rule_source}
    if rule_info is not None:
        payload["rule_info"] = rule_info
    status = _post(client, payload).json()["provenance"]["rule_status"]
    assert status["source"] == expected_source
    assert status["verification_performed_by_this_service"] is False
    assert status["rule_count"] == (0 if rule_source == "none" else 1)
    assert status["note"]


def test_default_app_labels_model_as_synthetic_demo():
    default_client = TestClient(create_app())  # builds the PP1 demo service
    prov = _post(default_client, {"context": BENIGN, "rule_source": "none"}).json()["provenance"]
    assert prov["detection_accuracy_claim"] == "none"
    assert prov["input_origin"] == "caller_supplied_request"
    assert "synthetic" in prov["model"]["note"]
    assert "not evidence" in prov["model"]["note"]


def test_injected_service_is_not_described_as_the_demo_model(client):
    prov = _post(client, {"context": BENIGN, "rule_source": "none"}).json()["provenance"]
    assert prov["detection_accuracy_claim"] == "none"
    assert "synthetic" not in prov["model"]["note"]


def test_rule_source_conflicts_are_rejected(client):
    none_with_rules = _post(
        client,
        {"context": BENIGN, "rule_source": "none", "rule_info": _rules("evt-benign")},
    )
    assert none_with_rules.status_code == 422
    assert none_with_rules.json()["error"] == "rule_source_conflict"

    mock_without_info = _post(client, {"context": BENIGN, "rule_source": "mock_fixture"})
    assert mock_without_info.status_code == 422
    assert mock_without_info.json()["error"] == "rule_info_required"

    mock_empty = _post(
        client,
        {
            "context": BENIGN,
            "rule_source": "verified",
            "rule_info": {"event_id": "evt-benign", "rules": []},
        },
    )
    assert mock_empty.status_code == 422
    assert mock_empty.json()["error"] == "rule_source_conflict"


def test_event_id_mismatch_is_rejected_without_leaking(client):
    response = _post(
        client,
        {
            "context": _suspicious("evt-a"),
            "rule_source": "mock_fixture",
            "rule_info": _rules("evt-b"),
        },
    )
    assert response.status_code == 422
    assert response.json()["error"] == "event_id_mismatch"
    for secret in (SECRET_TOKEN, SECRET_PASSWORD, SECRET_EMAIL, SECRET_IP):
        assert secret not in response.text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["metadata"].pop("path"),
        lambda c: c.pop("event_id"),
        lambda c: c.update(threat_score=5.0),
        lambda c: c.update(payload="not-a-mapping"),
        lambda c: c["metadata"].update(headers={"x": 1}),
    ],
)
def test_context_contract_violations_return_422(client, mutate):
    context = _suspicious()
    mutate(context)
    response = _post(client, {"context": context, "rule_source": "none"})
    assert response.status_code == 422
    assert response.json()["error"] == "input_contract_violation"
    assert SECRET_PASSWORD not in response.text
    assert SECRET_TOKEN not in response.text


def test_rule_contract_violation_returns_422(client):
    response = _post(
        client,
        {
            "context": BENIGN,
            "rule_source": "mock_fixture",
            "rule_info": {"event_id": "evt-benign", "rules": [{"severity": "high"}]},
        },
    )
    assert response.status_code == 422
    assert response.json()["error"] == "input_contract_violation"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"rule_source": "none"},
        {"context": BENIGN},
        {"context": BENIGN, "rule_source": "definitely-genuine"},
        {"context": "nope", "rule_source": "none"},
    ],
)
def test_malformed_requests_return_sanitized_422(client, body):
    response = _post(client, body)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert "input" not in response.json()


def test_validation_errors_do_not_echo_request_content(client):
    response = _post(
        client,
        {"context": BENIGN, "rule_source": "none", "extra": SECRET_PASSWORD},
    )
    assert response.status_code == 422
    assert SECRET_PASSWORD not in response.text

    bad_source = _post(client, {"context": BENIGN, "rule_source": SECRET_PASSWORD})
    assert bad_source.status_code == 422
    assert SECRET_PASSWORD not in bad_source.text


def test_invalid_json_returns_422_without_echo(client):
    response = client.post(
        "/v1/analyze",
        content=b'{"context": {"password": "' + SECRET_PASSWORD.encode() + b'"',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert SECRET_PASSWORD not in response.text


def test_oversized_and_unbounded_bodies_are_rejected(client):
    too_big = {"context": BENIGN, "rule_source": "none", "pad": "x" * (MAX_BODY_BYTES + 1)}
    assert _post(client, too_big).status_code == 413

    chunked = client.post(
        "/v1/analyze",
        content=iter([json.dumps({"context": BENIGN, "rule_source": "none"}).encode()]),
        headers={"content-type": "application/json"},
    )
    assert chunked.status_code == 411


def test_response_data_is_bounded(client):
    context = {
        "event_id": "evt-big",
        "metadata": {"method": "POST", "path": "/bulk"},
        "payload": {
            "blob": "y" * 5000,
            "items": [{"k": i} for i in range(400)],
            **{f"field_{i}": f"value-{i}" for i in range(200)},
        },
        "threat_score": 0.2,
    }
    response = _post(client, {"context": context, "rule_source": "none"})
    assert response.status_code == 200
    body = response.json()
    assert body["tokens"]["total"] > MAX_TOKENS_RETURNED
    assert body["tokens"]["returned"] == MAX_TOKENS_RETURNED
    assert body["tokens"]["truncated"] is True
    assert body["masked_representation"]["truncated"] is True
    payload = body["masked_representation"]["payload"]
    assert len(payload) <= 50
    # "blob" is unknown to the vocabulary: neither the key nor its 5000 chars appear.
    assert "blob" not in payload
    assert "y" * 20 not in json.dumps(body)
    assert body["tensor"]["payload_truncated"] is True


def test_masked_field_list_is_bounded(client):
    payload = {f"password_{i}": f"s{i}" for i in range(5)}
    payload.update({f"user_{i}": {"email": f"u{i}@example.com"} for i in range(80)})
    context = {
        "event_id": "evt-many",
        "metadata": {"method": "POST", "path": "/many"},
        "payload": payload,
    }
    response = _post(client, {"context": context, "rule_source": "none"})
    assert response.status_code == 200
    masking = response.json()["masking"]
    assert masking["masked_field_count"] == 80
    assert len(masking["masked_fields"]) == MAX_MASKED_FIELDS_RETURNED
    assert masking["masked_fields_truncated"] is True


def test_tensor_capacity_failure_returns_422(client):
    rules = {
        "event_id": "evt-benign",
        "rules": [{"rule_id": f"MOCK-R-{i}"} for i in range(200)],
    }
    response = _post(
        client, {"context": BENIGN, "rule_source": "mock_fixture", "rule_info": rules}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "pipeline_input_rejected"


def test_tensor_metadata_exposes_no_token_ids(client):
    body = _post(client, {"context": BENIGN, "rule_source": "none"}).json()
    tensor = body["tensor"]
    assert set(tensor) == {
        "max_length",
        "token_count_before_truncation",
        "tokens_in_tensor",
        "padded_positions",
        "payload_truncated",
        "unknown_token_count",
        "vocabulary_size",
    }
    assert tensor["tokens_in_tensor"] + tensor["padded_positions"] == tensor["max_length"]
    assert "input_ids" not in json.dumps(body)


def test_unready_model_returns_503(service):
    client = TestClient(create_app(service))
    assert client.get("/health").json() == {
        "status": "ok",
        "service": "tokenization_ai",
        "ready": True,
    }
    service.model = LightweightAnomalyModel()  # untrained
    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["ready"] is False
    response = _post(client, {"context": BENIGN, "rule_source": "none"})
    assert response.status_code == 503
    assert response.json()["error"] == "model_not_ready"


def test_unfitted_vocabulary_returns_503(service):
    client = TestClient(create_app(service))
    service.tensorizer.vocabulary._reset_to_reserved()  # simulate lost fit state
    service.tensorizer.vocabulary._fitted = False
    response = _post(client, {"context": BENIGN, "rule_source": "none"})
    assert response.status_code == 503


def test_unexpected_error_is_generic_and_not_logged_with_content(service, caplog):
    def boom(*_args, **_kwargs):
        raise Exception(f"leak {SECRET_PASSWORD}")

    service.analyze = boom
    client = TestClient(create_app(service), raise_server_exceptions=False)
    with caplog.at_level("DEBUG"):
        response = _post(client, {"context": _suspicious(), "rule_source": "none"})
    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"
    assert SECRET_PASSWORD not in response.text
    assert SECRET_PASSWORD not in caplog.text


def test_response_contract_is_stable(client):
    body = _post(
        client,
        {
            "context": BENIGN,
            "rule_source": "mock_fixture",
            "rule_info": _rules("evt-benign"),
        },
    ).json()
    assert set(body) == {
        "contract_version",
        "event_id",
        "provenance",
        "privacy",
        "masking",
        "masked_representation",
        "tokens",
        "tensor",
        "prediction",
        "timings",
    }
    assert body["contract_version"] == "member4-analysis-v2"
    assert set(body["privacy"]) == {
        "output_kind",
        "policy",
        "hidden_in_representation",
        "hidden_in_tokens",
        "note",
    }
    assert set(body["provenance"]) == {
        "input_origin",
        "rule_status",
        "model",
        "detection_accuracy_claim",
    }
    assert set(body["provenance"]["rule_status"]) == {
        "source",
        "label",
        "rule_count",
        "rule_ids",
        "rule_ids_truncated",
        "verification_performed_by_this_service",
        "note",
    }
    assert set(body["masking"]) == {
        "masked_field_count",
        "masked_fields",
        "masked_fields_truncated",
    }
    assert set(body["masked_representation"]) == {
        "method",
        "path",
        "headers",
        "payload",
        "truncated",
    }
    assert set(body["tokens"]) == {"total", "returned", "truncated", "items"}
    assert set(body["tokens"]["items"][0]) == {"token", "role"}
    assert set(body["prediction"]) == {"score", "model_version"}
    assert set(body["timings"]) == {"total_elapsed_ms", "stages"}
    assert set(body["timings"]["stages"][0]) == {"stage", "elapsed_ms", "success", "summary"}
