"""Member 1 -> Member 4 data-contract integration tests.

What this proves: a ``RawEvent`` that Member 1's real ``ingestion`` app
forwards downstream (interception path, real ``PIIMasker`` and broker) is
accepted by Member 4's real adapter, pipeline and HTTP API.

What this does NOT prove: a running multi-service deployment. Everything
runs in one process via ``TestClient`` with Member 1's own in-memory
broker and a capturing stand-in for the processing client (the same
pattern Member 1's tests use). There is no network hop, no Redis, no
Member 2 verified-rule output (rules here are absent or MOCK) and no
Member 3 handoff. All data is synthetic.
"""

import time

import pytest
from fastapi.testclient import TestClient

from ingestion.app import create_app as create_ingestion_app
from ingestion.broker import InMemoryBroker
from ingestion.pii_masker import REDACTED
from shared_contracts.interfaces import ProcessingClient
from shared_contracts.models import Label, Prediction, RawEvent
from tokenization_ai.adapters import (
    HTTP_METADATA_KEY,
    InputContractError,
    THREAT_SCORE_KEY,
    to_unified_api_context,
    to_verified_rule_info,
)
from tokenization_ai.app import create_app as create_member4_app
from tokenization_ai.demo import _build_demo_service

BEARER = "synthetic-bearer-M1-SECRET-111"
PASSWORD = "synthetic-password-M1-SECRET-222"
EMAIL = "synthetic.person@example.com"
UNKNOWN_FIELD_SECRET = "SENTINEL_M1_unknown_field_333"

INTERCEPTION = {
    "metadata": {
        "method": "POST",
        "path": "/orders",
        "content_type": "application/json",
        "headers": {"Authorization": f"Bearer {BEARER}"},
    },
    "payload": {
        "username": "admin",
        "password": PASSWORD,
        "contact": EMAIL,
        "free_text": UNKNOWN_FIELD_SECRET,
        "amount": 149.95,
    },
    "threat_score": 0.7,
}


class CapturingProcessing(ProcessingClient):
    """Records the event Member 1 forwards downstream (as Member 1's own tests do)."""

    def __init__(self) -> None:
        self.events: list[RawEvent] = []

    def process(self, event: RawEvent) -> Prediction:
        self.events.append(event)
        return Prediction(
            event_id=event.event_id,
            score=0.1,
            label=Label.NORMAL,
            threshold=0.8,
            model_version="m1-capture",
        )


def _wait_for(processing: CapturingProcessing, count: int = 1) -> None:
    deadline = time.monotonic() + 3.0
    while len(processing.events) < count and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(processing.events) >= count, "Member 1 did not forward an event downstream"


@pytest.fixture
def ingestion():
    processing = CapturingProcessing()
    client = TestClient(create_ingestion_app(processing=processing, broker=InMemoryBroker()))
    return client, processing


@pytest.fixture
def forwarded(ingestion):
    """(acknowledgement JSON, the real RawEvent Member 1 forwarded downstream)."""
    client, processing = ingestion
    response = client.post("/v1/ingest", json=INTERCEPTION)
    assert response.status_code == 202
    _wait_for(processing)
    return response.json(), processing.events[0]


@pytest.fixture(scope="module")
def member4_service():
    return _build_demo_service()


def test_member1_output_has_the_shape_member4_adapter_expects(forwarded):
    ack, event = forwarded
    assert isinstance(event, RawEvent)
    assert event.event_id == ack["event_id"]
    assert HTTP_METADATA_KEY in event.payload and THREAT_SCORE_KEY in event.payload
    # Member 1's own masking already ran on the payload.
    assert event.payload["password"] == REDACTED
    assert event.payload["contact"] == REDACTED
    assert PASSWORD not in event.model_dump_json()
    assert EMAIL not in event.model_dump_json()


def test_adapter_accepts_member1_event_and_preserves_identity(forwarded):
    _ack, event = forwarded
    context = to_unified_api_context(event)
    assert context.event_id == event.event_id
    assert context.method == "POST" and context.path == "/orders"
    assert context.threat_score == pytest.approx(0.7)
    assert HTTP_METADATA_KEY not in context.payload
    assert THREAT_SCORE_KEY not in context.payload
    assert context.payload["amount"] == 149.95


def test_member4_service_processes_member1_event(forwarded, member4_service):
    _ack, event = forwarded
    context = to_unified_api_context(event)
    rule_info = to_verified_rule_info({"event_id": event.event_id, "rules": []})

    result = member4_service.analyze(context, rule_info)

    assert result.prediction.event_id == event.event_id
    assert 0.0 <= result.prediction.score <= 1.0
    # Member 4 masks again by itself: it does not rely on Member 1 having done so.
    assert ("headers", "authorization") in result.masked_paths
    assert ("payload", "password") in result.masked_paths
    serialized = result.sequence.model_dump_json()
    for secret in (BEARER, PASSWORD, EMAIL):
        assert secret not in serialized


def test_member1_redaction_marker_is_handled_as_an_ordinary_literal(forwarded, member4_service):
    """Characterization: Member 4 does not special-case Member 1's marker.

    ``[REDACTED_PII]`` under an unrecognized key (``contact`` here) reaches
    Member 4 as plain text. It is safe (the original value is gone) but it is
    tokenized with role ``literal_api``, not ``masked``, because Member 4's
    masking provenance is path-based and never inferred from text.
    """
    _ack, event = forwarded
    result = member4_service.analyze(
        to_unified_api_context(event),
        to_verified_rule_info({"event_id": event.event_id, "rules": []}),
    )
    pairs = list(zip(result.sequence.tokens, result.sequence.roles))
    assert (REDACTED, "literal_api") in pairs
    assert (REDACTED, "masked") not in pairs


def test_member4_api_accepts_member1_event_and_stays_privacy_safe(forwarded, member4_service):
    _ack, event = forwarded
    client = TestClient(create_member4_app(member4_service))

    response = client.post(
        "/v1/analyze",
        json={"context": event.model_dump(mode="json"), "rule_source": "none"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["contract_version"] == "member4-analysis-v2"
    assert body["event_id"] == event.event_id
    assert body["provenance"]["rule_status"]["source"] == "none"
    assert body["privacy"]["output_kind"] == "privacy_safe_projection"
    assert 0.0 <= body["prediction"]["score"] <= 1.0
    for secret in (BEARER, PASSWORD, EMAIL, UNKNOWN_FIELD_SECRET):
        assert secret not in response.text
    rep = body["masked_representation"]
    assert rep["headers"]["authorization"] == "[MASKED_AUTH]"
    assert rep["payload"]["password"] == "[MASKED_SECRET]"
    assert "authorization" in {f.split(".")[-1] for f in body["masking"]["masked_fields"]}


def test_mock_rules_with_matching_event_id_are_labelled_mock(forwarded, member4_service):
    _ack, event = forwarded
    client = TestClient(create_member4_app(member4_service))
    response = client.post(
        "/v1/analyze",
        json={
            "context": event.model_dump(mode="json"),
            "rule_source": "mock_fixture",
            "rule_info": {"event_id": event.event_id, "rules": [{"rule_id": "MOCK-R-SQLI"}]},
        },
    )
    assert response.status_code == 200
    status = response.json()["provenance"]["rule_status"]
    assert status["source"] == "mock_fixture"
    assert status["verification_performed_by_this_service"] is False


def test_rule_info_for_a_different_event_is_rejected(forwarded, member4_service):
    _ack, event = forwarded
    client = TestClient(create_member4_app(member4_service))
    response = client.post(
        "/v1/analyze",
        json={
            "context": event.model_dump(mode="json"),
            "rule_source": "mock_fixture",
            "rule_info": {"event_id": "a-different-event", "rules": [{"rule_id": "MOCK-R-1"}]},
        },
    )
    assert response.status_code == 422
    assert response.json()["error"] == "event_id_mismatch"
    assert BEARER not in response.text


def test_member1_rejects_invalid_interception_so_nothing_reaches_member4(ingestion):
    client, processing = ingestion
    bad = {**INTERCEPTION, "threat_score": 5.0}
    response = client.post("/v1/ingest", json=bad)
    assert response.status_code == 422
    time.sleep(0.1)
    assert processing.events == []


def test_non_interception_member1_event_is_not_consumable_by_member4(ingestion, member4_service):
    """Member 1's standard (non-HTTP) path carries no HTTP metadata.

    This is a real contract limit between the two components: Member 4
    needs method/path, which only the interception path provides.
    """
    client, processing = ingestion
    response = client.post("/v1/ingest", json={"source": "sensor-1", "payload": {"temp": 0.95}})
    assert response.status_code == 202
    _wait_for(processing)
    event = processing.events[0]
    assert HTTP_METADATA_KEY not in event.payload

    with pytest.raises(InputContractError):
        to_unified_api_context(event)

    api = TestClient(create_member4_app(member4_service))
    rejected = api.post(
        "/v1/analyze",
        json={"context": event.model_dump(mode="json"), "rule_source": "none"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"] == "input_contract_violation"
