"""Unit tests for the Member 4 input adapters."""

import json
from datetime import datetime, timezone

import pytest

from shared_contracts.models import RawEvent
from tokenization_ai.adapters import (
    HTTP_METADATA_KEY,
    THREAT_SCORE_KEY,
    InputContractError,
    to_unified_api_context,
    to_verified_rule_info,
)

METADATA = {
    "method": "POST",
    "path": "/orders",
    "content_type": "application/json",
    "client_ip": None,
    "headers": {"x-request-id": "abc"},
}


def intercepted_event(**payload_overrides: object) -> RawEvent:
    """Build a RawEvent the way Member 1 publishes an intercepted request."""
    payload = {
        "email": "[REDACTED_PII]",
        "amount": 149.95,
        HTTP_METADATA_KEY: dict(METADATA),
        THREAT_SCORE_KEY: 0.88,
    }
    payload.update(payload_overrides)
    return RawEvent(event_id="evt-1", source="/orders", payload=payload)


def test_context_from_raw_event_separates_metadata_from_body() -> None:
    context = to_unified_api_context(intercepted_event())

    assert context.event_id == "evt-1"
    assert context.method == "POST"
    assert context.path == "/orders"
    assert context.content_type == "application/json"
    assert context.client_ip is None
    assert context.headers == {"x-request-id": "abc"}
    assert context.threat_score == 0.88
    assert context.payload == {"email": "[REDACTED_PII]", "amount": 149.95}


def test_context_preserves_the_raw_event_timestamp() -> None:
    event = intercepted_event()

    assert to_unified_api_context(event).timestamp == event.timestamp


def test_context_timestamp_is_optional() -> None:
    event = RawEvent(
        event_id="evt-1",
        source="/orders",
        payload={"email": "x", HTTP_METADATA_KEY: dict(METADATA)},
    )
    data = event.model_dump()
    del data["timestamp"]

    assert to_unified_api_context(data).timestamp is None


def test_context_from_envelope_preserves_timestamp() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    context = to_unified_api_context(
        {
            "event_id": "evt-2",
            "metadata": {"method": "GET", "path": "/items"},
            "payload": {},
            "timestamp": timestamp,
        }
    )

    assert context.timestamp == timestamp


def test_context_from_raw_event_dict_matches_raw_event() -> None:
    event = intercepted_event()

    assert to_unified_api_context(event.model_dump()) == to_unified_api_context(event)


def test_context_from_interception_envelope() -> None:
    context = to_unified_api_context(
        {
            "event_id": "evt-2",
            "metadata": {"method": "GET", "path": "/items"},
            "payload": {"page": 2},
            "threat_score": 0.1,
        }
    )

    assert context.event_id == "evt-2"
    assert context.method == "GET"
    assert context.path == "/items"
    assert context.content_type is None
    assert context.headers == {}
    assert context.payload == {"page": 2}
    assert context.threat_score == 0.1


def test_context_threat_score_is_optional() -> None:
    event = intercepted_event()
    del event.payload[THREAT_SCORE_KEY]

    assert to_unified_api_context(event).threat_score is None


def test_context_does_not_mutate_the_source_event() -> None:
    event = intercepted_event()
    before = event.model_dump()

    to_unified_api_context(event)

    assert event.model_dump() == before


def test_context_rejects_event_without_http_metadata() -> None:
    event = RawEvent(event_id="evt-1", source="sensor", payload={"amount": 1.0})

    with pytest.raises(InputContractError, match=HTTP_METADATA_KEY):
        to_unified_api_context(event)


@pytest.mark.parametrize("field", ["method", "path"])
def test_context_rejects_metadata_missing_required_field(field: str) -> None:
    metadata = dict(METADATA)
    del metadata[field]

    with pytest.raises(InputContractError, match=field):
        to_unified_api_context(intercepted_event(**{HTTP_METADATA_KEY: metadata}))


@pytest.mark.parametrize("field", ["event_id", "payload"])
def test_context_rejects_event_dict_missing_required_field(field: str) -> None:
    data = intercepted_event().model_dump()
    del data[field]

    with pytest.raises(InputContractError, match=field):
        to_unified_api_context(data)


@pytest.mark.parametrize("field", ["event_id", "payload"])
def test_context_rejects_envelope_missing_required_field(field: str) -> None:
    envelope = {
        "event_id": "evt-2",
        "metadata": {"method": "GET", "path": "/items"},
        "payload": {},
    }
    del envelope[field]

    with pytest.raises(InputContractError, match=field):
        to_unified_api_context(envelope)


def test_context_rejects_empty_method() -> None:
    metadata = {**METADATA, "method": ""}

    with pytest.raises(InputContractError, match="method"):
        to_unified_api_context(intercepted_event(**{HTTP_METADATA_KEY: metadata}))


def test_context_rejects_out_of_range_threat_score() -> None:
    with pytest.raises(InputContractError, match="threat_score"):
        to_unified_api_context(intercepted_event(**{THREAT_SCORE_KEY: 1.5}))


def test_context_rejects_unsupported_source_type() -> None:
    with pytest.raises(InputContractError):
        to_unified_api_context(["not", "an", "event"])  # type: ignore[arg-type]


def test_rule_info_from_mapping_keeps_extra_keys_as_attributes() -> None:
    info = to_verified_rule_info(
        {
            "event_id": "evt-1",
            "rules": [
                {"rule_id": "R-1", "field": "amount", "operator": "<=", "value": 500},
                {"rule_id": "R-2"},
            ],
        }
    )

    assert info.event_id == "evt-1"
    assert [rule.rule_id for rule in info.rules] == ["R-1", "R-2"]
    assert info.rules[0].attributes == {"field": "amount", "operator": "<=", "value": 500}
    assert info.rules[1].attributes == {}


def test_rule_info_from_json_matches_mapping() -> None:
    data = {"event_id": "evt-1", "rules": [{"rule_id": "R-1", "field": "amount"}]}
    expected = to_verified_rule_info(data)

    assert to_verified_rule_info(json.dumps(data)) == expected
    assert to_verified_rule_info(json.dumps(data).encode()) == expected


def test_rule_info_accepts_empty_rule_list() -> None:
    info = to_verified_rule_info({"event_id": "evt-1", "rules": []})

    assert info.rules == []


@pytest.mark.parametrize("field", ["event_id", "rules"])
def test_rule_info_rejects_missing_required_field(field: str) -> None:
    data = {"event_id": "evt-1", "rules": []}
    del data[field]

    with pytest.raises(InputContractError, match=field):
        to_verified_rule_info(data)


def test_rule_info_rejects_rule_without_rule_id() -> None:
    data = {"event_id": "evt-1", "rules": [{"rule_id": "R-1"}, {"field": "amount"}]}

    with pytest.raises(InputContractError, match=r"rules\[1\].*rule_id"):
        to_verified_rule_info(data)


def test_rule_info_rejects_non_list_rules() -> None:
    with pytest.raises(InputContractError, match="rules"):
        to_verified_rule_info({"event_id": "evt-1", "rules": {"rule_id": "R-1"}})


def test_rule_info_rejects_non_mapping_rule() -> None:
    with pytest.raises(InputContractError, match=r"rules\[0\]"):
        to_verified_rule_info({"event_id": "evt-1", "rules": ["R-1"]})


def test_rule_info_rejects_invalid_json() -> None:
    with pytest.raises(InputContractError, match="JSON"):
        to_verified_rule_info("{not json")


def test_rule_info_rejects_json_that_is_not_an_object() -> None:
    with pytest.raises(InputContractError):
        to_verified_rule_info("[1, 2, 3]")
