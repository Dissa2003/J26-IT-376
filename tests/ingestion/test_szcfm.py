"""SZ-CFM pipeline tests: schema, masking, protobuf, features, ring, inference, fusion, policy."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ingestion.app import create_app
from ingestion.broker import InMemoryBroker
from ingestion.feature_schema import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureExtractor,
    scan_payload,
    validate_vector,
)
from ingestion.inference import DevelopmentHeuristicBackend, InferenceStatus
from ingestion.pii_masker import PIIMasker, REDACTED
from ingestion.request_state import (
    DeliveryOutcome,
    RequestStateRing,
    RingFullError,
    SlotState,
    StaleTicketError,
)
from ingestion.szcfm import (
    CorrelationConflictError,
    EventValidationError,
    MalformedRequestError,
    PayloadTooLargeError,
    SZCFMPipeline,
)
from ingestion.szcfm_policy import SZCFMConfig, context_risk, fuse
from ingestion.szcfm_proto import ENVELOPE_MESSAGE_TYPE, SZCFMEnvelope
from ingestion.szcfm_schema import SZCFMEvent

FIXTURE = Path(__file__).parent / "fixtures" / "szcfm_example_event.json"
PRIOR_INDEX = FEATURE_NAMES.index("prior_threat_score")


def example() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def raw(data: dict) -> bytes:
    return json.dumps(data).encode()


# --- test backends (isolate the decision path from model behaviour) -----------


class FixedBackend:
    """Test double returning a fixed score; used only to drive policy branches."""

    model_version = "test-fixed"
    is_simulated = True
    feature_schema_version = "1"
    input_dim = FEATURE_COUNT

    def __init__(self, score: float, delay_s: float = 0.0) -> None:
        self.score, self.delay_s = score, delay_s

    def predict(self, vector):
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.score


class EchoPriorBackend(FixedBackend):
    """Returns the request's own prior_threat_score, so a mismatched result is detectable."""

    def __init__(self, jitter_s: float = 0.0) -> None:
        super().__init__(0.0)
        self.jitter_s = jitter_s

    def predict(self, vector):
        if self.jitter_s:
            time.sleep(self.jitter_s * ((vector[PRIOR_INDEX] * 997) % 1))
        return vector[PRIOR_INDEX]


class GatedBackend(FixedBackend):
    """Blocks inside predict until released; used for timeout/backpressure tests."""

    def __init__(self, score: float = 0.1) -> None:
        super().__init__(score)
        self.gate = threading.Event()
        self.started = threading.Semaphore(0)

    def predict(self, vector):
        self.started.release()
        self.gate.wait(5)
        return vector[PRIOR_INDEX] if vector[PRIOR_INDEX] else self.score


class FailingBackend(FixedBackend):
    def predict(self, vector):
        raise RuntimeError("model server unavailable")


class NaNBackend(FixedBackend):
    def predict(self, vector):
        return float("nan")


def pipeline(backend=None, **config) -> SZCFMPipeline:
    return SZCFMPipeline(SZCFMConfig(**config), backend=backend)


def run(p: SZCFMPipeline, data: dict, **kwargs):
    return p.process(p.parse(raw(data)), **kwargs)


# --- 1-4: schema ----------------------------------------------------------------


def test_valid_example_event_is_accepted_and_source_resolved_from_header() -> None:
    event = SZCFMEvent.model_validate(example())
    assert event.source == "web"
    assert event.payload.request_id == "req-20261022-000184"
    assert event.payload.traffic_context.current_request_size_bytes == 1842


def test_root_source_takes_precedence_over_header() -> None:
    data = example()
    data["source"] = "mobile-app"
    assert SZCFMEvent.model_validate(data).source == "mobile-app"


def test_missing_source_is_rejected_with_actionable_error() -> None:
    data = example()
    del data["metadata"]["headers"]["x-request-source"]
    p = pipeline()
    with pytest.raises(EventValidationError) as info:
        p.parse(raw(data))
    messages = " ".join(e["msg"] for e in info.value.errors)
    assert "source is required" in messages
    assert "x-request-source" in messages


def test_ingest_endpoint_no_longer_fails_with_rawevent_source_error() -> None:
    """The envelope that used to fail on RawEvent.source is now routed to SZ-CFM."""
    client = TestClient(create_app(broker=InMemoryBroker()))
    data = example()
    del data["threat_score"]
    ok = client.post("/v1/ingest", json=data)
    assert ok.status_code == 202
    assert ok.json()["security_decision"]["decision"] == "ALLOW"

    del data["metadata"]["headers"]["x-request-source"]
    bad = client.post("/v1/ingest", json=data)
    assert bad.status_code == 422
    assert "source is required" in json.dumps(bad.json())


@pytest.mark.parametrize(
    ("mutate", "loc"),
    [
        (lambda d: d["payload"]["traffic_context"].__setitem__("requests_per_minute", "many"), "requests_per_minute"),
        (lambda d: d["payload"]["authentication"].__setitem__("roles", "customer"), "roles"),
        (lambda d: d["payload"]["security_context"].__setitem__("ip_reputation_score", 1.5), "ip_reputation_score"),
        (lambda d: d["payload"]["client"].__setitem__("country", "Sri Lanka"), "country"),
        (lambda d: d["payload"]["traffic_context"].__setitem__("failed_logins_last_5_minutes", -1), "failed_logins_last_5_minutes"),
        (lambda d: d["metadata"].__setitem__("method", "BREW"), "method"),
        (lambda d: d["payload"].__setitem__("unexpected_block", {}), "unexpected_block"),
        (lambda d: d.__setitem__("extra_root", 1), "extra_root"),
    ],
)
def test_invalid_nested_types_and_unknown_fields_are_rejected(mutate, loc) -> None:
    data = example()
    mutate(data)
    with pytest.raises(EventValidationError) as info:
        pipeline().parse(raw(data))
    assert any(loc in e["loc"] for e in info.value.errors), info.value.errors


def test_validation_errors_do_not_echo_input_values() -> None:
    data = example()
    data["payload"]["authentication"]["roles"] = "secret-role-value-xyz"
    with pytest.raises(EventValidationError) as info:
        pipeline().parse(raw(data))
    assert "secret-role-value-xyz" not in json.dumps(info.value.errors)


def test_unsupported_schema_version_is_rejected() -> None:
    data = example()
    data["schema_version"] = "2"
    with pytest.raises(EventValidationError) as info:
        pipeline().parse(raw(data))
    assert "unsupported schema_version" in json.dumps(info.value.errors)
    data["schema_version"] = "1"
    assert pipeline().parse(raw(data)).schema_version == "1"


def test_malformed_json_and_non_finite_numbers_are_rejected() -> None:
    p = pipeline()
    with pytest.raises(MalformedRequestError):
        p.parse(b"{not json")
    with pytest.raises(MalformedRequestError):
        p.parse(b"[1, 2]")
    text = json.dumps(example()).replace('"threat_score": 0.12', '"threat_score": NaN')
    with pytest.raises(MalformedRequestError):
        p.parse(text.encode())


# --- 5: masking ---------------------------------------------------------------------


def test_recursive_masking_of_nested_objects_and_arrays() -> None:
    payload = {
        "orders": [
            {"items": [{"product_name": "Wireless Mouse", "card_number": "4111 1111 1111 1111"}]},
            {"customer": {"full_name": "Jane Doe", "contacts": [{"email": "jane@example.com"}]}},
        ],
        "auth": {"nested": {"password": "pw"}},
        "session_id": "sess-1",
        "headers": [{"cookie": "c=1"}, {"x-api-key": "k"}],
        "values": ["192.0.2.1", "plain text"],
    }
    original = copy.deepcopy(payload)
    masked = PIIMasker().mask(payload)
    assert masked["orders"][0]["items"][0] == {"product_name": "Wireless Mouse", "card_number": REDACTED}
    assert masked["orders"][1]["customer"]["full_name"] == REDACTED
    assert masked["orders"][1]["customer"]["contacts"][0]["email"] == REDACTED
    assert masked["auth"] == REDACTED
    assert masked["session_id"] == REDACTED
    assert masked["headers"] == [{"cookie": REDACTED}, {"x-api-key": REDACTED}]
    assert masked["values"] == [REDACTED, "plain text"]
    assert payload == original


def test_masking_rules_are_configurable() -> None:
    masker = PIIMasker(extra_pii_keys={"customer_id"}, extra_credential_keys={"otp"})
    masked = masker.mask({"customer_id": "C-1", "otp": "123456", "order_id": "O-1"})
    assert masked == {"customer_id": REDACTED, "otp": REDACTED, "order_id": "O-1"}
    assert PIIMasker().mask({"customer_id": "C-1"}) == {"customer_id": "C-1"}


def test_pipeline_masks_payload_and_drops_raw_authorization() -> None:
    data = example()
    data["metadata"]["authorization"] = "Bearer eyJhbGciOi.SECRET-TOKEN.sig"
    sink: list[bytes] = []
    p = SZCFMPipeline(SZCFMConfig(), envelope_sink=lambda view, ticket: sink.append(bytes(view)))
    result = run(p, data)
    body = result.model_dump_json()
    assert "SECRET-TOKEN" not in body
    assert "sess-a83f29" not in body
    assert b"SECRET-TOKEN" not in sink[0]
    assert b"sess-a83f29" not in sink[0]
    assert result.masked_payload["request_body"]["items"][0]["product_name"] == "Wireless Keyboard"


# --- 6-7: protobuf ------------------------------------------------------------------


def test_protobuf_envelope_round_trips_from_slot_bytes() -> None:
    captured: list[bytes] = []
    p = SZCFMPipeline(SZCFMConfig(), envelope_sink=lambda view, ticket: captured.append(bytes(view)))
    result = run(p, example())

    wire = captured[0]
    decoded = SZCFMEnvelope()
    decoded.ParseFromString(wire)
    assert result.serialization.success is True
    assert result.serialization.message_type == ENVELOPE_MESSAGE_TYPE
    assert decoded.request_id == "req-20261022-000184"
    assert decoded.correlation_id == "corr-7f83a91b2c"
    assert decoded.event_id == result.event_id
    assert decoded.source == "web"
    assert decoded.metadata.method == "POST"
    assert decoded.metadata.authorization_present is True
    assert decoded.metadata.authorization_scheme == "Bearer"
    assert decoded.metadata.headers["x-api-version"] == "v1"
    assert decoded.masked_payload["request_body"]["items"][1]["product_name"] == "Wireless Mouse"
    assert decoded.masked_payload["authentication"]["session_id"] == REDACTED
    assert list(decoded.feature_vector) == pytest.approx(result.feature_extraction.vector, abs=1e-6)
    assert decoded.has_prior_threat_score and decoded.prior_threat_score == 0.12
    assert decoded.SerializeToString(deterministic=True) == wire


def test_serialized_size_is_measured_from_real_bytes() -> None:
    captured: list[bytes] = []
    p = SZCFMPipeline(SZCFMConfig(), envelope_sink=lambda view, ticket: captured.append(bytes(view)))
    small = run(p, example())
    big_data = example()
    big_data["payload"]["request_body"]["note"] = "x" * 500
    big = run(p, big_data)

    assert small.serialization.bytes_size == len(captured[0])
    assert small.serialization.sha256 == hashlib.sha256(captured[0]).hexdigest()
    assert big.serialization.bytes_size == len(captured[1])
    assert big.serialization.bytes_size - small.serialization.bytes_size >= 500


def test_envelope_too_large_for_slot_is_reported_not_faked() -> None:
    p = pipeline(FixedBackend(0.1), ring_slot_bytes=256)
    result = run(p, example())
    assert result.serialization.success is False
    assert result.serialization.bytes_size == 0
    assert "exceeds slot capacity" in result.serialization.error
    assert result.security_decision.decision == "CHALLENGE"
    assert "ENVELOPE_SERIALIZATION_FAILED" in result.security_decision.reason_codes


def test_runtime_descriptor_matches_proto_file() -> None:
    proto = (Path(__file__).parents[2] / "ingestion" / "szcfm_context.proto").read_text()
    for message in SZCFMEnvelope.DESCRIPTOR.file.message_types_by_name.values():
        block = re.search(r"message %s \{(.*?)\n\}" % message.name, proto, re.S).group(1)
        declared = {name: int(num) for name, num in re.findall(r"(\w+) = (\d+);", block)}
        assert declared == {f.name: f.number for f in message.fields}, message.name


# --- 8-9: features ------------------------------------------------------------------


def test_feature_vector_order_and_dimension_are_stable() -> None:
    data = example()
    reordered = example()
    reordered["payload"] = dict(reversed(list(reordered["payload"].items())))
    reordered["payload"]["request_body"] = dict(reversed(list(reordered["payload"]["request_body"].items())))
    reordered["payload"]["traffic_context"] = dict(reversed(list(reordered["payload"]["traffic_context"].items())))
    extractor = FeatureExtractor(32_768)
    a = extractor.extract(SZCFMEvent.model_validate(data))
    b = extractor.extract(SZCFMEvent.model_validate(reordered))
    assert a.values == b.values
    assert len(a.values) == FEATURE_COUNT == len(FEATURE_NAMES) == 28
    assert a.names[0] == "ip_reputation_score" and a.names[-1] == "prior_threat_score_missing"
    assert a.as_dict()["ip_reputation_score"] == 0.12
    assert a.as_dict()["mfa_verified"] == 1.0
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in a.values)


def test_missing_context_blocks_use_explicit_defaults_and_indicators() -> None:
    data = example()
    for block in ("traffic_context", "security_context", "authentication"):
        del data["payload"][block]
    del data["threat_score"]
    result = FeatureExtractor(32_768).extract(SZCFMEvent.model_validate(data))
    features = result.as_dict()
    assert features["ip_reputation_score"] == 0.5
    assert features["security_context_missing"] == 1.0
    assert features["traffic_context_missing"] == 1.0
    assert features["authentication_missing"] == 1.0
    assert features["prior_threat_score_missing"] == 1.0
    assert features["requests_per_minute"] == 0.0
    assert set(result.missing_groups) == {"traffic_context", "security_context", "authentication", "prior_threat_score"}
    validate_vector(result.values)


def test_invalid_feature_vectors_are_rejected() -> None:
    good = [0.0] * FEATURE_COUNT
    validate_vector(good)
    with pytest.raises(ValueError, match="expected 28"):
        validate_vector(good[:-1])
    for bad in (float("nan"), float("inf"), 1.5, -0.1):
        vector = list(good)
        vector[3] = bad
        with pytest.raises(ValueError, match="method_is_delete"):
            validate_vector(vector)


def test_backend_with_mismatched_input_dimension_is_refused() -> None:
    class WrongDim(FixedBackend):
        input_dim = 5

    with pytest.raises(ValueError, match="5 inputs"):
        SZCFMPipeline(SZCFMConfig(), backend=WrongDim(0.1))


def test_signature_scan_detects_attacks_but_not_ordinary_punctuation() -> None:
    assert scan_payload({"q": "x' OR 1=1 --"}).matched_signatures
    assert "xss_script_tag" in scan_payload({"c": ["<script>alert(1)</script>"]}).matched_signatures
    assert "sqli_union_select" in scan_payload({"nested": {"k": "1 UNION SELECT pw FROM users"}}).matched_signatures
    assert scan_payload({"name": "O'Brien; Main St. (rear)"}).matched_signatures == ()


# --- 10-12, 15-16: request-state ring -------------------------------------------------


def test_ring_full_raises_instead_of_evicting_live_requests() -> None:
    ring = RequestStateRing(capacity=2, slot_bytes=64)
    a = ring.allocate("r1", "c1")
    b = ring.allocate("r2", "c2")
    with pytest.raises(RingFullError):
        ring.allocate("r3", "c3")
    assert ring.state_of(a) is SlotState.ALLOCATED and ring.state_of(b) is SlotState.ALLOCATED
    assert ring.full_rejections == 1
    ring.release(a)
    c = ring.allocate("r3", "c3")
    assert c.slot_index == a.slot_index and c.generation == a.generation + 1


def test_ring_full_returns_503_from_api() -> None:
    app = create_app(broker=InMemoryBroker(), szcfm_config=SZCFMConfig(ring_capacity=1))
    held = app.state.szcfm.ring.allocate("held", "held")
    client = TestClient(app)
    response = client.post("/v1/szcfm/events", json=example())
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    app.state.szcfm.ring.release(held)
    assert client.post("/v1/szcfm/events", json=example()).status_code == 200


def test_stale_slot_is_reclaimed_and_old_ticket_invalidated() -> None:
    ring = RequestStateRing(capacity=1, slot_bytes=64, stale_after_ns=1_000_000)
    old = ring.allocate("r1", "c1")
    time.sleep(0.01)
    new = ring.allocate("r2", "c2")
    assert ring.stale_reclaimed == 1
    assert new.generation == old.generation + 2
    assert ring.release(old) is False
    with pytest.raises(StaleTicketError):
        ring.mark_ai_pending(old)
    assert ring.state_of(new) is SlotState.ALLOCATED


def test_stale_result_after_slot_reuse_cannot_modify_new_request() -> None:
    ring = RequestStateRing(capacity=1, slot_bytes=64)
    first = ring.allocate("req-1", "corr-1")
    ring.mark_ai_pending(first)
    assert ring.wait_for_ai(first, 0.001)[0] is SlotState.AI_UNAVAILABLE  # timed out
    ring.release(first)
    second = ring.allocate("req-2", "corr-2")
    ring.mark_ai_pending(second)
    assert second.slot_index == first.slot_index

    outcome = ring.deliver_ai_result(first.slot_index, first.generation, "req-1", "corr-1", "late")
    assert outcome is DeliveryOutcome.STALE_GENERATION
    assert ring.state_of(second) is SlotState.AI_PENDING

    assert ring.deliver_ai_result(second.slot_index, second.generation, "req-2", "corr-2", "mine") is DeliveryOutcome.ACCEPTED
    assert ring.wait_for_ai(second, 0.1) == (SlotState.AI_COMPLETED, "mine")


def test_late_result_after_timeout_is_rejected() -> None:
    ring = RequestStateRing(capacity=1, slot_bytes=64)
    t = ring.allocate("req-1", "corr-1")
    ring.mark_ai_pending(t)
    ring.wait_for_ai(t, 0.001)
    assert ring.deliver_ai_result(t.slot_index, t.generation, "req-1", "corr-1", "x") is DeliveryOutcome.LATE


def test_result_arriving_before_wait_is_kept() -> None:
    ring = RequestStateRing(capacity=1, slot_bytes=64)
    t = ring.allocate("req-1", "corr-1")
    ring.mark_ai_pending(t)
    assert ring.deliver_ai_result(t.slot_index, t.generation, "req-1", "corr-1", "early") is DeliveryOutcome.ACCEPTED
    assert ring.wait_for_ai(t, 0) == (SlotState.AI_COMPLETED, "early")


def test_result_for_slot_not_awaiting_ai_is_rejected() -> None:
    ring = RequestStateRing(capacity=1, slot_bytes=64)
    t = ring.allocate("req-1", "corr-1")
    assert ring.deliver_ai_result(t.slot_index, t.generation, "req-1", "corr-1", "x") is DeliveryOutcome.NOT_AWAITING
    assert ring.deliver_ai_result(7, 1, "req-1", "corr-1", "x") is DeliveryOutcome.UNKNOWN_SLOT


def test_duplicate_ai_results_and_double_decisions_are_rejected() -> None:
    ring = RequestStateRing(capacity=1, slot_bytes=64)
    t = ring.allocate("req-1", "corr-1")
    ring.mark_ai_pending(t)
    args = (t.slot_index, t.generation, "req-1", "corr-1")
    assert ring.deliver_ai_result(*args, "first") is DeliveryOutcome.ACCEPTED
    assert ring.deliver_ai_result(*args, "second") is DeliveryOutcome.DUPLICATE
    assert ring.commit_decision(t, "ALLOW") is True
    assert ring.deliver_ai_result(*args, "third") is DeliveryOutcome.DUPLICATE
    assert ring.commit_decision(t, "BLOCK") is False
    assert ring.wait_for_ai(t, 0)[1] == "first"


def test_correlation_and_request_id_mismatch_are_rejected() -> None:
    ring = RequestStateRing(capacity=1, slot_bytes=64)
    t = ring.allocate("req-1", "corr-1")
    ring.mark_ai_pending(t)
    assert ring.deliver_ai_result(t.slot_index, t.generation, "req-1", "corr-OTHER", "x") is DeliveryOutcome.CORRELATION_MISMATCH
    assert ring.deliver_ai_result(t.slot_index, t.generation, "req-OTHER", "corr-1", "x") is DeliveryOutcome.REQUEST_ID_MISMATCH
    assert ring.state_of(t) is SlotState.AI_PENDING


def test_correlation_header_conflicting_with_body_is_rejected() -> None:
    p = pipeline()
    with pytest.raises(CorrelationConflictError):
        run(p, example(), header_correlation_id="corr-different")
    assert run(p, example(), header_correlation_id="corr-7f83a91b2c").correlation_id == "corr-7f83a91b2c"
    response = TestClient(create_app(broker=InMemoryBroker())).post(
        "/v1/szcfm/events", json=example(), headers={"X-Correlation-ID": "corr-different"}
    )
    assert response.status_code == 422


def test_ids_are_generated_when_absent() -> None:
    data = example()
    del data["payload"]["request_id"]
    del data["payload"]["correlation_id"]
    result = run(pipeline(), data, header_correlation_id="corr-from-header")
    assert result.request_id.startswith("req-") and len(result.request_id) > 10
    assert result.correlation_id == "corr-from-header"
    assert run(pipeline(), data).correlation_id.startswith("corr-")


def test_slot_memory_is_wiped_on_release() -> None:
    ring = RequestStateRing(capacity=1, slot_bytes=64)
    t = ring.allocate("r", "c")
    ring.store_envelope(t, b"secret-bytes")
    ring.release(t)
    assert bytes(ring._memory._raw_buffer[:12]) == b"\x00" * 12


# --- 11: concurrency -----------------------------------------------------------------


def test_concurrent_requests_receive_their_own_ai_results_across_slot_reuse() -> None:
    p = pipeline(EchoPriorBackend(jitter_s=0.004), ring_capacity=8, inference_workers=4, max_in_flight=64, ai_timeout_s=2.0)

    def submit(i: int):
        data = example()
        data["payload"]["request_id"] = f"req-{i}"
        data["payload"]["correlation_id"] = f"corr-{i}"
        data["threat_score"] = round(i / 1000, 3)
        return i, run(p, data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(1, 401)))

    generations = set()
    for i, result in results:
        assert result.request_id == f"req-{i}"
        assert result.correlation_id == f"corr-{i}"
        assert result.ai_assessment.status == "COMPLETED"
        assert result.ai_assessment.risk_score == pytest.approx(i / 1000)
        assert result.request_state.released is True
        generations.add((result.request_state.slot_index, result.request_state.generation))
    assert len(generations) == 400  # every request had its own (slot, generation)
    assert max(g for _, g in generations) > 1  # slots were reused
    assert p.ring.occupancy == 0
    counters = p.metrics.snapshot()["counters"]
    assert counters["ai_delivery_accepted"] == 400
    assert counters.get("errors", 0) == 0
    p.close()


def test_stale_inference_after_timeout_does_not_reach_reused_slot() -> None:
    backend = GatedBackend()
    p = pipeline(backend, ring_capacity=1, inference_workers=2, max_in_flight=4, ai_timeout_s=0.05)
    first = example()
    first["payload"]["request_id"], first["payload"]["correlation_id"] = "req-a", "corr-a"
    first["threat_score"] = 0.31
    result_a = run(p, first)
    assert result_a.ai_assessment.status == "TIMEOUT"
    assert result_a.request_state.released is True

    second = example()
    second["payload"]["request_id"], second["payload"]["correlation_id"] = "req-b", "corr-b"
    second["threat_score"] = 0.42
    holder = {}
    worker = threading.Thread(target=lambda: holder.setdefault("r", run(p, second)))
    worker.start()
    assert backend.started.acquire(timeout=2)  # A started earlier
    assert backend.started.acquire(timeout=2)  # B's inference is now running on slot 0, generation 2
    backend.gate.set()  # A's stale result and B's result complete
    worker.join(5)
    time.sleep(0.05)
    counters = p.metrics.snapshot()["counters"]
    # B timed out too (50 ms) or completed; either way A's result never matched B's slot.
    assert holder["r"].request_id == "req-b"
    if holder["r"].ai_assessment.status == "COMPLETED":
        assert holder["r"].ai_assessment.risk_score == pytest.approx(0.42)
    assert counters["ai_delivery_stale_generation"] >= 1
    assert result_a.request_state.generation == 1 and holder["r"].request_state.generation == 2
    p.close()


# --- 13-14, 23: inference failure modes ------------------------------------------------


def test_inference_timeout_is_not_treated_as_safe() -> None:
    p = pipeline(FixedBackend(0.01, delay_s=0.3), ai_timeout_s=0.02)
    result = run(p, example())
    assert result.ai_assessment.status == "TIMEOUT"
    assert result.ai_assessment.risk_score is None
    assert result.security_decision.decision == "CHALLENGE"
    assert "AI_UNAVAILABLE_CHALLENGE" in result.security_decision.reason_codes
    assert result.context_fusion.status == "DEGRADED"

    payments = example()
    payments["metadata"]["path"] = "/v1/payments/charge"
    assert run(p, payments).security_decision.decision == "BLOCK"  # FAIL_CLOSED path
    time.sleep(0.7)
    counters = p.metrics.snapshot()["counters"]
    # Results finishing after the request released its slot are stale; between
    # timeout and release they are late. Neither is ever accepted.
    assert counters.get("ai_delivery_late", 0) + counters.get("ai_delivery_stale_generation", 0) == 2
    assert counters.get("ai_delivery_accepted", 0) == 0
    assert counters["ai_timeouts"] == 2
    p.close()


def test_fail_open_is_only_used_when_configured_and_rules_pass() -> None:
    p = pipeline(FailingBackend(0.0), default_ai_failure_mode="FAIL_OPEN")
    assert run(p, example()).security_decision.decision == "ALLOW"
    attack = example()
    attack["payload"]["security_context"]["ip_reputation_score"] = 0.75
    assert run(p, attack).security_decision.decision == "CHALLENGE"


@pytest.mark.parametrize("backend", [FailingBackend(0.0), NaNBackend(0.0)])
def test_inference_failure_is_reported_and_not_allowed(backend) -> None:
    result = run(pipeline(backend), example())
    assert result.ai_assessment.status == "FAILED"
    assert result.ai_assessment.error
    assert result.security_decision.decision == "CHALLENGE"


def test_backpressure_rejects_inference_without_allowing() -> None:
    backend = GatedBackend()
    p = pipeline(backend, inference_workers=2, max_in_flight=2, ring_capacity=32, ai_timeout_s=0.2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: run(p, example()), range(8)))
    backend.gate.set()
    statuses = [r.ai_assessment.status for r in results]
    assert statuses.count("REJECTED_BACKPRESSURE") >= 6
    for r in results:
        assert r.security_decision.decision != "ALLOW"
    assert p.executor.rejected >= 6
    time.sleep(0.1)
    assert p.executor.in_flight == 0
    p.close()


def test_simulated_results_can_be_treated_as_unavailable() -> None:
    p = pipeline(DevelopmentHeuristicBackend(), simulated_ai_policy="unavailable")
    result = run(p, example())
    assert result.ai_assessment.simulated is True
    assert result.security_decision.decision == "CHALLENGE"
    assert result.security_decision.reasons[-1].detail.startswith("AI assessment SIMULATED_NOT_ACCEPTED")


# --- 17-19: policy -------------------------------------------------------------------


def test_fusion_formula() -> None:
    assert fuse(0.5, 1.0, 0.7) == pytest.approx(0.65)
    assert fuse(0.9, 0.0, 0.7) == pytest.approx(0.9)  # context never dilutes the model
    assert fuse(0.0, 0.0, 0.7) == 0.0
    features = {"ip_reputation_score": 1.0, "requests_last_10_seconds": 1.0, "failed_logins_last_5_minutes": 1.0, "authentication_anomaly": 1.0}
    assert context_risk(features) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "decision", "code"),
    [(0.10, "ALLOW", "MODEL_LOW_RISK"), (0.70, "CHALLENGE", "MODEL_ELEVATED_RISK"), (0.95, "BLOCK", "MODEL_HIGH_RISK")],
)
def test_model_driven_decisions(score, decision, code) -> None:
    result = run(pipeline(FixedBackend(score)), example())
    assert result.security_decision.decision == decision
    assert code in result.security_decision.reason_codes
    assert result.security_decision.deterministic_decision == "ALLOW"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda d: d["payload"]["security_context"].__setitem__("sql_injection_indicator", True), "SQL_INJECTION_INDICATOR"),
        (lambda d: d["payload"]["security_context"].__setitem__("xss_indicator", True), "XSS_INDICATOR"),
        (lambda d: d["payload"]["security_context"].__setitem__("rate_limit_exceeded", True), "RATE_LIMIT_EXCEEDED"),
        (lambda d: d["payload"]["security_context"].__setitem__("ip_reputation_score", 0.95), "IP_REPUTATION_BLOCK"),
        (lambda d: d["payload"]["request_body"].__setitem__("coupon", "x' OR '1'='1' --"), "PAYLOAD_ATTACK_SIGNATURE"),
    ],
)
def test_mandatory_rules_override_low_ai_risk(mutate, code) -> None:
    data = example()
    mutate(data)
    result = run(pipeline(FixedBackend(0.01)), data)
    assert result.security_decision.decision == "BLOCK"
    assert code in result.security_decision.reason_codes
    assert result.ai_assessment.status == "SKIPPED"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda d: d["metadata"].__setitem__("authorization", None), "AUTH_MISSING"),
        (lambda d: d["payload"].__delitem__("authentication"), "AUTH_CONTEXT_MISSING"),
        (lambda d: d["metadata"].__setitem__("authorization", "Basic abc"), "AUTH_SCHEME_MISMATCH"),
        (lambda d: d["metadata"].__setitem__("client_role", "admin"), "ROLE_NOT_GRANTED"),
        (lambda d: d["metadata"].__setitem__("path", "/v1/admin/users"), "ENDPOINT_ROLE_FORBIDDEN"),
    ],
)
def test_authentication_and_authorization_failures_block_independently_of_ai(mutate, code) -> None:
    data = example()
    mutate(data)
    result = run(pipeline(FixedBackend(0.0)), data)
    assert result.security_decision.decision == "BLOCK"
    assert code in result.security_decision.reason_codes
    reason = next(r for r in result.security_decision.reasons if r.code == code)
    assert reason.origin in {"authentication", "authorization"}


def test_public_path_does_not_require_authentication() -> None:
    data = example()
    data["metadata"]["path"] = "/v1/public/catalog"
    data["metadata"]["authorization"] = None
    del data["payload"]["authentication"]
    assert run(pipeline(FixedBackend(0.1)), data).security_decision.decision == "ALLOW"


def test_challenge_rules_mfa_and_anomaly() -> None:
    admin = example()
    admin["metadata"]["path"] = "/v1/admin/reports"
    admin["metadata"]["client_role"] = "admin"
    admin["payload"]["authentication"]["roles"] = ["admin"]
    admin["payload"]["authentication"]["mfa_verified"] = False
    result = run(pipeline(FixedBackend(0.05)), admin)
    assert result.security_decision.decision == "CHALLENGE"
    assert "MFA_REQUIRED" in result.security_decision.reason_codes

    anomaly = example()
    anomaly["payload"]["security_context"]["authentication_anomaly"] = True
    result = run(pipeline(FixedBackend(0.05)), anomaly)
    assert result.security_decision.decision == "CHALLENGE"
    assert result.security_decision.reason_codes[0] == "AUTHENTICATION_ANOMALY"


# --- 20-22: end to end, size, logs -------------------------------------------------------


class RecordingForwarder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, event, request_id, correlation_id) -> None:
        self.calls.append((request_id, correlation_id))


def test_end_to_end_api_processing_with_example_payload() -> None:
    broker = InMemoryBroker()
    published = []
    broker.subscribe(published.append)
    forwarder = RecordingForwarder()
    client = TestClient(create_app(broker=broker, szcfm_forwarder=forwarder))

    response = client.post("/v1/szcfm/events", json=example())
    assert response.status_code == 200
    body = response.json()
    assert body["event_status"] == "COMPLETED"
    assert body["request_id"] == "req-20261022-000184"
    assert body["correlation_id"] == "corr-7f83a91b2c"
    assert body["source"] == "web"
    assert body["serialization"]["success"] is True and body["serialization"]["bytes_size"] > 0
    assert body["feature_extraction"]["feature_count"] == 28
    assert body["request_state"] | {"ring_occupancy": None} == {
        "slot_allocated": True, "slot_index": 0, "generation": 1, "state": "DECIDED",
        "released": True, "ring_occupancy": None, "ring_capacity": 256,
    }
    assert body["ai_assessment"]["status"] == "COMPLETED"
    assert body["ai_assessment"]["model_version"] == "szcfm-dev-heuristic-1"
    assert body["ai_assessment"]["simulated"] is True
    assert body["context_fusion"]["request_context_matched"] is True
    assert body["security_decision"]["decision"] == "ALLOW"
    assert body["security_decision"]["committed"] is True
    assert body["forwarding"]["status"] == "FORWARDED"
    assert forwarder.calls == [("req-20261022-000184", "corr-7f83a91b2c")]
    assert body["publication"]["status"] == "PUBLISHED"
    assert published[0].payload["_szcfm"]["decision"] == "ALLOW"
    assert published[0].payload["authentication"]["session_id"] == REDACTED
    assert body["pipeline_metrics"]["execution_time_ms"] > 0

    metrics = client.get("/v1/szcfm/metrics").json()
    assert metrics["counters"]["requests_processed"] == 1
    assert metrics["ring"]["occupancy"] == 0
    schema = client.get("/v1/szcfm/feature-schema").json()
    assert [f["name"] for f in schema["features"]] == list(FEATURE_NAMES)


def test_blocked_request_is_not_forwarded() -> None:
    forwarder = RecordingForwarder()
    p = SZCFMPipeline(SZCFMConfig(), backend=FixedBackend(0.99), forwarder=forwarder)
    result = run(p, example())
    assert result.security_decision.decision == "BLOCK"
    assert result.forwarding.status == "NOT_PERMITTED"
    assert forwarder.calls == []


def test_oversized_requests_are_rejected() -> None:
    data = example()
    data["payload"]["request_body"]["blob"] = "a" * 40_000
    with pytest.raises(PayloadTooLargeError):
        pipeline().parse(raw(data))
    client = TestClient(create_app(broker=InMemoryBroker()))
    response = client.post("/v1/szcfm/events", content=raw(data), headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert client.get("/v1/szcfm/metrics").json()["counters"]["rejected_oversized"] == 1


def test_excessively_nested_body_is_rejected() -> None:
    data = example()
    nested: dict = {}
    cursor = nested
    for _ in range(40):
        cursor["n"] = {}
        cursor = cursor["n"]
    data["payload"]["request_body"] = nested
    with pytest.raises(EventValidationError, match="schema validation"):
        pipeline().parse(raw(data))


def test_no_secrets_in_logs_or_response(caplog) -> None:
    secrets = ["sk_live_TOPSECRET123", "sess-SECRET-777", "hunter2-SECRET", "4111111111111111", "198.51.100.42"]
    data = example()
    data["metadata"]["authorization"] = f"Bearer {secrets[0]}"
    data["payload"]["authentication"]["session_id"] = secrets[1]
    data["payload"]["request_body"]["password"] = secrets[2]
    data["payload"]["request_body"]["payment"] = {"card_number": secrets[3]}
    caplog.set_level(logging.DEBUG)
    client = TestClient(create_app(broker=InMemoryBroker(), szcfm_backend=FailingBackend(0.0)))
    response = client.post("/v1/szcfm/events", json=data)
    assert response.status_code == 200
    assert any("szcfm decision" in r.getMessage() for r in caplog.records)
    logged = "\n".join(r.getMessage() for r in caplog.records) + caplog.text
    for secret in secrets:
        assert secret not in logged
        assert secret not in response.text
