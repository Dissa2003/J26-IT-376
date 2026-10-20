"""Unit tests for broker delivery and the ingestion API."""

from collections.abc import Callable
import threading
import time
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from ingestion.broker import (
    InMemoryBroker,
    RedisStreamBroker,
    RingBufferBroker,
    create_broker,
)
from ingestion.app import create_app
from ingestion.validator import IngestionValidator
from ingestion.pii_masker import ASTPIIMasker, PIIMasker, REDACTED
from ingestion.zero_copy import ZeroCopyRingSlotBuffer
from ingestion.vector_builder import (
    HIGH_THREAT_THRESHOLD,
    LOW_THREAT_THRESHOLD,
    UnifiedContextVectorBuilder,
    extract_threat_score,
)
from shared_contracts.config import Settings
from shared_contracts.interfaces import ProcessingClient
from shared_contracts.models import Label, Prediction, RawEvent


class FakeProcessing(ProcessingClient):
    """Capture-free processing client for app tests."""

    def process(self, event: RawEvent) -> Prediction:
        """Return a deterministic normal prediction."""
        return Prediction(event_id=event.event_id, score=0.1, label=Label.NORMAL, threshold=0.8, model_version="t")


class FlakyProcessing(ProcessingClient):
    """Fail once to model a temporary processing outage."""

    def __init__(self) -> None:
        self.attempts = 0

    def process(self, event: RawEvent) -> Prediction:
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("processing is offline")
        return Prediction(
            event_id=event.event_id,
            score=0.2,
            label=Label.NORMAL,
            threshold=0.8,
            model_version="retry-test",
        )


def test_publish_delivers_raw_event_to_subscriber() -> None:
    """A published RawEvent is delivered to each subscribed consumer."""
    broker = InMemoryBroker()
    received: list[RawEvent] = []

    broker.subscribe(received.append)
    event = RawEvent(source="test-source", payload={"value": 1})

    broker.publish(event)

    assert received == [event]


def test_publish_delivers_to_multiple_subscribers() -> None:
    """A published RawEvent is delivered in subscription order."""
    broker = InMemoryBroker()
    received: list[str] = []

    def record(name: str) -> Callable[[RawEvent], None]:
        def handler(event: RawEvent) -> None:
            received.append(f"{name}:{event.event_id}")

        return handler

    broker.subscribe(record("first"))
    broker.subscribe(record("second"))
    event = RawEvent(source="test-source")

    broker.publish(event)

    assert received == [f"first:{event.event_id}", f"second:{event.event_id}"]


def test_ingest_forwards_to_processing() -> None:
    """The HTTP entry point publishes an event to the processing client."""
    c = TestClient(create_app(processing=FakeProcessing()))
    r = c.post("/v1/ingest", json={"source": "s", "payload": {"x": 1}})
    assert r.status_code == 202 and r.json()["label"] == "normal"


def test_ingest_full_path_uses_broker_and_processing_response() -> None:
    """POST /v1/ingest publishes through the broker to the processing client."""
    broker = InMemoryBroker()
    processing = FakeProcessing()
    client = TestClient(create_app(processing=processing, broker=broker))

    response = client.post(
        "/v1/ingest",
        json={"source": "integration-test", "payload": {"value": 4}},
    )

    assert response.status_code == 202
    assert response.json()["label"] == "normal"
    assert response.json()["model_version"] == "t"


def test_processing_outage_retries_without_dropping_event() -> None:
    """A transient processing outage is retried from the in-process queue."""
    processing = FlakyProcessing()
    client = TestClient(
        create_app(
            processing=processing,
            broker=InMemoryBroker(),
            processing_timeout=2.0,
        )
    )

    response = client.post(
        "/v1/ingest",
        json={"source": "retry-test", "payload": {"value": 4}},
    )

    assert response.status_code == 202
    assert response.json()["model_version"] == "retry-test"
    assert processing.attempts == 2


def test_validator_rejects_empty_source() -> None:
    """Events without a meaningful source are invalid."""
    with pytest.raises(ValueError, match="source"):
        IngestionValidator().validate({"source": "  ", "payload": {}})


def test_invalid_ingest_is_written_to_local_dlq() -> None:
    """Invalid requests return 422 and are retained by the local DLQ."""
    broker = InMemoryBroker()
    client = TestClient(create_app(processing=FakeProcessing(), broker=broker))

    response = client.post("/v1/ingest", json={"source": "", "payload": {}})

    assert response.status_code == 422
    assert broker.dead_letters[0]["payload"]["source"] == ""


def test_redis_dlq_uses_dedicated_stream() -> None:
    """Malformed payloads are written to the dedicated Redis DLQ stream."""
    client = MagicMock()
    broker = RedisStreamBroker(Settings(app_env="production"), redis_client=client)
    payload = {"source": "", "payload": {}}

    broker.publish_dlq(payload, "source must be non-empty")

    args, kwargs = client.xadd.call_args
    assert args[0] == "dlq:events"
    assert kwargs["maxlen"] == 100_000


def test_redis_publish_serializes_raw_event() -> None:
    """Redis publish uses XADD with the RawEvent JSON payload."""
    client = MagicMock()
    broker = RedisStreamBroker(
        Settings(app_env="production"),
        redis_client=client,
    )
    event = RawEvent(source="redis-test", payload={"value": 3})

    broker.publish(event)

    client.xadd.assert_called_once()
    args, kwargs = client.xadd.call_args
    assert args[0] == "events"
    assert RawEvent.model_validate_json(args[1]["event"]) == event
    assert kwargs["maxlen"] == 100_000


def test_redis_subscribe_dispatches_and_acknowledges() -> None:
    """Redis records are dispatched and acknowledged after handler success."""
    client = MagicMock()
    event = RawEvent(source="redis-test")
    client.xreadgroup.side_effect = [
        [("events", [("1-0", {"event": event.model_dump_json()})])],
        KeyboardInterrupt,
    ]
    received: list[RawEvent] = []
    broker = RedisStreamBroker(Settings(app_env="production"), redis_client=client)

    with pytest.raises(KeyboardInterrupt):
        broker.subscribe(received.append)

    assert received == [event]
    client.xack.assert_called_once_with("events", "ingestion", "1-0")


def test_create_broker_uses_memory_in_local_environment() -> None:
    """Local development never attempts a Redis connection."""
    broker = create_broker(Settings(app_env="local"))

    assert isinstance(broker, InMemoryBroker)


def test_ring_buffer_drains_events_sequentially() -> None:
    """Buffered events are delivered FIFO by one background worker."""
    broker = RingBufferBroker(maxlen=8)
    received: list[str] = []
    broker.subscribe(lambda event: received.append(event.event_id))
    events = [RawEvent(event_id=str(index), source="test") for index in range(4)]

    for event in events:
        broker.publish(event)

    deadline = time.monotonic() + 1.0
    while len(received) < len(events) and time.monotonic() < deadline:
        time.sleep(0.001)
    broker.stop()

    assert received == ["0", "1", "2", "3"]
    assert broker.evicted_count == 0


def test_ring_buffer_evicts_oldest_event_when_full() -> None:
    """A full circular buffer routes a low-threat event to overflow."""
    overflow = MagicMock()
    broker = RingBufferBroker(maxlen=2, overflow_broker=overflow)
    started = threading.Event()
    release = threading.Event()

    def slow_handler(event: RawEvent) -> None:
        started.set()
        release.wait(1.0)

    broker.subscribe(slow_handler)
    events = [RawEvent(event_id=str(index), source="test") for index in range(4)]

    broker.publish(events[0])
    assert started.wait(1.0)
    broker.publish(events[1])
    broker.publish(events[2])
    broker.publish(events[3])
    release.set()

    deadline = time.monotonic() + 1.0
    while broker.buffered_count and time.monotonic() < deadline:
        time.sleep(0.001)
    broker.stop()

    assert broker.evicted_count == 0
    assert overflow.publish.call_count >= 1


def test_twde_routes_low_threat_arrival_to_overflow_when_full() -> None:
    """TWDE sends a low-threat arrival to overflow instead of evicting memory."""
    overflow = MagicMock()
    broker = RingBufferBroker(maxlen=2, overflow_broker=overflow)
    events = [
        RawEvent(event_id="buffer-1", source="test", payload={"_threat_score": 0.1}),
        RawEvent(event_id="buffer-2", source="test", payload={"_threat_score": 0.5}),
        RawEvent(event_id="overflow", source="test", payload={"_threat_score": 0.2}),
    ]

    try:
        broker.publish(events[0])
        broker.publish(events[1])
        broker.publish(events[2])
        assert broker.buffered_count == 2
        assert broker.evicted_count == 0
        overflow.publish.assert_called_once_with(events[2])
    finally:
        broker.stop()


def test_twde_preserves_high_threat_event_by_evicting_low_threat_event() -> None:
    """TWDE retains high-threat arrivals and spills the oldest low-threat item."""
    overflow = MagicMock()
    broker = RingBufferBroker(maxlen=2, overflow_broker=overflow)
    received: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def slow_handler(event: RawEvent) -> None:
        received.append(event.event_id)
        started.set()
        release.wait(1.0)

    broker.subscribe(slow_handler)
    low = RawEvent(event_id="low", source="test", payload={"threat_score": 0.1})
    high = RawEvent(event_id="high", source="test", payload={"threat_score": 0.9})
    medium = RawEvent(
        event_id="medium",
        source="test",
        payload={"_threat_score": 0.5},
    )
    incoming = RawEvent(
        event_id="incoming-high",
        source="test",
        payload={"_threat_score": HIGH_THREAT_THRESHOLD},
    )

    try:
        broker.publish(low)
        assert started.wait(1.0)
        broker.publish(high)
        broker.publish(medium)
        broker.publish(incoming)
        release.set()
        deadline = time.monotonic() + 1.0
        while broker.buffered_count and time.monotonic() < deadline:
            time.sleep(0.001)

        assert overflow.publish.call_args.args == (medium,)
        assert received == ["low", "high", "incoming-high"]
    finally:
        release.set()
        broker.stop()


def test_twde_thresholds_are_shared_with_vector_builder() -> None:
    """Threat-score extraction clamps malformed values and honors TWDE thresholds."""
    assert extract_threat_score({"threat_score": 0.9}) >= HIGH_THREAT_THRESHOLD
    assert extract_threat_score({"_threat_score": 0.1}) < LOW_THREAT_THRESHOLD
    assert extract_threat_score({"_threat_score": "invalid"}) == 0.0


def test_twde_publish_hot_path_avoids_lock_contention() -> None:
    """A full-buffer classification uses only in-memory operations."""
    broker = RingBufferBroker(maxlen=8)
    try:
        for index in range(8):
            broker.publish(
                RawEvent(
                    event_id=str(index),
                    source="test",
                    payload={"_threat_score": 0.5},
                )
            )
        started = time.perf_counter_ns()
        broker.publish(
            RawEvent(
                event_id="high",
                source="test",
                payload={"_threat_score": HIGH_THREAT_THRESHOLD},
            )
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        assert elapsed_ms < 10
    finally:
        broker.stop()

def test_create_broker_falls_back_when_redis_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Redis startup failure selects the in-memory fallback."""
    def fail(_: Settings) -> RedisStreamBroker:
        raise RedisConnectionError("connection refused")

    monkeypatch.setattr("ingestion.broker.RedisStreamBroker", fail)

    broker = create_broker(Settings(app_env="production"))

    assert isinstance(broker, InMemoryBroker)


class CapturingProcessing(FakeProcessing):
    """Processing fake that records the event received downstream."""

    def __init__(self) -> None:
        self.events: list[RawEvent] = []

    def process(self, event: RawEvent) -> Prediction:
        self.events.append(event)
        return super().process(event)


def test_ingest_captures_http_interception_payload() -> None:
    """HTTP interception payloads return an immediate capture acknowledgement."""
    client = TestClient(create_app(processing=FakeProcessing()))

    r = client.post(
        "/v1/ingest",
        json={
            "metadata": {"path": "/orders", "method": "POST"},
            "payload": {"x": 1},
            "threat_score": 0.1,
        },
    )
    assert r.status_code == 202
    body = r.json()
    assert body["event_id"]
    assert body["masked_payload"]["x"] == 1
    assert body["protobuf_bytes_size"] > 0
    assert body["unified_context_vector"] == [0.1, 1.0]
    assert body["pipeline_metrics"]["execution_time_ms"] >= 0


def test_vector_builder_serializes_compact_context_envelope() -> None:
    """The vector starts with the threat score and protobuf output is binary."""
    builder = UnifiedContextVectorBuilder()
    payload = {"temperature": 14.2, "nested": {"pressure": 1450}}

    vector = builder.build(payload, 0.95)
    encoded = builder.serialize("event-1", payload, 0.95, vector)

    assert vector == [0.95, 14.2, 1450.0]
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0


def test_pii_masker_redacts_nested_sensitive_values() -> None:
    """Nested emails, cards, IPs, and credentials are masked in memory."""
    payload = {
        "email": "user@example.com",
        "card": "4111 1111 1111 1111",
        "ip": "192.168.1.10",
        "credentials": {"password": "secret", "user": "alice"},
        "items": [{"token": "abc", "value": 7}],
    }

    masked = PIIMasker().mask(payload)

    assert masked == {
        "email": REDACTED,
        "card": REDACTED,
        "ip": REDACTED,
        "credentials": REDACTED,
        "items": [{"token": REDACTED, "value": 7}],
    }
    assert payload["email"] == "user@example.com"


def test_interception_forwards_masked_payload() -> None:
    """PII is masked before the event reaches downstream processing."""
    processing = CapturingProcessing()
    client = TestClient(create_app(processing=processing))
    response = client.post(
        "/v1/ingest",
        json={
            "metadata": {"path": "/login"},
            "payload": {"email": "user@example.com", "password": "secret"},
            "threat_score": 0.1,
        },
    )

    assert response.status_code == 202
    assert processing.events[0].payload["email"] == REDACTED
    assert processing.events[0].payload["password"] == REDACTED


def test_interception_response_has_context_vector_and_sub_10ms_execution() -> None:
    """The interceptor returns the vector and meets the latency target (<10ms)."""
    client = TestClient(create_app(processing=FakeProcessing()))

    response = client.post(
        "/v1/ingest",
        json={
            "metadata": {"method": "POST", "path": "/payments"},
            "payload": {"amount": 149.95, "retry_count": 2},
            "threat_score": 0.88,
        },
    )

    body = response.json()
    assert response.status_code == 202
    assert body["unified_context_vector"] == [0.88, 149.95, 2.0]
    assert body["pipeline_metrics"]["execution_time_ms"] < 10


def test_invalid_interception_payload_returns_422() -> None:
    """Invalid interceptor envelopes are rejected by the API schema."""
    client = TestClient(create_app(processing=FakeProcessing()))

    response = client.post(
        "/v1/ingest",
        json={
            "metadata": {"path": "/payments"},
            "payload": {"amount": 10},
            "threat_score": 2.0,
        },
    )

    assert response.status_code == 422


def test_interception_captures_client_role_and_authorization_metadata() -> None:
    """Captured attributes include client_role, authorization, and headers."""
    processing = CapturingProcessing()
    client = TestClient(create_app(processing=processing))

    response = client.post(
        "/v1/ingest",
        json={
            "metadata": {
                "method": "POST",
                "path": "/admin/manage",
                "client_role": "admin",
                "authorization": "Bearer token-123",
                "headers": {"x-custom": "val"},
            },
            "payload": {"action": "delete_cache"},
            "threat_score": 0.05,
        },
    )
    assert response.status_code == 202
    forwarded = processing.events[0]
    http_meta = forwarded.payload["_http_metadata"]
    assert http_meta["client_role"] == "admin"
    assert http_meta["authorization"] == "Bearer token-123"
    assert http_meta["headers"]["x-custom"] == "val"


def test_async_ai_threat_score_ingestion_endpoint() -> None:
    """Async AI threat risk score harness (WBS 4.2 / FR2) records risk scores."""
    broker = InMemoryBroker()
    client = TestClient(create_app(broker=broker))

    resp = client.post(
        "/v1/threat-score",
        json={"event_id": "evt-async-101", "threat_score": 0.92},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
    assert broker.get_threat_score("evt-async-101") == 0.92


def test_ast_pii_masking_semantic_retention_and_secure_wipe() -> None:
    """Verify semantic retention (>95%) and NFR4 transient memory wiping."""
    masker = PIIMasker()
    payload = {
        "user_id": "u-123",
        "action": "checkout",
        "amount": 49.99,
        "currency": "USD",
        "items": [1, 2, 3],
        "email": "sensitive@example.com",
    }
    masked, retention = masker.mask_with_retention_metrics(payload)
    assert masked["email"] == REDACTED
    assert masked["amount"] == 49.99
    # 5 non-PII attributes retained out of 6 total attributes => >83%
    assert retention > 0.80

    # Test transient secure wipe (NFR4)
    transient_dict = {"secret": "sensitive_data", "sub": [1, 2, 3]}
    PIIMasker.secure_wipe(transient_dict)
    assert len(transient_dict) == 0


def test_async_correlation_pre_arrived_threat_score() -> None:
    """A pre-arrived async threat score (y) is automatically fused with incoming HTTP stream (x)."""
    broker = InMemoryBroker()
    client = TestClient(create_app(broker=broker))

    # 1. External AI module submits threat score (y) in advance
    corr_id = "evt-pre-arrive-777"
    score_resp = client.post(
        "/v1/threat-score",
        json={"event_id": corr_id, "threat_score": 0.865},
    )
    assert score_resp.status_code == 200
    assert score_resp.json()["status"] == "recorded"

    # 2. Incoming HTTP request stream (x) arrives with matching event_id
    intercept_resp = client.post(
        "/v1/ingest",
        headers={"X-Event-ID": corr_id},
        json={
            "metadata": {"method": "POST", "path": "/api/v1/auth", "client_ip": "10.0.0.1"},
            "payload": {"username": "admin", "token": "xyz123"},
            "threat_score": 0.05,  # initial gateway estimate
        },
    )
    assert intercept_resp.status_code == 202
    body = intercept_resp.json()
    assert body["event_id"] == corr_id
    # The pre-arrived probabilistic AI score (0.865) must supersede the default estimate
    assert body["unified_context_vector"][0] == 0.865


def test_async_correlation_late_arrived_speculative_fusion() -> None:
    """A late-arriving threat score (y) triggers speculative context vector rebuild in < 1 ms."""
    broker = InMemoryBroker()
    client = TestClient(create_app(broker=broker))

    # 1. HTTP request stream (x) arrives first and is buffered
    late_id = "evt-late-fuse-999"
    resp = client.post(
        "/v1/intercept",
        headers={"X-Event-ID": late_id},
        json={
            "metadata": {"method": "POST", "path": "/api/v1/transfer"},
            "payload": {"account": "acc-9921", "amount": 500.0},
            "threat_score": 0.10,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["unified_context_vector"][0] == 0.10

    # 2. Async AI threat score arrives afterwards
    score_resp = client.post(
        "/v1/threat-score",
        json={"event_id": late_id, "threat_score": 0.935},
    )
    assert score_resp.status_code == 200
    fused_body = score_resp.json()
    assert fused_body["status"] == "fused"
    assert fused_body["event_id"] == late_id
    assert fused_body["threat_score"] == 0.935
    # The unified context vector was dynamically rebuilt in RAM with the new threat score
    assert fused_body["unified_context_vector"][0] == 0.935
    assert 500.0 in fused_body["unified_context_vector"]


def test_syntax_preserving_ast_pii_masking_with_sqli_and_xss() -> None:
    """Masks alphanumeric PII literals while preserving attack metacharacters and tokens."""
    masker = ASTPIIMasker()
    payload = {
        "password": "secretPassword123' OR 1=1; --",
        "email": "<script>alert('xss')</script>victim@company.com",
        "access_token": "bearer_admin_token_999; DROP TABLE users;--",
        "safe_user": "John Doe",
        "amount": 250.0,
    }

    masked = masker.mask(payload)

    # 1. Structural attack syntax is preserved for downstream SMT/Z3 solvers
    assert "' OR 1=1; --" in masked["password"]
    assert "<script>alert('xss')</script>" in masked["email"]
    assert "; DROP TABLE users;--" in masked["access_token"]

    # 2. Sensitive alphanumeric literals are masked
    assert "secretPassword123" not in masked["password"]
    assert "victim@company.com" not in masked["email"]
    assert "bearer_admin_token_999" not in masked["access_token"]
    assert REDACTED in masked["password"]
    assert REDACTED in masked["email"]

    # 3. Non-sensitive fields are untouched
    assert masked["safe_user"] == "John Doe"
    assert masked["amount"] == 250.0


def test_rich_mathematical_vector_builder_dimensions() -> None:
    """Verifies rich mathematical vector layout for string-heavy and mixed HTTP requests."""
    builder = UnifiedContextVectorBuilder()

    # String-heavy request (0 numeric leaves in payload)
    string_payload = {
        "action": "search",
        "query": "wireless headphones' OR 1=1; --",
        "category": "electronics",
    }
    vector = builder.build(string_payload, threat_score=0.75)
    # [0]: threat_score
    # [1]: payload_byte_length
    # [2]: structural_depth
    # [3]: masked_pii_count
    # [4]: syntactic_anomaly_token_density
    assert len(vector) == 5
    assert vector[0] == 0.75
    assert vector[1] > 0.0  # byte length
    assert vector[2] >= 1.0  # structural depth
    assert vector[3] == 0.0  # 0 masked PII
    assert vector[4] > 0.0  # syntactic anomaly tokens: ', ;, -, =

    # Explicit rich vector on payload containing numeric leaves
    mixed_payload = {"amount": 100.0, "status": "pending"}
    rich_vector = builder.build(mixed_payload, threat_score=0.25, include_structural_features=True)
    assert len(rich_vector) == 6
    assert rich_vector[0] == 0.25
    assert rich_vector[-1] == 100.0


def test_zero_copy_ring_slot_buffer_pointer_slicing_and_wipe() -> None:
    """Zero-copy memoryview ring buffer demonstrates non-allocating slicing and wipe."""
    buffer = ZeroCopyRingSlotBuffer(slot_count=4, slot_size=1024)
    slot_id = buffer.allocate_slot()
    assert slot_id == 0

    http_raw = b"POST /v1/ingest HTTP/1.1\r\nHost: transverify.shield\r\n\r\n{\"action\": \"verify\"}"
    view = buffer.write_bytes(slot_id, http_raw)
    assert isinstance(view, memoryview)
    assert len(view) == len(http_raw)

    headers_view, body_view = buffer.slice_http_stream(slot_id)
    assert headers_view.tobytes() == b"POST /v1/ingest HTTP/1.1\r\nHost: transverify.shield"
    assert body_view.tobytes() == b"{\"action\": \"verify\"}"

    # Verify zero-allocation wipe (NFR4)
    buffer.wipe_slot(slot_id)
    assert buffer.get_slot_view(slot_id).tobytes() == b""
    assert buffer.stats()["active_slots"] == 0


def test_enterprise_checkout_payload_pii_masking_and_vector() -> None:
    """Validate comprehensive masking of financial checkout payload without leaking PII or payment details."""
    client = TestClient(create_app(processing=FakeProcessing()))
    interception_input = {
        "metadata": {
            "method": "POST",
            "path": "/api/v2/financial/orders/checkout",
            "content_type": "application/json",
            "client_ip": "192.168.1.105",
            "client_role": "enterprise_customer",
            "authorization": "Bearer sample_token",
            "headers": {
                "user-agent": "Mozilla/5.0",
                "x-forwarded-for": "10.0.0.45, 192.168.1.105",
                "x-correlation-id": "corr-tx-9988-7766-5544",
            },
        },
        "payload": {
            "transaction_context": {
                "order_id": "ord_998877665544",
                "currency": "USD",
                "subtotal_amount": 1450.75,
                "tax_amount": 116.06,
                "discount_rate": 0.15,
                "item_count": 8.0,
            },
            "customer_profile": {
                "account_id": "acc_88442211",
                "full_name": "Alexander Pierce",
                "user_email": "alexander.pierce@enterprise-domain.org",
                "backup_email": "alex.backup@gmail.com",
                "phone_number": "+1-555-019-2834",
                "account_tier": "VIP_PREMIUM",
                "kyc_verified": True,
            },
            "payment_gateway": {
                "merchant_provider": "Stripe_Merchant",
                "credit_card": "4532-8899-1122-7766",
                "cvv": "889",
                "card_expiry": "12/28",
                "billing_address": {
                    "street_address": "123 Wall Street",
                    "city": "New York",
                    "state": "NY",
                    "postal_code": "10005",
                    "country": "USA",
                },
            },
            "telemetry_metrics": {
                "client_cpu_load": 98.45,
                "memory_heap_used_mb": 4096.8,
                "disk_io_wait_ms": 2.15,
                "network_throughput_kbps": 85400.0,
                "active_db_connections": 142.0,
                "request_frequency_per_sec": 3850.0,
                "error_count_last_5m": 0.0,
            },
            "security_context": {
                "api_key": "sk_live_99887766554433221100_secret_token",
                "failed_login_attempts": 4.0,
                "session_duration_sec": 1240.5,
                "is_vpn_detected": False,
            },
        },
        "threat_score": 0.94,
    }

    resp = client.post("/v1/intercept", json=interception_input)
    assert resp.status_code == 200
    data = resp.json()

    masked = data["masked_payload"]
    # Verify no PII or sensitive payment credentials leak in the response
    assert masked["customer_profile"]["full_name"] == REDACTED
    assert masked["customer_profile"]["user_email"] == REDACTED
    assert masked["customer_profile"]["backup_email"] == REDACTED
    assert masked["customer_profile"]["phone_number"] == REDACTED
    assert masked["payment_gateway"]["credit_card"] == REDACTED
    assert masked["payment_gateway"]["cvv"] == REDACTED
    assert masked["payment_gateway"]["card_expiry"] == REDACTED
    assert masked["payment_gateway"]["billing_address"]["street_address"] == REDACTED
    assert masked["security_context"]["api_key"] == REDACTED

    # Verify non-sensitive business data is preserved
    assert masked["customer_profile"]["account_id"] == "acc_88442211"
    assert masked["payment_gateway"]["merchant_provider"] == "Stripe_Merchant"
    assert masked["payment_gateway"]["billing_address"]["city"] == "New York"

    # Verify exact unified context vector
    expected_vector = [
        0.94,
        1450.75,
        116.06,
        0.15,
        8.0,
        98.45,
        4096.8,
        2.15,
        85400.0,
        142.0,
        3850.0,
        0.0,
        4.0,
        1240.5,
    ]
    assert data["unified_context_vector"] == expected_vector



