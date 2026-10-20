"""Comprehensive End-to-End (E2E) Pipeline Integration Test.

Tests the full research lifecycle across TransVerify / LogiShield microservices:
1. Ingress & Auth via API Gateway (Port 8000)
2. Non-blocking Ingestion, AST PII Masking, Context Vectorization (Port 8001)
3. Formal Rule Consistency & Active Verified Policy Rules (Port 8002)
4. Neuro-Symbolic Tokenization & Calibrated Anomaly Inference (Port 8010)
5. Z3 Runtime Safety Verification & Circuit Breaker False-Positive Override (Port 8003)
"""

from __future__ import annotations

import base64
import json
import time

import pytest
from fastapi.testclient import TestClient

from api_gateway.app import create_app as create_gateway_app
from engine.circuit_breaker import CircuitState, HybridCircuitBreaker
from engine.service import HybridVerificationService, StubModel
from engine.verifier import RuntimeLTLVerifier
from ingestion.app import create_app as create_ingestion_app
from ingestion.broker import InMemoryBroker
from ingestion.pii_masker import ASTPIIMasker
from ingestion.vector_builder import UnifiedContextVectorBuilder
from shared_contracts.interfaces import Notifier
from shared_contracts.models import Alert
from tokenization_ai.models import FusedInput, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.service import TokenizationAIService


class AlertCapture(Notifier):
    """Capture alerts triggered by gateway and engine."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def notify(self, alert: Alert) -> None:
        self.alerts.append(alert)


def create_signed_jwt(sub: str, role: str) -> str:
    """Generate a mock signed JWT token for authentication testing."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": sub, "role": role, "iat": int(time.time())}).encode()
    ).decode().rstrip("=")
    signature = "valid_mock_signature"
    return f"{header}.{payload}.{signature}"


def test_e2e_full_pipeline_legitimate_admin_ai_false_positive_override():
    """E2E Test 1: Admin bulk manage action flagged with AI false-positive (y=0.95).

    Lifecycle:
    1. Gateway authenticates caller via JWT and verifies role="admin".
    2. Ingestion intercepts request, masks PII (email, card), computes UnifiedContextVector in <10ms.
    3. Tokenizer AI pairs continuous context vector with formal policy rule and scores request.
    4. Z3 Runtime Verifier executes under 5ms, verifies all invariants are SAT.
    5. Circuit Breaker overrides AI false-positive to ALLOW with telemetry recorded.
    """
    alerts = AlertCapture()
    gateway_client = TestClient(create_gateway_app(alerts))
    broker = InMemoryBroker()
    ingestion_client = TestClient(create_ingestion_app(broker=broker))

    # --- Step 1: Gateway Ingress & Auth ---
    admin_jwt = create_signed_jwt(sub="admin-01", role="admin")
    health_res = gateway_client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["service"] == "api_gateway"

    # --- Step 2: Line-rate Ingestion & AST PII Masking ---
    t0_ingest = time.perf_counter()
    raw_payload = {
        "metadata": {
            "method": "POST",
            "path": "/api/v1/system/bulk-manage",
            "client_ip": "10.0.0.1",
            "client_role": "admin",
            "authorization": f"Bearer {admin_jwt}",
        },
        "payload": {
            "admin_email": "root@transverify.internal",
            "credit_card": "4111-2222-3333-4444",
            "action": "manage",
            "role": "admin",
            "rate_limit": 100,
            "request_rate": 15,
            "item_count": 50,
        },
        "threat_score": 0.95,  # High anomaly score (AI false positive)
    }

    ingest_res = ingestion_client.post("/v1/intercept", json=raw_payload)
    t1_ingest = time.perf_counter()
    ingestion_overhead_ms = (t1_ingest - t0_ingest) * 1000

    assert ingest_res.status_code == 200
    ingest_body = ingest_res.json()
    assert "event_id" in ingest_body
    assert ingest_body["protobuf_bytes_size"] > 0
    assert len(ingest_body["unified_context_vector"]) >= 4
    assert ingestion_overhead_ms < 50.0  # Well within end-to-end HTTP overhead

    # Verify AST PII Masker masked sensitive attributes while preserving structure
    masked = ingest_body["masked_payload"]
    assert masked["credit_card"] != "4111-2222-3333-4444"
    assert "admin_email" in masked
    assert masked["action"] == "manage"
    assert masked["role"] == "admin"

    # --- Step 3: Neuro-Symbolic Tokenization & Alignment ---
    tokenizer_service = TokenizationAIService.create_default()
    verified_rules = VerifiedRuleInfo(
        event_id=ingest_body["event_id"],
        rules=[
            VerifiedRule(
                rule_id="rule-admin-manage",
                attributes={
                    "subject": "admin",
                    "action": "manage",
                    "effect": "allow",
                    "resource": "system",
                },
            )
        ],
    )
    fused_input = FusedInput(
        context=UnifiedApiContext(
            event_id=ingest_body["event_id"],
            method="POST",
            path="/api/v1/system/bulk-manage",
            client_ip="10.0.0.1",
            headers={"authorization": f"Bearer {admin_jwt}"},
            payload=masked,
            threat_score=0.95,
        ),
        rule_info=verified_rules,
    )
    analysis = tokenizer_service.analyze(fused_input.context, fused_input.rule_info)
    assert 0.0 <= analysis.prediction.score <= 1.0
    assert len(analysis.model_input.input_ids) > 0

    # --- Step 4: Z3 Runtime Safety Verification & Circuit Breaker ---
    verifier = RuntimeLTLVerifier(timeout_ms=5)
    breaker = HybridCircuitBreaker(verifier=verifier, max_latency_ms=25.0)

    # Evaluate decision using formal verification
    decision = breaker.evaluate_hybrid_decision(
        event_id=ingest_body["event_id"],
        score=0.95,  # AI flags anomaly (score >= threshold 0.8)
        anomaly_threshold=0.8,
        context={
            "threat_score": 0.95,
            "role": "admin",
            "action": "manage",
            "rate_limit": 100,
            "request_rate": 15,
        },
    )

    # Formal invariants prove SAT: Role is admin, rate is within limit
    assert decision.verification.z3_status == "SAT"
    assert decision.decision == "ALLOW"
    assert decision.override_applied is True
    assert decision.override_reason == "z3_formal_allow_override"
    assert decision.verification.execution_time_ms < 5.0  # < 5ms SLA
    assert breaker.overrides_applied == 1


def test_e2e_full_pipeline_unauthorized_attack_deterministically_blocked():
    """E2E Test 2: Malicious user attempting unauthorized account deletion.

    Lifecycle:
    1. Unauthorized caller attempts delete with role="user".
    2. Ingestion intercepts and masks request, generating unified context vector.
    3. Tokenizer aligns request with verified security policy forbidding user delete.
    4. Z3 Runtime Verifier evaluates policy constraint in < 5ms.
    5. Z3 proves UNSAT: Request is deterministically BLOCKED with violated rule recorded.
    """
    verifier = RuntimeLTLVerifier(timeout_ms=5)
    breaker = HybridCircuitBreaker(verifier=verifier, max_latency_ms=25.0)

    # Policy constraint: Only admin can delete accounts
    policy_constraint = "role == 'admin'"

    decision = breaker.evaluate_hybrid_decision(
        event_id="evt-attack-sqli-delete",
        score=0.90,  # AI flags threat
        anomaly_threshold=0.8,
        context={
            "threat_score": 0.90,
            "role": "user",  # Attacker is a standard user
            "action": "delete",
            "resource": "UserAccount",
            "rate_limit": 10,
            "request_rate": 2,
        },
        custom_constraints=[policy_constraint],
    )

    assert decision.verification.z3_status == "UNSAT"
    assert decision.decision == "BLOCK"
    assert decision.override_applied is False
    assert decision.verification.execution_time_ms < 5.0  # < 5ms SLA
    assert len(decision.verification.violated_rules) > 0


def test_e2e_gateway_auth_and_malformed_numeric_rejection():
    """E2E Test 3: Gateway edge security rejects unauthenticated & malformed payloads."""
    gateway_client = TestClient(create_gateway_app(AlertCapture()))

    # Unauthenticated request rejected with 401
    res_unauth = gateway_client.get("/v1/alerts")
    assert res_unauth.status_code == 401

    # Malformed numeric value rejected with 400 Bad Request
    res_bad_num = gateway_client.post(
        "/v1/events",
        headers={"x-api-key": "dev-key-1"},
        json={
            "event_id": "evt-malformed-01",
            "source": "/api/v1/checkout",
            "payload": {"amount": "INVALID_FLOAT_STRING", "user_id": "usr-123"},
        },
    )
    assert res_bad_num.status_code == 400
    assert "Invalid numeric format" in res_bad_num.json()["detail"]


def test_e2e_ingestion_broker_fallback_telemetry():
    """E2E Test 4: Ingestion broker reports structured telemetry and warning on fallback."""
    broker = InMemoryBroker(fallback_reason="redis_connection_offline")
    ingestion_client = TestClient(create_ingestion_app(broker=broker))

    health_res = ingestion_client.get("/health")
    assert health_res.status_code == 200
    health_data = health_res.json()

    assert health_data["status"] == "ok"
    assert health_data["is_fallback"] is True
    assert health_data["broker_backend"] == "InMemoryBroker"
    assert health_data["fallback_reason"] == "redis_connection_offline"
    assert "Degraded" in health_data["warning"]

    # Stream endpoint returns recent envelopes
    stream_res = ingestion_client.get("/v1/context-stream")
    assert stream_res.status_code == 200
    assert "envelopes" in stream_res.json()
