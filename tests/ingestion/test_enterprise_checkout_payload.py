"""Integration test for large enterprise transaction checkout payload processing."""

from __future__ import annotations

import time
from fastapi.testclient import TestClient

from ingestion.app import create_app
from ingestion.broker import InMemoryBroker


def test_large_enterprise_checkout_payload_validation_and_benchmarking():
    """Verify that the enterprise checkout payload is ingested, masked, and context-fused in < 10ms."""
    broker = InMemoryBroker()
    client = TestClient(create_app(broker=broker))

    test_payload = {
        "metadata": {
            "method": "POST",
            "path": "/api/v2/enterprise/transactions/checkout-secure",
            "content_type": "application/json",
            "client_ip": "192.168.1.105",
            "client_role": "admin",
            "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.sample_enterprise_admin_token",
        },
        "payload": {
            "transaction_context": {
                "order_id": "ord_998877665544_secure",
                "subtotal_amount": 1450.75,
                "tax_amount": 116.06,
                "discount_rate": 0.15,
                "item_count": 8.0,
                "currency": "USD",
            },
            "customer_profile": {
                "full_name": "Alexander Pierce",
                "user_email": "alexander.pierce@enterprise-domain.org",
                "phone_number": "+1-555-019-2834",
                "account_tier": "VIP_PREMIUM",
                "backup_email": "alex.backup@external-mail.net",
            },
            "payment_gateway": {
                "credit_card": "4532-8899-1122-7766",
                "cvv": "889",
                "card_expiry": "12/28",
                "billing_address": "123 Cybernetic Way, Silicon Valley, CA",
            },
            "telemetry_metrics": {
                "client_cpu_load": 98.45,
                "memory_heap_used_mb": 4096.8,
                "request_frequency_per_sec": 3850.0,
                "network_latency_ms": 12.4,
            },
        },
        "threat_score": 0.94,
    }

    # Test via POST /v1/ingest
    t0 = time.perf_counter()
    resp = client.post("/v1/ingest", json=test_payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 202
    body = resp.json()

    # 1. PII Redaction Validation
    masked_customer = body["masked_payload"]["customer_profile"]
    assert masked_customer["full_name"] == "[REDACTED_PII]"
    assert masked_customer["user_email"] == "[REDACTED_PII]"
    assert masked_customer["backup_email"] == "[REDACTED_PII]"
    assert masked_customer["phone_number"] == "[REDACTED_PII]"
    assert masked_customer["account_tier"] == "VIP_PREMIUM"

    masked_payment = body["masked_payload"]["payment_gateway"]
    assert masked_payment["credit_card"] == "[REDACTED_PII]"
    assert masked_payment["cvv"] == "[REDACTED_PII]"
    assert masked_payment["card_expiry"] == "[REDACTED_PII]"
    assert masked_payment["billing_address"] == "[REDACTED_PII]"

    # 2. Syntax & Non-sensitive Structure Preservation
    tx_context = body["masked_payload"]["transaction_context"]
    assert tx_context["order_id"] == "ord_998877665544_secure"
    assert tx_context["currency"] == "USD"
    assert tx_context["subtotal_amount"] == 1450.75

    # 3. Unified Context Vector Dimensions and Values
    vector = body["unified_context_vector"]
    assert len(vector) == 9
    assert vector[0] == 0.94  # threat_score
    assert vector[1] == 1450.75  # subtotal_amount
    assert vector[2] == 116.06  # tax_amount
    assert vector[3] == 0.15  # discount_rate
    assert vector[4] == 8.0  # item_count
    assert vector[5] == 98.45  # client_cpu_load
    assert vector[6] == 4096.8  # memory_heap_used_mb
    assert vector[7] == 3850.0  # request_frequency_per_sec
    assert vector[8] == 12.4  # network_latency_ms

    # 4. Protobuf Binary Serialization
    assert body["protobuf_bytes_size"] > 0

    # 5. SLA Enforcement (< 10ms target)
    assert body["pipeline_metrics"]["execution_time_ms"] < 10.0
    assert elapsed_ms < 50.0  # Total HTTP overhead within bounds
