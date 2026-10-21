"""Tests for the dashboard ingestion proxy in tokenization_ai.app."""

import pytest
from fastapi.testclient import TestClient

from ingestion.app import create_app as create_ingestion_app
from ingestion.broker import InMemoryBroker
from tokenization_ai.app import create_app as create_logishield_app
from tokenization_ai.demo import _build_demo_service


@pytest.fixture
def standalone_client():
    """Client pointing to a non-existent port to test 502 handling."""
    app = create_logishield_app(
        service=_build_demo_service(),
        ingestion_url="http://127.0.0.1:59999",
    )
    return TestClient(app)


def test_ingestion_proxy_handles_service_offline_gracefully(standalone_client):
    res = standalone_client.get("/api/ingestion/health")
    assert res.status_code == 502
    body = res.json()
    assert body["error"] == "ingestion_service_unavailable"
    assert "Ingestion service is unavailable" in body["detail"]


def test_ingestion_proxy_intercept_end_to_end():
    # Spin up in-memory ingestion app and test direct proxy contract
    ingestion_app = create_ingestion_app(broker=InMemoryBroker())
    ingestion_client = TestClient(ingestion_app)

    # Verify the payload structure that the dashboard page submits
    payload = {
        "metadata": {
            "method": "POST",
            "path": "/v1/orders",
            "content_type": "application/json",
            "client_ip": "198.51.100.42",
            "client_role": "customer",
            "headers": {"Authorization": "Bearer secret-token-123"},
        },
        "payload": {
            "email": "test@example.com",
            "password": "super-secret-password",
            "amount": 149.95,
        },
        "threat_score": 0.25,
    }

    res = ingestion_client.post("/v1/intercept", json=payload)
    assert res.status_code == 200
    ack = res.json()
    assert "event_id" in ack
    assert ack["masked_payload"]["password"] == "[REDACTED_PII]"
    assert ack["masked_payload"]["email"] == "[REDACTED_PII]"
    assert ack["masked_payload"]["amount"] == 149.95
    assert ack["protobuf_bytes_size"] > 0
    assert len(ack["unified_context_vector"]) >= 2
    assert ack["unified_context_vector"][0] == 0.25
    assert ack["pipeline_metrics"]["execution_time_ms"] >= 0
