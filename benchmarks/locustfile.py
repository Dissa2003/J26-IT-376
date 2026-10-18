"""Locust automated high-throughput stress-testing script (WBS 4.5 | NFR1 & NFR2).

Configured for 200-second stress-test runs generating 10,000 to 20,000 requests/sec.
Target endpoint: POST /v1/ingest (Line-rate HTTP Interception & Context Fusion).
"""

import random
import time
from locust import HttpUser, task, between, events

METHODS = ["POST", "PUT", "PATCH", "GET"]
PATHS = [
    "/api/v1/orders",
    "/api/v1/payments",
    "/api/v1/auth/token",
    "/api/v1/users/profile",
    "/api/v1/checkout",
]
ROLES = ["admin", "analyst", "guest", "service_worker", "customer"]


class IngestionPipelineUser(HttpUser):
    # Minimal wait time to simulate line-rate continuous traffic
    wait_time = between(0.0001, 0.001)

    @task(8)
    def submit_interception_payload(self):
        """High-frequency traffic stream (x) merged with AI threat score (y)."""
        threat_score = round(random.betavariate(1.5, 5.0), 3)  # Real-world skewed threat distribution
        payload = {
            "metadata": {
                "method": random.choice(METHODS),
                "path": random.choice(PATHS),
                "content_type": "application/json",
                "client_ip": f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                "client_role": random.choice(ROLES),
                "authorization": f"Bearer token-{random.randint(100000, 999999)}",
                "headers": {
                    "x-forwarded-for": "10.0.0.1",
                    "x-request-id": f"req-{random.randint(10000, 99999)}",
                },
            },
            "payload": {
                "email": f"user{random.randint(1, 50000)}@enterprise.com",
                "password": f"pwd-secret-{random.randint(100, 999)}",
                "card_number": "4111 1111 1111 1111",
                "transaction_amount": round(random.uniform(5.0, 999.0), 2),
                "item_count": random.randint(1, 15),
                "status_code": 200,
            },
            "threat_score": threat_score,
        }

        with self.client.post(
            "/v1/ingest",
            json=payload,
            catch_response=True,
            name="POST /v1/ingest (Interception)",
        ) as response:
            if response.status_code == 202:
                data = response.json()
                if "unified_context_vector" in data and "protobuf_bytes_size" in data:
                    response.success()
                else:
                    response.failure("Malformed interception acknowledgement")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def submit_async_ai_risk_score(self):
        """Asynchronous AI risk score harness (WBS 4.2 / FR2)."""
        event_id = f"evt-stream-{random.randint(10000, 99999)}"
        score = round(random.random(), 3)
        self.client.post(
            "/v1/threat-score",
            json={"event_id": event_id, "threat_score": score},
            name="POST /v1/threat-score (Async y)",
        )

    @task(1)
    def check_health(self):
        """Low-latency liveness verification."""
        self.client.get("/health", name="GET /health")
