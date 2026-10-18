"""Dataset loaders and test harnesses for TransVerify Ingestion Benchmarks (WBS 4.5).

Supports:
1. HTTP CSIC 2010 Dataset: Normal (36k) vs Malicious (25k: SQLi, XSS, Parameter Tampering, Buffer Overflow).
2. UNSW-NB15 Anomaly Feature mapping.
3. Synthetic High-Throughput Enterprise Microservice Traffic Streams (x, y).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class IngestionBenchmarkRecord:
    source: str
    metadata: dict[str, Any]
    payload: dict[str, Any]
    threat_score: float
    attack_type: str | None = None
    is_anomaly: bool = False


class CSIC2010DataLoader:
    """Generates and loads HTTP CSIC 2010 request representations."""

    ATTACK_VECTORS = [
        ("sqli", "SELECT * FROM users WHERE id = '1' OR '1'='1'--", 0.95),
        ("xss", "<script>alert('XSS_PAYLOAD')</script>", 0.88),
        ("tampering", "admin=true&role=superuser&bypass=1", 0.78),
        ("buffer_overflow", "A" * 2048, 0.92),
    ]

    NORMAL_TEMPLATES = [
        {"action": "view_cart", "item_id": 104, "quantity": 1, "currency": "EUR"},
        {"action": "search", "query": "wireless headphones", "page": 2, "limit": 20},
        {"action": "checkout", "subtotal": 85.50, "tax": 7.20, "shipping": 5.00},
        {"action": "account_info", "account_id": 99281, "verified": True},
    ]

    def generate_csic_stream(
        self, normal_count: int = 36000, anomaly_count: int = 25000
    ) -> Iterator[IngestionBenchmarkRecord]:
        """Generate simulated CSIC 2010 stream with 36k normal and 25k malicious events."""
        records = []
        for i in range(normal_count):
            tpl = dict(random.choice(self.NORMAL_TEMPLATES))
            tpl["user_ref"] = f"usr-{i}"
            records.append(
                IngestionBenchmarkRecord(
                    source="/csic2010/normal",
                    metadata={
                        "method": "POST",
                        "path": "/api/shop",
                        "client_ip": f"10.0.{i % 255}.{(i // 255) % 255}",
                    },
                    payload=tpl,
                    threat_score=round(random.uniform(0.01, 0.25), 3),
                    attack_type=None,
                    is_anomaly=False,
                )
            )

        for i in range(anomaly_count):
            attack_type, payload_str, threat = random.choice(self.ATTACK_VECTORS)
            records.append(
                IngestionBenchmarkRecord(
                    source="/csic2010/anomalous",
                    metadata={
                        "method": "POST",
                        "path": "/api/shop",
                        "client_ip": f"198.51.100.{i % 254}",
                    },
                    payload={"input": payload_str, "attack_class": attack_type, "length": len(payload_str)},
                    threat_score=threat,
                    attack_type=attack_type,
                    is_anomaly=True,
                )
            )

        random.shuffle(records)
        yield from records


class UNSWNB15DataLoader:
    """Simulates UNSW-NB15 flow feature attributes mapped into HTTP context."""

    def generate_flow_stream(self, count: int = 10000) -> Iterator[IngestionBenchmarkRecord]:
        for i in range(count):
            is_attack = random.random() < 0.3
            threat_score = round(random.uniform(0.70, 0.99) if is_attack else random.uniform(0.01, 0.20), 3)
            payload = {
                "dur": round(random.uniform(0.001, 5.0), 4),
                "sbytes": random.randint(100, 50000),
                "dbytes": random.randint(100, 100000),
                "sttl": random.choice([31, 64, 254]),
                "sload": round(random.uniform(1000.0, 500000.0), 2),
                "dpkts": random.randint(1, 500),
            }
            yield IngestionBenchmarkRecord(
                source="/unsw-nb15/flows",
                metadata={"method": "POST", "path": "/telemetry/flow"},
                payload=payload,
                threat_score=threat_score,
                attack_type="anomalous_flow" if is_attack else None,
                is_anomaly=is_attack,
            )


class SyntheticEnterpriseStream:
    """Generates continuous paired enterprise traffic streams (x, y)."""

    def generate(self, batch_size: int = 1000) -> list[dict[str, Any]]:
        batch = []
        for i in range(batch_size):
            score = round(random.random(), 3)
            batch.append({
                "metadata": {
                    "method": "POST",
                    "path": "/v1/transactions",
                    "content_type": "application/json",
                    "client_ip": f"172.16.{random.randint(1, 10)}.{random.randint(1, 254)}",
                    "client_role": "billing_service",
                    "authorization": "Bearer eyJhbGciOi...",
                },
                "payload": {
                    "user_id": f"usr-{i}",
                    "email": f"client-{i}@firm.org",
                    "password": "temporary-token-string",
                    "credit_card": "4111 2222 3333 4444",
                    "amount": round(random.uniform(10.0, 5000.0), 2),
                    "fee": round(random.uniform(0.5, 15.0), 2),
                    "rate": 1.15,
                },
                "threat_score": score,
            })
        return batch


def load_csic_2010(normal_count: int = 36000, anomaly_count: int = 25000, limit: int | None = None) -> list[IngestionBenchmarkRecord]:
    """Helper to load CSIC 2010 records."""
    loader = CSIC2010DataLoader()
    stream = loader.generate_csic_stream(normal_count=normal_count, anomaly_count=anomaly_count)
    if limit is not None:
        result = []
        for item in stream:
            result.append(item)
            if len(result) >= limit:
                break
        return result
    return list(stream)


def load_unsw_nb15(count: int = 10000, limit: int | None = None) -> list[IngestionBenchmarkRecord]:
    """Helper to load UNSW-NB15 records."""
    loader = UNSWNB15DataLoader()
    stream = loader.generate_flow_stream(count=count)
    if limit is not None:
        result = []
        for item in stream:
            result.append(item)
            if len(result) >= limit:
                break
        return result
    return list(stream)


def generate_synthetic_microservice_traffic(count: int = 1000) -> list[dict[str, Any]]:
    """Helper to generate synthetic paired traffic streams (x, y)."""
    return SyntheticEnterpriseStream().generate(batch_size=count)
