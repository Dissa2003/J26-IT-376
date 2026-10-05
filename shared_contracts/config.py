"""Environment-based config loader (reads .env if present, real env vars win)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@dataclass(frozen=True)
class Settings:
    app_env: str = "local"
    log_level: str = "INFO"
    ingestion_url: str = "http://localhost:8001"
    processing_url: str = "http://localhost:8002"
    engine_url: str = "http://localhost:8003"
    gateway_url: str = "http://localhost:8000"
    redis_url: str = "redis://localhost:6379/0"
    anomaly_threshold: float = 0.8
    model_version: str = "stub-0.1"
    api_keys: tuple[str, ...] = field(default_factory=lambda: ("dev-key-1",))
    webhook_url: str = ""


def get_settings() -> Settings:
    _load_dotenv()
    e = os.environ.get
    d = Settings()
    return Settings(
        app_env=e("APP_ENV", d.app_env),
        log_level=e("LOG_LEVEL", d.log_level),
        ingestion_url=e("INGESTION_URL", d.ingestion_url),
        processing_url=e("PROCESSING_URL", d.processing_url),
        engine_url=e("ENGINE_URL", d.engine_url),
        gateway_url=e("GATEWAY_URL", d.gateway_url),
        redis_url=e("REDIS_URL", d.redis_url),
        anomaly_threshold=float(e("ANOMALY_THRESHOLD", d.anomaly_threshold)),
        model_version=e("MODEL_VERSION", d.model_version),
        api_keys=tuple(k.strip() for k in e("API_KEYS", ",".join(d.api_keys)).split(",") if k.strip()),
        webhook_url=e("WEBHOOK_URL", d.webhook_url),
    )