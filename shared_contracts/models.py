"""Wire contracts shared by all modules. Changes here need approval from ALL members (see README)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Label(str, Enum):
    NORMAL = "normal"
    ANOMALY = "anomaly"


class Severity(str, Enum):
    LOW = "low"
    HIGH = "high"
    CRITICAL = "critical"


class RawEvent(BaseModel):
    """Ingestion -> Processing."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    timestamp: datetime = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class FeatureVector(BaseModel):
    """Processing -> Engine."""
    event_id: str
    features: dict[str, float]
    schema_version: str = SCHEMA_VERSION


class Prediction(BaseModel):
    """Engine -> Processing -> Ingestion (response path)."""
    event_id: str
    score: float = Field(ge=0.0, le=1.0)
    label: Label
    threshold: float
    model_version: str


class Alert(BaseModel):
    """Engine -> API Gateway."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    severity: Severity
    score: float
    message: str
    created_at: datetime = Field(default_factory=_now)


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None