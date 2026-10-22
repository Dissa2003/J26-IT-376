"""Canonical SZ-CFM event envelope and pipeline result models.

The envelope keeps the existing interception shape (``metadata`` + ``payload`` +
optional ``threat_score``) so that interceptor clients do not change, but every
level is validated strictly: unknown fields are rejected instead of being
silently dropped, numeric context values are bounded, and only the free-form
``request_body`` accepts arbitrary (finite, JSON-compatible) content.

``source`` contract: ``RawEvent`` already defines ``source`` at the root, so the
envelope accepts it there. When an interceptor does not set it, the source is
taken from the gateway-captured ``x-request-source`` header. If neither is
present the event is rejected with an explicit error instead of the bare
"Field required" that ``RawEvent`` produced for interception payloads.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EVENT_SCHEMA_VERSION = "1"
SUPPORTED_EVENT_SCHEMA_VERSIONS = frozenset({EVENT_SCHEMA_VERSION})
SOURCE_HEADER = "x-request-source"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SOURCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]{1,128}$")
_HTTP_METHODS = {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}
MAX_BODY_DEPTH = 32

_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _check_json_value(value: Any, path: str, depth: int) -> None:
    """Reject non-JSON, non-finite, or excessively nested values."""
    if depth > MAX_BODY_DEPTH:
        raise ValueError(f"{path} exceeds maximum nesting depth {MAX_BODY_DEPTH}")
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} contains an empty or non-string key")
            _check_json_value(child, f"{path}.{key}", depth + 1)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _check_json_value(child, f"{path}[{index}]", depth + 1)
        return
    raise ValueError(f"{path} has an unsupported value type")


class EnvelopeMetadata(BaseModel):
    """HTTP request metadata captured by the gateway/interceptor."""

    model_config = _STRICT

    method: str
    path: str = Field(min_length=1, max_length=2048)
    content_type: str = Field(default="application/json", min_length=1, max_length=255)
    client_ip: str | None = Field(default=None, max_length=64)
    client_role: str | None = Field(default=None, max_length=64)
    authorization: str | None = Field(default=None, max_length=8192)
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def _method(cls, value: str) -> str:
        method = value.upper()
        if method not in _HTTP_METHODS:
            raise ValueError(f"method must be one of {sorted(_HTTP_METHODS)}")
        return method

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("path must start with '/'")
        return value

    @field_validator("headers")
    @classmethod
    def _headers(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 100:
            raise ValueError("at most 100 headers are accepted")
        normalized: dict[str, str] = {}
        for name, header_value in value.items():
            if not _HEADER_NAME.fullmatch(name):
                raise ValueError(f"invalid header name: {name!r}")
            if len(header_value) > 4096:
                raise ValueError(f"header {name!r} exceeds 4096 characters")
            normalized[name.lower()] = header_value
        return normalized


class ClientInfo(BaseModel):
    """Client device descriptors (client-controlled)."""

    model_config = _STRICT

    client_id: str | None = Field(default=None, max_length=128)
    device_type: str | None = Field(default=None, max_length=64)
    operating_system: str | None = Field(default=None, max_length=64)
    browser: str | None = Field(default=None, max_length=64)
    country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")


class AuthenticationContext(BaseModel):
    """Authentication result asserted by the gateway's identity layer."""

    model_config = _STRICT

    type: str = Field(min_length=1, max_length=32)
    user_id: str | None = Field(default=None, max_length=128)
    roles: list[str] = Field(default_factory=list, max_length=32)
    session_id: str | None = Field(default=None, max_length=256)
    mfa_verified: bool = False


class TrafficContext(BaseModel):
    """Gateway-computed traffic counters. All values are non-negative counts."""

    model_config = _STRICT

    requests_per_minute: int = Field(ge=0)
    requests_last_10_seconds: int = Field(ge=0)
    failed_logins_last_5_minutes: int = Field(ge=0)
    previous_requests_from_ip: int = Field(ge=0)
    endpoint_requests_last_minute: int = Field(ge=0)
    average_request_size_bytes: int = Field(ge=0)
    current_request_size_bytes: int = Field(ge=0)


class SecurityContext(BaseModel):
    """Gateway-computed security indicators."""

    model_config = _STRICT

    ip_reputation_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rate_limit_exceeded: bool = False
    authentication_anomaly: bool = False
    sql_injection_indicator: bool = False
    xss_indicator: bool = False


class EnvelopePayload(BaseModel):
    """Request payload and gateway context blocks."""

    model_config = _STRICT

    request_id: str | None = None
    correlation_id: str | None = None
    client: ClientInfo | None = None
    authentication: AuthenticationContext | None = None
    request_body: dict[str, Any] = Field(default_factory=dict)
    traffic_context: TrafficContext | None = None
    security_context: SecurityContext | None = None

    @field_validator("request_id", "correlation_id")
    @classmethod
    def _identifier(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER.fullmatch(value):
            raise ValueError(
                "must be 1-128 characters of letters, digits, '.', '_', ':' or '-'"
            )
        return value

    @field_validator("request_body")
    @classmethod
    def _body(cls, value: dict[str, Any]) -> dict[str, Any]:
        _check_json_value(value, "request_body", 1)
        return value


class SZCFMEvent(BaseModel):
    """Canonical SZ-CFM ingestion envelope (schema version 1)."""

    model_config = _STRICT

    schema_version: str = EVENT_SCHEMA_VERSION
    source: str | None = None
    metadata: EnvelopeMetadata
    payload: EnvelopePayload
    threat_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "Optional upstream prior risk estimate. It is used as one input "
            "feature only; it is never treated as the AI assessment."
        ),
    )

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, value: str) -> str:
        if value not in SUPPORTED_EVENT_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {value!r}; supported: "
                f"{sorted(SUPPORTED_EVENT_SCHEMA_VERSIONS)}"
            )
        return value

    @model_validator(mode="after")
    def _resolve_source(self) -> "SZCFMEvent":
        source = self.source or self.metadata.headers.get(SOURCE_HEADER)
        if not source:
            raise ValueError(
                "source is required: set the root 'source' field or the "
                f"metadata.headers['{SOURCE_HEADER}'] header"
            )
        if not _SOURCE.fullmatch(source):
            raise ValueError(
                "source must be 1-64 characters of letters, digits, '.', '_' or '-'"
            )
        self.source = source
        return self


# --- Result models -----------------------------------------------------------


class SerializationResult(BaseModel):
    format: Literal["protobuf"] = "protobuf"
    success: bool
    bytes_size: int = Field(ge=0)
    sha256: str | None = None
    schema_version: str
    message_type: str
    error: str | None = None


class FeatureExtractionResult(BaseModel):
    schema_version: str
    feature_count: int
    feature_names: list[str]
    vector: list[float]
    missing_groups: list[str]


class RequestStateResult(BaseModel):
    slot_allocated: bool
    slot_index: int | None = None
    generation: int | None = None
    state: str
    released: bool
    ring_occupancy: int
    ring_capacity: int


class AIAssessmentResult(BaseModel):
    status: str
    model_version: str | None = None
    simulated: bool | None = None
    risk_score: float | None = None
    threat_level: str | None = None
    error: str | None = None


class ContextFusionResult(BaseModel):
    status: str
    request_context_matched: bool
    stale_result: bool
    duplicate_result: bool
    delivery_outcome: str | None = None
    context_risk: float | None = None
    fused_risk: float | None = None
    formula: str
    contributing_inputs: list[str]


class DecisionReason(BaseModel):
    code: str
    origin: Literal["authentication", "authorization", "mandatory_rule", "model", "availability", "policy"]
    effect: Literal["ALLOW", "CHALLENGE", "BLOCK"]
    detail: str


class SecurityDecisionResult(BaseModel):
    decision: Literal["ALLOW", "CHALLENGE", "BLOCK"]
    reason_codes: list[str]
    reasons: list[DecisionReason]
    deterministic_decision: Literal["ALLOW", "CHALLENGE", "BLOCK"]
    ai_failure_mode: str | None = None
    committed: bool


class ForwardingResult(BaseModel):
    attempted: bool
    status: str
    detail: str | None = None


class PipelineMetrics(BaseModel):
    execution_time_ms: float
    validation_time_ms: float
    masking_time_ms: float
    serialization_time_ms: float
    feature_extraction_time_ms: float
    ai_inference_time_ms: float | None
    ai_wait_time_ms: float | None
    context_fusion_time_ms: float
    decision_time_ms: float


class SZCFMResult(BaseModel):
    """Complete pipeline result returned by ``POST /v1/szcfm/events``."""

    event_id: str
    request_id: str
    correlation_id: str
    source: str
    ingested_at: str
    event_status: Literal["COMPLETED", "FAILED"]
    masked_payload: dict[str, Any]
    serialization: SerializationResult
    feature_extraction: FeatureExtractionResult
    request_state: RequestStateResult
    ai_assessment: AIAssessmentResult
    context_fusion: ContextFusionResult
    security_decision: SecurityDecisionResult
    forwarding: ForwardingResult
    publication: dict[str, Any]
    pipeline_metrics: PipelineMetrics
