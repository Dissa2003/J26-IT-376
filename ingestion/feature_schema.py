"""Versioned, fixed-order Unified Context Vector for SZ-CFM (feature schema v1).

Every feature is a float in [0, 1]. The order is defined once by ``FEATURES``
and never depends on payload key order. Count-like inputs use
``log1p(x) / log1p(cap)`` so that values spanning several orders of magnitude
stay comparable to the binary indicators. Missing context blocks are encoded
with an explicit neutral default plus a ``*_missing`` indicator feature, so a
missing value is never confused with a genuine zero.

The vector contains no identifiers, tokens, session values, IP addresses or
free text: only derived numeric signals.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .szcfm_schema import MAX_BODY_DEPTH, SZCFMEvent

FEATURE_SCHEMA_VERSION = "1"

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
_AUTH_PATH = re.compile(r"/(login|logout|auth|oauth|token|session|password)(/|$)", re.IGNORECASE)
_ADMIN_PATH = re.compile(r"/admin(/|$)", re.IGNORECASE)

# Baseline signature set (implementation choice, not exhaustive). Each pattern
# targets a specific attack form; bare quotes or semicolons do not match, so
# ordinary text such as "O'Brien" is not flagged.
ATTACK_SIGNATURES: dict[str, re.Pattern[str]] = {
    "sqli_union_select": re.compile(r"\bunion\b\s+(all\s+)?\bselect\b", re.IGNORECASE),
    "sqli_tautology": re.compile(r"['\"]\s*(or|and)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+", re.IGNORECASE),
    "sqli_stacked_ddl": re.compile(r";\s*(drop|truncate|alter|delete|insert|update)\s+\w+", re.IGNORECASE),
    "sqli_comment_terminator": re.compile(r"['\"]\s*(--|#|/\*)"),
    "xss_script_tag": re.compile(r"<\s*script\b", re.IGNORECASE),
    "xss_event_handler": re.compile(r"\bon(error|load|click|mouseover|focus)\s*=", re.IGNORECASE),
    "xss_js_uri": re.compile(r"javascript\s*:", re.IGNORECASE),
    "path_traversal": re.compile(r"\.\./|\.\.\\|%2e%2e%2f", re.IGNORECASE),
}
_SYNTAX_CHARS = frozenset("'\";<>=()`$%{}|&")


def _log_scaled(value: float, cap: float) -> float:
    return min(1.0, math.log1p(max(0.0, value)) / math.log1p(cap))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    units: str
    normalization: str
    missing_default: float | None


@dataclass(frozen=True)
class PayloadScan:
    """Attack-signature scan of the client-controlled request body."""

    matched_signatures: tuple[str, ...]
    syntax_density: float
    depth: int
    string_chars: int


@dataclass(frozen=True)
class FeatureVectorResult:
    schema_version: str
    names: tuple[str, ...]
    values: tuple[float, ...]
    missing_groups: tuple[str, ...]
    payload_scan: PayloadScan

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("ip_reputation_score", "Gateway IP reputation risk", "probability-like score", "as given, [0,1]", 0.5),
    FeatureSpec("security_context_missing", "security_context block absent", "indicator", "0/1", None),
    FeatureSpec("method_is_mutating", "HTTP method changes state (POST/PUT/PATCH/DELETE)", "indicator", "0/1", None),
    FeatureSpec("method_is_delete", "HTTP method is DELETE", "indicator", "0/1", None),
    FeatureSpec("path_depth", "Number of path segments", "segments", "min(n/10, 1)", None),
    FeatureSpec("path_is_admin", "Path contains an /admin segment", "indicator", "0/1", None),
    FeatureSpec("path_is_auth", "Path is an authentication endpoint", "indicator", "0/1", None),
    FeatureSpec("request_size", "Current request size", "bytes", "log1p(x)/log1p(max_request_bytes)", 0.0),
    FeatureSpec("request_size_ratio", "Current / average request size", "ratio", "min(ratio/10, 1); avg=0 -> 1 if current>0", 0.0),
    FeatureSpec("requests_per_minute", "Requests from the client in the last minute", "requests/min", "log1p(x)/log1p(600)", 0.0),
    FeatureSpec("requests_last_10_seconds", "Burst count in the last 10 seconds", "requests/10s", "log1p(x)/log1p(100)", 0.0),
    FeatureSpec("failed_logins_last_5_minutes", "Failed logins in the last 5 minutes", "count", "min(x/10, 1)", 0.0),
    FeatureSpec("endpoint_requests_last_minute", "Requests to this endpoint in the last minute", "requests/min", "log1p(x)/log1p(600)", 0.0),
    FeatureSpec("previous_requests_from_ip", "Historic requests seen from the IP", "count", "log1p(x)/log1p(10000)", 0.0),
    FeatureSpec("traffic_context_missing", "traffic_context block absent", "indicator", "0/1", None),
    FeatureSpec("rate_limit_exceeded", "Gateway rate limit exceeded", "indicator", "0/1", 0.0),
    FeatureSpec("authentication_anomaly", "Gateway authentication anomaly", "indicator", "0/1", 0.0),
    FeatureSpec("sql_injection_indicator", "Gateway SQL injection indicator", "indicator", "0/1", 0.0),
    FeatureSpec("xss_indicator", "Gateway XSS indicator", "indicator", "0/1", 0.0),
    FeatureSpec("payload_attack_signature", "Ingestion's own signature scan matched", "indicator", "0/1", None),
    FeatureSpec("payload_syntax_density", "Share of syntax metacharacters in body strings", "ratio", "[0,1]; no strings -> 0", None),
    FeatureSpec("payload_depth", "Request body nesting depth", "levels", f"min(depth/{MAX_BODY_DEPTH}, 1)", None),
    FeatureSpec("authorization_present", "Authorization header present", "indicator", "0/1", None),
    FeatureSpec("authentication_missing", "authentication block absent", "indicator", "0/1", None),
    FeatureSpec("mfa_verified", "MFA verified for the session", "indicator", "0/1", 0.0),
    FeatureSpec("role_not_granted", "Claimed client_role is not among the authenticated roles", "indicator", "0/1", 0.0),
    FeatureSpec("prior_threat_score", "Upstream prior risk estimate (root threat_score)", "score", "as given, [0,1]", 0.0),
    FeatureSpec("prior_threat_score_missing", "Root threat_score absent", "indicator", "0/1", None),
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURES)
FEATURE_COUNT = len(FEATURES)


def scan_payload(body: Mapping[str, Any]) -> PayloadScan:
    """Scan every string key and value of the body for attack signatures."""
    matched: set[str] = set()
    total = syntax = 0

    def visit(value: Any, depth: int) -> int:
        nonlocal total, syntax
        if isinstance(value, str):
            total += len(value)
            syntax += sum(1 for ch in value if ch in _SYNTAX_CHARS)
            for name, pattern in ATTACK_SIGNATURES.items():
                if name not in matched and pattern.search(value):
                    matched.add(name)
            return depth
        if isinstance(value, Mapping):
            return max([visit(k, depth) for k in value] + [visit(v, depth + 1) for v in value.values()], default=depth)
        if isinstance(value, list):
            return max((visit(v, depth + 1) for v in value), default=depth)
        return depth

    depth = visit(dict(body), 1) if body else 0
    return PayloadScan(
        matched_signatures=tuple(sorted(matched)),
        syntax_density=(syntax / total) if total else 0.0,
        depth=depth,
        string_chars=total,
    )


def role_not_granted(event: SZCFMEvent) -> bool:
    """True when metadata.client_role is not one of the authenticated roles."""
    auth = event.payload.authentication
    role = event.metadata.client_role
    if role is None or auth is None:
        return False
    return role.casefold() not in {r.casefold() for r in auth.roles}


class FeatureExtractor:
    """Deterministic feature extraction for feature schema version 1."""

    schema_version = FEATURE_SCHEMA_VERSION
    names = FEATURE_NAMES

    def __init__(self, max_request_bytes: int) -> None:
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        self._max_request_bytes = max_request_bytes

    def extract(self, event: SZCFMEvent) -> FeatureVectorResult:
        meta, payload = event.metadata, event.payload
        security, traffic, auth = payload.security_context, payload.traffic_context, payload.authentication
        scan = scan_payload(payload.request_body)
        missing = [
            name
            for name, block in (
                ("security_context", security),
                ("traffic_context", traffic),
                ("authentication", auth),
            )
            if block is None
        ]
        if event.threat_score is None:
            missing.append("prior_threat_score")

        segments = [s for s in meta.path.split("?")[0].split("/") if s]
        ratio = 0.0
        if traffic is not None:
            avg, cur = traffic.average_request_size_bytes, traffic.current_request_size_bytes
            ratio = min(cur / avg, 10.0) / 10.0 if avg > 0 else (1.0 if cur > 0 else 0.0)
        values: dict[str, float] = {
            "ip_reputation_score": security.ip_reputation_score if security else 0.5,
            "security_context_missing": float(security is None),
            "method_is_mutating": float(meta.method in _MUTATING),
            "method_is_delete": float(meta.method == "DELETE"),
            "path_depth": min(len(segments) / 10.0, 1.0),
            "path_is_admin": float(bool(_ADMIN_PATH.search(meta.path))),
            "path_is_auth": float(bool(_AUTH_PATH.search(meta.path))),
            "request_size": _log_scaled(traffic.current_request_size_bytes, self._max_request_bytes) if traffic else 0.0,
            "request_size_ratio": ratio,
            "requests_per_minute": _log_scaled(traffic.requests_per_minute, 600) if traffic else 0.0,
            "requests_last_10_seconds": _log_scaled(traffic.requests_last_10_seconds, 100) if traffic else 0.0,
            "failed_logins_last_5_minutes": min(traffic.failed_logins_last_5_minutes / 10.0, 1.0) if traffic else 0.0,
            "endpoint_requests_last_minute": _log_scaled(traffic.endpoint_requests_last_minute, 600) if traffic else 0.0,
            "previous_requests_from_ip": _log_scaled(traffic.previous_requests_from_ip, 10_000) if traffic else 0.0,
            "traffic_context_missing": float(traffic is None),
            "rate_limit_exceeded": float(security.rate_limit_exceeded) if security else 0.0,
            "authentication_anomaly": float(security.authentication_anomaly) if security else 0.0,
            "sql_injection_indicator": float(security.sql_injection_indicator) if security else 0.0,
            "xss_indicator": float(security.xss_indicator) if security else 0.0,
            "payload_attack_signature": float(bool(scan.matched_signatures)),
            "payload_syntax_density": _clamp(scan.syntax_density),
            "payload_depth": min(scan.depth / MAX_BODY_DEPTH, 1.0),
            "authorization_present": float(bool(meta.authorization)),
            "authentication_missing": float(auth is None),
            "mfa_verified": float(auth.mfa_verified) if auth else 0.0,
            "role_not_granted": float(role_not_granted(event)),
            "prior_threat_score": event.threat_score if event.threat_score is not None else 0.0,
            "prior_threat_score_missing": float(event.threat_score is None),
        }
        vector = tuple(float(values[name]) for name in FEATURE_NAMES)
        validate_vector(vector)
        return FeatureVectorResult(
            schema_version=FEATURE_SCHEMA_VERSION,
            names=FEATURE_NAMES,
            values=vector,
            missing_groups=tuple(missing),
            payload_scan=scan,
        )


def validate_vector(vector: tuple[float, ...] | list[float], expected_dim: int = FEATURE_COUNT) -> None:
    """Raise ``ValueError`` unless the vector has the schema dimension and finite [0,1] values."""
    if len(vector) != expected_dim:
        raise ValueError(f"feature vector has {len(vector)} values, expected {expected_dim}")
    for name, value in zip(FEATURE_NAMES, vector):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"feature {name} is {value!r}; expected a finite value in [0, 1]")

