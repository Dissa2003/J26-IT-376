"""SZ-CFM configuration, context-fusion formula and final security policy.

Nothing here comes from a published SZ-CFM specification (none exists in the
repository). The fusion formula, thresholds and rule set are documented
implementation choices; see ``docs/szcfm/README.md``.

Precedence: BLOCK > CHALLENGE > ALLOW. Authentication, authorization and
mandatory rules are evaluated before and independently of the model, so a low
model score can never lift a deterministic BLOCK or CHALLENGE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from .feature_schema import role_not_granted
from .szcfm_schema import DecisionReason, SZCFMEvent

Decision = Literal["ALLOW", "CHALLENGE", "BLOCK"]
AIFailureMode = Literal["FAIL_CLOSED", "CHALLENGE", "FAIL_OPEN"]
_RANK = {"ALLOW": 0, "CHALLENGE": 1, "BLOCK": 2}


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_map(name: str, default: dict[str, str]) -> dict[str, str]:
    """Parse ``prefix=value;prefix=value`` pairs."""
    raw = os.environ.get(name)
    if raw is None:
        return dict(default)
    pairs = (item.split("=", 1) for item in raw.split(";") if "=" in item)
    return {k.strip(): v.strip() for k, v in pairs}


@dataclass(frozen=True)
class SZCFMConfig:
    max_request_bytes: int = 32_768
    ring_capacity: int = 256
    ring_slot_bytes: int = 65_536
    stale_after_s: float = 30.0
    ai_timeout_s: float = 0.050
    inference_workers: int = 4
    max_in_flight: int = 64
    # Fusion: fused = max(ai, w * ai + (1 - w) * context_risk)
    ai_weight: float = 0.7
    block_threshold: float = 0.85
    challenge_threshold: float = 0.60
    ip_reputation_block: float = 0.90
    ip_reputation_challenge: float = 0.70
    failed_login_challenge: int = 10
    public_path_prefixes: tuple[str, ...] = ("/health", "/v1/public")
    role_required_prefixes: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"/v1/admin": ("admin",)}
    )
    mfa_required_prefixes: tuple[str, ...] = ("/v1/admin",)
    default_ai_failure_mode: AIFailureMode = "CHALLENGE"
    ai_failure_mode_by_prefix: dict[str, str] = field(
        default_factory=lambda: {"/v1/admin": "FAIL_CLOSED", "/v1/payments": "FAIL_CLOSED"}
    )
    # "accept": simulated scores are used (development); "unavailable": they are
    # treated like a missing AI result (recommended until a real model exists).
    simulated_ai_policy: Literal["accept", "unavailable"] = "accept"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ai_weight <= 1.0:
            raise ValueError("ai_weight must be in [0, 1]")
        if not 0.0 < self.challenge_threshold < self.block_threshold <= 1.0:
            raise ValueError("require 0 < challenge_threshold < block_threshold <= 1")
        modes = {self.default_ai_failure_mode, *self.ai_failure_mode_by_prefix.values()}
        if not modes <= {"FAIL_CLOSED", "CHALLENGE", "FAIL_OPEN"}:
            raise ValueError(f"unknown AI failure mode in {sorted(modes)}")
        if self.simulated_ai_policy not in ("accept", "unavailable"):
            raise ValueError("simulated_ai_policy must be 'accept' or 'unavailable'")
        if self.ai_timeout_s <= 0 or self.max_request_bytes < 1:
            raise ValueError("ai_timeout_s and max_request_bytes must be positive")

    @classmethod
    def from_env(cls) -> "SZCFMConfig":
        d = cls()
        e = os.environ.get
        roles = _env_map("SZCFM_ROLE_REQUIRED", {k: ",".join(v) for k, v in d.role_required_prefixes.items()})
        return cls(
            max_request_bytes=int(e("SZCFM_MAX_REQUEST_BYTES", d.max_request_bytes)),
            ring_capacity=int(e("SZCFM_RING_CAPACITY", d.ring_capacity)),
            ring_slot_bytes=int(e("SZCFM_RING_SLOT_BYTES", d.ring_slot_bytes)),
            stale_after_s=float(e("SZCFM_STALE_AFTER_S", d.stale_after_s)),
            ai_timeout_s=float(e("SZCFM_AI_TIMEOUT_S", d.ai_timeout_s)),
            inference_workers=int(e("SZCFM_INFERENCE_WORKERS", d.inference_workers)),
            max_in_flight=int(e("SZCFM_MAX_IN_FLIGHT", d.max_in_flight)),
            ai_weight=float(e("SZCFM_AI_WEIGHT", d.ai_weight)),
            block_threshold=float(e("SZCFM_BLOCK_THRESHOLD", d.block_threshold)),
            challenge_threshold=float(e("SZCFM_CHALLENGE_THRESHOLD", d.challenge_threshold)),
            ip_reputation_block=float(e("SZCFM_IP_REPUTATION_BLOCK", d.ip_reputation_block)),
            ip_reputation_challenge=float(e("SZCFM_IP_REPUTATION_CHALLENGE", d.ip_reputation_challenge)),
            failed_login_challenge=int(e("SZCFM_FAILED_LOGIN_CHALLENGE", d.failed_login_challenge)),
            public_path_prefixes=_env_list("SZCFM_PUBLIC_PATHS", d.public_path_prefixes),
            role_required_prefixes={k: tuple(r.strip() for r in v.split(",") if r.strip()) for k, v in roles.items()},
            mfa_required_prefixes=_env_list("SZCFM_MFA_REQUIRED_PATHS", d.mfa_required_prefixes),
            default_ai_failure_mode=e("SZCFM_AI_FAILURE_MODE", d.default_ai_failure_mode),  # type: ignore[arg-type]
            ai_failure_mode_by_prefix=_env_map("SZCFM_AI_FAILURE_MODE_BY_PATH", d.ai_failure_mode_by_prefix),
            simulated_ai_policy=e("SZCFM_SIMULATED_AI_POLICY", d.simulated_ai_policy),  # type: ignore[arg-type]
        )

    def ai_failure_mode(self, path: str) -> AIFailureMode:
        """Longest matching path prefix wins."""
        best, mode = -1, self.default_ai_failure_mode
        for prefix, value in self.ai_failure_mode_by_prefix.items():
            if _prefix_match(path, prefix) and len(prefix) > best:
                best, mode = len(prefix), value  # type: ignore[assignment]
        return mode


def _prefix_match(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


FUSION_FORMULA = "fused_risk = max(ai_risk, w*ai_risk + (1-w)*context_risk); context_risk = 0.5*ip_rep + 0.2*burst + 0.2*failed_logins + 0.1*auth_anomaly"


def context_risk(features: dict[str, float]) -> float:
    """Deterministic risk from gateway context, in [0, 1] (implementation choice)."""
    value = (
        0.5 * features["ip_reputation_score"]
        + 0.2 * features["requests_last_10_seconds"]
        + 0.2 * features["failed_logins_last_5_minutes"]
        + 0.1 * features["authentication_anomaly"]
    )
    return min(1.0, max(0.0, value))


def fuse(ai_risk: float, ctx_risk: float, ai_weight: float) -> float:
    """Context may raise the model's risk but never dilute it."""
    return max(ai_risk, ai_weight * ai_risk + (1.0 - ai_weight) * ctx_risk)


def threat_level(risk: float, config: SZCFMConfig) -> str:
    if risk >= config.block_threshold:
        return "HIGH"
    if risk >= config.challenge_threshold:
        return "MEDIUM"
    return "LOW"


@dataclass
class PolicyOutcome:
    reasons: list[DecisionReason] = field(default_factory=list)

    def add(self, code: str, origin: str, effect: Decision, detail: str) -> None:
        self.reasons.append(DecisionReason(code=code, origin=origin, effect=effect, detail=detail))  # type: ignore[arg-type]

    @property
    def decision(self) -> Decision:
        effects = [r.effect for r in self.reasons if r.effect != "ALLOW"]
        return max(effects, key=_RANK.__getitem__, default="ALLOW")  # type: ignore[return-value]


class PolicyEngine:
    def __init__(self, config: SZCFMConfig) -> None:
        self.config = config

    def evaluate_deterministic(self, event: SZCFMEvent, matched_signatures: tuple[str, ...]) -> PolicyOutcome:
        """Authentication, authorization and mandatory rules (no model input)."""
        cfg, meta = self.config, event.metadata
        auth, security, traffic = event.payload.authentication, event.payload.security_context, event.payload.traffic_context
        out = PolicyOutcome()
        public = any(_prefix_match(meta.path, p) for p in cfg.public_path_prefixes)

        # Authentication
        if not public:
            if not meta.authorization:
                out.add("AUTH_MISSING", "authentication", "BLOCK", "protected path without an Authorization header")
            if auth is None:
                out.add("AUTH_CONTEXT_MISSING", "authentication", "BLOCK", "protected path without an authentication context")
            elif meta.authorization:
                scheme = meta.authorization.split(" ", 1)[0]
                if scheme.casefold() != auth.type.casefold():
                    out.add("AUTH_SCHEME_MISMATCH", "authentication", "BLOCK", "authorization scheme differs from the authenticated type")

        # Authorization
        if role_not_granted(event):
            out.add("ROLE_NOT_GRANTED", "authorization", "BLOCK", "claimed client_role is not an authenticated role")
        granted = {r.casefold() for r in auth.roles} if auth else set()
        for prefix, roles in cfg.role_required_prefixes.items():
            if _prefix_match(meta.path, prefix) and not granted & {r.casefold() for r in roles}:
                out.add("ENDPOINT_ROLE_FORBIDDEN", "authorization", "BLOCK", f"{prefix} requires one of {sorted(roles)}")
        if any(_prefix_match(meta.path, p) for p in cfg.mfa_required_prefixes) and not (auth and auth.mfa_verified):
            out.add("MFA_REQUIRED", "authorization", "CHALLENGE", "path requires a verified MFA session")

        # Mandatory security rules
        if security is not None:
            if security.sql_injection_indicator:
                out.add("SQL_INJECTION_INDICATOR", "mandatory_rule", "BLOCK", "gateway reported a SQL injection indicator")
            if security.xss_indicator:
                out.add("XSS_INDICATOR", "mandatory_rule", "BLOCK", "gateway reported an XSS indicator")
            if security.rate_limit_exceeded:
                out.add("RATE_LIMIT_EXCEEDED", "mandatory_rule", "BLOCK", "gateway rate limit exceeded")
            if security.ip_reputation_score >= cfg.ip_reputation_block:
                out.add("IP_REPUTATION_BLOCK", "mandatory_rule", "BLOCK", f"ip reputation >= {cfg.ip_reputation_block}")
            elif security.ip_reputation_score >= cfg.ip_reputation_challenge:
                out.add("IP_REPUTATION_CHALLENGE", "mandatory_rule", "CHALLENGE", f"ip reputation >= {cfg.ip_reputation_challenge}")
            if security.authentication_anomaly:
                out.add("AUTHENTICATION_ANOMALY", "mandatory_rule", "CHALLENGE", "gateway reported an authentication anomaly")
        if matched_signatures:
            out.add("PAYLOAD_ATTACK_SIGNATURE", "mandatory_rule", "BLOCK", f"request body matched {list(matched_signatures)}")
        if traffic is not None and traffic.failed_logins_last_5_minutes >= cfg.failed_login_challenge:
            out.add("FAILED_LOGIN_BURST", "mandatory_rule", "CHALLENGE", f"failed logins >= {cfg.failed_login_challenge}")
        return out

    def apply_model(self, outcome: PolicyOutcome, fused_risk: float, simulated: bool) -> None:
        cfg = self.config
        note = " (simulated development model)" if simulated else ""
        if fused_risk >= cfg.block_threshold:
            outcome.add("MODEL_HIGH_RISK", "model", "BLOCK", f"fused risk {fused_risk:.4f} >= {cfg.block_threshold}{note}")
        elif fused_risk >= cfg.challenge_threshold:
            outcome.add("MODEL_ELEVATED_RISK", "model", "CHALLENGE", f"fused risk {fused_risk:.4f} >= {cfg.challenge_threshold}{note}")
        else:
            outcome.add("MODEL_LOW_RISK", "model", "ALLOW", f"fused risk {fused_risk:.4f} < {cfg.challenge_threshold}{note}")

    def apply_ai_unavailable(self, outcome: PolicyOutcome, path: str, status: str) -> AIFailureMode:
        mode = self.config.ai_failure_mode(path)
        effect: Decision = {"FAIL_CLOSED": "BLOCK", "CHALLENGE": "CHALLENGE", "FAIL_OPEN": "ALLOW"}[mode]  # type: ignore[assignment]
        outcome.add(f"AI_UNAVAILABLE_{mode}", "availability", effect, f"AI assessment {status}; failure mode {mode}")
        return mode
