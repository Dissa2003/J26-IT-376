"""Redis cache for verify-request decisions.

The cache only wraps ``PolicyVerificationEngine.check_request``; it never
changes a decision. Any Redis problem (not running, timeout, bad data) is
treated as a cache miss so the request falls through to Z3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from .models import RequestDecisionReport, SecurityPolicy

log = logging.getLogger(__name__)

KEY_PREFIX = "policy-verifier:verify-request:v1:"
DEFAULT_TTL_SECONDS = 3600
# After a Redis failure, skip Redis for this long instead of paying a connect
# timeout on every request (a stopped WSL instance times out rather than refusing).
RETRY_AFTER_SECONDS = 30.0
SOCKET_TIMEOUT_SECONDS = 0.1


def cache_key(
    policies: Sequence[SecurityPolicy],
    subject_id: str,
    roles: Sequence[str],
    action: str,
    resource: str,
) -> str:
    """MD5 of "{policy_text}|{subject}|{action}|{resource}".

    ``policy_text`` covers every submitted policy (the decision depends on all of
    them) and ``subject`` includes the role assignment (the decision depends on
    it), so two requests share a key only if Z3 would see identical input.
    """
    policy_text = json.dumps(
        [
            [p.policy_id, p.subject, p.action, p.resource, p.effect.value,
             p.interpretation.value, p.raw_text]
            for p in policies
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    subject = json.dumps([subject_id, sorted(set(roles))], ensure_ascii=False, separators=(",", ":"))
    raw = f"{policy_text}|{subject}|{action}|{resource}"
    return KEY_PREFIX + hashlib.md5(raw.encode("utf-8")).hexdigest()


class RequestDecisionCache:
    def __init__(
        self,
        client: Any | None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        retry_after_seconds: float = RETRY_AFTER_SECONDS,
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._retry_after = retry_after_seconds
        self._skip_until = 0.0

    @classmethod
    def from_url(cls, url: str | None) -> "RequestDecisionCache":
        if not url:
            return cls(None)
        # No client-side retries: redis-py's default backoff waits seconds before
        # giving up, and RETRY_AFTER_SECONDS already handles a Redis that is down.
        client = redis.Redis.from_url(
            url,
            socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
            socket_timeout=SOCKET_TIMEOUT_SECONDS,
            retry=Retry(NoBackoff(), 0),
        )
        return cls(client)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _available(self) -> bool:
        return self._client is not None and time.monotonic() >= self._skip_until

    def _failed(self, operation: str, exc: Exception) -> None:
        self._skip_until = time.monotonic() + self._retry_after
        log.warning(
            "Redis %s failed (%s); using Z3 directly for %.0fs",
            operation, type(exc).__name__, self._retry_after,
        )

    def get(self, key: str) -> RequestDecisionReport | None:
        if not self._available():
            return None
        try:
            raw = self._client.get(key)
        except (redis.RedisError, OSError) as exc:
            self._failed("read", exc)
            return None
        if raw is None:
            return None
        try:
            return RequestDecisionReport.model_validate_json(raw)
        except ValueError:
            log.warning("Ignoring unreadable cached decision for %s", key)
            return None

    def set(self, key: str, report: RequestDecisionReport) -> None:
        # An indeterminate solver result may not repeat, so it is never cached.
        if report.reason_code == "solver_unknown" or not self._available():
            return
        try:
            self._client.setex(key, self._ttl, report.model_dump_json())
        except (redis.RedisError, OSError) as exc:
            self._failed("write", exc)
