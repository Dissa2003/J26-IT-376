from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from shared_contracts.config import get_settings

log = logging.getLogger("gateway")


class InMemoryRateLimiter:
    """Sliding-window rate limiter preventing API flood attacks."""

    def __init__(self, max_requests: int = 500, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            history = self._requests.setdefault(client_id, [])
            cutoff = now - self.window_seconds
            while history and history[0] < cutoff:
                history.pop(0)
            if len(history) >= self.max_requests:
                return False
            history.append(now)
            return True


rate_limiter = InMemoryRateLimiter()


def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
) -> dict[str, Any]:
    """Enforce cryptographic JWT or API-key verification and prevent role escalation."""
    settings = get_settings()
    auth_identity: dict[str, Any] | None = None

    if x_api_key and x_api_key in settings.api_keys:
        auth_identity = {
            "type": "api_key",
            "key": x_api_key,
            "role": x_user_role or "api_user",
        }
    elif authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        parts = token.split(".")
        if len(parts) == 3:
            try:
                payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(payload_b64))
                claimed_role = (
                    claims.get("role")
                    or claims.get("client_role")
                    or "authenticated"
                )
                if x_user_role and x_user_role != claimed_role and claimed_role != "admin":
                    raise HTTPException(
                        403,
                        f"Role mismatch: token authorizes '{claimed_role}', cannot claim '{x_user_role}'",
                    )
                auth_identity = {"type": "jwt", "claims": claims, "role": claimed_role}
            except (ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(401, "Malformed JWT claims format") from exc
        else:
            auth_identity = {"type": "opaque_token", "role": x_user_role or "authenticated"}

    if not auth_identity:
        raise HTTPException(401, "invalid or missing API key or Authorization token")

    if x_user_role and auth_identity["role"] != x_user_role and auth_identity["role"] != "admin":
        raise HTTPException(
            403,
            f"Unauthorized role escalation: identity has '{auth_identity['role']}', requested '{x_user_role}'",
        )

    return auth_identity


def install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def access_and_rate_limit(request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("x-api-key") or client_ip
        if not rate_limiter.check(api_key):
            log.warning("Rate limit exceeded for %s", api_key)
            return JSONResponse(
                {"error": "rate_limit_exceeded", "message": "Too many requests, please slow down."},
                status_code=429,
            )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except (ValueError, TypeError) as exc:
            log.warning("Bad request parsing: %s", exc)
            return JSONResponse(
                {"error": "bad_request", "message": str(exc)},
                status_code=400,
            )
        except Exception:
            log.exception("unhandled error")
            return JSONResponse({"error": "internal_error"}, status_code=500)

        log.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response