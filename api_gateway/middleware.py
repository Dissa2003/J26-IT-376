from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from shared_contracts.config import get_settings

log = logging.getLogger("gateway")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key not in get_settings().api_keys:
        raise HTTPException(401, "invalid or missing API key")


def install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error")
            return JSONResponse({"error": "internal_error"}, status_code=500)
        log.info("%s %s -> %s (%.1fms)", request.method, request.url.path,
                 response.status_code, (time.perf_counter() - start) * 1000)
        return response