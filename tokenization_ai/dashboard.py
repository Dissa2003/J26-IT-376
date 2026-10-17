"""Static host for the LogiShield dashboard at ``/dashboard``.

Presentation only. This module serves a small single-page client from the
separate ``dashboard_static`` directory and adds no analysis logic, no state
and no POST endpoint. The client talks to the *existing* ``POST /v1/analyze``
and ``GET /health`` of the same origin.

The original ``/demo`` page and its assets (``static/``) are untouched and
keep their own asset allowlist. Dashboard assets use a separate fixed
allowlist: the requested name is only ever looked up in a dict, never joined
into a path.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response

from shared_contracts.models import ErrorResponse

_STATIC_DIR = Path(__file__).parent / "dashboard_static"

DASHBOARD_ASSETS: dict[str, str] = {
    "dashboard.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "shell.js": "text/javascript; charset=utf-8",
    "api.js": "text/javascript; charset=utf-8",
    "ui.js": "text/javascript; charset=utf-8",
    "charts.js": "text/javascript; charset=utf-8",
    "routes.js": "text/javascript; charset=utf-8",
    "session.js": "text/javascript; charset=utf-8",
    "overview-model.js": "text/javascript; charset=utf-8",
    "page-overview.js": "text/javascript; charset=utf-8",
    "reports-model.js": "text/javascript; charset=utf-8",
    "export-format.js": "text/javascript; charset=utf-8",
    "evidence.js": "text/javascript; charset=utf-8",
    "page-reports.js": "text/javascript; charset=utf-8",
    "page-anomaly.js": "text/javascript; charset=utf-8",
    "page-logs.js": "text/javascript; charset=utf-8",
    "page-components.js": "text/javascript; charset=utf-8",
    "page-policy.js": "text/javascript; charset=utf-8",
    "page-policy-data.js": "text/javascript; charset=utf-8",
    "page-pending.js": "text/javascript; charset=utf-8",
    "logo.svg": "image/svg+xml",
}

_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def install_dashboard(app: FastAPI) -> None:
    """Register ``GET /dashboard`` and ``GET /dashboard/assets/{name}`` (hidden from OpenAPI)."""

    @app.get("/dashboard", include_in_schema=False)
    def dashboard_page():
        return HTMLResponse(
            (_STATIC_DIR / "index.html").read_text(encoding="utf-8"), headers=_HEADERS
        )

    @app.get("/dashboard/assets/{name}", include_in_schema=False)
    def dashboard_asset(name: str):
        media_type = DASHBOARD_ASSETS.get(name)  # fixed allowlist; no path joining on input
        if media_type is None:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(error="not_found", detail="Unknown dashboard asset.").model_dump(),
            )
        return Response((_STATIC_DIR / name).read_bytes(), media_type=media_type, headers=_HEADERS)
