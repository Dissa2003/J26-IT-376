from __future__ import annotations

import httpx
from fastapi import Depends, FastAPI, HTTPException

from shared_contracts.config import get_settings
from shared_contracts.interfaces import Notifier
from shared_contracts.models import Alert, Prediction, RawEvent

from .middleware import install_middleware, require_api_key
from .notifiers import LogNotifier


def create_app(notifier: Notifier | None = None, ingestion_url: str | None = None) -> FastAPI:
    s = get_settings()
    notifier = notifier or LogNotifier()
    ingestion_url = (ingestion_url or s.ingestion_url).rstrip("/")
    alerts: list[Alert] = []
    app = FastAPI(title="api_gateway")
    install_middleware(app)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "api_gateway"}

    @app.post("/v1/events", dependencies=[Depends(require_api_key)], status_code=202)
    def submit_event(event: RawEvent) -> Prediction:
        try:
            r = httpx.post(f"{ingestion_url}/v1/ingest", content=event.model_dump_json(),
                           headers={"content-type": "application/json"}, timeout=10)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(exc.response.status_code, exc.response.text) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"ingestion unavailable: {exc}") from exc
        return Prediction.model_validate(r.json())

    @app.get("/v1/alerts", dependencies=[Depends(require_api_key)])
    def list_alerts() -> list[Alert]:
        return alerts

    # Internal (network-private) endpoint called by the engine.
    @app.post("/internal/alerts", status_code=201)
    def receive_alert(alert: Alert) -> Alert:
        alerts.append(alert)
        notifier.notify(alert)
        return alert

    return app


app = create_app()