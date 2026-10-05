from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException

from shared_contracts.config import get_settings
from shared_contracts.http_clients import HttpProcessingClient
from shared_contracts.interfaces import ProcessingClient
from shared_contracts.models import Prediction, RawEvent

from .broker import InMemoryBroker

log = logging.getLogger("ingestion")


def create_app(processing: ProcessingClient | None = None) -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    processing = processing or HttpProcessingClient(settings.processing_url)
    broker = InMemoryBroker()
    results: dict[str, Prediction] = {}

    def forward(event: RawEvent) -> None:
        results[event.event_id] = processing.process(event)

    broker.subscribe(forward)
    app = FastAPI(title="ingestion")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "ingestion"}

    @app.post("/v1/ingest", status_code=202)
    def ingest(event: RawEvent) -> Prediction:
        try:
            broker.publish(event)
        except (httpx.HTTPError, ValueError) as exc:
            log.exception("downstream failure")
            raise HTTPException(502, f"downstream failure: {exc}") from exc
        return results[event.event_id]

    return app


app = create_app()