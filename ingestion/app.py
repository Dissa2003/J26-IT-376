"""FastAPI entry point for event ingestion."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException

from shared_contracts.config import get_settings
from shared_contracts.http_clients import HttpProcessingClient
from shared_contracts.interfaces import ProcessingClient
from shared_contracts.models import Prediction, RawEvent

from .broker import create_broker
from shared_contracts.interfaces import EventConsumer, EventProducer
from .validator import IngestionValidator

log = logging.getLogger("ingestion")


def create_app(
    processing: ProcessingClient | None = None,
    broker: EventProducer & EventConsumer | None = None,
) -> FastAPI:
    """Create the ingestion API with an injectable processing client."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    processing = processing or HttpProcessingClient(settings.processing_url)
    broker = broker or create_broker(settings)
    validator = IngestionValidator()
    results: dict[str, Prediction] = {}

    def forward(event: RawEvent) -> None:
        results[event.event_id] = processing.process(event)

    broker.subscribe(forward)
    app = FastAPI(title="ingestion")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return the service health status."""
        return {"status": "ok", "service": "ingestion"}

    @app.post("/v1/ingest", status_code=202)
    def ingest(payload: dict[str, Any] = Body(...)) -> Prediction:
        """Publish an event and return the downstream prediction."""
        try:
            event = validator.validate(payload)
        except ValueError as exc:
            reason = str(exc)
            publish_dlq = getattr(broker, "publish_dlq", None)
            if publish_dlq is not None:
                try:
                    publish_dlq(payload, reason)
                except Exception:
                    log.exception("Unable to write invalid event to the DLQ")
            raise HTTPException(422, reason) from exc
        try:
            broker.publish(event)
        except (httpx.HTTPError, ValueError) as exc:
            log.exception("downstream failure")
            raise HTTPException(502, f"downstream failure: {exc}") from exc
        return results[event.event_id]

    return app


app = create_app()