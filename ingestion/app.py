"""FastAPI entry point for resilient event ingestion and HTTP interception."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from shared_contracts.config import get_settings
from shared_contracts.http_clients import HttpProcessingClient
from shared_contracts.interfaces import EventConsumer, EventProducer, ProcessingClient
from shared_contracts.models import ErrorResponse, Prediction, RawEvent

from .broker import InMemoryBroker, create_broker
from .validator import IngestionValidator

log = logging.getLogger("ingestion")
INGESTION_TAG = "Data Ingestion"


class AcceptedEvent(BaseModel):
    """Acknowledgement returned when downstream processing is still pending."""

    status: str = Field(default="accepted", examples=["accepted"])
    event_id: str = Field(description="Identifier retained for asynchronous retry")


class HttpRequestMetadata(BaseModel):
    """Metadata captured with an intercepted HTTP request."""

    method: str = Field(default="POST", min_length=1)
    path: str = Field(default="/v1/ingest", min_length=1)
    content_type: str = Field(default="application/json", min_length=1)
    client_ip: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class InterceptionRequest(BaseModel):
    """Explicit wire schema for request metadata, payload, and threat score."""

    metadata: HttpRequestMetadata = Field(default_factory=HttpRequestMetadata)
    payload: dict[str, Any] = Field(
        description="Raw request payload attributes captured as x."
    )
    threat_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilistic AI threat score y in the range [0, 1].",
    )


class InterceptionAcknowledgement(BaseModel):
    """Immediate capture acknowledgement with nanosecond timing metrics."""

    status: str = "accepted"
    event_id: str
    received_at: datetime
    acknowledged_at: datetime
    capture_latency_ns: int = Field(ge=0)
    payload_bytes: int = Field(ge=0)


@dataclass
class _ProcessingTask:
    """A queued event and the future completed by the processing worker."""

    event: RawEvent
    result: Future[Prediction]


class _ProcessingDispatcher:
    """Submit events to processing asynchronously and retry transient failures."""

    def __init__(self, processing: ProcessingClient) -> None:
        """Start the single worker that drains the durable-in-process queue."""
        self._processing = processing
        self._tasks: Queue[_ProcessingTask] = Queue()
        self._worker = threading.Thread(
            target=self._run,
            name="ingestion-processing-dispatcher",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        event: RawEvent,
        result: Future[Prediction] | None = None,
    ) -> Future[Prediction]:
        """Queue an event without performing downstream I/O on the broker thread."""
        task_result = result or Future()
        self._tasks.put(_ProcessingTask(event, task_result))
        return task_result

    def _run(self) -> None:
        """Retry each event until processing succeeds or the application stops."""
        while True:
            task = self._tasks.get()
            delay = 0.25
            while not task.result.done():
                try:
                    prediction = self._processing.process(task.event)
                except Exception:
                    log.exception(
                        "Processing unavailable for event %s; retrying in %.2fs",
                        task.event.event_id,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 5.0)
                else:
                    task.result.set_result(prediction)


def create_app(
    processing: ProcessingClient | None = None,
    broker: EventProducer & EventConsumer | None = None,
    processing_timeout: float = 5.0,
) -> FastAPI:
    """Create the ingestion API with injectable processing and broker clients."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    processing_client = processing or HttpProcessingClient(settings.processing_url)
    event_broker = broker or create_broker(settings)
    validator = IngestionValidator()
    dispatcher = _ProcessingDispatcher(processing_client)
    pending: dict[str, Future[Prediction]] = {}

    def forward(event: RawEvent) -> None:
        """Queue a consumed event for asynchronous processing."""
        result = pending.get(event.event_id)
        pending[event.event_id] = dispatcher.submit(event, result)

    if isinstance(event_broker, InMemoryBroker):
        event_broker.subscribe(forward)
    else:
        threading.Thread(
            target=event_broker.subscribe,
            args=(forward,),
            name="ingestion-event-consumer",
            daemon=True,
        ).start()

    app = FastAPI(
        title="Dissanayaka Ingestion API",
        description=(
            "Validates incoming real-time events, publishes them to the configured "
            "stream broker, and forwards them to the processing pipeline."
        ),
        version="1.0.0",
        openapi_tags=[
            {
                "name": INGESTION_TAG,
                "description": "Validate and publish real-time events.",
            }
        ],
    )

    @app.get(
        "/health",
        tags=[INGESTION_TAG],
        summary="Check ingestion service health",
        description="Returns a liveness response for the ingestion service.",
    )
    def health() -> dict[str, str]:
        """Return the service health status."""
        return {"status": "ok", "service": "ingestion"}

    @app.post(
        "/v1/ingest",
        tags=[INGESTION_TAG],
        summary="Ingest or capture a real-time event",
        description=(
            "Accepts either a RawEvent-compatible payload or an HTTP interception "
            "payload with metadata and a threat score."
        ),
        status_code=202,
        response_model=Prediction | AcceptedEvent | InterceptionAcknowledgement,
        responses={
            422: {
                "model": ErrorResponse,
                "description": "The event payload failed ingestion validation.",
            },
            502: {
                "model": ErrorResponse,
                "description": "The ingestion broker is unavailable.",
            },
        },
    )
    def ingest(
        payload: dict[str, Any] = Body(..., description="Ingestion request payload."),
    ) -> Prediction | AcceptedEvent | InterceptionAcknowledgement:
        """Validate, publish, and await processing for standard events."""
        started_ns = time.perf_counter_ns()
        interception = "metadata" in payload and "threat_score" in payload
        if interception:
            try:
                request = InterceptionRequest.model_validate(payload)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            received_at = datetime.now(timezone.utc)
            event = RawEvent(
                source=request.metadata.path,
                payload={
                    **request.payload,
                    "_http_metadata": request.metadata.model_dump(),
                    "_threat_score": request.threat_score,
                },
            )
            try:
                event_broker.publish(event)
            except (httpx.HTTPError, RedisError, OSError) as exc:
                log.exception("Unable to publish intercepted event %s", event.event_id)
                raise HTTPException(502, f"ingestion broker unavailable: {exc}") from exc
            acknowledged_at = datetime.now(timezone.utc)
            return InterceptionAcknowledgement(
                event_id=event.event_id,
                received_at=received_at,
                acknowledged_at=acknowledged_at,
                capture_latency_ns=max(0, time.perf_counter_ns() - started_ns),
                payload_bytes=len(event.model_dump_json()),
            )

        try:
            event = validator.validate(payload)
        except ValueError as exc:
            reason = str(exc)
            publish_dlq = getattr(event_broker, "publish_dlq", None)
            if publish_dlq is not None:
                try:
                    publish_dlq(payload, reason)
                except Exception:
                    log.exception("Unable to write invalid event to the DLQ")
            raise HTTPException(422, reason) from exc

        result: Future[Prediction] = Future()
        pending[event.event_id] = result
        try:
            event_broker.publish(event)
            return result.result(timeout=processing_timeout)
        except TimeoutError:
            log.warning(
                "Processing did not respond within %.2fs; event %s remains queued",
                processing_timeout,
                event.event_id,
            )
            return AcceptedEvent(event_id=event.event_id)
        except (httpx.HTTPError, RedisError, OSError) as exc:
            log.exception("Unable to publish event %s", event.event_id)
            raise HTTPException(502, f"ingestion broker unavailable: {exc}") from exc

    return app


app = create_app()
