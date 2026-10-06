"""FastAPI entry point for resilient event ingestion and HTTP interception."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
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
from .pii_masker import PIIMasker
from .validator import IngestionValidator
from .vector_builder import UnifiedContextVectorBuilder

log = logging.getLogger("ingestion")
INGESTION_TAG = "Data Ingestion"


class AcceptedEvent(BaseModel):
    """Acknowledgement returned when downstream processing is still pending."""

    status: str = Field(default="accepted", examples=["accepted"])
    event_id: str = Field(description="Identifier retained for asynchronous retry")

    model_config = {
        "json_schema_extra": {
            "example": {"status": "accepted", "event_id": "evt-queued-001"}
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "metadata": {
                        "method": "POST",
                        "path": "/orders",
                        "content_type": "application/json",
                    },
                    "payload": {
                        "email": "customer@example.com",
                        "password": "do-not-send-me",
                        "amount": 149.95,
                    },
                    "threat_score": 0.88,
                }
            ]
        }
    }


class InterceptionAcknowledgement(BaseModel):
    """Immediate capture acknowledgement with serialized context metadata."""

    event_id: str
    masked_payload: dict[str, Any]
    protobuf_bytes_size: int = Field(ge=0)
    unified_context_vector: list[float]
    pipeline_metrics: dict[str, float]

    model_config = {
        "json_schema_extra": {
            "example": {
                "event_id": "evt-001",
                "masked_payload": {
                    "email": "[REDACTED_PII]",
                    "password": "[REDACTED_PII]",
                    "amount": 149.95,
                },
                "protobuf_bytes_size": 128,
                "unified_context_vector": [0.88, 149.95],
                "pipeline_metrics": {"execution_time_ms": 1.42},
            }
        }
    }


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
    pii_masker = PIIMasker()
    vector_builder = UnifiedContextVectorBuilder()
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
            202: {
                "description": "Event accepted and captured for downstream processing.",
                "content": {
                    "application/json": {
                        "examples": {
                            "captured": {
                                "summary": "Masked event with context vector",
                                "value": {
                                    "event_id": "evt-001",
                                    "masked_payload": {
                                        "email": "[REDACTED_PII]",
                                        "amount": 149.95,
                                    },
                                    "protobuf_bytes_size": 128,
                                    "unified_context_vector": [0.88, 149.95],
                                    "pipeline_metrics": {
                                        "execution_time_ms": 1.42
                                    },
                                },
                            },
                            "queued": {
                                "summary": "Standard event awaiting processing",
                                "value": {
                                    "status": "accepted",
                                    "event_id": "evt-queued-001",
                                },
                            },
                        }
                    }
                },
            },
            422: {
                "model": ErrorResponse,
                "description": "The event payload failed ingestion validation.",
                "content": {
                    "application/json": {
                        "example": {
                            "error": "validation_error",
                            "detail": "source must be non-empty",
                        }
                    }
                },
            },
            502: {
                "model": ErrorResponse,
                "description": "The ingestion broker is unavailable.",
                "content": {
                    "application/json": {
                        "example": {
                            "error": "bad_gateway",
                            "detail": "ingestion broker unavailable",
                        }
                    }
                },
            },
        },
    )
    def ingest(
        payload: dict[str, Any] = Body(
            ...,
            description=(
                "RawEvent payload or interceptor envelope. Interceptor payloads "
                "are synchronously PII-masked before vectorization and publication."
            ),
            examples={
                "interception_with_pii": {
                    "summary": "Interceptor request containing PII",
                    "value": {
                        "metadata": {
                            "method": "POST",
                            "path": "/orders",
                            "content_type": "application/json",
                        },
                        "payload": {
                            "email": "customer@example.com",
                            "password": "secret-value",
                            "client_ip": "192.0.2.10",
                            "amount": 149.95,
                        },
                        "threat_score": 0.88,
                    },
                },
                "invalid_event": {
                    "summary": "Invalid event",
                    "value": {"source": "", "payload": {"amount": "unknown"}},
                },
            },
        ),
    ) -> Prediction | AcceptedEvent | InterceptionAcknowledgement:
        """Validate, publish, and await processing for standard events."""
        started_ns = time.perf_counter_ns()
        interception = "metadata" in payload and "threat_score" in payload
        if interception:
            try:
                request = InterceptionRequest.model_validate(payload)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            masked_payload = pii_masker.mask(request.payload)
            vector = vector_builder.build(masked_payload, request.threat_score)
            event = RawEvent(
                source=request.metadata.path,
                payload={
                    **masked_payload,
                    "_http_metadata": request.metadata.model_dump(),
                    "_threat_score": request.threat_score,
                },
            )
            protobuf_bytes = vector_builder.serialize(
                event.event_id,
                masked_payload,
                request.threat_score,
                vector,
            )
            try:
                event_broker.publish(event)
            except (httpx.HTTPError, RedisError, OSError) as exc:
                log.exception("Unable to publish intercepted event %s", event.event_id)
                raise HTTPException(502, f"ingestion broker unavailable: {exc}") from exc
            return InterceptionAcknowledgement(
                event_id=event.event_id,
                masked_payload=masked_payload,
                protobuf_bytes_size=len(protobuf_bytes),
                unified_context_vector=vector,
                pipeline_metrics={
                    "execution_time_ms": (
                        max(0, time.perf_counter_ns() - started_ns) / 1_000_000
                    )
                },
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
        event = event.model_copy(
            update={"payload": pii_masker.mask(event.payload)}
        )

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
