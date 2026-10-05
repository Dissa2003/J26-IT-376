"""FastAPI entry point for resilient event ingestion."""

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

from shared_contracts.config import get_settings
from shared_contracts.http_clients import HttpProcessingClient
from shared_contracts.interfaces import EventConsumer, EventProducer, ProcessingClient
from shared_contracts.models import Prediction, RawEvent

from .broker import InMemoryBroker, RedisStreamBroker, create_broker
from .validator import IngestionValidator

log = logging.getLogger("ingestion")


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

    app = FastAPI(title="ingestion")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return the service health status."""
        return {"status": "ok", "service": "ingestion"}

    @app.post("/v1/ingest", status_code=202)
    def ingest(payload: dict[str, Any] = Body(...)) -> Prediction | dict[str, str]:
        """Validate, publish, and await a bounded window for processing."""
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
            return {"status": "accepted", "event_id": event.event_id}
        except (httpx.HTTPError, ValueError) as exc:
            log.exception("Unable to queue event %s", event.event_id)
            raise HTTPException(503, f"event queued for retry: {exc}") from exc

    return app


app = create_app()
