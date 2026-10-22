"""Asynchronous AI threat inference for SZ-CFM.

No trained threat-detection model is connected to the ingestion runtime. The
LogiShield model in ``tokenization_ai`` is documented there as a demo fitted on
two synthetic fixtures, and module boundaries forbid importing it. Two backends
are therefore provided, and every result states whether it is simulated:

* :class:`DevelopmentHeuristicBackend` - a hand-weighted logistic function of the
  feature vector. ``is_simulated = True``. It exists to exercise the
  asynchronous correlation path; it is not a detection model.
* :class:`InferenceClientBackend` - adapts the shared ``InferenceClient``
  contract (for example ``HttpInferenceClient`` -> engine ``/v1/predict``). The
  caller must state whether the remote model is simulated; the engine's
  ``StubModel`` is.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from shared_contracts.interfaces import InferenceClient
from shared_contracts.models import FeatureVector

from .feature_schema import FEATURE_COUNT, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, validate_vector


class InferenceStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    REJECTED_BACKPRESSURE = "REJECTED_BACKPRESSURE"
    SKIPPED = "SKIPPED"
    LATE = "LATE"


class InferenceBackend(Protocol):
    model_version: str
    is_simulated: bool
    feature_schema_version: str
    input_dim: int

    def predict(self, vector: Sequence[float]) -> float:
        """Return a risk score in [0, 1] for one feature vector."""


@dataclass(frozen=True)
class InferenceTask:
    """One inference job; carries the identity needed to correlate its result."""

    slot_index: int
    generation: int
    request_id: str
    correlation_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class InferenceResult:
    slot_index: int
    generation: int
    request_id: str
    correlation_id: str
    status: InferenceStatus
    risk_score: float | None
    model_version: str
    simulated: bool
    latency_ns: int
    error: str | None = None


class DevelopmentHeuristicBackend:
    """SIMULATED model: fixed hand-chosen weights over feature schema v1."""

    model_version = "szcfm-dev-heuristic-1"
    is_simulated = True
    feature_schema_version = FEATURE_SCHEMA_VERSION
    input_dim = FEATURE_COUNT

    _WEIGHTS = {
        "ip_reputation_score": 2.5,
        "requests_last_10_seconds": 1.0,
        "failed_logins_last_5_minutes": 2.0,
        "request_size_ratio": 0.8,
        "rate_limit_exceeded": 1.5,
        "authentication_anomaly": 2.0,
        "sql_injection_indicator": 3.0,
        "xss_indicator": 3.0,
        "payload_attack_signature": 3.0,
        "payload_syntax_density": 2.0,
        "role_not_granted": 2.0,
        "authentication_missing": 1.0,
        "path_is_admin": 0.5,
        "mfa_verified": -0.5,
        "prior_threat_score": 1.0,
    }
    _BIAS = -3.5

    def __init__(self) -> None:
        self._weights = [self._WEIGHTS.get(name, 0.0) for name in FEATURE_NAMES]

    def predict(self, vector: Sequence[float]) -> float:
        logit = self._BIAS + sum(w * x for w, x in zip(self._weights, vector))
        return 1.0 / (1.0 + math.exp(-logit))


class InferenceClientBackend:
    """Adapter for the shared ``InferenceClient`` contract (named features)."""

    feature_schema_version = FEATURE_SCHEMA_VERSION
    input_dim = FEATURE_COUNT

    def __init__(self, client: InferenceClient, *, model_version: str, is_simulated: bool) -> None:
        self._client = client
        self.model_version = model_version
        self.is_simulated = is_simulated

    def predict(self, vector: Sequence[float]) -> float:
        prediction = self._client.predict(
            FeatureVector(
                event_id="szcfm-inference",
                features=dict(zip(FEATURE_NAMES, vector)),
                schema_version=FEATURE_SCHEMA_VERSION,
            )
        )
        self.model_version = prediction.model_version
        return prediction.score


ResultCallback = Callable[[InferenceResult], None]


class AsyncInferenceExecutor:
    """Bounded worker pool with non-blocking admission control.

    ``submit`` never waits: when ``max_in_flight`` tasks are already queued or
    running, it returns ``None`` and the caller treats the AI as unavailable.
    The in-flight permit is held until the backend call actually returns, so a
    request that timed out still counts against capacity while its inference
    runs (Python threads cannot be interrupted).
    """

    def __init__(self, backend: InferenceBackend, *, workers: int = 4, max_in_flight: int = 64) -> None:
        if workers < 1 or max_in_flight < workers:
            raise ValueError("require workers >= 1 and max_in_flight >= workers")
        if backend.feature_schema_version != FEATURE_SCHEMA_VERSION or backend.input_dim != FEATURE_COUNT:
            raise ValueError(
                f"backend expects feature schema {backend.feature_schema_version} "
                f"with {backend.input_dim} inputs; extractor produces schema "
                f"{FEATURE_SCHEMA_VERSION} with {FEATURE_COUNT}"
            )
        self.backend = backend
        self.max_in_flight = max_in_flight
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="szcfm-inference")
        self._permits = threading.BoundedSemaphore(max_in_flight)
        self._lock = threading.Lock()
        self._in_flight = 0
        self.rejected = 0

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def submit(self, task: InferenceTask, on_result: ResultCallback) -> Future[InferenceResult] | None:
        if not self._permits.acquire(blocking=False):
            with self._lock:
                self.rejected += 1
            return None
        with self._lock:
            self._in_flight += 1
        try:
            future = self._pool.submit(self._run, task)
        except BaseException:
            self._finish()
            raise

        def done(f: Future[InferenceResult]) -> None:
            self._finish()
            if not f.cancelled():
                on_result(f.result())

        future.add_done_callback(done)
        return future

    def _finish(self) -> None:
        with self._lock:
            self._in_flight -= 1
        self._permits.release()

    def _run(self, task: InferenceTask) -> InferenceResult:
        start = time.perf_counter_ns()
        status, score, error = InferenceStatus.COMPLETED, None, None
        try:
            validate_vector(task.vector, self.backend.input_dim)
            score = float(self.backend.predict(task.vector))
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"model returned out-of-range score {score!r}")
        except Exception as exc:  # reported as FAILED, never as a safe score
            status, score = InferenceStatus.FAILED, None
            error = f"{type(exc).__name__}: {exc}"[:300]
        return InferenceResult(
            slot_index=task.slot_index,
            generation=task.generation,
            request_id=task.request_id,
            correlation_id=task.correlation_id,
            status=status,
            risk_score=score,
            model_version=self.backend.model_version,
            simulated=self.backend.is_simulated,
            latency_ns=time.perf_counter_ns() - start,
            error=error,
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
