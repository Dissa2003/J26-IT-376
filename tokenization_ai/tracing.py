"""Per-stage tracing of the Member 4 pipeline for debugging and auditability.

Each traced request gets its own :class:`PipelineTrace`, created fresh by
:meth:`PipelineTracer.start`. There is no module-level or instance-level
mutable store keyed by event id or otherwise shared across requests --
a trace is only ever mutated by the single caller holding the object
returned from ``start`` -- so concurrent or interleaved requests cannot
accidentally read or append to each other's trace.

Privacy contract
-----------------
A :class:`StageTrace` may only ever carry:

- the stage name (one of ``PIPELINE_STAGES``)
- an elapsed time in milliseconds
- a success/failure flag
- an exception's *type name* on failure (e.g. ``"ValueError"``), never
  its message or arguments, since a message can legitimately quote
  identifiers or lengths that a caller might still consider sensitive
  in combination
- a ``summary`` mapping restricted to small integers/booleans (counts,
  flags) -- never raw field values, field names/paths, header values,
  tokens, or any other upstream content

Callers (see ``service.py``) are responsible for only ever passing
count-shaped summaries into :meth:`PipelineTrace.record`; this module
does not itself inspect or sanitize ``summary`` values, so it cannot by
itself prevent a caller from passing something unsafe -- see
``service.py``'s stage summaries for the values actually recorded in
this codebase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

STAGE_FUSION = "fusion"
STAGE_MASKING = "masking"
STAGE_NORMALIZATION = "normalization"
STAGE_TOKENIZATION = "tokenization"
STAGE_TENSORIZATION = "tensorization"
STAGE_MODEL_INFERENCE = "model_inference"

PIPELINE_STAGES = (
    STAGE_FUSION,
    STAGE_MASKING,
    STAGE_NORMALIZATION,
    STAGE_TOKENIZATION,
    STAGE_TENSORIZATION,
    STAGE_MODEL_INFERENCE,
)


@dataclass(frozen=True)
class StageTrace:
    """One stage's timing and safe structural summary for one request.

    ``elapsed_ms`` is always ``>= 0`` -- it is derived from
    ``time.perf_counter_ns()``, a monotonic clock, so it can never be
    negative or go backward even if the wall clock is adjusted mid-request.
    """

    stage: str
    elapsed_ms: float
    success: bool
    summary: dict[str, int | bool] = field(default_factory=dict)
    error_type: str | None = None


class PipelineTrace:
    """One request's ordered, append-only collection of :class:`StageTrace` entries.

    Identified only by ``event_id`` (already known to every Member 4
    caller; not itself sensitive). Owned exclusively by whoever received
    it from :meth:`PipelineTracer.start` -- nothing in this module hands
    the same instance to two different requests.
    """

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        self._stages: list[StageTrace] = []

    @property
    def stages(self) -> tuple[StageTrace, ...]:
        return tuple(self._stages)

    @property
    def total_elapsed_ms(self) -> float:
        """Sum of every recorded stage's elapsed time, in milliseconds."""
        return sum(stage.elapsed_ms for stage in self._stages)

    def record(
        self,
        stage: str,
        elapsed_ms: float,
        *,
        success: bool = True,
        summary: dict[str, int | bool] | None = None,
        error_type: str | None = None,
    ) -> None:
        """Append one stage's result. See the module docstring's privacy contract."""
        self._stages.append(
            StageTrace(
                stage=stage,
                elapsed_ms=elapsed_ms,
                success=success,
                summary=dict(summary or {}),
                error_type=error_type,
            )
        )


class PipelineTracer:
    """Stateless factory for per-request :class:`PipelineTrace` objects.

    Holds no per-request state itself (no dict keyed by event id), so
    there is nothing here that could mix up two requests' traces --
    each call to :meth:`start` returns a brand-new, independent object.
    """

    def start(self, event_id: str) -> PipelineTrace:
        """Begin a new, empty trace for one request."""
        return PipelineTrace(event_id)

    def run_stage(
        self,
        trace: PipelineTrace | None,
        stage: str,
        fn: Callable[[], object],
        *,
        summary_fn: Callable[[object], dict[str, int | bool]] | None = None,
    ) -> object:
        """Call ``fn()``, optionally timing and recording it into ``trace``.

        If ``trace`` is ``None``, calls ``fn()`` directly with no timing
        overhead and no side effect -- this is what makes tracing truly
        optional for existing callers. Otherwise times ``fn()`` with
        ``time.perf_counter_ns()`` (monotonic; immune to wall-clock
        adjustments) and records one :class:`StageTrace`:

        - on success, ``summary_fn(result)`` (if given) supplies the safe
          structural summary to record alongside the timing
        - on failure, only ``type(exc).__name__`` is recorded -- never
          ``str(exc)`` -- then the exception is re-raised unchanged so
          callers see the identical error they would without tracing

        Returns ``fn()``'s result (success case) or raises (failure case).
        """
        if trace is None:
            return fn()
        start = time.perf_counter_ns()
        try:
            result = fn()
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            trace.record(stage, elapsed_ms, success=False, error_type=type(exc).__name__)
            raise
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        summary = summary_fn(result) if summary_fn is not None else {}
        trace.record(stage, elapsed_ms, success=True, summary=summary)
        return result
