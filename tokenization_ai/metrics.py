"""Runtime metrics for the Member 4 pipeline, derived from a :class:`PipelineTrace`.

Scope -- runtime measurements only
-----------------------------------
Everything in this module measures *how long the pipeline took and how
big its intermediate artifacts were* for whatever input it was actually
given. That is true and useful regardless of whether the input was a
synthetic demo fixture or real traffic.

This module deliberately does **not** compute precision, recall, F1, or
any other detection-accuracy metric. Member 4 has no real labeled
dataset as of this phase (see the Phase 9/10 planning notes and
``test_service.py``'s synthetic-PP1-prototype test docstring), so there
is no ground truth to score predictions against. A number that *looks*
like a quality metric but is computed from hand-picked synthetic labels
would invite exactly the kind of "it works" misreading this project is
trying to avoid -- so that computation does not exist here at all. Any
future accuracy/precision/recall evaluation must be implemented as its
own clearly-labeled, clearly-sourced addition once a real labeled
dataset exists, not folded into this module's output.

Every :class:`RuntimeMetrics` instance reports measurements for exactly
one request; it carries no label distinguishing "synthetic demo" from
"real traffic" because this module has no way to know which it was
given -- callers (e.g. ``demo.py``) are responsible for stating that
context wherever these numbers are displayed or logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .tracing import STAGE_MODEL_INFERENCE, STAGE_TENSORIZATION, PipelineTrace


@dataclass(frozen=True)
class RuntimeMetrics:
    """Measured runtime performance for one pipeline execution.

    ``token_count`` and ``truncated`` are ``None`` whenever the supplied
    trace has no tensorization-stage entry (e.g. the pipeline failed
    before reaching that stage) -- never fabricated or inferred from
    something else.
    """

    event_id: str
    stage_elapsed_ms: dict[str, float]
    total_elapsed_ms: float
    model_inference_ms: float | None
    token_count: int | None
    truncated: bool | None


def runtime_metrics_from_trace(trace: PipelineTrace) -> RuntimeMetrics:
    """Build a :class:`RuntimeMetrics` from one request's completed trace.

    Reads only values already present on ``trace`` -- every number here
    is a direct measurement recorded by ``tracing.PipelineTracer``
    during the actual pipeline run, not a re-derived or estimated one.
    """
    stage_elapsed_ms = {stage.stage: stage.elapsed_ms for stage in trace.stages}

    tensorization = next(
        (stage for stage in trace.stages if stage.stage == STAGE_TENSORIZATION and stage.success),
        None,
    )
    token_count = tensorization.summary.get("token_count") if tensorization else None
    truncated = (
        bool(tensorization.summary["truncated"])
        if tensorization is not None and "truncated" in tensorization.summary
        else None
    )

    model_stage = next(
        (stage for stage in trace.stages if stage.stage == STAGE_MODEL_INFERENCE and stage.success),
        None,
    )
    model_inference_ms = model_stage.elapsed_ms if model_stage is not None else None

    return RuntimeMetrics(
        event_id=trace.event_id,
        stage_elapsed_ms=stage_elapsed_ms,
        total_elapsed_ms=trace.total_elapsed_ms,
        model_inference_ms=model_inference_ms,
        token_count=token_count,
        truncated=truncated,
    )


def aggregate_runtime_metrics(metrics: Iterable[RuntimeMetrics]) -> dict[str, float]:
    """Mean/min/max total pipeline latency across a batch of :class:`RuntimeMetrics`.

    Purely descriptive statistics over whatever timings it is given --
    carries no claim about detection accuracy or real-world traffic
    volume; see the module docstring. Raises ``ValueError`` if ``metrics``
    is empty, since a mean/min/max of nothing is not a meaningful value
    to silently report as e.g. ``0.0``.
    """
    totals = [item.total_elapsed_ms for item in metrics]
    if not totals:
        raise ValueError("aggregate_runtime_metrics requires at least one RuntimeMetrics")
    return {
        "count": len(totals),
        "mean_ms": sum(totals) / len(totals),
        "min_ms": min(totals),
        "max_ms": max(totals),
    }
