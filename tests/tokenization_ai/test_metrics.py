"""Unit tests for tokenization_ai.metrics."""

import pytest

from tokenization_ai.metrics import aggregate_runtime_metrics, runtime_metrics_from_trace
from tokenization_ai.tracing import PipelineTrace


def _full_trace(event_id: str = "e1") -> PipelineTrace:
    trace = PipelineTrace(event_id)
    trace.record("fusion", 1.0, summary={"rule_count": 0})
    trace.record("masking", 2.0, summary={"masked_field_count": 1})
    trace.record("normalization", 1.5, summary={"masked_field_count": 1})
    trace.record("tokenization", 3.0, summary={"token_count": 10, "masked_token_count": 1})
    trace.record("tensorization", 2.5, summary={"token_count": 10, "truncated": False})
    trace.record("model_inference", 4.0)
    return trace


def test_runtime_metrics_from_trace_sums_total_elapsed_ms() -> None:
    metrics = runtime_metrics_from_trace(_full_trace())

    assert metrics.total_elapsed_ms == pytest.approx(14.0)


def test_runtime_metrics_from_trace_reports_per_stage_elapsed_ms() -> None:
    metrics = runtime_metrics_from_trace(_full_trace())

    assert metrics.stage_elapsed_ms == {
        "fusion": 1.0,
        "masking": 2.0,
        "normalization": 1.5,
        "tokenization": 3.0,
        "tensorization": 2.5,
        "model_inference": 4.0,
    }


def test_runtime_metrics_from_trace_reports_model_inference_ms() -> None:
    metrics = runtime_metrics_from_trace(_full_trace())

    assert metrics.model_inference_ms == 4.0


def test_runtime_metrics_from_trace_reports_token_count_and_truncated() -> None:
    metrics = runtime_metrics_from_trace(_full_trace())

    assert metrics.token_count == 10
    assert metrics.truncated is False


def test_runtime_metrics_from_trace_preserves_event_id() -> None:
    metrics = runtime_metrics_from_trace(_full_trace("distinct-event"))

    assert metrics.event_id == "distinct-event"


def test_runtime_metrics_from_trace_with_no_tensorization_stage_leaves_fields_none() -> None:
    trace = PipelineTrace("e1")
    trace.record("fusion", 1.0)

    metrics = runtime_metrics_from_trace(trace)

    assert metrics.token_count is None
    assert metrics.truncated is None
    assert metrics.model_inference_ms is None


def test_runtime_metrics_from_trace_with_a_failed_tensorization_stage_leaves_fields_none() -> None:
    # A failed stage's summary is not trustworthy structural output (the
    # stage may have failed partway through producing it), so it must
    # never be surfaced as if it were a real measurement.
    trace = PipelineTrace("e1")
    trace.record(
        "tensorization",
        1.0,
        success=False,
        error_type="ValueError",
        summary={"token_count": 999, "truncated": True},
    )

    metrics = runtime_metrics_from_trace(trace)

    assert metrics.token_count is None
    assert metrics.truncated is None


def test_runtime_metrics_from_trace_with_an_empty_trace_has_zero_total() -> None:
    metrics = runtime_metrics_from_trace(PipelineTrace("e1"))

    assert metrics.total_elapsed_ms == 0.0
    assert metrics.stage_elapsed_ms == {}


# -- aggregate_runtime_metrics --------------------------------------------------


def test_aggregate_runtime_metrics_computes_count_mean_min_max() -> None:
    fast = runtime_metrics_from_trace(_fixed_total_trace(2.0))
    slow = runtime_metrics_from_trace(_fixed_total_trace(8.0))

    aggregate = aggregate_runtime_metrics([fast, slow])

    assert aggregate == {"count": 2, "mean_ms": 5.0, "min_ms": 2.0, "max_ms": 8.0}


def test_aggregate_runtime_metrics_rejects_an_empty_iterable() -> None:
    with pytest.raises(ValueError):
        aggregate_runtime_metrics([])


def _fixed_total_trace(total_ms: float) -> PipelineTrace:
    trace = PipelineTrace("e1")
    trace.record("fusion", total_ms)
    return trace
