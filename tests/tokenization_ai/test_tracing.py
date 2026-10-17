"""Unit tests for tokenization_ai.tracing."""

import pytest

from tokenization_ai.tracing import (
    PIPELINE_STAGES,
    STAGE_FUSION,
    PipelineTrace,
    PipelineTracer,
)


def test_pipeline_stages_lists_all_six_stages_in_order() -> None:
    assert PIPELINE_STAGES == (
        "fusion",
        "masking",
        "normalization",
        "tokenization",
        "tensorization",
        "model_inference",
    )


def test_start_returns_an_empty_trace_for_the_given_event_id() -> None:
    trace = PipelineTracer().start("e1")

    assert trace.event_id == "e1"
    assert trace.stages == ()
    assert trace.total_elapsed_ms == 0.0


def test_start_returns_a_distinct_trace_object_each_call() -> None:
    tracer = PipelineTracer()

    first = tracer.start("e1")
    second = tracer.start("e1")
    first.record("fusion", 1.0)

    assert first is not second
    assert second.stages == ()


def test_two_traces_for_different_events_do_not_share_state() -> None:
    tracer = PipelineTracer()
    trace_a = tracer.start("event-a")
    trace_b = tracer.start("event-b")

    trace_a.record("fusion", 1.0, summary={"rule_count": 1})
    trace_b.record("fusion", 2.0, summary={"rule_count": 2})

    assert len(trace_a.stages) == 1
    assert len(trace_b.stages) == 1
    assert trace_a.stages[0].summary == {"rule_count": 1}
    assert trace_b.stages[0].summary == {"rule_count": 2}
    assert trace_a.event_id == "event-a"
    assert trace_b.event_id == "event-b"


def test_record_appends_in_order_and_total_elapsed_sums_all_stages() -> None:
    trace = PipelineTrace("e1")

    trace.record("fusion", 1.5)
    trace.record("masking", 2.5)

    assert [stage.stage for stage in trace.stages] == ["fusion", "masking"]
    assert trace.total_elapsed_ms == 4.0


def test_record_defaults_to_success_true_and_empty_summary() -> None:
    trace = PipelineTrace("e1")

    trace.record("fusion", 1.0)

    stage = trace.stages[0]
    assert stage.success is True
    assert stage.summary == {}
    assert stage.error_type is None


def test_record_stores_failure_details_without_a_message() -> None:
    trace = PipelineTrace("e1")

    trace.record("masking", 1.0, success=False, error_type="ValueError")

    stage = trace.stages[0]
    assert stage.success is False
    assert stage.error_type == "ValueError"


# -- run_stage ----------------------------------------------------------------


def test_run_stage_with_no_trace_calls_fn_directly_and_returns_its_result() -> None:
    tracer = PipelineTracer()
    calls = []

    result = tracer.run_stage(None, STAGE_FUSION, lambda: calls.append(1) or "ok")

    assert result == "ok"
    assert calls == [1]


def test_run_stage_with_a_trace_records_one_successful_stage() -> None:
    tracer = PipelineTracer()
    trace = tracer.start("e1")

    result = tracer.run_stage(trace, STAGE_FUSION, lambda: 42, summary_fn=lambda r: {"value": r})

    assert result == 42
    assert len(trace.stages) == 1
    stage = trace.stages[0]
    assert stage.stage == STAGE_FUSION
    assert stage.success is True
    assert stage.summary == {"value": 42}
    assert stage.elapsed_ms >= 0.0


def test_run_stage_elapsed_ms_is_never_negative() -> None:
    tracer = PipelineTracer()
    trace = tracer.start("e1")

    tracer.run_stage(trace, STAGE_FUSION, lambda: None)

    assert trace.stages[0].elapsed_ms >= 0.0


def test_run_stage_without_summary_fn_records_an_empty_summary() -> None:
    tracer = PipelineTracer()
    trace = tracer.start("e1")

    tracer.run_stage(trace, STAGE_FUSION, lambda: "result")

    assert trace.stages[0].summary == {}


def test_run_stage_reraises_the_original_exception_unchanged() -> None:
    tracer = PipelineTracer()
    trace = tracer.start("e1")

    def _boom() -> None:
        raise ValueError("sensitive-looking detail: password=hunter2")

    with pytest.raises(ValueError, match="sensitive-looking detail"):
        tracer.run_stage(trace, STAGE_FUSION, _boom)


def test_run_stage_records_failure_without_the_exception_message() -> None:
    tracer = PipelineTracer()
    trace = tracer.start("e1")

    def _boom() -> None:
        raise ValueError("sensitive-looking detail: password=hunter2")

    with pytest.raises(ValueError):
        tracer.run_stage(trace, STAGE_FUSION, _boom)

    stage = trace.stages[0]
    assert stage.success is False
    assert stage.error_type == "ValueError"
    assert stage.summary == {}
    # The raw exception text must never have been copied anywhere onto the trace.
    assert "hunter2" not in repr(stage)
    assert "password" not in repr(stage)


def test_run_stage_with_no_trace_propagates_exceptions_without_recording_anything() -> None:
    tracer = PipelineTracer()

    def _boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        tracer.run_stage(None, STAGE_FUSION, _boom)
