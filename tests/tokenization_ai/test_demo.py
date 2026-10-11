"""Tests for the Member 4 PP1 demonstration (tokenization_ai.demo).

These tests verify the demo runs successfully end to end using its
clearly-labeled synthetic fixtures and produces structurally valid
output -- they are not a research result in themselves.
"""

from tokenization_ai import demo
from tokenization_ai.metrics import RuntimeMetrics
from tokenization_ai.models import AnomalyPrediction
from tokenization_ai.tracing import PIPELINE_STAGES, PipelineTrace


def test_run_demo_returns_one_result_per_demo_event() -> None:
    results = demo.run_demo()

    assert len(results) == len(demo._DEMO_EVENTS)


def test_run_demo_result_event_ids_match_the_fixture_event_ids() -> None:
    results = demo.run_demo()

    result_event_ids = {event_id for event_id, _, _, _ in results}
    fixture_event_ids = {context.event_id for context, _ in demo._DEMO_EVENTS}
    assert result_event_ids == fixture_event_ids


def test_run_demo_produces_a_bounded_prediction_for_every_event() -> None:
    results = demo.run_demo()

    for _event_id, prediction, _trace, _metrics in results:
        assert isinstance(prediction, AnomalyPrediction)
        assert 0.0 <= prediction.score <= 1.0
        assert prediction.model_version == "ngram-linear-v1-trained"


def test_run_demo_produces_a_complete_trace_for_every_event() -> None:
    results = demo.run_demo()

    for _event_id, _prediction, trace, _metrics in results:
        assert isinstance(trace, PipelineTrace)
        assert tuple(stage.stage for stage in trace.stages) == PIPELINE_STAGES
        assert all(stage.success for stage in trace.stages)


def test_run_demo_produces_runtime_metrics_with_non_negative_timings() -> None:
    results = demo.run_demo()

    for _event_id, _prediction, _trace, metrics in results:
        assert isinstance(metrics, RuntimeMetrics)
        assert metrics.total_elapsed_ms >= 0.0
        assert all(elapsed >= 0.0 for elapsed in metrics.stage_elapsed_ms.values())
        assert metrics.token_count is not None
        assert metrics.token_count > 0


def test_demo_fixtures_use_a_mock_rule_id_not_a_genuine_verification_marker() -> None:
    # The mock fixture must be unmistakably labeled as mock, so nobody
    # downstream can mistake it for genuine Member 2 verified-rule output.
    rule_infos = [rule_info for _, rule_info in demo._DEMO_EVENTS]
    all_rule_ids = [rule.rule_id for rule_info in rule_infos for rule in rule_info.rules]

    assert all(rule_id.startswith("MOCK-") for rule_id in all_rule_ids)


def test_run_demo_does_not_leak_raw_sensitive_demo_payload_values_into_any_trace() -> None:
    results = demo.run_demo()

    for _event_id, _prediction, trace, _metrics in results:
        trace_text = repr(trace.stages)
        assert "hunter2" not in trace_text
        assert "demo-not-a-real-token" not in trace_text


def test_main_runs_without_raising(capsys) -> None:
    demo.main()

    captured = capsys.readouterr()
    assert "SYNTHETIC DATA ONLY" in captured.out
    assert "MOCK data" in captured.out
