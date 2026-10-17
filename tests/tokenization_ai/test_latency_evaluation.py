"""Tests for the KPI 4 latency benchmark framework.

These verify the harness (statistics, workload, boundaries, safeguards,
reporting) and that KPI 1-3 evidence is untouched. They deliberately do NOT
assert a latency threshold: timing results come from the scored benchmark
run, not from unit tests on whatever machine runs them.
"""

import ast
import inspect
import json
from pathlib import Path

import pytest

from tokenization_ai.evaluation import latency, runner
from tokenization_ai.evaluation.latency import (
    BOUNDARIES,
    MIX,
    PROTOCOL_BATCHES,
    PROTOCOL_REPS,
    STRATA,
    percentile,
    summarize,
)

# SHA-256 of the frozen KPI 1-3 evidence, as written to the original PRE_RUN_MANIFEST.json
# before the KPI 1-3 scored run. KPI 4 work must not change any of them.
KPI13_FROZEN_HASHES = {
    "fixtures/curated_eval.json": "153166a4cc79f8b18fba7fb5fa603ec528390012df30108f18436482065eea44",
    "fixtures/generated_eval.json": "bdccc918823a114398768c0c5431b301c09ecda6196ce3f9ecd1f0f7e1a76d2d",
    "fixtures/dev_curated.json": "26abdff5e7d71f35fddad0eb887bd622ecdc25cb2bcf530c22459b00932ece25",
    "fixtures/dev_generated.json": "c0be00b0c97a6fd1f771804b60ea9df6a7f7da8536439e68ef56a2d253b844a1",
    "fixtures/invalid_cases.json": "ca95382be7f785a831ac428eb7630d899c5e8a326d0ef2107d720fcc5e3cb589",
    "PROTOCOL.md": "bfc926c074a8d20cb6753a19712dd89a8dba8e54b390f32781a3c46b06781d2d",
}


# --- statistics --------------------------------------------------------------


def test_percentile_is_nearest_rank():
    data = [float(i) for i in range(1, 101)]
    assert percentile(data, 95) == 95.0
    assert percentile(data, 99) == 99.0
    assert percentile(data, 100) == 100.0
    assert percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_reports_every_required_statistic():
    s = summarize([4.0, 1.0, 3.0, 2.0])
    assert s["n"] == 4
    assert s["mean_ms"] == 2.5 and s["median_ms"] == 2.5
    assert s["min_ms"] == 1.0 and s["max_ms"] == 4.0
    assert s["std_ms"] == pytest.approx(1.2909944, rel=1e-6)
    assert {"p95_ms", "p99_ms"} <= set(s)


def test_batch_mean_interval():
    assert latency.batch_mean_interval([1.0]) is None
    assert latency.batch_mean_interval([2.0] * 5) == [2.0, 2.0]
    lo, hi = latency.batch_mean_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo < 3.0 < hi


# --- protocol / workload -----------------------------------------------------


def test_protocol_minimums_and_documented_constants():
    assert PROTOCOL_REPS >= 1000 and PROTOCOL_BATCHES >= 3
    text = latency.PROTOCOL_FILE.read_text(encoding="utf-8")
    assert "**1000**" in text and "**5**" in text  # reps and batches as written in the protocol
    assert "S6_mix_all_valid" in text and "5.0 ms" in text
    for boundary in ("B1", "B1T", "B2", "B3"):
        assert f"**{boundary}**" in text
    assert latency.PRIMARY == {"boundary": "B1", "stratum": MIX}
    assert latency.TARGET_MS == 5.0
    for stratum in (*STRATA, MIX):
        assert stratum in text


def test_workload_uses_existing_frozen_cases_and_excludes_the_capacity_rejection():
    strata = latency.load_workloads()
    assert set(strata) == {*STRATA, MIX}
    assert len(strata[MIX]) == 30
    assert latency.CAPACITY_CASE_ID not in {i.case_id for i in strata[MIX]}
    for name, ids in STRATA.items():
        assert [i.case_id for i in strata[name]] == ids


def test_every_workload_item_is_a_valid_prediction_request_on_all_boundaries():
    from fastapi.testclient import TestClient

    from tokenization_ai.app import create_app
    from tokenization_ai.demo import _build_demo_service

    service = _build_demo_service()
    client = TestClient(create_app(service))
    for items in latency.load_workloads().values():
        for item in items:
            assert 0.0 <= service.predict(item.context, item.rules).score <= 1.0
            response = client.post("/v1/analyze", content=item.body, headers={"content-type": "application/json"})
            assert response.status_code == 200, item.case_id


def test_kpi_1_3_evidence_is_unchanged():
    assert runner.fixture_hashes() == KPI13_FROZEN_HASHES


def test_latency_module_does_not_import_the_model_or_modify_kpi_files():
    imported = set()
    for node in ast.walk(ast.parse(inspect.getsource(latency))):
        if isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert "..model" not in imported
    source = inspect.getsource(latency)
    assert "PROTOCOL.md" in source and "write_text(" in source
    # The only files it writes are latency_* reports and its own manifest.
    for name in ("kpi_report", "PRE_RUN_MANIFEST.json\""):
        assert name not in source.replace("LATENCY_PRE_RUN_MANIFEST.json", "")


# --- harness safeguards ------------------------------------------------------


def test_error_responses_abort_the_benchmark_instead_of_being_timed():
    item = latency.Item("x", "x", None, None, {}, {}, b"")
    with pytest.raises(RuntimeError):
        latency._timed_loop(lambda _: 500, [item], reps=5, warmup=1, check=lambda s: s == 200)
    with pytest.raises(RuntimeError):
        latency._timed_loop(lambda _: 500, [item], reps=5, warmup=0, check=lambda s: s == 200)


def test_timed_loop_reenables_garbage_collection():
    import gc

    item = latency.Item("x", "x", None, None, {}, {}, b"")
    samples, _ = latency._timed_loop(lambda _: None, [item], reps=50, warmup=2)
    assert len(samples) == 50 and all(s >= 0 for s in samples)
    assert gc.isenabled()


def test_timer_overhead_is_small_and_positive():
    overhead = latency.timer_overhead_ms(2000)
    assert 0 < overhead < 0.01  # well under 10 microseconds


# --- end-to-end smoke (plumbing only; NOT protocol conformant) ----------------


@pytest.fixture(scope="module")
def smoke():
    return latency.run_benchmark(batches=2, reps=12, warmup=3, warmup_http=3, include_http=True)


def test_smoke_run_is_labelled_non_conformant_and_complete(smoke):
    report, raw = smoke
    assert report["config"]["protocol_conformant"] is False
    assert report["primary"]["protocol_conformant_run"] is False
    assert set(report["results"]) == set(BOUNDARIES)
    for boundary in ("B1", "B1T", "B2", "B3"):
        for stratum in (*STRATA, MIX):
            entry = report["results"][boundary][stratum]
            assert entry["pooled"]["n"] == 2 * 12
            assert entry["batches"] == 2
            assert len(raw["samples"][boundary][stratum]) == 2
    assert "health_floor" in report["results"]["B3"]
    assert report["primary"]["boundary"] == "B1" and report["primary"]["stratum"] == MIX
    assert report["environment"]["gc_disabled_during_timed_loops"] is True
    assert report["timer_overhead_ms"] > 0
    assert set(report["informational_stage_means_ms"][MIX]) >= {"fusion", "model_inference"}


def test_report_never_labels_pipeline_time_as_model_inference_or_accuracy(smoke):
    report, _ = smoke
    text = latency.render_markdown(report)
    assert "Not isolated ML-model inference" in report["scope_statement"]
    assert "proposal target, not an achieved production result" in text
    assert "informational" in text.lower()
    assert "Limitations" in text
    assert "target_met" in text


# --- CLI ---------------------------------------------------------------------


def test_cli_manifest_is_written_once_and_never_overwritten(tmp_path: Path):
    assert latency.main(["--out-dir", str(tmp_path), "--write-manifest"]) == 0
    assert latency.main(["--out-dir", str(tmp_path), "--write-manifest"]) == 2
    manifest = json.loads((tmp_path / "LATENCY_PRE_RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["written_before_first_scored_run"] is True
    assert {"LATENCY_PROTOCOL.md", "latency.py"} <= set(manifest["sha256"])
    for name, digest in KPI13_FROZEN_HASHES.items():
        assert manifest["sha256"][f"kpi1_3/{name}"] == digest
