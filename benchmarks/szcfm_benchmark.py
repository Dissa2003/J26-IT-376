"""Reproducible in-process benchmark of the SZ-CFM pipeline.

Conditions: the example event in tests/ingestion/fixtures/szcfm_example_event.json,
the SIMULATED DevelopmentHeuristicBackend, default SZCFMConfig except where stated,
time.perf_counter_ns() timing, a warm-up pass before every measurement. No network,
no HTTP layer (the FastAPI/TestClient overhead is excluded on purpose).

Baseline: the pre-existing legacy interception path (PIIMasker.mask +
UnifiedContextVectorBuilder.build + .serialize) on the same payload. It performs
no request-state tracking, no inference and no decision, so it is a lower bound
for "ingestion work only", not an equivalent pipeline.

Run: python -m benchmarks.szcfm_benchmark [--requests 2000] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.pii_masker import PIIMasker  # noqa: E402
from ingestion.szcfm import SZCFMPipeline  # noqa: E402
from ingestion.szcfm_policy import SZCFMConfig  # noqa: E402
from ingestion.vector_builder import UnifiedContextVectorBuilder  # noqa: E402

FIXTURE = ROOT / "tests" / "ingestion" / "fixtures" / "szcfm_example_event.json"


def _summary(samples_ns: list[int]) -> dict[str, float]:
    s = sorted(samples_ns)
    return {
        "n": len(s),
        "mean_ms": round(statistics.fmean(s) / 1e6, 4),
        "p50_ms": round(s[len(s) // 2] / 1e6, 4),
        "p95_ms": round(s[int(len(s) * 0.95)] / 1e6, 4),
        "p99_ms": round(s[int(len(s) * 0.99)] / 1e6, 4),
        "max_ms": round(s[-1] / 1e6, 4),
    }


def bench_legacy(raw: bytes, n: int) -> dict:
    masker, builder = PIIMasker(), UnifiedContextVectorBuilder()

    def once() -> int:
        t = time.perf_counter_ns()
        data = json.loads(raw)
        masked = masker.mask(data["payload"])
        vector = builder.build(masked, data["threat_score"])
        builder.serialize("evt", masked, data["threat_score"], vector)
        return time.perf_counter_ns() - t

    for _ in range(200):
        once()
    return _summary([once() for _ in range(n)])


def bench_szcfm(raw: bytes, n: int, concurrency: int) -> dict:
    config = SZCFMConfig(ring_capacity=max(64, concurrency * 2), max_in_flight=max(64, concurrency * 2))
    pipeline = SZCFMPipeline(config)
    stage = {k: [] for k in ("total", "serialization", "features", "ai_inference", "ai_wait", "fusion")}

    def once(_: int):
        t = time.perf_counter_ns()
        result = pipeline.process(pipeline.parse(raw))
        elapsed = time.perf_counter_ns() - t
        return elapsed, result

    for i in range(200):
        once(i)
    pipeline.metrics = type(pipeline.metrics)()  # reset counters after warm-up

    start = time.perf_counter_ns()
    if concurrency == 1:
        results = [once(i) for i in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(once, range(n)))
    wall_s = (time.perf_counter_ns() - start) / 1e9

    decisions: dict[str, int] = {}
    ai_status: dict[str, int] = {}
    for elapsed, r in results:
        m = r.pipeline_metrics
        stage["total"].append(elapsed)
        stage["serialization"].append(int(m.serialization_time_ms * 1e6))
        stage["features"].append(int(m.feature_extraction_time_ms * 1e6))
        stage["fusion"].append(int(m.context_fusion_time_ms * 1e6))
        if m.ai_inference_time_ms is not None:
            stage["ai_inference"].append(int(m.ai_inference_time_ms * 1e6))
        if m.ai_wait_time_ms is not None:
            stage["ai_wait"].append(int(m.ai_wait_time_ms * 1e6))
        decisions[r.security_decision.decision] = decisions.get(r.security_decision.decision, 0) + 1
        ai_status[r.ai_assessment.status] = ai_status.get(r.ai_assessment.status, 0) + 1
    stats = pipeline.stats()
    pipeline.close()
    return {
        "concurrency": concurrency,
        "requests": n,
        "wall_time_s": round(wall_s, 3),
        "throughput_rps": round(n / wall_s, 1),
        "latency": {k: _summary(v) for k, v in stage.items() if v},
        "decisions": decisions,
        "ai_status": ai_status,
        "ring": stats["ring"],
        "rejected_backpressure": stats["inference"]["rejected_backpressure"],
        "errors": stats["counters"].get("errors", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    raw = FIXTURE.read_bytes()
    report = {
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "model": "szcfm-dev-heuristic-1 (SIMULATED)",
        "baseline_legacy_interception": bench_legacy(raw, args.requests),
        "szcfm": [bench_szcfm(raw, args.requests, c) for c in (1, 8, 32)],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
