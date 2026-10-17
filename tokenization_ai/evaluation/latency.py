"""KPI 4 latency benchmark (see LATENCY_PROTOCOL.md).

Measures the REAL LogiShield implementation at three separate boundaries:

- B1  in-process pipeline: ``TokenizationAIService.predict`` (no tracing),
      plus B1T (the same call with the production tracer enabled);
- B2  adapter-inclusive: raw envelope/rule dicts -> adapters -> ``predict``;
- B3  HTTP: ``POST /v1/analyze`` on the real app, loopback, keep-alive client.

"Inference time" in the proposal is measured here as end-to-end prediction
time. It is never presented as isolated ML-model inference time.

    python -m tokenization_ai.evaluation.latency --out-dir <dir> --write-manifest
    python -m tokenization_ai.evaluation.latency --out-dir <dir>
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..adapters import to_unified_api_context, to_verified_rule_info
from . import runner

LATENCY_PROTOCOL_VERSION = "1.0.0"
PROTOCOL_FILE = Path(__file__).parent / "LATENCY_PROTOCOL.md"
THIS_FILE = Path(__file__)

TARGET_MS = 5.0
PROTOCOL_BATCHES = 5
PROTOCOL_REPS = 1000
PROTOCOL_WARMUP = 300
PROTOCOL_WARMUP_HTTP = 200

CAPACITY_CASE_ID = "G-D2-capacity-with-ten-rules"
STRATA: dict[str, list[str]] = {
    "S1_small_no_rules": ["E01-benign-numeric"],
    "S2_masked_with_rule": ["E02-suspicious-login-mock-rule"],
    "S3_medium_seeded_20_fields": ["gen-eval-08"],
    "S4_large_seeded_80_fields": ["gen-eval-12"],
    "S5_rule_heavy_truncated": ["G-D1-truncation-with-nine-rules"],
    # S6 is resolved at load time: every valid case except the capacity rejection.
}
MIX = "S6_mix_all_valid"
PRIMARY = {"boundary": "B1", "stratum": MIX}
BOUNDARIES = ("B1", "B1T", "B2", "B3")

_T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

LIMITATIONS = [
    "Single machine, single OS and Python build; the result does not transfer to other hardware.",
    "Synthetic payloads of at most 128 tokens after truncation; real traffic and SR-BH were not used.",
    "Python-level timing of a pure-Python/numpy pipeline; other processes on the machine were not controlled "
    "(no CPU pinning, priority change or isolation).",
    "B3 runs client and server in one process on loopback, sharing one interpreter lock and CPU.",
    "End-to-end prediction time is measured, not isolated ML-model inference time; per-stage means are informational.",
    "The model is fitted on two synthetic fixtures; latency says nothing about detection quality.",
    "Garbage collection is disabled during timed loops, which slightly flatters allocation-heavy paths.",
    "Not a throughput, concurrency or production latency-SLA result; cold start and error paths are excluded.",
    "The mean hides tail behaviour; p95, p99 and max are reported for that reason.",
]


# --- statistics --------------------------------------------------------------


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def summarize(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min_ms": ordered[0],
        "p95_ms": percentile(ordered, 95),
        "p99_ms": percentile(ordered, 99),
        "max_ms": ordered[-1],
    }


def batch_mean_interval(batch_means: list[float]) -> list[float] | None:
    k = len(batch_means)
    if k < 2 or (k - 1) not in _T_CRIT_95:
        return None
    mean = statistics.fmean(batch_means)
    half = _T_CRIT_95[k - 1] * statistics.stdev(batch_means) / math.sqrt(k)
    return [mean - half, mean + half]


# --- workload ----------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    case_id: str
    event_id: str
    context: Any
    rules: Any
    raw_context: dict[str, Any]
    raw_rules: dict[str, Any]
    body: bytes


def load_workloads(fixture_dir: Path = runner.FIXTURE_DIR) -> dict[str, list[Item]]:
    docs = runner.load_fixtures(fixture_dir)
    ordered = [case for name in runner.EVAL_FILES for case in docs[name]["cases"]]
    by_id = {case["id"]: case for case in ordered}

    def build(case: dict[str, Any]) -> Item:
        raw_context = case["input"]["context"]
        context = to_unified_api_context(raw_context)
        rule_source = case["input"].get("rule_info")
        raw_rules = rule_source if rule_source is not None else {"event_id": context.event_id, "rules": []}
        request = {"context": raw_context, "rule_source": "mock_fixture" if rule_source else "none"}
        if rule_source:
            request["rule_info"] = rule_source
        return Item(
            case_id=case["id"],
            event_id=context.event_id,
            context=context,
            rules=to_verified_rule_info(raw_rules),
            raw_context=raw_context,
            raw_rules=raw_rules,
            body=json.dumps(request).encode("utf-8"),
        )

    strata = {name: [build(by_id[i]) for i in ids] for name, ids in STRATA.items()}
    strata[MIX] = [build(c) for c in ordered if c["id"] != CAPACITY_CASE_ID]
    return strata


# --- measurement -------------------------------------------------------------


def timer_overhead_ms(iterations: int = 20000) -> float:
    pc = time.perf_counter_ns
    total = 0
    for _ in range(iterations):
        t0 = pc()
        total += pc() - t0
    return total / iterations / 1e6


def _timed_loop(call: Callable[[Item], Any], items: list[Item], reps: int, warmup: int,
                check: Callable[[Any], bool] | None = None) -> tuple[list[float], int]:
    n = len(items)
    for i in range(warmup):
        result = call(items[i % n])
        if check is not None and not check(result):
            raise RuntimeError("benchmark request failed during warm-up; refusing to time error responses")
    samples = [0.0] * reps
    bad = 0
    pc = time.perf_counter_ns
    gc.collect()
    gc.disable()
    try:
        for i in range(reps):
            item = items[i % n]
            t0 = pc()
            result = call(item)
            samples[i] = (pc() - t0) / 1e6
            if check is not None and not check(result):
                bad += 1
    finally:
        gc.enable()
    if bad:
        raise RuntimeError(f"{bad} benchmark request(s) returned an error status; aborting")
    return samples, bad


class _LoopbackServer:
    def __init__(self, app: Any) -> None:
        import uvicorn

        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.port = 0

    def __enter__(self) -> "_LoopbackServer":
        self._thread.start()
        deadline = time.time() + 15
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("loopback server did not start")
            time.sleep(0.02)
        self.port = self._server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def environment() -> dict[str, Any]:
    versions = {}
    for name in ("fastapi", "starlette", "uvicorn", "httpx", "numpy", "pydantic"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    power = None
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=5)
        power = out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - best-effort metadata only
        pass
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_versions": versions,
        "windows_power_scheme": power,
        "process_priority_changed": False,
        "cpu_pinning": False,
        "gc_disabled_during_timed_loops": True,
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_hashes() -> dict[str, str]:
    hashes = {
        "LATENCY_PROTOCOL.md": file_sha256(PROTOCOL_FILE),
        "latency.py": file_sha256(THIS_FILE),
    }
    hashes.update({f"kpi1_3/{k}": v for k, v in runner.fixture_hashes().items()})
    return hashes


def run_benchmark(
    batches: int = PROTOCOL_BATCHES,
    reps: int = PROTOCOL_REPS,
    warmup: int = PROTOCOL_WARMUP,
    warmup_http: int = PROTOCOL_WARMUP_HTTP,
    include_http: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the benchmark. Returns ``(report, raw_samples)``."""
    from ..app import create_app
    from ..demo import _build_demo_service

    strata = load_workloads()
    service = _build_demo_service()
    app = create_app(service)
    conformant = (
        batches >= PROTOCOL_BATCHES and reps >= PROTOCOL_REPS and warmup >= PROTOCOL_WARMUP
        and (warmup_http >= PROTOCOL_WARMUP_HTTP or not include_http) and include_http
    )

    predict = service.predict
    tracer = service.tracer

    def b1(item: Item) -> Any:
        return predict(item.context, item.rules)

    def b1t(item: Item) -> Any:
        trace = tracer.start(item.event_id)
        predict(item.context, item.rules, trace=trace)
        return trace

    def b2(item: Item) -> Any:
        return predict(to_unified_api_context(item.raw_context), to_verified_rule_info(item.raw_rules))

    raw: dict[str, dict[str, list[list[float]]]] = {b: {s: [] for s in strata} for b in BOUNDARIES}
    raw["B3"]["health_floor"] = []
    stage_totals: dict[str, dict[str, list[float]]] = {s: {} for s in strata}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    overhead = timer_overhead_ms()

    client = None
    server_cm = _LoopbackServer(app) if include_http else None
    try:
        if server_cm is not None:
            import httpx

            server_cm.__enter__()
            client = httpx.Client(base_url=f"http://127.0.0.1:{server_cm.port}", timeout=15.0)
        headers = {"content-type": "application/json"}

        def b3(item: Item) -> int:
            return client.post("/v1/analyze", content=item.body, headers=headers).status_code

        health_item = [Item("health", "health", None, None, {}, {}, b"")]

        def b3_health(_: Item) -> int:
            return client.get("/health").status_code

        for _batch in range(batches):
            for name, items in strata.items():
                samples, _ = _timed_loop(b1, items, reps, warmup)
                raw["B1"][name].append(samples)
            for name, items in strata.items():
                samples, _ = _timed_loop(b1t, items, reps, warmup)
                raw["B1T"][name].append(samples)
                # Stage means come from the production tracer, read OUTSIDE the timed region.
                for item in items:
                    trace = b1t(item)
                    for stage in trace.stages:
                        stage_totals[name].setdefault(stage.stage, []).append(stage.elapsed_ms)
            for name, items in strata.items():
                samples, _ = _timed_loop(b2, items, reps, warmup)
                raw["B2"][name].append(samples)
            if include_http:
                for name, items in strata.items():
                    samples, _ = _timed_loop(b3, items, reps, warmup_http, check=lambda s: s == 200)
                    raw["B3"][name].append(samples)
                samples, _ = _timed_loop(b3_health, health_item, reps, warmup_http, check=lambda s: s == 200)
                raw["B3"]["health_floor"].append(samples)
    finally:
        if client is not None:
            client.close()
        if server_cm is not None:
            server_cm.__exit__(None, None, None)

    # Token counts (single untimed traced pass per case).
    token_counts: dict[str, Any] = {}
    for name, items in strata.items():
        counts = []
        for item in items:
            trace = tracer.start(item.event_id)
            predict(item.context, item.rules, trace=trace)
            stage = next(s for s in trace.stages if s.stage == "tensorization")
            counts.append(stage.summary["token_count"])
        token_counts[name] = {"cases": len(counts), "min": min(counts), "median": statistics.median(counts), "max": max(counts)}

    results: dict[str, Any] = {}
    for boundary, per_stratum in raw.items():
        results[boundary] = {}
        for stratum, batch_lists in per_stratum.items():
            if not batch_lists:
                continue
            pooled = [x for batch in batch_lists for x in batch]
            batch_means = [statistics.fmean(b) for b in batch_lists]
            results[boundary][stratum] = {
                "pooled": summarize(pooled),
                "batch_means_ms": batch_means,
                "batch_mean_95_interval_ms": batch_mean_interval(batch_means),
                "batches": len(batch_lists),
                "reps_per_batch": len(batch_lists[0]),
            }

    primary_result = results[PRIMARY["boundary"]][PRIMARY["stratum"]]
    pooled_mean = primary_result["pooled"]["mean_ms"]
    worst_batch = max(primary_result["batch_means_ms"])
    primary = {
        **PRIMARY,
        "definition": "pooled mean of B1 end-to-end prediction time over the S6 mixed workload",
        "pooled_mean_ms": pooled_mean,
        "worst_batch_mean_ms": worst_batch,
        "target_ms": TARGET_MS,
        "criterion": "pooled mean <= target AND every batch mean <= target",
        "target_met": pooled_mean <= TARGET_MS and worst_batch <= TARGET_MS,
        "protocol_conformant_run": conformant,
        "interpretation": (
            "On this machine, Python build and synthetic workload, the in-process LogiShield pipeline's "
            "average end-to-end prediction time was within the 5 ms target. This is not a production SLA."
            if pooled_mean <= TARGET_MS and worst_batch <= TARGET_MS
            else "The in-process average exceeded the 5 ms target on this machine and workload."
        ),
    }
    stage_means = {
        s: {stage: statistics.fmean(v) for stage, v in stages.items()} for s, stages in stage_totals.items()
    }
    report = {
        "latency_protocol_version": LATENCY_PROTOCOL_VERSION,
        "scope_statement": (
            "KPI 4 single-machine synthetic-workload latency measurement of end-to-end prediction time. "
            "Not isolated ML-model inference, not a production SLA, not detection accuracy, "
            "not Formal Verification/3 integration."
        ),
        "target_ms": TARGET_MS,
        "config": {
            "batches": batches,
            "reps_per_stratum_per_batch": reps,
            "warmup_in_process": warmup,
            "warmup_http": warmup_http,
            "http_included": include_http,
            "protocol_conformant": conformant,
            "strata": {name: [i.case_id for i in items] if len(items) < 10 else f"{len(items)} cases" for name, items in strata.items()},
            "tensor_max_length": service.tensorizer.max_length,
            "model_version": service.model.model_version,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "environment": environment(),
        "timer_overhead_ms": overhead,
        "token_counts": token_counts,
        "primary": primary,
        "results": results,
        "informational_stage_means_ms": stage_means,
        "limitations": LIMITATIONS,
    }
    return report, {
        "unit": "ms",
        "layout": "raw[boundary][stratum] = list of batches, each a list of per-call times",
        "samples": raw,
    }


# --- reporting ---------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    p = report["primary"]
    cfg = report["config"]
    env = report["environment"]
    lines = [
        "# LogiShield KPI 4 latency report",
        "",
        f"Latency protocol {report['latency_protocol_version']} · protocol-conformant run: **{cfg['protocol_conformant']}**",
        "",
        "> " + report["scope_statement"],
        "> The 5 ms figure is a proposal target, not an achieved production result.",
        "",
        "## Primary result",
        "",
        f"- Boundary **{p['boundary']}** (in-process pipeline, no tracing), workload **{p['stratum']}**.",
        f"- Pooled mean: **{p['pooled_mean_ms']:.4f} ms** (worst batch mean {p['worst_batch_mean_ms']:.4f} ms) "
        f"vs target ≤ {p['target_ms']:g} ms → target_met = **{p['target_met']}**.",
        f"- {p['interpretation']}",
        "",
        f"Configuration: {cfg['batches']} batches × {cfg['reps_per_stratum_per_batch']} calls per stratum "
        f"(warm-up {cfg['warmup_in_process']} / {cfg['warmup_http']} HTTP); tensor max_length "
        f"{cfg['tensor_max_length']}; model `{cfg['model_version']}`.",
        "",
        "## All boundaries and strata (pooled over batches, milliseconds)",
        "",
        "| Boundary | Stratum | n | mean | median | std | p95 | p99 | max | batch-mean 95% interval |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for boundary in BOUNDARIES:
        for stratum, m in report["results"].get(boundary, {}).items():
            s = m["pooled"]
            ci = m["batch_mean_95_interval_ms"]
            ci_text = "n/a" if not ci else f"{ci[0]:.4f}–{ci[1]:.4f}"
            lines.append(
                f"| {boundary} | {stratum} | {s['n']} | {s['mean_ms']:.4f} | {s['median_ms']:.4f} | "
                f"{s['std_ms']:.4f} | {s['p95_ms']:.4f} | {s['p99_ms']:.4f} | {s['max_ms']:.4f} | {ci_text} |"
            )
    lines += [
        "",
        "Boundary meanings: **B1** in-process pipeline · **B1T** B1 with the production tracer · "
        "**B2** adapters + B1 · **B3** real HTTP `POST /v1/analyze` on loopback (includes client, JSON, "
        "projection; `health_floor` is a `GET /health` transport floor and is not subtracted).",
        "",
        "## Token counts per stratum (after masking/normalization, before truncation)",
        "",
        "| Stratum | cases | min | median | max |",
        "|---|---|---|---|---|",
    ]
    for stratum, t in report["token_counts"].items():
        lines.append(f"| {stratum} | {t['cases']} | {t['min']} | {t['median']} | {t['max']} |")
    lines += [
        "",
        "## Informational per-stage means from the production tracer (B1T runs, ms)",
        "",
        "These are per-stage instrumentation times, **not** the KPI. `model_inference` is the isolated "
        "model-scoring stage; the KPI covers the whole pipeline.",
        "",
    ]
    stage_names = ["fusion", "masking", "normalization", "tokenization", "tensorization", "model_inference"]
    lines.append("| Stratum | " + " | ".join(stage_names) + " |")
    lines.append("|---|" + "---|" * len(stage_names))
    for stratum, stages in report["informational_stage_means_ms"].items():
        lines.append("| " + stratum + " | " + " | ".join(f"{stages.get(s, float('nan')):.4f}" for s in stage_names) + " |")
    lines += [
        "",
        "## Environment",
        "",
        f"- Python {env['python'].splitlines()[0]} ({env['implementation']}), {env['platform']}, "
        f"{env['processor']}, {env['cpu_count']} logical CPUs.",
        f"- Packages: {env['package_versions']}",
        f"- Power scheme: {env['windows_power_scheme']}",
        f"- Timer overhead (perf_counter_ns pair): {report['timer_overhead_ms'] * 1000:.3f} µs; "
        "GC disabled during timed loops; no CPU pinning or priority change.",
        f"- Run window: {cfg['started_utc']} → {cfg['finished_utc']}",
        "",
        "## Limitations",
        "",
    ] + [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tokenization_ai.evaluation.latency")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--write-manifest", action="store_true", help="write LATENCY_PRE_RUN_MANIFEST.json and exit")
    parser.add_argument("--kpi13-manifest", type=Path, help="original KPI 1-3 PRE_RUN_MANIFEST.json to verify")
    parser.add_argument("--smoke", action="store_true", help="tiny non-conformant run for plumbing checks")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest:
        path = args.out_dir / "LATENCY_PRE_RUN_MANIFEST.json"
        if path.exists():
            print(f"refusing to overwrite {path}")
            return 2
        path.write_text(
            json.dumps(
                {
                    "latency_protocol_version": LATENCY_PROTOCOL_VERSION,
                    "written_before_first_scored_run": True,
                    "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sha256": manifest_hashes(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")
        return 0

    targets = [args.out_dir / n for n in ("latency_report.json", "latency_report.md", "latency_raw_samples.json.gz")]
    existing = [t for t in targets if t.exists()]
    if existing:
        print("refusing to overwrite:", *existing)
        return 2

    if args.smoke:
        report, raw = run_benchmark(batches=2, reps=20, warmup=5, warmup_http=5)
    else:
        report, raw = run_benchmark()

    hashes = manifest_hashes()
    pre_run = args.out_dir / "LATENCY_PRE_RUN_MANIFEST.json"
    checks: dict[str, Any] = {"current_sha256": hashes}
    if pre_run.exists():
        recorded = json.loads(pre_run.read_text(encoding="utf-8"))["sha256"]
        checks["matches_latency_pre_run_manifest"] = recorded == hashes
    if args.kpi13_manifest and args.kpi13_manifest.exists():
        original = json.loads(args.kpi13_manifest.read_text(encoding="utf-8"))["sha256"]
        current = runner.fixture_hashes()
        checks["kpi1_3_fixtures_match_original_pre_run_manifest"] = original == current
        checks["kpi1_3_mismatched_files"] = sorted(k for k in set(original) | set(current) if original.get(k) != current.get(k))
    report["integrity"] = checks

    (args.out_dir / "latency_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    (args.out_dir / "latency_report.md").write_text(render_markdown(report), encoding="utf-8")
    with gzip.open(args.out_dir / "latency_raw_samples.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(raw, handle, separators=(",", ":"))
    prim = report["primary"]
    print(f"PRIMARY {prim['boundary']} {prim['stratum']}: pooled mean {prim['pooled_mean_ms']:.4f} ms, "
          f"worst batch mean {prim['worst_batch_mean_ms']:.4f} ms, target_met={prim['target_met']}, "
          f"conformant={prim['protocol_conformant_run']}")
    print("wrote reports to", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
