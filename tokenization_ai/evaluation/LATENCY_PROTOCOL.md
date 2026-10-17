# LogiShield KPI 4 Latency Evaluation Protocol

**Latency protocol version:** 1.0.0
**Scope:** Hybrid Structure-Aware Tokenization & Lightweight AI Prediction (LogiShield only).
**Covers:** KPI 4, "average inference time ≤ 5 ms" (proposal Table 2: *measure end-to-end prediction time*).
**Independent of:** `PROTOCOL.md` (KPI 1–3, v1.0.0). That file and its fixtures are frozen and are not modified here.

> The 5 ms figure is a **proposal target, not an achieved result**. Everything below is a
> **single-machine, synthetic-workload, in-process/loopback** measurement. It is not a
> production latency SLA, not a throughput or concurrency result, and says nothing about
> detection quality, Formal Verification integration or Runtime Verification integration.

## 1. Pre-registration rules

1. This file, `latency.py`, and the stratum definitions below are frozen **before** the scored
   benchmark. Their SHA-256 hashes are written to `LATENCY_PRE_RUN_MANIFEST.json` first.
2. The workload, boundaries, repetition counts, statistics and the primary criterion are not
   changed after seeing results. A change requires a new version (1.1.0, …) and a changelog entry.
3. No measurement is discarded, trimmed or "cleaned". Slow outliers are kept and reported.
4. KPI 1–3 evidence (fixtures, `PROTOCOL.md`, original reports) is read only. The original
   pre-run manifest hashes are re-verified and reported; no original report is overwritten.

## 2. What "inference time" means here (and what it does not)

The proposal's wording is "average inference time" with the method "measure end-to-end
prediction time". This protocol therefore measures **end-to-end prediction time of the real LogiShield
pipeline** and labels it as such. It is **not** the isolated cost of the ML model: the model
(`LightweightAnomalyModel.predict`) is one of six stages. The production tracer's per-stage
times (including `model_inference`) are reported separately as *informational* stage means.

## 3. Timing boundaries (measured separately; never merged)

| ID | Name | Starts | Ends | Includes | Excludes |
|---|---|---|---|---|---|
| **B1** | In-process pipeline | `service.predict(context, rule_info)` called with already-validated objects | prediction returned | fusion, masking, normalization, tokenization, tensorization, model scoring; **no tracing** | adapters, JSON, HTTP, privacy projection, process/start-up, model/vocabulary fitting |
| **B1T** | B1 with tracing | as B1, plus `tracer.start(...)` and `trace=` passed | prediction returned | B1 + per-stage timing instrumentation | as B1 |
| **B2** | Adapter-inclusive | raw envelope/RawEvent dict + rule dict | prediction returned | `to_unified_api_context`, `to_verified_rule_info`, then B1 | JSON, HTTP, projection |
| **B3** | HTTP endpoint | request bytes handed to the HTTP client (`POST /v1/analyze`) | full response body received by the client | uvicorn, ASGI, middleware, request validation, adapters, traced pipeline, privacy projection, response JSON serialization, loopback TCP, client-side overhead | client-side JSON encoding of the request (done before timing) and decoding of the response; network beyond loopback |

B3 runs the **real** application (`create_app(service)`) in an in-process uvicorn server thread on
`127.0.0.1` (ephemeral port) with a keep-alive `httpx` client in the same process, so client and
server compete for the same interpreter lock and CPU. This is a stated limitation. A
`GET /health` round trip is measured as a *transport floor* and is not subtracted from anything.

Clock: `time.perf_counter_ns()`. The cost of the timer itself is calibrated and reported.
Garbage collection is collected before and **disabled during** each timed loop, then re-enabled.

## 4. Workload (frozen strata)

Inputs are read-only references to existing frozen KPI 1–3 evaluation cases, so no new data is
invented and no fixture is changed. All data are synthetic; rules are mock.

| Stratum | Case(s) | Purpose |
|---|---|---|
| `S1_small_no_rules` | `E01-benign-numeric` | small request, no rules, no masking |
| `S2_masked_with_rule` | `E02-suspicious-login-mock-rule` | masking of header and field, one mock rule |
| `S3_medium_seeded_20_fields` | `gen-eval-08` | seeded, 20 top-level payload fields |
| `S4_large_seeded_80_fields` | `gen-eval-12` | seeded, 80 fields, truncated to `max_length` |
| `S5_rule_heavy_truncated` | `G-D1-truncation-with-nine-rules` | nine rules, payload truncated around the rule segment |
| `S6_mix_all_valid` | all 30 valid cases of `curated_eval` + `generated_eval`, **excluding** `G-D2-capacity-with-ten-rules`, round-robin in fixture order | **primary workload** |

Configuration: the demo service used by `python -m tokenization_ai` — `_build_demo_service()`
(model fitted on two synthetic fixtures; `max_length=128`). One configuration only.

## 5. Procedure

- Warm-up per stratum per boundary, not timed: **300** calls (B1, B1T, B2), **200** (B3).
- Measured repetitions: **1000** per stratum per batch.
- Batches: **5**, each run in the fixed order: B1 → B1T → B2 → B3 over S1…S6, then the B3 `/health`
  floor. The B3 server thread is started once, before batch 1, and stays idle during B1/B1T/B2.
- Every B3 response must be HTTP 200; any non-200 aborts the run (a benchmark of error
  responses would be meaningless). Workload cases are exactly those that produce a prediction.
- Statistics per (boundary, stratum): n, mean, median, sample standard deviation (ddof=1), min,
  p95, p99 (nearest-rank), max — pooled over all batches, plus per-batch means and a 95% t-interval
  over the 5 batch means (df=4, t=2.776).

## 6. Target interpretation (pre-registered)

- **Primary statistic:** pooled arithmetic mean of **B1** per-call time over **`S6_mix_all_valid`**.
- **Criterion:** `target_met` is true iff the pooled mean ≤ **5.0 ms** **and** every batch mean ≤ 5.0 ms.
- Reported alongside, not substituted for the primary: B2 and B3 means, p95/p99/max, and every
  stratum. A B3 mean exceeding 5 ms does not retroactively change the primary result; it is reported
  as measured and discussed.
- `target_met` means only "on this machine, with this Python build and this synthetic workload, the
  in-process LogiShield pipeline's average end-to-end prediction time was within 5 ms".

## 7. Exclusions

Cold start / first call / import time; model and vocabulary fitting; process start; error and
capacity-rejection paths (a rejection is not a prediction); concurrency, throughput and load;
real network, TLS, proxies; GPU; any machine other than the one recorded in the report.

## 8. Known limitations

1. One machine, one OS, one Python build; results do not transfer.
2. Synthetic payloads of at most 128 tokens after truncation; real traffic and SR-BH are not used.
3. Python-level timing of a pure-Python/numpy pipeline; other processes on the machine are not
   controlled (no CPU pinning, priority change, or isolation); this is reported, not hidden.
4. B3 shares one process and one interpreter lock between client and server.
5. The model is fitted on two fixtures; latency does not depend on detection quality.
6. GC is disabled during timed loops, which slightly flatters allocation-heavy paths.
7. The mean hides tail behaviour; p95/p99/max are reported for that reason.

## Changelog

- 1.0.0 — initial latency protocol, written before the first scored benchmark.
