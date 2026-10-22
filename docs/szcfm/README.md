# SZ-CFM — request / AI-assessment correlation in `ingestion/`

SZ-CFM correlates each intercepted HTTP request with an asynchronous AI threat
assessment and produces one ALLOW / CHALLENGE / BLOCK decision per request.

**Status of the specification.** The repository contains no SZ-CFM research
specification. Everything below — the fusion formula, thresholds, rule set,
state machine and feature schema — is an *implementation choice* documented
here, not an established research result.

**Status of the model.** No trained threat-detection model is connected. The
default backend, `szcfm-dev-heuristic-1`, is a hand-weighted logistic function
and every result it produces is reported with `"simulated": true`. Genuine AI
inference is not claimed.

## Endpoints (ingestion service, port 8001)

| Endpoint | Purpose |
|---|---|
| `POST /v1/szcfm/events` | Canonical SZ-CFM endpoint. Returns `SZCFMResult` (200), or 400 malformed JSON, 413 oversized, 422 schema/correlation error, 503 ring full (`Retry-After: 1`) |
| `POST /v1/ingest` | Still serves the legacy `RawEvent` and `metadata`+`payload`+`threat_score` shapes unchanged. A `metadata`+`payload` envelope **without** root `threat_score` (or any body with `schema_version`) is now routed to SZ-CFM; previously it fell through to `RawEvent` and failed with `source: Field required` |
| `GET /v1/szcfm/metrics` | Counters, latency p50/p95/max per stage, ring occupancy, inference in-flight (queue depth), backpressure rejections, throughput, error rate |
| `GET /v1/szcfm/feature-schema` | Ordered feature definitions (below) |

`/v1/intercept` and `/v1/threat-score` are unchanged legacy endpoints.

## Pipeline

| Stage | Module | What actually happens |
|---|---|---|
| A ingest | `app.py`, `szcfm.py`, `szcfm_schema.py` | Body streamed with a hard byte limit (`Content-Length` checked first); JSON parsed with NaN/Infinity rejected; strict Pydantic validation (`extra="forbid"` at every level except `request_body`); server-generated `event_id`; `request_id`/`correlation_id` taken from the payload, else the `X-Correlation-ID` header, else generated (header ≠ body → 422); wall-clock ingestion timestamp; monotonic `perf_counter_ns` for durations |
| B mask | `pii_masker.py` | Existing recursive masker, now configurable (`extra_credential_keys`, `extra_pii_keys`, `non_personal_name_keys`). Added defaults: `session_id`, `cookie`, `set_cookie`, `x_api_key`, `client_secret`, `session_token`; `product_name`/`item_name`/`brand_name`/`category_name` are no longer treated as personal names. The raw `Authorization` value is reduced to *present* + *scheme* and never stored, logged or serialized |
| D features | `feature_schema.py` | 28 fixed-order features, all in [0, 1] (below). Computed before C because the envelope carries the vector |
| E slot | `request_state.py` | Generation-stamped slot from a bounded ring |
| C serialize | `szcfm_proto.py`, `szcfm_context.proto` | Real protobuf `SZCFMEnvelope`; `SerializeToString(deterministic=True)`; `bytes_size = len(bytes)`; SHA-256 of the same bytes; bytes written once into the slot's pre-allocated memory, consumers get a read-only `memoryview` |
| H1 rules | `szcfm_policy.py` | Authentication, authorization and mandatory rules — no model input |
| F inference | `inference.py` | Submitted to a bounded worker pool **unless H1 already decided BLOCK** (then `SKIPPED`). Non-blocking admission: when `max_in_flight` is reached the request gets `REJECTED_BACKPRESSURE` immediately |
| G fusion | `request_state.py`, `szcfm.py` | Result accepted only if slot index, generation, request_id and correlation_id all match and the slot is `AI_PENDING` |
| H2 decide | `szcfm_policy.py`, `request_state.py` | Model or availability reason added; decision committed exactly once; forwarded only on ALLOW; masked event published to the broker; slot wiped and released |

### Canonical event schema (v1)

The interceptor shape is kept (`metadata`, `payload`, optional `threat_score`).
Additions: optional root `schema_version` (only `"1"` accepted) and optional
root `source`.

* `source` belongs at the **root**, as in `RawEvent`. If absent it is taken
  from the gateway-captured `metadata.headers["x-request-source"]`. If both are
  absent the event is rejected: *"source is required: set the root 'source'
  field or the metadata.headers['x-request-source'] header"*.
* Root `threat_score` is an optional upstream *prior*. It is one feature
  (`prior_threat_score`); it is never used as the AI assessment.
* Trust split (assumption): `metadata` (method, path, client IP, role,
  authorization) and `payload.authentication`, `traffic_context`,
  `security_context` are asserted by the gateway; `request_body`, `client` and
  request headers are client-controlled. Only `request_body` is free-form.
* Validation errors list `loc`, `msg`, `type` and never echo input values.
* Limits: 32 768 request bytes (configurable), body nesting ≤ 32, ≤ 100
  headers, identifier pattern `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`.

### Feature schema v1

| # | Feature | Units | Normalization | Missing |
|---|---|---|---|---|
| 0 | ip_reputation_score | score | as given | 0.5 + #1 |
| 1 | security_context_missing | 0/1 | | |
| 2 | method_is_mutating | 0/1 | POST/PUT/PATCH/DELETE | |
| 3 | method_is_delete | 0/1 | | |
| 4 | path_depth | segments | min(n/10, 1) | |
| 5 | path_is_admin | 0/1 | `/admin` segment | |
| 6 | path_is_auth | 0/1 | login/auth/token/... segment | |
| 7 | request_size | bytes | log1p(x)/log1p(max_request_bytes) | 0 + #14 |
| 8 | request_size_ratio | ratio | min(cur/avg, 10)/10 | 0 + #14 |
| 9 | requests_per_minute | req/min | log1p(x)/log1p(600) | 0 + #14 |
| 10 | requests_last_10_seconds | req/10 s | log1p(x)/log1p(100) | 0 + #14 |
| 11 | failed_logins_last_5_minutes | count | min(x/10, 1) | 0 + #14 |
| 12 | endpoint_requests_last_minute | req/min | log1p(x)/log1p(600) | 0 + #14 |
| 13 | previous_requests_from_ip | count | log1p(x)/log1p(10000) | 0 + #14 |
| 14 | traffic_context_missing | 0/1 | | |
| 15–18 | rate_limit_exceeded, authentication_anomaly, sql_injection_indicator, xss_indicator | 0/1 | | 0 + #1 |
| 19 | payload_attack_signature | 0/1 | ingestion's own scan of `request_body` | |
| 20 | payload_syntax_density | ratio | metacharacters / string chars | 0 if no strings |
| 21 | payload_depth | levels | min(depth/32, 1) | |
| 22 | authorization_present | 0/1 | | |
| 23 | authentication_missing | 0/1 | | |
| 24 | mfa_verified | 0/1 | | 0 + #23 |
| 25 | role_not_granted | 0/1 | client_role ∉ authenticated roles | 0 |
| 26 | prior_threat_score | score | as given | 0 + #27 |
| 27 | prior_threat_score_missing | 0/1 | | |

No identifiers, tokens, session values, IPs or free text enter the vector.
`validate_vector` enforces the dimension and finite values in [0, 1]; the
inference executor refuses a backend whose `feature_schema_version` or
`input_dim` differs.

The signature scan (`ATTACK_SIGNATURES`) is a small baseline set — UNION
SELECT, quote-tautology, stacked DDL, quote-comment terminator, `<script`,
inline event handlers, `javascript:`, path traversal. It is not comprehensive
and has not been evaluated for false positive/negative rates.

## Request-state ring

* **Lock-based, not lock-free.** One ring lock guards a FIFO free list of slot
  indices; one lock per slot guards its state. The two are only nested in
  stale reclamation (ring → slot), so there is no lock-order inversion.
* Ownership: an index popped from the free list belongs to one request until
  released. Every operation takes a `SlotTicket(slot_index, generation,
  request_id, correlation_id)` and re-checks all four fields under the slot
  lock.
* Full ring: **no eviction of live requests** (the legacy
  `ZeroCopyRingSlotBuffer.allocate_slot` evicts; SZ-CFM does not call it).
  Slots older than `stale_after_s` are reclaimed with a generation bump;
  otherwise `RingFullError` → HTTP 503.
* States: `FREE → ALLOCATED → AI_PENDING → AI_COMPLETED | AI_UNAVAILABLE → DECIDED → FREE`;
  `ALLOCATED → DECIDED` when AI is skipped.
* Delivery outcomes: `ACCEPTED`, `STALE_GENERATION` (slot reused or already
  released), `REQUEST_ID_MISMATCH`, `CORRELATION_MISMATCH`, `DUPLICATE`, `LATE`
  (after timeout, before release), `NOT_AWAITING`, `UNKNOWN_SLOT`. Each is
  counted in `/v1/szcfm/metrics` as `ai_delivery_<outcome>`.
* A result arriving before the request starts waiting is kept (the slot event
  is set); a result for a request that never allocated a slot cannot be
  matched — external pre-arrival is not supported.
* Memory: each slot owns 64 KiB of one pre-allocated `bytearray`
  (256 × 64 KiB = 16 MiB by default). Release zero-fills the region.
  "Zero-copy" is limited to this: protobuf serialization allocates a `bytes`
  object, which is copied once into the slot; readers then get views without
  further copies.

## Fusion and decision

```
context_risk = 0.5*ip_reputation + 0.2*burst_10s + 0.2*failed_logins + 0.1*auth_anomaly   (normalized features)
fused_risk   = max(ai_risk, w*ai_risk + (1-w)*context_risk)      w = 0.7
```

Context can raise the model's risk but never lower it. With
`challenge_threshold = 0.60` and `block_threshold = 0.85`: fused ≥ 0.85 →
BLOCK, ≥ 0.60 → CHALLENGE, else ALLOW. The development model is not
calibrated, so these thresholds have no probabilistic meaning yet.

Deterministic rules (evaluated first, independent of the model):

| Code | Effect |
|---|---|
| AUTH_MISSING, AUTH_CONTEXT_MISSING, AUTH_SCHEME_MISMATCH (non-public paths) | BLOCK |
| ROLE_NOT_GRANTED, ENDPOINT_ROLE_FORBIDDEN (`/v1/admin` → admin) | BLOCK |
| SQL_INJECTION_INDICATOR, XSS_INDICATOR, RATE_LIMIT_EXCEEDED, IP_REPUTATION_BLOCK (≥ 0.9), PAYLOAD_ATTACK_SIGNATURE | BLOCK |
| MFA_REQUIRED (`/v1/admin`), IP_REPUTATION_CHALLENGE (≥ 0.7), AUTHENTICATION_ANOMALY, FAILED_LOGIN_BURST (≥ 10) | CHALLENGE |
| ENVELOPE_SERIALIZATION_FAILED | CHALLENGE |

Precedence is BLOCK > CHALLENGE > ALLOW over all reasons, so a low model score
never lifts a deterministic BLOCK/CHALLENGE.

AI unavailable (`TIMEOUT`, `FAILED`, `REJECTED_BACKPRESSURE`, or a simulated
result when `simulated_ai_policy="unavailable"`) adds
`AI_UNAVAILABLE_<mode>`: `FAIL_CLOSED` → BLOCK, `CHALLENGE` → CHALLENGE
(default), `FAIL_OPEN` → ALLOW only if no other rule fired. Defaults:
`/v1/admin` and `/v1/payments` are FAIL_CLOSED; everything else CHALLENGE.
REVIEW is not implemented (the existing engine only knows ALLOW/BLOCK).

Every decision returns `reasons[]` with `code`, `origin`
(`authentication`/`authorization`/`mandatory_rule`/`model`/`availability`/`policy`),
`effect` and `detail`, plus `deterministic_decision` and `contributing_inputs`.

## Deadline

`ai_timeout_s` (default 50 ms) bounds how long the request thread waits for
the AI result (boundary: submit → result accepted, reported as
`ai_wait_time_ms`). It is a **soft** deadline in CPython: the timeout is
enforced when the waiting thread runs again, and a result delivered by then is
used. Under CPU/GIL saturation (32 concurrent threads) waits of up to ~147 ms
were observed with a 50 ms timeout. A timed-out inference keeps its worker and
in-flight permit until the backend call returns (threads cannot be
interrupted).

## Configuration (`SZCFM_*` environment variables)

`MAX_REQUEST_BYTES`, `RING_CAPACITY`, `RING_SLOT_BYTES`, `STALE_AFTER_S`,
`AI_TIMEOUT_S`, `INFERENCE_WORKERS`, `MAX_IN_FLIGHT`, `AI_WEIGHT`,
`BLOCK_THRESHOLD`, `CHALLENGE_THRESHOLD`, `IP_REPUTATION_BLOCK`,
`IP_REPUTATION_CHALLENGE`, `FAILED_LOGIN_CHALLENGE`, `PUBLIC_PATHS` (comma
list), `ROLE_REQUIRED` (`/prefix=role,role;...`), `MFA_REQUIRED_PATHS`,
`AI_FAILURE_MODE`, `AI_FAILURE_MODE_BY_PATH` (`/prefix=MODE;...`),
`SIMULATED_AI_POLICY` (`accept` | `unavailable`).

## Running

```bash
python -m ingestion                                    # port 8001
curl -X POST localhost:8001/v1/szcfm/events -H "content-type: application/json" \
     --data-binary @tests/ingestion/fixtures/szcfm_example_event.json
pytest tests/ingestion                                 # 101 tests incl. 68 SZ-CFM
python -m benchmarks.szcfm_benchmark --requests 2000   # in-process benchmark
```

## Observed benchmark (2026-10-22, one run)

Windows 11, CPython 3.13.3, Intel64 Family 6 Model 190; in-process, no HTTP;
example payload; simulated model; 200 warm-up requests; 2000 measured.

| Configuration | Throughput | p50 total | p95 total | p50 AI wait | p50 inference |
|---|---|---|---|---|---|
| Legacy interception path (baseline, does less work) | — | 0.61 ms | 2.51 ms | — | — |
| SZ-CFM, 1 thread | 520 req/s | 1.44 ms | 4.48 ms | 0.16 ms | 0.016 ms |
| SZ-CFM, 8 threads | 388 req/s | 16.7 ms | 49.0 ms | 14.8 ms | 0.019 ms |
| SZ-CFM, 32 threads | 420 req/s | 71.5 ms | 111.0 ms | 69.8 ms | 0.022 ms |

Interpretation limited to what was measured: single-threaded, SZ-CFM costs
about 0.8 ms more than the legacy path at p50, of which the thread handoff to
the inference pool is ~0.15 ms. Throughput does not scale with threads under
the GIL; added concurrency turns into waiting. No zero-copy, lock-free or
line-rate claim follows from these numbers.

## Known limitations / open questions

* No real model; thresholds uncalibrated; the fusion weights are arbitrary.
* No external asynchronous result endpoint for SZ-CFM; the legacy
  `/v1/threat-score` remains unauthenticated and unversioned.
* The forwarder is a hook; no reverse proxy to a protected API is configured.
* The gateway's `/v1/events` still posts `RawEvent` (root `source` required)
  and is not routed to SZ-CFM (outside the ingestion module).
* Legacy behaviour retained for compatibility and covered by existing tests:
  `/v1/ingest` interception forwards `_http_metadata.authorization` unmasked;
  `RingBufferBroker` is described as lock-free but performs non-atomic
  check-then-act; `/health` reports a constant `hotpath_cache_latency_ms`.
* Single process only: ring state is in memory, not shared across replicas.
