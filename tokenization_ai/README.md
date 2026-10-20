# LogiShield — Hybrid Structure-Aware Tokenization & Lightweight AI Prediction

Research ID **J-26-IT-376** · LogiShield: Neuro-Symbolic AI Verification for API Gateways

> **Status in one paragraph.** This package is a working, tested **proof of concept** of
> LogiShield's component: it fuses an API request with verified-rule information, masks
> sensitive data, tokenizes the result with an explicit structure-aware grammar, prepares a
> fixed-size tensor and scores it with a small trained model. It is **not** a validated
> intrusion detector, it has **not** been integrated with Formal Verification's or Runtime Verification's
> components, and every reported measurement is from **synthetic** data on a single machine.
> See [Limitations](#9-scientific-limitations) and [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md).

*Documentation was last verified against the repository at commit `ea50a49` (branch
`feature/logishield-tokenization-ai`; the masking hardening of audit finding D3 is included).
Counts and numbers below come from the stated source files or from the evidence reports named
in §8; nothing here was re-measured when writing it.*

---

## 1. Research purpose and contribution

**Component responsibility (proposal):** preprocess a unified API context together with
verified symbolic rule information into an AI-ready representation, run a lightweight model,
and emit an anomaly-aware signal for the next stage (Runtime Verification's runtime verification).

**Research gap addressed (proposal §1.3):** little evidence exists for a low-latency
preprocessing pipeline that combines API context with *already-verified symbolic rule
information*, preserves both semantic and symbolic structure during tokenization, and prepares
aligned model-ready tensors.

**What this package contributes (implemented):**

1. **Provenance-tracked privacy masking** — a token's "masked" role comes from the trusted
   masker's *path* record, never from matching text, so attacker text that merely looks like
   a placeholder is never trusted.
2. **Role-aware structure-aware tokenization** — every token carries an authoritative role
   (`control`, `literal_api`, `literal_rule`, `masked`); API and rule vocabularies are
   separate namespaces.
3. **Rule-preserving tensor truncation** — only payload content is shrunk; framing and the
   whole verified-rule segment are kept, or the request is rejected.
4. **A fail-closed, privacy-safe inspection API and demo** layered on the unchanged pipeline.
5. **A pre-registered, reproducible synthetic evaluation** of KPIs 1–4.

## 2. Processing pipeline (implemented)

```
Unified API context  +  verified-rule information
   │  adapters.py        to_unified_api_context / to_verified_rule_info
   ▼
1 Input fusion           fusion.py        InputFusion.fuse      (event-id match enforced)
2 Privacy masking        masking.py       PrivacyMasker.mask_with_provenance
3 Normalization          normalization.py Normalizer.normalize_with_provenance
4 Tokenization           tokenizer.py     HybridTokenizer.tokenize  → tokens + roles
5 Token IDs + tensor     vocabulary.py, tensorizer.py   → input_ids + attention_mask
6 Lightweight prediction model.py         LightweightAnomalyModel.predict → score in [0,1]
   │  service.py   TokenizationAIService.predict / analyze   (orchestration + tracing)
   ▼
AnomalyPrediction(event_id, score, model_version)      ← handoff to Runtime Verification NOT implemented
```

| Stage | What it does | Source |
|---|---|---|
| Input fusion | Joins context and rules by `event_id`; a mismatch raises `InputContractError` | `fusion.py` |
| Masking | Replaces sensitive values with `[MASKED_AUTH]`, `[MASKED_SECRET]`, `[MASKED_PII]`, `[MASKED_IP]`; records masked **paths** | `masking.py` |
| Normalization | Upper-case method; canonical path; lower-case header names; sorted payload keys; stripped strings. Re-keys masked paths | `normalization.py` |
| Tokenization | Emits a fixed grammar of control tags plus literals; keys sorted, list order kept | `tokenizer.py` |
| Vocabulary | Reserved ids for control tags and mask placeholders; literal ids learned once from a training corpus and frozen; unseen literals → `[UNK]` | `vocabulary.py` |
| Tensorization | Fixed `max_length`; padding + attention mask; structure-aware truncation of payload only; `ValueError` if framing + rules cannot fit | `tensorizer.py` |
| Model | Hashed unigram + bigram features → logistic regression (NumPy); `predict` refuses to run untrained | `model.py` |
| Tracing | Per-stage timings and count-only summaries; never raw values | `tracing.py`, `metrics.py` |

Not implemented (so not claimed): learned **embeddings** (token ids feed hashed n-gram
features; there is no embedding table), a **rule-aligned signal**, a **Runtime Verification output
adapter** (`adapters.to_runtime_verification_output` raises `NotImplementedError`), an anomaly **label /
threshold**, and the request-only vs rule-fused **baseline comparison**.

## 3. Input and output structures

**Input** (`models.py`, built by `adapters.py`):

- `UnifiedApiContext`: `event_id`, `method`, `path`, `content_type`, `client_ip`, `headers`,
  `payload` (object), `threat_score` (0–1, optional), `timestamp`.
  Accepted forms: Ingestion interception envelope (`event_id`, `metadata{method,path,…}`,
  `payload`, `threat_score`) or a `RawEvent` whose payload carries `_http_metadata` /
  `_threat_score`.
- `VerifiedRuleInfo`: `event_id` and `rules: [{rule_id, …attributes}]`. **This is LogiShield's
  own input structure; it is not Formal Verification's actual output format** (see
  [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md)).

`threat_score` is carried in the context but is **not tokenized and not used by the model**.
`client_ip` and `content_type` are likewise not tokenized (`client_ip` is masked).

**Outputs:**

| Output | Where | Notes |
|---|---|---|
| `AnomalyPrediction{event_id, score, model_version}` | `service.predict` | Score ∈ [0,1]; not calibrated |
| `TokenSequence{tokens, roles}` / `ModelInput{input_ids, attention_mask}` | internal | ids are never returned by the API |
| `PipelineAnalysis` | `service.analyze` | prediction + normalized record + masked paths + tokens + tensor |
| API response `logishield-analysis-v2` | `POST /v1/analyze` | privacy-safe projection, see §6 |
| Trace / runtime metrics | `tracing.py`, `metrics.py` | counts and timings only |

### Anomaly signal (proposed output format)

`anomaly_signal.py` adds `AnomalySignal`, a strict, immutable wrapper around `AnomalyPrediction`
(`to_anomaly_signal`, `to_anomaly_signal_dict`). Fields: `schema_version` (`"anomaly-signal-1"`),
`event_id`, `score` (finite, 0–1), `model_version`, and the fixed metadata
`score_semantics = "uncalibrated_model_output"`, `is_probability = false`, `calibrated = false`.
Unknown fields are rejected. There is no label, threshold, decision, `threat_score` or
rule-aligned signal.

```json
{"schema_version": "anomaly-signal-1", "event_id": "synthetic-example-1", "score": 0.25,
 "model_version": "synthetic-example-model", "score_semantics": "uncalibrated_model_output",
 "is_probability": false, "calibrated": false}
```

*(Synthetic illustration, not output of any running system.)*

- **Status:** a **proposed Member 4-owned format**. It is **not** a completed Member 3
  integration; nothing sends it anywhere, and `adapters.to_runtime_verification_output` is still unimplemented.
- **The score is uncalibrated and is not an attack probability.** KPI 5 remains unmeasured.
- **Member 3 must still agree** on score interpretation (and whether an uncalibrated score is
  acceptable), who owns any threshold, transport, and failure handling (fail-open or
  fail-closed). Field-name mapping onto Member 3's request is deliberately not done here.
- **Evidence:** this module is not in the KPI source-hash inventory, so the archived KPI 1–4
  reports do not cover it. Any future evaluation that uses it needs updated source provenance
  under a new protocol version; the historical evidence is not rewritten.

### Policy report compatibility adapter (isolated, not integrated)

`policy_report_adapter.py::policy_reports_to_verified_rule_info(context, reports,
applicable_policy_ids=...)` converts **caller-supplied** Member 2-style report mappings (JSON form)
into the existing `VerifiedRuleInfo`. It is an isolated compatibility adapter: it does not import
or call Member 2, and nothing in the pipeline, API or dashboard uses it.

- **`event_id` comes only from the supplied `UnifiedApiContext`.** A report that carries its own
  `event_id` is rejected.
- **Applicability is asserted by the caller.** `applicable_policy_ids` must equal the reports'
  `policy_id`s exactly (explicit, unique, non-empty). Member 4 never decides which policies apply
  to an HTTP request.
- **Only policy-level eligibility is accepted:** `is_valid` true, `rule_status == "eligible"`,
  top-level and solver status `UNSAT` for an `entailment` query with `is_entailment` true (UNSAT
  is the expected result because Member 2 checks the *negated* query), and no conflicts. This is
  **not** runtime request verification: Member 2 checks entailment on a fixed demonstration
  individual.
- **Allowlist only:** `subject`, `action`, `resource`, `effect`, `interpretation`. Raw policy text,
  formulas, traces, constraints and conditions are never copied; a policy with conditions is
  rejected rather than silently turned into an unconditional rule.
- **Strict and content-safe:** malformed, ambiguous or ineligible input raises
  `PolicyReportError` / `PolicyReportNotEligibleError` naming a report index and field, never a
  supplied value. There is no fallback to mock rules. Report order is preserved, and rules that
  do not fit the tensor raise (the existing tensorizer behaviour) instead of being truncated.
- **Unverified:** prediction quality with real policies (the fitted vocabulary and model only know
  mock tokens), automatic policy applicability, Member 2 integration, and KPI 5. Test fixtures are
  synthetic and may drift from Member 2's real output. The module is outside the archived KPI
  source-hash inventory; any evaluation using it needs updated provenance under a new protocol version.

## 4. Privacy masking, normalization and token roles

- **Masking is a heuristic, name- and shape-based blocklist.** It recognizes sensitive key
  names (e.g. `password`, `token`, `api_key`, `secret_key`, `session`, `jwt`, `email`,
  `phone`, `card`, `ssn`, `national_id`, `date_of_birth`, `address`, `first_name`,
  `client_ip`), authentication names (`authorization`, `cookie`, `x-api-key`,
  `x-session-id`, …) in headers, payloads and query strings, email / payment-card / IP /
  `Bearer …` / JWT-shaped *values*, and sensitive query-string parameters in the path.
  Key matching ignores case, `-`/`_`/`.`/space differences, surrounding whitespace and
  camelCase (`accessToken`, `Secret-Key`, `dateOfBirth`). Names are matched exactly or by a
  whole-word suffix, so `token_count`, `session_timeout` and `sort_key` are left alone.
  **Anything not on these lists is not masked** (for example a secret under an arbitrary
  field name, or a secret or personal value inside a URL path segment). Do not use real
  credentials. The list was extended after the PP1 privacy audit (finding D3). That extension
  was **not** part of the original KPI 1–3 / KPI 4 runs; KPI 4 was re-measured on the
  hardened code (§8), and KPI 1–3 were only re-run for reproducibility (§9.7).
- **Roles are authoritative.** `tokenizer.py` assigns `control` / `literal_api` /
  `literal_rule` / `masked` at the moment each token is emitted; the vocabulary and tensorizer
  read them rather than infer them. Ingestion's `[REDACTED_PII]` marker is **not** recognized as
  masked: it is tokenized as an ordinary `literal_api` (documented by a characterization test).
- **Grammar** (summary; full specification in `evaluation/PROTOCOL.md` §3):
  `[SEQ_START] method path headers payload rules [SEQ_END]`, with `[API:*]` tags for
  context/payload and `[RULE:*]` tags for rules; masked values follow
  `[API:VAL_MASKED_STR]` / `[API:HEADER_VALUE_MASKED]`.

## 5. Setup

Requirements: Python 3.12+ (developed and measured on 3.14.7), the packages in the root
`requirements.txt`, **plus NumPy** (kept out of the root requirements file; LogiShield only):

```powershell
cd C:\Users\kavin\Documents\LogiShield-Research\J26-IT-376
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r tokenization_ai\requirements-ml.txt     # numpy>=1.26
```

`fastapi`, `uvicorn`, `httpx`, `pydantic`, `pytest` come from the root `requirements.txt`.
The commands below call `.venv\Scripts\python.exe` directly so activation is optional.
Run everything from the repository root.

## 6. Running the backend, API and browser demo

**Start the service** (binds to `127.0.0.1:8004` only; no authentication — local use):

```powershell
.venv\Scripts\python.exe -m tokenization_ai
```

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8004/demo` | Browser demonstration (static HTML/CSS/JS, same origin) |
| `http://127.0.0.1:8004/health` | `{"status":"ok","service":"tokenization_ai","ready":true}`; 503 if the model/vocabulary is not ready |
| `http://127.0.0.1:8004/docs` | Auto-generated OpenAPI page |
| `POST http://127.0.0.1:8004/v1/analyze` | Analysis endpoint |

**`POST /v1/analyze`** request body:

| Field | Required | Meaning |
|---|---|---|
| `context` | yes | Interception envelope or `RawEvent`-shaped object (§3) |
| `rule_source` | yes | `"none"`, `"mock_fixture"`, or `"verified"` (**caller-asserted**; this service never verifies rules) |
| `rule_info` | for `mock_fixture`/`verified` | `{event_id, rules:[{rule_id,…}]}`; `event_id` must equal `context.event_id` |

```powershell
$body = @'
{"context":{"event_id":"demo-1","metadata":{"method":"POST","path":"/login",
  "headers":{"Authorization":"Bearer synthetic-not-a-token"}},
  "payload":{"username":"admin","password":"synthetic-pass"}},
 "rule_source":"none"}
'@
Invoke-RestMethod -Uri http://127.0.0.1:8004/v1/analyze -Method Post -ContentType application/json -Body $body
```

Response (`contract_version: "logishield-analysis-v2"`) contains `event_id`, `provenance`
(rule-source label, `verification_performed_by_this_service: false`, model note,
`detection_accuracy_claim: "none"`), `privacy`, `masking`, `masked_representation`, `tokens`,
`tensor` (metadata only, no ids), `prediction`, and `timings`. Errors use
`{"error": <code>, "detail": <generic text>}` (422 contract/validation, 411 missing
`Content-Length`, 413 body > 64 KB, 503 not ready).

**Privacy projection (implemented in `app.py`).** The model scores the complete
post-masking input, but the *response* shows only: reserved control/mask tokens, standard
HTTP methods, literals the fitted vocabulary already knows, and values at masked paths.
Everything else becomes a placeholder such as `[UNRECOGNIZED_VALUE]`. This does not change
what the model receives.

**Other entry points**

```powershell
.venv\Scripts\python.exe -m tokenization_ai.demo     # console run of the two synthetic fixtures with timings
```

## 7. Two-minute PP1 demonstration

Everything below was exercised in earlier verified runs; scores and counts are deterministic
for the bundled demo model. **Say at the start:** *"All data is synthetic, rules are mock,
and the model is fitted on two synthetic examples. This demonstrates the pipeline, not
real-world detection."*

**Preparation (before the panel):** `.venv\Scripts\python.exe -m tokenization_ai`, open
`http://127.0.0.1:8004/demo`. Keep a recorded backup of this run.

### Function 1 — Input fusion, privacy masking and normalization (~60 s)

1. Select **Suspicious API request** (mock rule fixture is labelled "not genuine Formal Verification
   output").
2. In *Request sent to the API* point out the raw synthetic `Authorization` header and
   `password`. Click **Analyze**.
3. Show, in order: **Input context** → **Input fusion** (1 rule, "Verified by this service:
   no") → **Privacy masking & normalization**.

**Expected (verified earlier):** *Fields masked: 2* (`headers.authorization`,
`payload.password`); representation shows `[MASKED_AUTH]` and `[MASKED_SECRET]`; header name
lower-cased (`authorization`); raw secrets absent from every response-derived panel.

**Concepts:** privacy-by-design masking with provenance, event-level fusion, normalization,
fail-closed output projection.

### Function 2 — Structure-aware tokenization, tensorization and prediction (~60 s)

1. Stay on the same result; scroll to **Structure-aware tokenization**: show the colored
   token roles (control / API literal / **masked** / rule literal).
2. **Tensor preparation**: 44 real tokens in a fixed length of 128, 84 padded.
3. **Lightweight AI prediction**: score and per-stage timings.
4. Select **Benign**, click **Analyze**, compare.

**Expected (verified earlier):** suspicious → 44 tokens, score ≈ **0.9991**; benign → 26
tokens, score ≈ **0.0016**; total pipeline time a fraction of a millisecond.

**Concepts:** authoritative token roles, namespaced vocabulary, fixed-size tensors, hashed
n-gram logistic regression.

**What the score is — say this out loud:** it separates the two bundled fixtures because
the model was fitted on exactly those two. It is a *demonstration that the stages are wired
end to end*, not a detection capability. In an informal 5-input probe during the audit a
normal login scored higher than attack-like requests; do not describe the score as accuracy.

**Likely viva questions:** why path-based provenance instead of text matching? What happens to
a secret under an unrecognized key? Why no embeddings? How is truncation kept safe for rules?
What do 0.34 ms (current code; 0.32 ms before the D3 masking fix) include and exclude? What would KPI 5 need?

## 8. Tests and research evidence

**Tests** (from the repository root):

```powershell
.venv\Scripts\python.exe -m pytest tests/tokenization_ai -q     # 871 passed at cf1f200 (current)
.venv\Scripts\python.exe -m pytest -q                           # 897 passed at cf1f200 (full repository)
```

Counts observed per commit (each is a record of that commit's run; older rows are kept as
history, not corrected):

| Commit | Member 4 suite | Full repository | What changed |
|---|---|---|---|
| `ea50a49` | 560 passed | 586 passed | Counts at the post-D3 KPI 4 re-evaluation (the commit the KPI 4 figures are tied to) |
| `a76ba11` | 673 passed | 699 passed | Verified baseline after the dashboard work |
| `964813b` | 710 passed | 736 passed | Added `anomaly_signal.py` (+37 tests) |
| `cf1f200` | **871 passed** | **897 passed** | Added `policy_report_adapter.py` (+161 tests); **current** |

Each run reported one pre-existing Starlette/httpx deprecation warning (third-party). Unit and
framework tests establish behaviour, not research KPI attainment or detection quality.

**Reproducing the evaluation** (writes reports to the directory you give; never overwrite
original evidence — use an empty directory):

```powershell
.venv\Scripts\python.exe -m tokenization_ai.evaluation.generate --check                 # fixtures match their seeds
.venv\Scripts\python.exe -m tokenization_ai.evaluation --out-dir <new-dir> --write-manifest
.venv\Scripts\python.exe -m tokenization_ai.evaluation --out-dir <new-dir>               # KPI 1-3
.venv\Scripts\python.exe -m tokenization_ai.evaluation.latency --out-dir <new-dir> --write-manifest
.venv\Scripts\python.exe -m tokenization_ai.evaluation.latency --out-dir <new-dir>      # KPI 4 (~3 min)
```

### KPI summary (synthetic; targets are proposal targets, not real-world results)

| KPI | Target | Reported result | Basis |
|---|---|---|---|
| 1 Input alignment | ≥ 90% | **31 / 31** valid synthetic events (Wilson 95%: 89.0–100%); 5 / 5 invalid events rejected, reported separately | `evaluation/PROTOCOL.md` §4 |
| 2 Token mapping | ≥ 90% | **31 / 31** sequences matched; 3719 / 3719 token positions; masking provenance 31 / 31 | same |
| 3 Tensor correctness | ≥ 95% | **30 / 30** tensors (Wilson 88.7–100%); 1 / 1 capacity rejection as specified | same |
| 4 Average prediction time | ≤ 5 ms | **Current code (`ea50a49`, post-D3): pre-registered B1 mean 0.3413 ms** (worst batch mean 0.3481 ms; 5 batches × 1000 = 5000 calls). Historical pre-D3 baseline: 0.3200 ms (worst batch 0.3371 ms). Both on one machine and a synthetic workload; target met in both | `evaluation/LATENCY_PROTOCOL.md` |
| 5 Downstream usability | ≥ 90% | **Pending genuine Runtime Verification integration — not measured** | [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) |

**KPI 4 in detail.** The same frozen latency protocol and harness were run twice:

| Pooled mean, S6 mixed workload (ms) | Pre-D3 (original) | Post-D3 (current code) |
|---|---|---|
| **B1** in-process pipeline (primary KPI) | **0.3200** | **0.3413** (+0.0213, ≈ +6.7%) |
| B1T same, with tracer | 0.3393 | 0.3605 |
| B2 adapters + pipeline | 0.3326 | 0.3580 |
| B3 real HTTP, loopback | 2.7696 | 2.7007 |
| B3 large payload S4: mean / p95 | 4.4607 / 5.2659 | 4.5444 / 5.1237 |
| Masking stage mean (informational) | 0.0896 | 0.1145 |

- **What B1 is:** the in-process **end-to-end pipeline time** (fusion, masking,
  normalization, tokenization, tensorization and model scoring, no tracing). It is **not**
  isolated model inference (the model-scoring stage alone is ≈ 0.075 ms mean in the
  informational stage breakdown) and it is **not** deployed or real-world latency: it excludes
  adapters, JSON, HTTP and the privacy projection.
- **The increase** is consistent with the extra key normalization and matching added to
  `masking.py`, but it comes from single runs on one non-isolated machine, so small
  differences cannot be attributed precisely; B3 values moved in both directions, i.e.
  run-to-run noise.
- **HTTP tail latency:** B3 includes the client, JSON and the privacy projection, and client
  and server share one process. Its mixed-workload mean is under 5 ms (p95 4.46 ms, p99
  4.89 ms post-D3), but the large-payload stratum exceeds 5 ms at p95 (5.12 ms) and a single
  37 ms outlier occurred. These are reported as measured; the pre-registered primary result
  remains the B1 mean.
- **Scope:** synthetic workload of ≤ 128 tokens after truncation, one machine (Windows 10,
  8 logical CPUs, "Ultimate Performance" power scheme). No production-SLA, throughput or
  concurrency claim is made.

**Evidence locations (outside the repository, on the author's machine):**

| Evidence | Path |
|---|---|
| KPI 1–3 pre-run manifest, report (JSON/MD), run logs | `C:\Users\kavin\Downloads\LogiShield_PP1_Coding_Evidence\KPI_Evaluation\` (`PRE_RUN_MANIFEST.json`, `kpi_report.json`, `kpi_report.md`) |
| KPI 4 **original (pre-D3)** benchmark: the historical baseline, produced by the code before the masking fix (`masking.py` sha256 `d23ae1f9…`) | same folder (`LATENCY_PRE_RUN_MANIFEST.json`, `latency_report.json`, `latency_report.md`, `latency_raw_samples.json.gz`) |
| KPI 4 **post-D3** re-evaluation of the current code at `ea50a49` (`masking.py` sha256 `8eb0f66a…`): same protocol and harness, new directory, nothing overwritten | `C:\Users\kavin\Downloads\LogiShield_PP1_Coding_Evidence\KPI_Evaluation\Post_D3_Latency\` (`POST_D3_TRACEABILITY_PRE_RUN.json`, `LATENCY_PRE_RUN_MANIFEST.json`, `latency_report.json`, `latency_report.md`, `latency_raw_samples.json.gz`, `README_post_D3_comparison.txt`, `terminal_*.txt`) |
| Browser interaction log and screenshots | `C:\Users\kavin\Downloads\LogiShield_PP1_Coding_Evidence\` (`browser_interaction_log.txt`, `screenshot_*.png`) |
| Frozen protocols and fixtures (in repo) | `tokenization_ai/evaluation/PROTOCOL.md`, `LATENCY_PROTOCOL.md`, `fixtures/` — **do not edit**; a change needs a new protocol version |

## 9. Scientific limitations

1. **Synthetic only.** Cases are hand-designed coverage scenarios (17 curated, 12 seeded, 2
   designed), not a sample of traffic. N = 31, so even a perfect score has a Wilson lower bound
   (89.0% / 88.7%) below the 90% / 95% targets. The result means "handled as specified in this
   set", not "statistically shown to meet the target".
2. **Same-author oracle.** Expected outputs are hand-written plus a second encoder by the same
   author; the evaluation detects regressions and deviations from the written design, not a
   flawed design.
3. **Unknown-token rate 40.41%** (99 / 245 literal positions, non-truncated sequences) because
   the evaluation vocabulary is fitted on a deliberately small development corpus (395
   entries). It describes that corpus, not real traffic.
4. **No exact-fit tensor cases** (length mix: 21 padded, 0 exact-fit, 9 truncated). KPI 3 feeds
   the tensorizer *expected* sequences; the tokenizer→tensor path is covered by a secondary
   consistency count (30 / 30).
5. **The model is a synthetic proof of concept.** The bundled model is fitted on two
   hand-labelled fixtures (`demo.py`); no precision, recall, F1 or any real-world
   anomaly-detection metric exists, and none is claimed. The SR-BH adapter
   (`dataset_adapters.py`) has been tested on synthetic rows only; no dataset was obtained.
6. **Mock rules.** `MOCK-*` rule fixtures stand in for verified rules. Formal-verification
   correctness is not evaluated here.
7. **Masking is a blocklist, not a guarantee** (§4). The audit's D3 gaps were fixed in
   `masking.py` and covered by `tests/tokenization_ai/test_masking_hardening.py`, but
   unlisted field names, secrets in URL path segments, and free text remain unmasked.
   Evidence versioning: the original KPI 1–3 and KPI 4 reports were produced by the earlier
   `masking.py` (source hash `d23ae1f9…`). Re-running the KPI 1–3 evaluation on the
   unchanged frozen fixtures reproduced the identical results hash (`f353779d…`), which
   confirms only that those 31 cases are unaffected; it is not a measurement of the new
   masking rules. KPI 4 **was** re-measured on the hardened code (§8, 0.3413 ms). The
   Ingestion header exposure is noted in INTEGRATION_STATUS.
8. **Latency** is one machine (Windows 10, 8 logical CPUs, Windows "Ultimate Performance"
   power scheme), Python-level timing, GC disabled in timed loops, synthetic payloads of ≤ 128
   tokens after truncation; not a throughput, concurrency or production-SLA result. The
   post-D3 figure (0.3413 ms) is one protocol-conformant run; the pre-D3 baseline (0.3200 ms)
   was a separate run, so the ≈ +6.7% difference is indicative, not a controlled estimate of
   the masking cost.
9. **Score semantics.** The score is uncalibrated and is not a probability of attack.
10. **Integration.** No Formal Verification (Member 2) or Runtime Verification (Member 3)
    integration exists at runtime; KPI 5 is unmeasured. The two standalone modules added after
    the KPI evidence (`anomaly_signal.py`, `policy_report_adapter.py`) do not change this:
    schema compatibility is not integration.
11. **No held-out detection-quality result yet.** The held-out evaluation infrastructure
    (`heldout_eval/`, request-only versus simulated-rule versus character n-gram arms) is
    implemented and tested on development fixtures only; the confirmatory experiment has not
    been run, so no request-only versus rule-fused comparison result exists. Scores are
    uncalibrated and are not attack probabilities.

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: numpy` (tests or service) | NumPy is not in root `requirements.txt`. `pip install -r tokenization_ai\requirements-ml.txt` |
| `No module named tokenization_ai` | Run from the repository root (`...\J26-IT-376`); `python -m` resolves the package from the current directory |
| `[Errno 10048]` / port 8004 in use | An earlier server is still running. Stop it, or find it with `Get-NetTCPConnection -LocalPort 8004` |
| `/demo` shows "Demo scenarios unavailable" | The page loaded but `/demo/assets/scenarios.json` failed; reload, confirm the server is on 127.0.0.1:8004 and check `/health` |
| Analyze fails: "Could not reach the LogiShield service" | Server not running or a different port; start it as in §6 |
| `/health` returns 503 / `model_not_ready` | Vocabulary or model not fitted; the default demo service fits both at start-up, so restart the process |
| `411 length_required` from a custom client | The endpoint requires a `Content-Length` header; chunked uploads are rejected |
| `422 invalid_event_id` | `event_id` must be 1–64 characters of letters, digits, `.`, `_`, `:` or `-` |
| `422 input_contract_violation` | Missing `metadata.method`/`path`, non-object payload, rule without `rule_id`, or `threat_score` outside 0–1 |
| `422 pipeline_input_rejected` | Mandatory framing plus the full rule segment exceeds `max_length` (128) |
| Evaluation CLI "refusing to overwrite" | Intentional: reports and pre-run manifests are never overwritten; use an empty directory |
| Browser blocks the page | The demo ships a strict Content-Security-Policy; use the page from `http://127.0.0.1:8004/demo` directly (not an iframe or file://) |

## 11. File map

| Path | Role |
|---|---|
| `adapters.py`, `models.py` | Input contracts and data models |
| `fusion.py`, `masking.py`, `normalization.py` | Stages 1–3 |
| `tokenizer.py`, `vocabulary.py`, `tensorizer.py` | Stages 4–5 |
| `model.py`, `service.py` | Model and orchestration |
| `tracing.py`, `metrics.py` | Per-stage tracing and runtime metrics |
| `app.py`, `__main__.py`, `static/` | HTTP API and browser demo |
| `demo.py` | Console demo and the fixtures that fit the bundled model |
| `anomaly_signal.py` | Standalone `AnomalySignal` output schema (proposed format). **Not wired into any running service**: the API, dashboard and `service.py` do not use it |
| `policy_report_adapter.py` | Standalone adapter from caller-supplied Member 2-style policy reports to `VerifiedRuleInfo`. **Not wired into any running service** and not Member 2 integration |

`anomaly_signal.py` and `policy_report_adapter.py` are exercised only by their own tests
(`tests/tokenization_ai/test_anomaly_signal.py`, `test_policy_report_adapter.py`). Both sit outside
the archived KPI source-hash inventory, so the historical KPI 1-4 evidence, which stays tied to
its original evaluation protocol, does not cover them; a future evaluation using either module needs
updated source provenance under a new protocol version.
| `dataset_adapters.py` | SR-BH 2020 row adapter (not run on real data) |
| `evaluation/` | Protocols, frozen fixtures, KPI 1–4 runners |
| `INTEGRATION_STATUS.md` | Cross-member integration state |
| `LICENSE`, `NOTICE.md` | Licence and contribution notice |
