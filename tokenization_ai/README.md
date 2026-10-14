# Member 4 — Hybrid Structure-Aware Tokenization & Lightweight AI Prediction

Research ID **J-26-IT-376** · LogiShield: Neuro-Symbolic AI Verification for API Gateways

> **Status in one paragraph.** This package is a working, tested **proof of concept** of
> Member 4's component: it fuses an API request with verified-rule information, masks
> sensitive data, tokenizes the result with an explicit structure-aware grammar, prepares a
> fixed-size tensor and scores it with a small trained model. It is **not** a validated
> intrusion detector, it has **not** been integrated with Member 2's or Member 3's
> components, and every reported measurement is from **synthetic** data on a single machine.
> See [Limitations](#9-scientific-limitations) and [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md).

*Documentation was verified against the repository at commit `de07872` (branch
`feature/member4-tokenization-ai`). Counts and numbers below come from the stated source
files or from the evidence reports named in §8; nothing here was re-measured when writing it.*

---

## 1. Research purpose and contribution

**Component responsibility (proposal):** preprocess a unified API context together with
verified symbolic rule information into an AI-ready representation, run a lightweight model,
and emit an anomaly-aware signal for the next stage (Member 3's runtime verification).

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
AnomalyPrediction(event_id, score, model_version)      ← handoff to Member 3 NOT implemented
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
features; there is no embedding table), a **rule-aligned signal**, a **Member 3 output
adapter** (`adapters.to_member3_output` raises `NotImplementedError`), an anomaly **label /
threshold**, and the request-only vs rule-fused **baseline comparison**.

## 3. Input and output structures

**Input** (`models.py`, built by `adapters.py`):

- `UnifiedApiContext`: `event_id`, `method`, `path`, `content_type`, `client_ip`, `headers`,
  `payload` (object), `threat_score` (0–1, optional), `timestamp`.
  Accepted forms: Member 1 interception envelope (`event_id`, `metadata{method,path,…}`,
  `payload`, `threat_score`) or a `RawEvent` whose payload carries `_http_metadata` /
  `_threat_score`.
- `VerifiedRuleInfo`: `event_id` and `rules: [{rule_id, …attributes}]`. **This is Member 4's
  own input structure; it is not Member 2's actual output format** (see
  [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md)).

`threat_score` is carried in the context but is **not tokenized and not used by the model**.
`client_ip` and `content_type` are likewise not tokenized (`client_ip` is masked).

**Outputs:**

| Output | Where | Notes |
|---|---|---|
| `AnomalyPrediction{event_id, score, model_version}` | `service.predict` | Score ∈ [0,1]; not calibrated |
| `TokenSequence{tokens, roles}` / `ModelInput{input_ids, attention_mask}` | internal | ids are never returned by the API |
| `PipelineAnalysis` | `service.analyze` | prediction + normalized record + masked paths + tokens + tensor |
| API response `member4-analysis-v2` | `POST /v1/analyze` | privacy-safe projection, see §6 |
| Trace / runtime metrics | `tracing.py`, `metrics.py` | counts and timings only |

## 4. Privacy masking, normalization and token roles

- **Masking is heuristic.** It recognizes sensitive key names (e.g. `password`, `token`,
  `api_key`, `secret`, `email`, `phone`, `card`, `ssn`, `client_ip`), authentication header
  names (`authorization`, `cookie`, `x-api-key`, …), email/payment-card/IP-shaped *values*,
  and sensitive query-string parameters in the path. **Unrecognized field names are not
  masked** — for example, a payload key such as `authorization`, `secret_key`, `apikey`,
  `session`, `jwt` or `address` currently passes through (checked with synthetic values in
  the final audit; this is a known gap, listed in §9). Do not use real credentials.
- **Roles are authoritative.** `tokenizer.py` assigns `control` / `literal_api` /
  `literal_rule` / `masked` at the moment each token is emitted; the vocabulary and tensorizer
  read them rather than infer them. Member 1's `[REDACTED_PII]` marker is **not** recognized as
  masked: it is tokenized as an ordinary `literal_api` (documented by a characterization test).
- **Grammar** (summary; full specification in `evaluation/PROTOCOL.md` §3):
  `[SEQ_START] method path headers payload rules [SEQ_END]`, with `[API:*]` tags for
  context/payload and `[RULE:*]` tags for rules; masked values follow
  `[API:VAL_MASKED_STR]` / `[API:HEADER_VALUE_MASKED]`.

## 5. Setup

Requirements: Python 3.12+ (developed and measured on 3.14.7), the packages in the root
`requirements.txt`, **plus NumPy** (kept out of the root requirements file; Member 4 only):

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

Response (`contract_version: "member4-analysis-v2"`) contains `event_id`, `provenance`
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

1. Select **Suspicious API request** (mock rule fixture is labelled "not genuine Member 2
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
What does 0.32 ms include and exclude? What would KPI 5 need?

## 8. Tests and research evidence

**Tests** (from the repository root):

```powershell
.venv\Scripts\python.exe -m pytest tests/tokenization_ai -q     # 419 passed at de07872
.venv\Scripts\python.exe -m pytest -q                           # 445 passed at de07872 (full repository)
```

(Counts are those observed at `de07872`; one pre-existing Starlette/httpx deprecation warning
appears.) Unit and framework tests establish behaviour, not research KPI attainment.

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
| 4 Average prediction time | ≤ 5 ms | pre-registered **B1 mean 0.3200 ms** (worst batch mean 0.3371 ms; 5 batches × 1000 = 5000 calls) on one machine and synthetic workload | `evaluation/LATENCY_PROTOCOL.md` |
| 5 Downstream usability | ≥ 90% | **Pending genuine Member 3 integration — not measured** | [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) |

KPI 4 detail (same run): B2 (adapters + pipeline) mixed mean 0.333 ms; **B3 (real HTTP on
loopback, client and server in one process) mixed mean 2.77 ms**, large-payload mean 4.46 ms
with p95 5.27 ms. The 0.32 ms figure is the in-process pipeline only and is **end-to-end
prediction time, not isolated model inference** (model scoring stage ≈ 0.075 ms mean in the
informational stage breakdown).

**Evidence locations (outside the repository, on the author's machine):**

| Evidence | Path |
|---|---|
| KPI 1–3 pre-run manifest, report (JSON/MD), run logs | `C:\Users\kavin\Downloads\LogiShield_PP1_Coding_Evidence\KPI_Evaluation\` (`PRE_RUN_MANIFEST.json`, `kpi_report.json`, `kpi_report.md`) |
| KPI 4 manifest, report, raw per-call samples | same folder (`LATENCY_PRE_RUN_MANIFEST.json`, `latency_report.json`, `latency_report.md`, `latency_raw_samples.json.gz`) |
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
7. **Masking recall gaps** (§4) and the Member 1 header exposure noted in INTEGRATION_STATUS.
8. **Latency** is one machine (Windows 10, 8 logical CPUs, Windows "Ultimate Performance"
   power scheme), Python-level timing, GC disabled in timed loops, synthetic payloads of ≤ 128
   tokens after truncation; not a throughput, concurrency or production-SLA result.
9. **Score semantics.** The score is uncalibrated and is not a probability of attack.
10. **Integration.** No Member 2 or Member 3 integration exists; KPI 5 is unmeasured.

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: numpy` (tests or service) | NumPy is not in root `requirements.txt`. `pip install -r tokenization_ai\requirements-ml.txt` |
| `No module named tokenization_ai` | Run from the repository root (`...\J26-IT-376`); `python -m` resolves the package from the current directory |
| `[Errno 10048]` / port 8004 in use | An earlier server is still running. Stop it, or find it with `Get-NetTCPConnection -LocalPort 8004` |
| `/demo` shows "Demo scenarios unavailable" | The page loaded but `/demo/assets/scenarios.json` failed; reload, confirm the server is on 127.0.0.1:8004 and check `/health` |
| Analyze fails: "Could not reach the Member 4 service" | Server not running or a different port; start it as in §6 |
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
| `dataset_adapters.py` | SR-BH 2020 row adapter (not run on real data) |
| `evaluation/` | Protocols, frozen fixtures, KPI 1–4 runners |
| `INTEGRATION_STATUS.md` | Cross-member integration state |
| `LICENSE`, `NOTICE.md` | Licence and contribution notice |
