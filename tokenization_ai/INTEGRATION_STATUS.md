# LogiShield — Integration Status

*Verified against the repository at commit `de07872` (branch `feature/member4-tokenization-ai`)
and, for Formal Verification and Runtime Verification, against the already-fetched remote refs `origin/feature/policy_verification`
(tip `7e9f54c`) and `origin/dev` (tip `a247ba0`). Other members' code was **read only**. This
document describes what exists today; it does not define any contract on another member's behalf.*

> **Reading guide.** Everything from "Summary" down is the original audit at `de07872` and is kept
> unchanged as a historical record. The section dated 2026-10-21 describes the current state on the
> integration branch; the section dated 2026-10-19 is kept as the record of Member 4 at `cf1f200`.

## Current status (2026-10-21, D4-7G3, branch `integration/member4-dev-sync` on top of `f8282f5`)

This branch merges `dev` (`e9e040a`) with Member 4 (`87365a2`). Other members' code and `shared_contracts`
were read and called in tests only; none was modified.

**Fixed in this phase**

| Change | What it does | Evidence |
|---|---|---|
| `model.py` restored to Member 4's verified version | Removes `initialize_calibrated` / `create_calibrated`, which filled the weights from `RandomState(42)` and set `is_trained=True`: random weights are neither trained nor calibrated, and the flag bypassed the model's refusal to predict before `fit` | `tests/tokenization_ai/test_model_initialization_safety.py` |
| `TokenizationAIService.create_default()` | Keeps the API that `tests/test_e2e_pipeline.py` calls, but now returns the standalone demo pipeline: vocabulary and model genuinely fitted (gradient descent) on the two synthetic demo fixtures. Its score shows only that the stages run; it is not calibrated and not a detection model | same file; `tests/test_e2e_pipeline.py` passes unchanged |
| `request_decision_adapter.py` (new) | Converts Formal Verification's real `POST /api/v1/verify-request` body (`verify-request-response-v1`) into rule input: one rule with the enumerated `decision` and `reason_code`, `per_policy_outcomes="unavailable"`, `event_association="caller_asserted"`. `event_id` comes only from the API context (the contract has none); free text, formulas and subject ids are never copied; a decision inconsistent with the reported solver statuses is rejected | `tests/tokenization_ai/test_request_decision_adapter.py` (round trips through Formal Verification's real API for all decision kinds) |
| `runtime_handoff.py` (new) and `adapters.to_runtime_verification_output` | Returns `runtime-verification-input-1`: the strict `AnomalySignal` plus exactly the `event_id` and `score` that Runtime Verification's existing `HybridVerificationService.evaluate_runtime` and circuit breaker take. No threshold, label or decision is added; the score stays an uncalibrated model output | `tests/tokenization_ai/test_runtime_handoff.py` (a fitted prediction passed into Runtime Verification's real code) |
| Archived KPI 1-4 files | Restored to the archived bytes, LF pinned by `tokenization_ai/evaluation/.gitattributes`; all three archived pre-run manifests verify | D4-7G2 |

**Link status (what is genuinely connected)**

| Link | Status |
|---|---|
| Ingestion → LogiShield | Tested in-process with Ingestion's real app output (`test_ingestion_integration.py`). Not a networked or deployed connection |
| Formal Verification → LogiShield | **Adapter implemented and tested against the real API; not wired.** Nothing calls it at runtime. Missing from the contract: an `event_id` (or a request correlation id) and a mapping from an HTTP request to the subject, roles, action and resource that Formal Verification checks. The decision is request-level; per-policy outcomes are not available. The fitted demo vocabulary does not know the decision tokens (they map to the unknown token), so they carry no learned weight |
| LogiShield → Runtime Verification | **Handoff implemented and argument-compatible with Runtime Verification's real code; not wired.** Nothing in the running system calls it. Still to agree with Runtime Verification: the threshold for an uncalibrated score, transport, and failure handling |
| Runtime path | **Not integrated.** The gateway does not route requests to LogiShield; `docker-compose.yml` has no LogiShield service; `engine/service.py` still scores with `StubModel`; `tests/test_e2e_pipeline.py` computes a LogiShield analysis but passes a hard-coded `score=0.95` to the circuit breaker |
| KPI 5 | **Still unmeasured** |

**Model used anywhere at runtime:** none. LogiShield's model runs only inside its own service (`python -m
tokenization_ai`, port 8004, the demo model fitted on two synthetic fixtures) and in tests. No group runtime path uses
a LogiShield prediction. The held-out evaluation (`heldout_eval/`) is implemented and tested on development fixtures
only; no confirmatory result exists.

**Outside Member 4's scope (needs the owners):** routing requests to LogiShield in the gateway; a LogiShield service in
`docker-compose.yml`; replacing `StubModel` or feeding LogiShield's score into `evaluate_runtime`; the score threshold;
an event correlation id in Formal Verification's response; replacing the hard-coded score in `tests/test_e2e_pipeline.py`.

## Current status (2026-10-19, Member 4 at `cf1f200`)

Two standalone Member 4 modules were added after the audit. **Neither is wired into the running
service, API or dashboard**, and neither completes an integration.

| Module | What it is | What it is not |
|---|---|---|
| `anomaly_signal.py` (`AnomalySignal`, commit `964813b`) | A strict, immutable output format carrying `event_id`, `score`, `model_version` and fixed metadata (`uncalibrated_model_output`, `is_probability=false`, `calibrated=false`) | Not an agreed Member 3 contract. No label, threshold, decision or `threat_score`. Nothing sends it anywhere. `adapters.to_member3_output` (renamed `to_runtime_verification_output` on `dev`) is still `NotImplementedError` |
| `policy_report_adapter.py` (commit `cf1f200`) | A strict conversion of **caller-supplied** Member 2-style policy report mappings to `VerifiedRuleInfo`. `event_id` comes only from the supplied context; the caller asserts which policies apply; only policy-level eligible reports are accepted | Not Member 2 integration. It does not import or call Member 2, does not decide applicability, and policy eligibility is not runtime request verification. Test reports are synthetic and may drift from Member 2's real output |

Current link status:

| Link | Status |
|---|---|
| Member 1 → Member 4 | Unchanged: tested in-process only |
| Member 2 → Member 4 | **Not integrated.** An isolated adapter exists, but no real Member 2 output is consumed, no event linkage or applicability rule is agreed, and real-policy prediction quality is unverified (the fitted vocabulary and model know only mock rule tokens) |
| Member 4 → Member 3 | **Not integrated.** An isolated output format exists, but score interpretation, threshold ownership, transport and failure handling are not agreed with Member 3 |
| KPI 5 | **Still unmeasured.** Schema compatibility alone would not measure or prove downstream usability |

Research limits that still apply:
- Scores are uncalibrated and are not attack probabilities.
- There is no held-out detection-quality evaluation and no request-only versus rule-fused baseline yet.
- The historical KPI 1-4 evidence stays tied to its original evaluation protocol and is not rewritten.
- Both new modules are outside the archived KPI source-hash inventory; any future evaluation that
  uses them needs updated source provenance under a new protocol version.

Test results per commit are recorded in `README.md` section 8 (current: 871 Member 4 and 897
repository tests passing at `cf1f200`).

## Summary

| Link | Status | One-line truth |
|---|---|---|
| Ingestion → LogiShield | **Data contract tested in-process** | Real `ingestion` app output is accepted; not a deployed or networked integration |
| Formal Verification → LogiShield | **Not integrated; schemas differ** | LogiShield consumes a structure it defined; Formal Verification's report has a different shape and no event link |
| LogiShield → Runtime Verification | **Not integrated; no contract** | LogiShield outputs a prediction; `to_runtime_verification_output` raises `NotImplementedError` |
| KPI 5 (downstream usability) | **Cannot be measured honestly yet** | Needs a real Runtime Verification input contract |
| Full group system | **Not demonstrated** | LogiShield runs standalone on `127.0.0.1:8004`; it is not in `docker-compose.yml` |

---

## 1. Ingestion → LogiShield: what is genuinely tested

**Source:** `ingestion/app.py` (`POST /v1/ingest`, interception path), `ingestion/pii_masker.py`;
LogiShield side `tokenization_ai/adapters.py::to_unified_api_context`.
**Test:** `tests/tokenization_ai/test_ingestion_integration.py` (9 tests).

What the tests actually do: instantiate Ingestion's real `create_app` with its in-memory broker,
POST an interception request, capture the `RawEvent` that Ingestion forwards downstream, then pass
that real object through LogiShield's adapter, service and `/v1/analyze` API.

| Verified | Evidence |
|---|---|
| Forwarded `RawEvent` carries `_http_metadata` and `_threat_score`; LogiShield's adapter accepts it and preserves event id, method, path, threat score | `test_adapter_accepts_ingestion_event_and_preserves_identity` |
| LogiShield processes it and masks again independently (Authorization header, `password`) | `test_logishield_service_processes_ingestion_event` |
| API response is privacy-safe for that event | `test_logishield_api_accepts_ingestion_event_and_stays_privacy_safe` |
| Ingestion rejection (bad `threat_score`) means nothing reaches LogiShield | `test_ingestion_rejects_invalid_interception_…` |
| Mismatched rule event id is rejected | `test_rule_info_for_a_different_event_is_rejected` |

**Not verified / limits**

- Everything runs in one process with a capturing stand-in for the processing client (the same
  pattern Ingestion's own tests use). There is **no network hop, no Redis broker, no
  docker-compose run** involving LogiShield.
- The `/v1/ingest` standard (non-interception) path produces events **without HTTP metadata**;
  LogiShield cannot consume them (`InputContractError`, test
  `test_non_interception_ingestion_event_is_not_consumable_by_logishield`).
- **Observation for Ingestion:** the forwarded event keeps request headers (e.g. `Authorization`)
  unmasked inside `payload["_http_metadata"]["headers"]`; only the payload body is passed through
  `PIIMasker`. LogiShield masks headers itself, so its output is safe, but any other consumer of
  that event would see them.
- Ingestion's `[REDACTED_PII]` marker reaches LogiShield as an ordinary string and gets role
  `literal_api`, not `masked` (by design: LogiShield's provenance is path-based; characterised in
  `test_ingestion_redaction_marker_is_handled_as_an_ordinary_literal`).

**Decision needed from Ingestion (low urgency):** whether headers should be masked before
forwarding; whether a single agreed marker constant should be shared.

---

## 2. Formal Verification → LogiShield: schema differences and required decisions

### What LogiShield currently accepts (its own definition)

`tokenization_ai/models.py` and `adapters.to_verified_rule_info`:

```json
{ "event_id": "<same id as the API context>",
  "rules": [ { "rule_id": "<string>", "<any attribute>": "<json value>" } ] }
```

Rule attributes are tokenized as-is. In all tests and demos these rules are `MOCK-*` fixtures.
Nothing in this repository establishes that Formal Verification produces this structure.

### What Formal Verification's code actually produces (read from `origin/feature/policy_verification`)

`src/policy_verifier/models.py::VerificationReport`, returned by
`src/policy_verifier/verifier.py::PolicyVerificationEngine.process_and_verify` and exposed at
`POST /api/v1/verify-policies` (`api/routes.py`):

```json
{ "is_valid": true,
  "status": "SAT" | "UNSAT",
  "fol_expressions": ["Allow(P1)", "Deny(P2)"],
  "conflicting_policies": ["P2"] | null,
  "execution_time_ms": 0.05 }
```

Input to that endpoint is a list of `SecurityPolicy{policy_id, subject, action, resource,
effect, raw_text}`; `routes.py` returns HTTP 400 with the report when the status is `UNSAT`.

### Differences (facts from the code above)

| Aspect | LogiShield expects | Formal Verification produces |
|---|---|---|
| Event linkage | `event_id` required and must equal the API context's | **No `event_id`**; a report describes a policy set, not a request |
| Rule list | `rules[]` each with `rule_id` + attributes | `fol_expressions` (strings such as `Allow(P1)`) and `conflicting_policies` (ids); no per-rule attribute objects |
| Verdict | none (rules are inert input) | `is_valid` / `status` SAT-UNSAT |
| Field names | `rule_id` | `policy_id` (in `SecurityPolicy`) |
| Timing | – | `execution_time_ms` |

A `VerificationReport` posted to LogiShield's `to_verified_rule_info` would be rejected for the
missing `event_id`/`rules` fields. Formal Verification also has a `ConflictDetector` (`conflict_detector.py`)
that uses Z3; in the files read, `process_and_verify` — the function `routes.py` calls — builds
`fol_expressions` and sets SAT/UNSAT from the presence of allow and deny effects, and does not
call `ConflictDetector`. Formal Verification should confirm which path is intended to be the verified-rule
output.

### Decisions required from Formal Verification (not assumed here)

1. **What is the verified-rule output that LogiShield should consume?** `VerificationReport`, a
   per-policy record, or a different structure?
2. **How does a verification result relate to a single API event?** Which request attributes
   select applicable policies, and who attaches `event_id`? (Today there is no linkage.)
3. **Which fields should reach the model?** e.g. `fol_expressions`, policy ids, SAT/UNSAT, the
   conflicting policy list.
4. **Is an UNSAT report ever delivered, or only SAT?** (`routes.py` turns UNSAT into HTTP 400.)
5. **Stable vocabulary:** are policy ids/FOL strings a closed set suitable for a trained
   vocabulary, or open text?
6. **Delivery mechanism:** HTTP call, shared module, or message — and its version.

**LogiShield next step once answered:** write a narrowly scoped adapter in `adapters.py` and
re-run the frozen-protocol evaluation as a *new protocol version*. Until then, LogiShield results
must be reported as "mock rules".

---

## 3. LogiShield → Runtime Verification: outputs, missing contract, adapter stub

### What LogiShield outputs today

| Output | Form |
|---|---|
| Anomaly prediction | `AnomalyPrediction{event_id, score∈[0,1], model_version}` (`models.py`) |
| AI-ready representation | `ModelInput{input_ids, attention_mask}` (internal; not returned by the API) |
| Trace / metrics | per-stage timings and counts (`tracing.py`, `metrics.py`) |
| Inspection response | `logishield-analysis-v2` JSON (`app.py`) — a privacy-safe view for humans, **not** a handoff contract |

### What is missing

- **No Runtime Verification input contract exists** in the repository. `shared_contracts` defines only the
  legacy generic models (`RawEvent`, `FeatureVector`, `Prediction`, `Alert`); none carries LogiShield's
  output.
- **Runtime Verification's implementation is a placeholder** on `main`, `dev` and
  `origin/feature/policy_verification`: `engine/service.py` contains `StubModel` (score = clipped
  maximum feature) and a threshold; there is no LTL / Z3 runtime rule engine, circuit breaker or
  verdict/XAI trace in what I inspected.
- `tokenization_ai/adapters.py::to_runtime_verification_output(prediction)` is an **intentional
  `NotImplementedError`** because no agreed contract exists.
- The proposal's **"Rule-aligned Signal"** and **"Next Component Input"** outputs have no
  implementation and no definition of what they contain.
- Shared contract changes need approval from all members (`README.md` branching rules,
  `.github/CODEOWNERS`).

### Decisions required from Runtime Verification

1. Input format and transport of the anomaly-aware signal (fields, types, versioning).
2. Is a single score sufficient, or are a label/threshold, rule-alignment information or token-level
   attributions required?
3. Latency and error-handling expectations (what happens on a LogiShield rejection or `503`).
4. Whether the signal is consumed per request synchronously or asynchronously.

---

## 4. Why KPI 5 cannot yet be honestly measured

Proposal Table 2: *"Downstream output usability — useful downstream anomaly signal ≥ 90%;
measurement: check compatibility with next verification component."*

| Needed | Exists? |
|---|---|
| A documented Runtime Verification input contract | No |
| A running Runtime Verification consumer (or its agreed schema) | No (placeholder `StubModel` only) |
| Definition of "usable" (schema validity vs. decision usefulness) | No agreed definition; the proposal's KPI name and measurement method describe different things |
| Defined rule-aligned signal / next-component input | No |

Any percentage computed today would be measured against a contract LogiShield invented, i.e. a
self-assessment. **KPI 5 is reported as pending genuine Runtime Verification integration.** What can be done
independently without claiming KPI 5: document LogiShield's output schema as a *proposal*, check
that each of the proposal's five named outputs is produced (currently: prediction yes,
AI-ready representation internal only, rule-aligned signal no, next-component input no,
trace partial), and keep that list clearly labelled as completeness, not usability.

---

## 5. Integration risks, owners and proposed next steps

| # | Risk | Severity | Owner of decision | Proposed next step |
|---|---|---|---|---|
| R1 | Formal Verification's report schema differs from LogiShield's rule input; no event linkage | **High** | Formal Verification (+ LogiShield for the adapter) | Agree the verified-rule output and event association (§2 questions); then implement the adapter and a new evaluation protocol version |
| R2 | No Runtime Verification contract or runtime; `to_runtime_verification_output` unimplemented | **High** | Runtime Verification (+ all members for `shared_contracts`) | Agree the handoff schema (§3 questions); then implement the adapter and measure KPI 5 |
| R3 | Rule-aligned signal and next-component input undefined | Medium | Members 2, 3, 4 jointly | Define semantics in the contract discussion |
| R4 | Standard (non-HTTP) ingestion events are unusable by LogiShield | Medium | Ingestion | Confirm that only interception events are intended to reach LogiShield |
| R5 | Ingestion leaves headers unmasked in forwarded events | Medium (privacy) | Ingestion | Mask headers before forwarding, or document that downstream must |
| R6 | LogiShield not packaged: absent from `docker-compose.yml`; Dockerfile installs root requirements only (no NumPy) | Medium | LogiShield + whoever owns compose | Add `numpy` handling and a service entry once contracts settle (needs shared approval) |
| R7 | Masking is a heuristic blocklist. Audit finding D3 (credential-like payload keys, session header, whitespace-padded header names) was fixed in `masking.py` with regression tests; unlisted field names and secrets in path segments remain unmasked | Medium → Low-Medium (privacy) | LogiShield | Keep the lists under review; any further extension is a documented policy change with a new evaluation protocol version if frozen fixtures are affected |
| R8 | LogiShield's model is fitted on two synthetic fixtures; score has no detection meaning | Medium (scientific) | LogiShield | Train/evaluate on a labelled dataset (SR-BH adapter exists) with a leakage-safe split |
| R9 | `/v1/analyze` requires `Content-Length` and caps bodies at 64 KB | Low | LogiShield / Runtime Verification | Revisit when the real transport is chosen |
| R10 | Component demonstrated standalone only; "complete system" must not be claimed | – | All | Present integration as *planned*, with the contract questions above as the plan |

## 6. What can and cannot be said at PP1

**Can say (evidence-backed):** LogiShield's pipeline, API and demo work; Ingestion's real output is
accepted by LogiShield in-process; KPIs 1–4 have synthetic, pre-registered measurements with the
limitations in `README.md` §9.

**Cannot say:** that Formal Verification's verified rules are consumed; that Runtime Verification receives or uses
LogiShield's output; that KPI 5 is met; that the system detects real intrusions; or that the group
system is integrated end to end.
