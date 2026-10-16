# Member 4 — Integration Status

*Verified against the repository at commit `de07872` (branch `feature/member4-tokenization-ai`)
and, for Member 2/3, against the already-fetched remote refs `origin/feature/policy_verification`
(tip `7e9f54c`) and `origin/dev` (tip `a247ba0`). Other members' code was **read only**. This
document describes what exists today; it does not define any contract on another member's behalf.*

## Summary

| Link | Status | One-line truth |
|---|---|---|
| Member 1 → Member 4 | **Data contract tested in-process** | Real `ingestion` app output is accepted; not a deployed or networked integration |
| Member 2 → Member 4 | **Not integrated; schemas differ** | Member 4 consumes a structure it defined; Member 2's report has a different shape and no event link |
| Member 4 → Member 3 | **Not integrated; no contract** | Member 4 outputs a prediction; `to_member3_output` raises `NotImplementedError` |
| KPI 5 (downstream usability) | **Cannot be measured honestly yet** | Needs a real Member 3 input contract |
| Full group system | **Not demonstrated** | Member 4 runs standalone on `127.0.0.1:8004`; it is not in `docker-compose.yml` |

---

## 1. Member 1 → Member 4: what is genuinely tested

**Source:** `ingestion/app.py` (`POST /v1/ingest`, interception path), `ingestion/pii_masker.py`;
Member 4 side `tokenization_ai/adapters.py::to_unified_api_context`.
**Test:** `tests/tokenization_ai/test_member1_integration.py` (9 tests).

What the tests actually do: instantiate Member 1's real `create_app` with its in-memory broker,
POST an interception request, capture the `RawEvent` that Member 1 forwards downstream, then pass
that real object through Member 4's adapter, service and `/v1/analyze` API.

| Verified | Evidence |
|---|---|
| Forwarded `RawEvent` carries `_http_metadata` and `_threat_score`; Member 4's adapter accepts it and preserves event id, method, path, threat score | `test_adapter_accepts_member1_event_and_preserves_identity` |
| Member 4 processes it and masks again independently (Authorization header, `password`) | `test_member4_service_processes_member1_event` |
| API response is privacy-safe for that event | `test_member4_api_accepts_member1_event_and_stays_privacy_safe` |
| Member 1 rejection (bad `threat_score`) means nothing reaches Member 4 | `test_member1_rejects_invalid_interception_…` |
| Mismatched rule event id is rejected | `test_rule_info_for_a_different_event_is_rejected` |

**Not verified / limits**

- Everything runs in one process with a capturing stand-in for the processing client (the same
  pattern Member 1's own tests use). There is **no network hop, no Redis broker, no
  docker-compose run** involving Member 4.
- The `/v1/ingest` standard (non-interception) path produces events **without HTTP metadata**;
  Member 4 cannot consume them (`InputContractError`, test
  `test_non_interception_member1_event_is_not_consumable_by_member4`).
- **Observation for Member 1:** the forwarded event keeps request headers (e.g. `Authorization`)
  unmasked inside `payload["_http_metadata"]["headers"]`; only the payload body is passed through
  `PIIMasker`. Member 4 masks headers itself, so its output is safe, but any other consumer of
  that event would see them.
- Member 1's `[REDACTED_PII]` marker reaches Member 4 as an ordinary string and gets role
  `literal_api`, not `masked` (by design: Member 4's provenance is path-based; characterised in
  `test_member1_redaction_marker_is_handled_as_an_ordinary_literal`).

**Decision needed from Member 1 (low urgency):** whether headers should be masked before
forwarding; whether a single agreed marker constant should be shared.

---

## 2. Member 2 → Member 4: schema differences and required decisions

### What Member 4 currently accepts (its own definition)

`tokenization_ai/models.py` and `adapters.to_verified_rule_info`:

```json
{ "event_id": "<same id as the API context>",
  "rules": [ { "rule_id": "<string>", "<any attribute>": "<json value>" } ] }
```

Rule attributes are tokenized as-is. In all tests and demos these rules are `MOCK-*` fixtures.
Nothing in this repository establishes that Member 2 produces this structure.

### What Member 2's code actually produces (read from `origin/feature/policy_verification`)

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

| Aspect | Member 4 expects | Member 2 produces |
|---|---|---|
| Event linkage | `event_id` required and must equal the API context's | **No `event_id`**; a report describes a policy set, not a request |
| Rule list | `rules[]` each with `rule_id` + attributes | `fol_expressions` (strings such as `Allow(P1)`) and `conflicting_policies` (ids); no per-rule attribute objects |
| Verdict | none (rules are inert input) | `is_valid` / `status` SAT-UNSAT |
| Field names | `rule_id` | `policy_id` (in `SecurityPolicy`) |
| Timing | – | `execution_time_ms` |

A `VerificationReport` posted to Member 4's `to_verified_rule_info` would be rejected for the
missing `event_id`/`rules` fields. Member 2 also has a `ConflictDetector` (`conflict_detector.py`)
that uses Z3; in the files read, `process_and_verify` — the function `routes.py` calls — builds
`fol_expressions` and sets SAT/UNSAT from the presence of allow and deny effects, and does not
call `ConflictDetector`. Member 2 should confirm which path is intended to be the verified-rule
output.

### Decisions required from Member 2 (not assumed here)

1. **What is the verified-rule output that Member 4 should consume?** `VerificationReport`, a
   per-policy record, or a different structure?
2. **How does a verification result relate to a single API event?** Which request attributes
   select applicable policies, and who attaches `event_id`? (Today there is no linkage.)
3. **Which fields should reach the model?** e.g. `fol_expressions`, policy ids, SAT/UNSAT, the
   conflicting policy list.
4. **Is an UNSAT report ever delivered, or only SAT?** (`routes.py` turns UNSAT into HTTP 400.)
5. **Stable vocabulary:** are policy ids/FOL strings a closed set suitable for a trained
   vocabulary, or open text?
6. **Delivery mechanism:** HTTP call, shared module, or message — and its version.

**Member 4 next step once answered:** write a narrowly scoped adapter in `adapters.py` and
re-run the frozen-protocol evaluation as a *new protocol version*. Until then, Member 4 results
must be reported as "mock rules".

---

## 3. Member 4 → Member 3: outputs, missing contract, adapter stub

### What Member 4 outputs today

| Output | Form |
|---|---|
| Anomaly prediction | `AnomalyPrediction{event_id, score∈[0,1], model_version}` (`models.py`) |
| AI-ready representation | `ModelInput{input_ids, attention_mask}` (internal; not returned by the API) |
| Trace / metrics | per-stage timings and counts (`tracing.py`, `metrics.py`) |
| Inspection response | `member4-analysis-v2` JSON (`app.py`) — a privacy-safe view for humans, **not** a handoff contract |

### What is missing

- **No Member 3 input contract exists** in the repository. `shared_contracts` defines only the
  legacy generic models (`RawEvent`, `FeatureVector`, `Prediction`, `Alert`); none carries Member 4's
  output.
- **Member 3's implementation is a placeholder** on `main`, `dev` and
  `origin/feature/policy_verification`: `engine/service.py` contains `StubModel` (score = clipped
  maximum feature) and a threshold; there is no LTL / Z3 runtime rule engine, circuit breaker or
  verdict/XAI trace in what I inspected.
- `tokenization_ai/adapters.py::to_member3_output(prediction)` is an **intentional
  `NotImplementedError`** because no agreed contract exists.
- The proposal's **"Rule-aligned Signal"** and **"Next Component Input"** outputs have no
  implementation and no definition of what they contain.
- Shared contract changes need approval from all members (`README.md` branching rules,
  `.github/CODEOWNERS`).

### Decisions required from Member 3

1. Input format and transport of the anomaly-aware signal (fields, types, versioning).
2. Is a single score sufficient, or are a label/threshold, rule-alignment information or token-level
   attributions required?
3. Latency and error-handling expectations (what happens on a Member 4 rejection or `503`).
4. Whether the signal is consumed per request synchronously or asynchronously.

---

## 4. Why KPI 5 cannot yet be honestly measured

Proposal Table 2: *"Downstream output usability — useful downstream anomaly signal ≥ 90%;
measurement: check compatibility with next verification component."*

| Needed | Exists? |
|---|---|
| A documented Member 3 input contract | No |
| A running Member 3 consumer (or its agreed schema) | No (placeholder `StubModel` only) |
| Definition of "usable" (schema validity vs. decision usefulness) | No agreed definition; the proposal's KPI name and measurement method describe different things |
| Defined rule-aligned signal / next-component input | No |

Any percentage computed today would be measured against a contract Member 4 invented, i.e. a
self-assessment. **KPI 5 is reported as pending genuine Member 3 integration.** What can be done
independently without claiming KPI 5: document Member 4's output schema as a *proposal*, check
that each of the proposal's five named outputs is produced (currently: prediction yes,
AI-ready representation internal only, rule-aligned signal no, next-component input no,
trace partial), and keep that list clearly labelled as completeness, not usability.

---

## 5. Integration risks, owners and proposed next steps

| # | Risk | Severity | Owner of decision | Proposed next step |
|---|---|---|---|---|
| R1 | Member 2's report schema differs from Member 4's rule input; no event linkage | **High** | Member 2 (+ Member 4 for the adapter) | Agree the verified-rule output and event association (§2 questions); then implement the adapter and a new evaluation protocol version |
| R2 | No Member 3 contract or runtime; `to_member3_output` unimplemented | **High** | Member 3 (+ all members for `shared_contracts`) | Agree the handoff schema (§3 questions); then implement the adapter and measure KPI 5 |
| R3 | Rule-aligned signal and next-component input undefined | Medium | Members 2, 3, 4 jointly | Define semantics in the contract discussion |
| R4 | Standard (non-HTTP) ingestion events are unusable by Member 4 | Medium | Member 1 | Confirm that only interception events are intended to reach Member 4 |
| R5 | Member 1 leaves headers unmasked in forwarded events | Medium (privacy) | Member 1 | Mask headers before forwarding, or document that downstream must |
| R6 | Member 4 not packaged: absent from `docker-compose.yml`; Dockerfile installs root requirements only (no NumPy) | Medium | Member 4 + whoever owns compose | Add `numpy` handling and a service entry once contracts settle (needs shared approval) |
| R7 | Masking is a heuristic blocklist. Audit finding D3 (credential-like payload keys, session header, whitespace-padded header names) was fixed in `masking.py` with regression tests; unlisted field names and secrets in path segments remain unmasked | Medium → Low-Medium (privacy) | Member 4 | Keep the lists under review; any further extension is a documented policy change with a new evaluation protocol version if frozen fixtures are affected |
| R8 | Member 4's model is fitted on two synthetic fixtures; score has no detection meaning | Medium (scientific) | Member 4 | Train/evaluate on a labelled dataset (SR-BH adapter exists) with a leakage-safe split |
| R9 | `/v1/analyze` requires `Content-Length` and caps bodies at 64 KB | Low | Member 4 / Member 3 | Revisit when the real transport is chosen |
| R10 | Component demonstrated standalone only; "complete system" must not be claimed | – | All | Present integration as *planned*, with the contract questions above as the plan |

## 6. What can and cannot be said at PP1

**Can say (evidence-backed):** Member 4's pipeline, API and demo work; Member 1's real output is
accepted by Member 4 in-process; KPIs 1–4 have synthetic, pre-registered measurements with the
limitations in `README.md` §9.

**Cannot say:** that Member 2's verified rules are consumed; that Member 3 receives or uses
Member 4's output; that KPI 5 is met; that the system detects real intrusions; or that the group
system is integrated end to end.
