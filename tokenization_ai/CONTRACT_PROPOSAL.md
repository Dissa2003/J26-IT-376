# LogiShield — Cross-Member Interface Contract **Proposal**

**Status: PROPOSAL FOR DISCUSSION. Nothing in this document has been agreed with Formal Verification or
Runtime Verification, nothing here is implemented, and no cross-member integration beyond what §1 states
exists.** All JSON in §4 and §5 is labelled *PROPOSED EXAMPLE*: it is illustrative, contains
invented identifiers, and is not an output of any running system or a verification or
prediction result.

| | |
|---|---|
| Author component | LogiShield — Hybrid Structure-Aware Tokenization & Lightweight AI Prediction |
| Written against | repository commit `6ccf1df` (branch `feature/logishield-tokenization-ai`); Formal Verification code at `origin/feature/policy_verification` @ `7e9f54c`; Ingestion code at `origin/feature/ingestion` @ `7d82a16` / `origin/dev` @ `a247ba0`; no Runtime Verification branch exists on `origin` |
| Related documents | `INTEGRATION_STATUS.md` (current state), `README.md` (usage and evidence) |
| Proposal version | 0.1 (draft) |

---

## 1. Integration status today (facts, not claims)

| Link | State | Evidence |
|---|---|---|
| Ingestion → LogiShield | **Tested in-process only.** Real `ingestion` app output is accepted by LogiShield's adapter and API. No network hop, Redis, or docker-compose run involving LogiShield. | `tests/tokenization_ai/test_ingestion_integration.py` (9 tests) |
| Formal Verification → LogiShield | **Unresolved.** Formal Verification's output schema differs from what LogiShield accepts; there is no event correlation. No adapter exists. | §2 |
| LogiShield → Runtime Verification | **Pending.** No Runtime Verification input contract exists in the repository and no runtime_verification implementation was found on any remote ref. `adapters.to_runtime_verification_output` raises `NotImplementedError`. | `adapters.py`; `engine/service.py` is the generic `StubModel` |
| KPI 5 (downstream usability) | **Not measurable**; no real consumer or agreed schema | `README.md` §8 |

## 2. Formal Verification → LogiShield: what exists and how it differs

### 2.1 What LogiShield currently accepts (LogiShield's own definition)

`tokenization_ai/models.py::VerifiedRuleInfo`, built by `adapters.to_verified_rule_info`:

```text
VerifiedRuleInfo
  event_id : str  (required, must equal the API context's event_id)
  rules    : list[VerifiedRule]
VerifiedRule
  rule_id    : str (required)
  attributes : dict[str, any JSON]   # every key of the rule object other than "rule_id"
```

Rules are tokenized as inert data (`[RULE:ID] <rule_id> [RULE:ATTRS_START] …`). LogiShield never
evaluates them. In all tests and demos these are `MOCK-*` fixtures.

### 2.2 What Formal Verification's code produces (read from `origin/feature/policy_verification`)

`src/policy_verifier/models.py`, `verifier.py`, `api/routes.py`:

```text
POST /api/v1/verify-policies
  request : list[SecurityPolicy]
              SecurityPolicy{ policy_id, subject, action, resource,
                              effect: "Allow"|"Deny", raw_text }
  response: VerificationReport
              { is_valid: bool,
                status: "SAT" | "UNSAT",
                fol_expressions: list[str]          # e.g. "Allow(P1)", "Deny(P2)"
                conflicting_policies: list[str]|null,
                execution_time_ms: float }
  UNSAT -> HTTP 400 with {"message": ..., "report": <VerificationReport>}
```

### 2.3 Exact mismatches

| # | Aspect | LogiShield expects | Formal Verification provides | Consequence |
|---|---|---|---|---|
| 1 | **Event correlation** | `event_id` on the rule information, equal to the context's | **No `event_id`** anywhere in `src/`, `api/`, `storage/`; a report describes a *policy set* | LogiShield's adapter would reject a `VerificationReport` (missing `event_id`, `rules`) |
| 2 | **Rule representation** | `rules[]` of objects (`rule_id` + attributes) | `fol_expressions` strings (`"Allow(P1)"`) and `conflicting_policies` ids; policy objects are *inputs* to the endpoint, not part of the response | A lossy or invented mapping unless a representation is agreed |
| 3 | **Identity naming** | `rule_id` | `policy_id` (inputs); embedded inside FOL strings in outputs | Preservation of policy identity must be decided (§3.2) |
| 4 | **Applicability to a request** | Rules are assumed to apply to the event they accompany | No mapping from an HTTP request (`method`, `path`, headers, payload) to a policy's `subject`/`action`/`resource` | Nobody currently decides *which* policies are relevant to *this* request |
| 5 | **Verdict semantics** | None; rules are inert tokens | `is_valid` / `status` for the whole set; `UNSAT` returns HTTP 400 | Unclear whether LogiShield ever sees an `UNSAT` report, and what it would mean |
| 6 | **Pipeline wiring** | consumed by an adapter | separate FastAPI app (`api/routes.py`) not wired into the legacy `processing/` stage | Delivery mechanism undecided |

Observations Formal Verification should confirm (read from the code; not asserted as defects): the path
used by `routes.py` (`PolicyVerificationEngine.process_and_verify`) derives SAT/UNSAT from the
presence of Allow and Deny effects and does not call `ConflictDetector` (the Z3 component);
`storage/database.py::save_log` is not called by that path.

## 3. Proposed minimal Formal Verification → LogiShield interface (for Formal Verification's decision)

*Formal Verification has not accepted any of this. It is the smallest change that would let LogiShield
consume real output without inventing semantics.*

### 3.1 Principle

LogiShield should receive **verified-rule information that is already correlated to one API
event**, and treat it as opaque, structured data. Deciding *which* policies apply to a request
and *what a verification result means* is Formal Verification's responsibility, not LogiShield's.

### 3.2 Proposed structure — "Verified Rule Info v1" (PROPOSED)

| Field | Type | Req. | Meaning / source |
|---|---|---|---|
| `schema_version` | string | yes | e.g. `"verified-rule-info-1"` |
| `event_id` | string | yes | The API event this applies to (same as the Ingestion/LogiShield context id) |
| `verification` | object | yes | Result of Formal Verification's verification for the applicable policy set |
| `verification.status` | `"SAT"`\|`"UNSAT"` | yes | From `VerificationReport.status` |
| `verification.is_valid` | bool | yes | From `VerificationReport.is_valid` |
| `verification.execution_time_ms` | number | no | From `VerificationReport` |
| `rules` | array | yes (may be empty) | One entry per **applicable** policy |
| `rules[].rule_id` | string | yes | **Preserves `policy_id` unchanged** |
| `rules[].effect` | `"Allow"`\|`"Deny"` | yes | From `SecurityPolicy.effect` |
| `rules[].expression` | string | no | The corresponding FOL string (`"Allow(P1)"`) |
| `rules[].in_conflict` | bool | no | True if the policy id appears in `conflicting_policies` |
| `rules[].subject/action/resource` | strings | no | Optional descriptive attributes |

**How LogiShield would use it (once agreed):** a small adapter in `tokenization_ai/adapters.py`
maps this onto the existing `VerifiedRuleInfo` — `rule_id` unchanged, the remaining fields as
`attributes`, and the `verification` summary carried as additional attributes or a documented
extension. No change to the tokenizer or model is required for the rule *content* itself.

**PROPOSED EXAMPLE (illustrative; not an actual Formal Verification output; identifiers invented):**

```json
{
  "schema_version": "verified-rule-info-1",
  "event_id": "evt-example-0001",
  "verification": { "status": "SAT", "is_valid": true, "execution_time_ms": 0.05 },
  "rules": [
    { "rule_id": "P1", "effect": "Allow", "expression": "Allow(P1)", "in_conflict": false,
      "subject": "Admin", "action": "Delete", "resource": "UserAccount" }
  ]
}
```

### 3.3 Handling of SAT/UNSAT and HTTP 400 (open decision, options only)

| Option | Description | Open implication |
|---|---|---|
| A | Deliver both SAT and UNSAT reports as data with `status`; transport uses HTTP 200 | LogiShield tokenizes the status as a signal |
| B | UNSAT is an error (HTTP 400, as today): LogiShield never receives UNSAT | Then the "rule" signal can only ever be SAT; the model sees no conflicts |
| C | Formal Verification resolves conflicts before delivery and sends only the effective rule set | Policy identity of conflicting rules is lost unless carried separately |

LogiShield has no preference that is scientifically neutral; the choice changes what the model can
learn. This must be decided by Formal Verification together with Runtime Verification.

### 3.4 What must remain unchanged

- `event_id` correlation is mandatory; LogiShield rejects mismatches (existing behaviour,
  `fusion.py`).
- Rule content is never masked by LogiShield (existing behaviour); Formal Verification must not place
  secrets in rule attributes.
- Rule literals unknown to the fitted vocabulary map to `[UNK]` (existing behaviour); a
  closed, stable policy-id/effect vocabulary would make training more meaningful (question Q2.5).

## 4. LogiShield's current actual output (fact)

### 4.1 Programmatic output

`tokenization_ai/models.py::AnomalyPrediction`, returned by `TokenizationAIService.predict`:

| Field | Type | Notes |
|---|---|---|
| `event_id` | string | Same id as the input context |
| `score` | float ∈ [0, 1] | Output of a small model fitted on two synthetic fixtures; **uncalibrated; not a probability of attack** |
| `model_version` | string | Currently `ngram-linear-v1-trained` |

Also available internally (not delivered anywhere): `ModelInput{input_ids, attention_mask}`,
`TokenSequence{tokens, roles}`, per-stage trace (`tracing.py`).

### 4.2 HTTP inspection output

`POST /v1/analyze` returns the privacy-safe `logishield-analysis-v2` document (`app.py`:
`event_id`, `provenance`, `privacy`, `masking`, `masked_representation`, `tokens`, `tensor`,
`prediction{score, model_version}`, `timings`). It is a human inspection view with a
fail-closed privacy projection, **not** a machine handoff contract.

### 4.3 Not produced today

| Item (from the proposal's output list) | Status |
|---|---|
| Anomaly **label** / threshold | Not produced (score only) |
| **Rule-aligned signal** | Not implemented; no definition exists |
| **Next component input** | `to_runtime_verification_output` is `NotImplementedError` |
| Structured, persisted logs | Per-request in-memory trace only |
| Token-level attribution / explanation | Not implemented |

## 5. Proposed minimal LogiShield → Runtime Verification interface (for discussion)

*No Runtime Verification input contract exists today (`shared_contracts` contains only the legacy
`RawEvent`, `FeatureVector`, `Prediction`, `Alert`; `engine/service.py` is a generic
`StubModel`). Runtime Verification has not been consulted.*

### 5.1 Fields: existing vs proposed

| Field | Type | Status | Notes |
|---|---|---|---|
| `schema_version` | string | **proposed** | e.g. `"anomaly-signal-1"` |
| `event_id` | string | **existing** | `AnomalyPrediction.event_id` |
| `score` | number [0,1] | **existing** | `AnomalyPrediction.score` |
| `model_version` | string | **existing** | `AnomalyPrediction.model_version` |
| `model_note` | string | **proposed** | e.g. "fitted on synthetic fixtures; not calibrated" — prevents over-trusting the score |
| `label` | `"normal"`\|`"anomaly"` | **proposed, not implemented** | Requires a threshold decision (§5.3) |
| `threshold` | number | **proposed, not implemented** | Who owns it: LogiShield or Runtime Verification? |
| `rule_context.rule_count` | integer | **proposed**, derivable from input | counts only |
| `rule_context.rule_ids` | array of string | **proposed**, derivable from input | policy ids as received |
| `rule_context.verification_status` | `"SAT"`\|`"UNSAT"`\|`"absent"` | **proposed**, depends on §3 | echoes Formal Verification's input; LogiShield does **not** verify |
| `rule_aligned_signal` | undefined | **proposed placeholder, not implemented** | Meaning undefined — see Q3.3. Must not be fabricated |
| `trace.stages[]` | array of {stage, elapsed_ms, success} | **existing internally** (`tracing.py`), not delivered | counts and timings only, never raw values |
| `degraded` / `error` | object | **proposed** | failure signalling, §5.4 |

**Never included** (existing privacy policy): raw payload values, unmasked headers, token ids
or tokens.

### 5.2 PROPOSED EXAMPLE (illustrative; invented values; not a real prediction)

```json
{
  "schema_version": "anomaly-signal-1",
  "event_id": "evt-example-0001",
  "score": 0.00,
  "model_version": "ngram-linear-v1-trained",
  "model_note": "example only; real scores come from the deployed model",
  "label": null,
  "threshold": null,
  "rule_context": {
    "rule_count": 1,
    "rule_ids": ["P1"],
    "verification_status": "SAT"
  },
  "rule_aligned_signal": null,
  "trace": { "stages": [ { "stage": "model_inference", "elapsed_ms": 0.0, "success": true } ] }
}
```

`null` marks fields that are **not implemented today**; the numeric values are placeholders,
not measurements.

### 5.3 Label and threshold (open)

The current model score is not calibrated and was fitted on two synthetic fixtures; any
threshold chosen today would be arbitrary. Options: (a) LogiShield delivers `score` only and
Runtime Verification owns the threshold; (b) LogiShield delivers a label after a threshold is fixed from a
validated evaluation (not available yet); (c) neither, with Runtime Verification using the score as a soft
signal. The legacy `shared_contracts.models.Prediction` carries `label`, `threshold` and
`model_version`; whether Runtime Verification intends to reuse it must be confirmed.

### 5.4 Failure and timeout behaviour (proposed defaults, to be confirmed)

| Situation | Today (LogiShield) | Proposal |
|---|---|---|
| Input contract violation | HTTP 422 `input_contract_violation`, generic text | Return a structured error; Runtime Verification treats the event as "not scored" |
| Model/vocabulary not ready | HTTP 503 `model_not_ready` | Same; Runtime Verification must define fail-open vs fail-closed |
| Payload cannot fit `max_length` after framing + rules | HTTP 422 `pipeline_input_rejected` | Runtime Verification decides whether this is itself a risk signal |
| Timeouts | Not defined | Runtime Verification states its timeout budget; the measured in-process mean is 0.34 ms on one machine, but HTTP latency on loopback was a mean of ~2.7 ms (see README §8) |

LogiShield can only supply a score or an explicit "not scored" status; **the Allow/Block
decision belongs to Runtime Verification**.

### 5.5 Transport and versioning (open)

LogiShield currently runs as a standalone local service (`127.0.0.1:8004`), is not in
`docker-compose.yml`, and has no versioned handoff endpoint. Options: HTTP endpoint (e.g. a
future `POST /v1/signal`), an in-process interface, or a message. Any shared Python type in
`shared_contracts` requires approval from all four members (`README.md` branching rules).

## 6. Decisions required

### 6.1 From Formal Verification

| # | Question |
|---|---|
| Q2.1 | Which structure is the verified-rule output LogiShield should consume: `VerificationReport`, the structure in §3.2, or another? |
| Q2.2 | **Event correlation:** who attaches `event_id`, and where in the pipeline? |
| Q2.3 | **Applicability:** how is an HTTP request (`method`, `path`, headers, payload, authenticated role) mapped to `subject`/`action`/`resource`, and which policies are therefore delivered with an event? |
| Q2.4 | **Verdicts:** are UNSAT reports delivered, or only SAT? Is the HTTP 400 behaviour intended for downstream consumers? |
| Q2.5 | **Policy identity:** is `policy_id` stable across requests and versions, and is the policy-id/effect set closed (suitable for a trained vocabulary)? |
| Q2.6 | Which fields should LogiShield tokenize: `fol_expressions`, effects, `status`, `conflicting_policies`? |
| Q2.7 | Which verification path is intended for the verified output (`process_and_verify` or `ConflictDetector`)? |
| Q2.8 | Required types, nullability, `schema_version`, and delivery mechanism (HTTP call, shared module, message) |
| Q2.9 | Failure/timeout behaviour of your service, and its latency budget |

### 6.2 From Runtime Verification

| # | Question |
|---|---|
| Q3.1 | Where is your implementation, and which branch should be inspected? |
| Q3.2 | Required input fields, types, nullability, `schema_version`, and transport |
| Q3.3 | What do "rule-aligned signal" and "next component input" (proposal §3.6) mean concretely for your stage? |
| Q3.4 | Is `score` sufficient, or are `label`/`threshold`, token-level attribution or trace needed? Who owns the threshold? |
| Q3.5 | Synchronous per request or asynchronous? Timeout budget? |
| Q3.6 | On LogiShield failure (422/503/timeout): fail-open, fail-closed, or degraded mode? |
| Q3.7 | **KPI 5:** what makes LogiShield's output "usable" for you — schema validity only, or measurable influence on your verdicts? What acceptance test would you run? |

### 6.3 Within the group (shared contracts)

| # | Question |
|---|---|
| Q4.1 | Will shared types live in `shared_contracts` (needs all four members' approval) or stay per-interface? |
| Q4.2 | Which member owns adding LogiShield to `docker-compose.yml`, and how is the NumPy dependency handled? |

## 7. KPI 5 implications

KPI 5 ("downstream output usability ≥ 90%"; proposal method: *check compatibility with the
next verification component*) can be measured **only** after Runtime Verification supplies a real contract
or consumer. Until then this proposal does not produce a KPI 5 value, and any compatibility
check against §5 would be a self-assessment against a contract LogiShield wrote. Once Q3.2–Q3.7
are answered, a protocol would be pre-registered in the same way as KPI 1–4 (a new, versioned
protocol file; frozen KPI 1–4 evidence untouched).

## 8. Implementation plan

| # | Task | Owner | Depends on | Notes |
|---|---|---|---|---|
| 1 | Send §6 questions; record answers in this document | LogiShield | – | Nothing to implement yet |
| 2 | Agree event correlation, applicability and UNSAT handling | **Members 2 + 4** (+3 for Q2.4) | Q2.1–Q2.4 | Decision, not code |
| 3 | Implement adapter: agreed Formal Verification structure → `VerifiedRuleInfo` in `tokenization_ai/adapters.py`, with tests against Formal Verification's real output | LogiShield | Task 2 | Entirely inside LogiShield once agreed; no shared-contract change unless transport requires it |
| 4 | Agree Runtime Verification input contract, label/threshold ownership, failure handling | **Members 3 + 4** | Q3.1–Q3.6 | Decision, not code |
| 5 | Implement `to_runtime_verification_output` and a versioned handoff endpoint | LogiShield | Task 4 | Inside LogiShield, plus Runtime Verification's consumer |
| 6 | Genuine integration test (real consumer, real or recorded Formal Verification output) | **Members 2, 3, 4** | Tasks 3, 5 | Separate from the in-process Ingestion test |
| 7 | Pre-register and run KPI 5 protocol | LogiShield (+3) | Task 6 | Separate protocol version; do not reuse KPI 1–4 files |
| 8 | Add LogiShield to docker-compose / shared requirements | Members 1–4 | Q4.1–Q4.2 | Needs shared-contract approval |
| 9 | Re-run KPI 1–4 under a new protocol version if input structure changes | LogiShield | Task 3 | Frozen v1.0.0 evidence stays unmodified |

**Independent LogiShield work that does not need other members** (not part of this proposal):
masking policy review, evaluation-set strengthening under a new protocol version, SR-BH
evaluation.

## 9. Non-goals and caveats

- This document does not define, assume or imply Formal Verification's or Runtime Verification's behaviour beyond
  what is read from their code, and it does not claim they have reviewed it.
- No formal-verification result, prediction measurement or KPI 5 value appears here; all
  numbers other than those cited from existing evidence (`README.md` §8) are placeholders.
- The synthetic proof-of-concept model's score is not a detection result.
