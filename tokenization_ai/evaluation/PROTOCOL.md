# Member 4 KPI Evaluation Protocol — KPIs 1–3

**Protocol version:** 1.0.0  
**Scope:** Hybrid Structure-Aware Tokenization & Lightweight AI Prediction (Member 4 only).  
**Covers:** KPI 1 Input Alignment, KPI 2 Semantic/Symbolic Token Mapping, KPI 3 Aligned Tensor Correctness.  
**Out of scope:** KPI 4 (latency), KPI 5 (downstream usability), any detection-quality metric.

> The proposal's targets (≥90%, ≥90%, ≥95%) are **targets, not results**. Everything
> here is a *synthetic specification-coverage* evaluation. A pass rate is the share of
> curated scenarios the implementation handles as specified. It is **not** real-world
> accuracy, not precision/recall/F1, not formal-rule verification correctness, and not
> evidence of Member 2 or Member 3 integration.

## 0. Sources and their limits

| Source | Used for |
|---|---|
| Member 4 proposal §3.2–3.5, Table 2 | KPI names, targets, one-line measurement methods |
| `tokenizer.py` / `masking.py` / `normalization.py` docstrings | The *design specification* of the grammar (section 3 below) |

The proposal fixes **no** sample size, split or dataset (§3.2, Appendix 1). This protocol
therefore chooses them and says so. Table 2's methods are one line each; the definitions
below are this protocol's operationalisation, not text from the proposal.

## 1. Pre-registration rules

1. This file and all fixtures under `fixtures/` are frozen **before** the first scored run.
   Their SHA-256 hashes are written to a manifest (`--write-manifest`) before scoring and
   repeated in every report.
2. After a scored run, fixtures and expectations are not edited to improve a score. A change
   requires a new protocol version (1.1.0, …), a changelog entry here, and the previous
   report kept.
3. Failed cases are always preserved in the report.
4. Fixture QA before scoring is allowed, but only against the *specification* (the grammar
   acceptor in `spec.py` and the reference encoder cross-check), never against production
   output.

## 2. Evaluation data

| Group | Role | Expected answers come from | Cases |
|---|---|---|---|
| `curated_eval` | KPI 1–3 scoring | Hand-written token lists, masked paths and fused records | 17 valid |
| `generated_eval` | KPI 1–3 scoring | `spec.reference_encode` (independent encoder written from section 3) from a fixed seed | 12 seeded + 2 designed |
| `invalid_cases` | KPI 1 rejection (reported separately) | Contract rule each case violates | 5 |
| `dev_curated` + `dev_generated` | **Fitting the vocabulary only; never scored** | Hand-written / reference | 6 + 12 |

**Size justification.** No statistical power claim is made. 17 curated cases are the
smallest set that covers every grammar production, every masking rule class, every
normalization rule and every identified attacker-lookalike case once or twice. The 12
seeded cases give variable length (≈40 to >300 tokens) so that padding, exact fit and
truncation are all exercised. The 2 designed cases force rule-segment preservation under
truncation and a capacity rejection. The result is reported with a Wilson 95% interval to
make small-sample uncertainty explicit. No case was added to reach a target.

**Seeds.** Dev generation seed `1101`, eval generation seed `2202` (`generate.py`).
Frozen JSON is committed; a test regenerates from the seed and requires identical output.

**Vocabulary / unknown tokens.** The vocabulary is fitted once, on the dev groups only
(through the production tokenizer), `min_frequency=1`, no size cap. Evaluation cases are
never used for fitting, so literals first seen at evaluation map to `[UNK]`. That unknown-token
rate is **reported as a separate measurement and is not counted as a failure**: an unknown
literal mapped to `[UNK]` is the specified behaviour. The *independent* expectation of
which literals are unknown is derived from the dev groups' hand-written/reference
expected sequences, not from the vocabulary under test.

**Tensor configuration.** `max_length = 128` (the demo service's value). One configuration only.

## 3. Specification of the representation (oracle basis)

Written here so the oracle does not depend on running the tokenizer.

**Pipeline order:** adapters → fusion → masking (with provenance) → normalization → tokenization → tensorization.

**Normalization (applied before tokenization):** method upper-cased and stripped; path
stripped, single leading `/`, repeated `/` collapsed, trailing `/` removed (except root);
header names lower-cased and stripped (values untouched); payload object keys sorted;
string values stripped; numbers/bools/null unchanged.

**Masking:** a value is replaced by a placeholder when (a) its key is a sensitive key →
`[MASKED_SECRET]` (password, token, api_key, secret, …) or PII key → `[MASKED_PII]` (email,
phone, card, ssn, …); (b) a header is an authentication header (authorization, cookie,
x-api-key, …) → `[MASKED_AUTH]`; (c) a string value is email- or payment-card-shaped →
`[MASKED_PII]`, IP-shaped → `[MASKED_IP]`; (d) a sensitive query parameter inside the path →
its value becomes the URL-encoded placeholder inside the path string. Masked locations are
recorded as *paths* (`["headers","authorization"]`, `["payload","users",0,"email"]`, `["path"]`,
`["client_ip"]`). Verified-rule information is never masked. Unknown field names and
non-recognized values are **not** masked (a documented limitation, kept as a test).

**Token grammar** (`C`=control, `A`=API literal, `R`=rule literal, `M`=masked value):

```
SEQ     := C[SEQ_START] C[API:METHOD] A(method)
           C[API:PATH_START] (C[API:PATH_SEG] A(segment))* C[API:PATH_END]
           C[API:HEADERS_START] header* C[API:HEADERS_END]
           C[API:PAYLOAD_START] OBJ(API) C[API:PAYLOAD_END]
           C[RULES_START] rule* C[RULES_END] C[SEQ_END]
header  := C[API:HEADER_NAME] A(name)
           ( C[API:HEADER_VALUE] A(value) | C[API:HEADER_VALUE_MASKED] M(placeholder) )   ; names sorted
rule    := C[RULE_START] C[RULE:ID] R(id) C[RULE:ATTRS_START] OBJ(RULE) C[RULE:ATTRS_END] C[RULE_END]
OBJ(ns) := C[ns:OBJ_START] (C[ns:KEY] lit(key) VALUE(ns))* C[ns:OBJ_END]                ; keys sorted
LIST(ns):= C[ns:LIST_START] VALUE(ns)* C[ns:LIST_END]                                   ; order kept
VALUE   := OBJ | LIST | C[ns:VAL_STR] lit | C[API:VAL_MASKED_STR] M(placeholder)         ; masked only in API ns
         | C[ns:VAL_INT] lit | C[ns:VAL_FLOAT] lit | C[ns:VAL_BOOL] lit("true"|"false") | C[ns:VAL_NULL]
lit     := A in the API namespace, R in the rule namespace
```

A genuinely masked location gets role `M`; text that merely *looks like* a placeholder
under an unmasked key stays `A`. A masked query parameter stays an `A` literal (it is part of
the last path segment). Not tokenized: `client_ip`, `content_type`, `threat_score`.

**Tensorization:** output length exactly `max_length`; attention mask is `1`×n then `0`×(L−n);
padding id = pad id; non-truncated sequences encode token-for-token (unknown literal → `[UNK]`;
control and masked tokens are never `[UNK]`). When n_tokens > L only payload *content* is
shrunk, by dropping whole fields / list elements; framing and the entire rule segment are
kept; the result stays structurally balanced. If framing + rule segment cannot fit even with
an empty payload, a `ValueError` is the specified outcome (a *capacity rejection*).

## 4. KPI definitions

### KPI 1 — Input alignment (target ≥ 90%)

*Table 2 method: "check combined API context + verified rule information consistency".*

- **Unit:** one valid evaluation event (raw Member-1-style input + rule information).
- **Numerator:** events for which `adapters → fusion` yields a fused record **exactly equal**
  to the hand-written expected record: event id, method, path, headers, payload,
  content type, client ip, threat score, and the ordered list of rules (id + attributes).
  Fusion/adapters must *preserve*, not normalize (e.g. a lower-case method stays lower-case here).
- **Denominator:** all valid events (`curated_eval` + `generated_eval`). **Invalid events are not in the denominator.**
- **Reported separately — invalid-event rejection:** for 5 events, each violating a *different*
  contract rule, the expected outcome is an `InputContractError`. Reported as `k/5`, never
  merged into the alignment percentage.
- **Check type:** independent expected-output.
- **Threat to validity:** fusion is close to identity; this KPI mostly tests the adapters and
  field preservation. A high value is expected and is weak evidence of anything beyond that.
  Rules are mock; consistency with *genuinely verified* rules is untested.

### KPI 2 — Semantic and symbolic token mapping (target ≥ 90%)

*Table 2 method: "compare tokens against expected structure-aware outputs".*

"Semantic" is operationalised **narrowly**: correct *structural position, token text and
authoritative role* per the section 3 grammar. The tokenizer does no language understanding
and none is evaluated.

- **Unit:** one valid event, taken through masking, normalization and tokenization.
- **Primary numerator:** events whose token sequence **and** role sequence equal the
  hand-written expected sequence exactly. **Denominator:** all valid events.
- **Token-level match:** over all expected positions (a length mismatch counts the missing
  positions as unmatched), the share where token text **and** role both equal expected.
- **Role correctness:** among positions whose text matched, the share with the expected role.
- **Masking provenance (separate):** per event, equality of the expected and actual set of
  masked paths; plus micro precision/recall over (event, path) pairs.
- **Implementation-invariant check (reported apart from the KPI):** the *actual* sequence is
  accepted by the independent grammar acceptor in `spec.py`.
- **Threat to validity:** the oracle was written by the same team that wrote the tokenizer,
  from the same design; it detects regressions and unintended behaviour, not a wrong *design*.
  The generated group's expected output comes from a second implementation by the same author
  (correlated errors possible); its reference encoder is cross-checked against the hand-written
  fixtures before scoring.

### KPI 3 — Aligned tensor correctness (target ≥ 95%)

*Table 2 method: "validate tensor shapes and feature mapping".*

- **Unit:** one tensor produced from an **expected** (independent) token sequence by the
  production vocabulary and tensorizer, so tensorizer defects are not confounded with tokenizer
  defects. (A secondary pipeline-consistency count compares it with the tensor from the
  production token sequence.)
- **Numerator:** tensors passing **every applicable** check below. **Denominator:** all valid
  events whose expected outcome is a tensor.
- **Checks.** *invariant* = implementation self-consistency; *independent* = compared with the
  specification-derived expectation.
  | Check | Type | Applies to |
  |---|---|---|
  | `length` ids and mask are exactly `max_length` | invariant | all |
  | `mask_binary_prefix` mask is 1…1 0…0 | invariant | all |
  | `ids_in_range` every id in [0, vocab size) | invariant | all |
  | `padding_ids` positions with mask 0 hold the pad id | invariant | all |
  | `event_id` preserved | invariant | all |
  | `deterministic` second call identical | invariant | all |
  | `valid_count` n = expected length (or ≤ L when truncated) | independent | all |
  | `token_id_consistency` decode(ids[:n]) = expected tokens with unseen literals → `[UNK]` | independent | non-truncated |
  | `reserved_never_unk` control/masked tokens never `[UNK]` | independent | non-truncated |
  | `framing_preserved` tokens outside the payload value unchanged | independent | truncated |
  | `rule_segment_preserved` whole rule segment unchanged | independent | truncated |
  | `payload_balanced` object/list tags balanced | independent | truncated |
  | `payload_subsequence` kept payload tokens are an in-order subsequence of the expected payload | independent | truncated |
- **Reported separately:** unknown-literal rate (observed vs. independently expected), the
  count of capacity rejections (expected `ValueError`, correct if raised), and the secondary
  pipeline-consistency count.
- **Threat to validity:** the vocabulary comes from a tiny synthetic dev corpus; unknown rate is
  therefore a property of this corpus, not of real traffic. Balanced-tag counting cannot
  distinguish a literal that textually equals a tag (no truncated case contains one).

## 5. Reporting rules

- Every KPI reports numerator, denominator, percentage, Wilson 95% interval, target, and
  `target_met`, **plus** the group breakdown (curated vs. generated vs. designed).
- `target_met: true` only means "the synthetic scenario set was handled as specified".
- A KPI whose ground truth is insufficient is reported `not_measurable`, not given a number.
- No report states or implies detection accuracy, formal verification, or Member 2/3 integration.

## 6. Known limitations

1. Synthetic, hand-picked, small N; coverage, not sampling.
2. Same-author oracle; independent in *process* (hand-written, grammar acceptor), not in *authorship*.
3. Mock rules only. Member 2's real rule schema is unavailable.
4. SR-BH not used (no local dataset; none downloaded).
5. One tensor configuration and one tiny dev vocabulary.
6. The anomaly model is not exercised by KPIs 1–3 and is unchanged.

## Changelog

- 1.0.0 — initial protocol, written before the first scored run.
