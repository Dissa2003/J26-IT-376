# Held-out evaluation protocol (Member 4)

**Protocol version:** `heldout-eval-1.2.0-draft`  **Status: DRAFT, not frozen.**
**Template version:** `tmpl-1`  **Generator (implemented):** `synthgen-3`  **Catalog (implemented):** `synthetic-catalog-3`
(G1 and G2, sections 12.2 and 12.2.1; the earlier `synthgen-2` / `synthetic-catalog-2` pilot is exploratory and burned, section 12.4)
**Code (this folder):** `splits.py`, `synthetic_catalog.py`, `generator.py`, `audit.py`, `policy_sim.py`, `policy_diagnostics.py`  **Simulated rule catalog:** `simpol-1`
**Tests:** `tests/tokenization_ai/test_heldout_splits.py`, `test_synthetic_catalog.py`,
`test_synthetic_generator.py`, `test_heldout_audit.py`, `test_policy_sim.py`

## Change history

| Version | Change |
|---|---|
| `1.0.0-draft` | First draft: SR-BH as the candidate dataset; leakage-safe group splitting (`splits.py`). |
| `1.1.0-draft` | **Synthetic API requests are the primary evaluation source.** SR-BH is deferred and not downloaded. Added the scenario catalog, deterministic generator and non-model audit, and a pilot stage. The comparison arms, primary metric, splitting rules and leakage controls are unchanged. The `1.0.0-draft` text remains in git history. |
| `1.1.0-draft` (D3a-R refinement, same draft version) | **Dataset quality refinement after reviewing the first pilot's six descriptive warnings** (generator `synthgen-1` -> `synthgen-2`, catalog `-1` -> `-2`). See section 2.6 for what was changed and why, section 2.7 for the intentional associations that remain, and section 4.1 for the pilot record. The audit gained a template-supported exclusivity measure and an independent content-type semantics check. Splitting code is unchanged. |
| `1.1.0-draft` (D3b, same draft version) | **Simulated policy rule catalog `simpol-1`** (`policy_sim.py`) and its descriptive diagnostics (`policy_diagnostics.py`) for Arm B. Rules are deterministic functions of observable request properties, carry explicit simulated provenance and no verification status, and are not connected to any API. See section 3.1. |
| `1.2.0-draft` (D3c-1) | **Scientific design decisions recorded; documentation only.** Approves generator `synthgen-3` with two planned corrections (G1, G2), keeps `simpol-1` unchanged, marks the seed-5101 pilot as exploratory/burned, fixes the seed plan, the full-study scale, the equal-`max_length` rule, the request-to-context mapping `request_to_context-v1`, the Arm C definition, the primary-metric comparison plan and the pre-run manifest plan. See section 12 and the chronological record in section 13. **No code, generator, split, catalog or dataset changed in this step, and nothing is frozen.** |
| `1.2.0-draft` (D3c-2) | **Generator `synthgen-3` / catalog `synthetic-catalog-3` implemented (G1 and G2 only).** Normal `/files/...` paths (`crud_read`, `hn_dotted_paths`) and comparable DELETE structures in both classes (`crud_write`; `traversal_like`, `sqli_like`, `cmdi_like`, `markup_like`). No label rule, weight, pool, split code, `simpol-1`, `policy_sim.py`, `policy_diagnostics.py` or audit threshold changed. The default dataset seed is now the development seed 6101. Design note, development-seed results and remaining limitations are in section 12.2.1. The confirmatory dataset is not generated and nothing is frozen. |
| `1.2.0-draft` (D4-0) | **Descriptive structural-coverage diagnostics added to `audit.py`; audit protocol label corrected.** `structural_coverage` is a new, separate field of the audit report (section 12.2.2). Existing audit checks, thresholds, errors, warnings, `construction_checks_passed`, fingerprints and split logic are unchanged. The report's `protocol` label now reads `heldout-eval-1.2.0-draft` (it read `1.1.0-draft`); `splits.PROTOCOL_VERSION` is deliberately untouched. No generator, catalog, `simpol-1` or split change; nothing is frozen. |
| `1.2.0-draft` (D4-1) | **`request_to_context-v1` implemented** (`request_context.py`, section 12.9.1). Pure mapping of a synthetic request to the pipeline's input context; no production module, generator, catalog, audit, split or rule change; no model run; nothing is frozen. |
| `1.2.0-draft` (D4-2) | **`budget-rule-1` implemented** (`token_budget.py`, section 12.8.1): the training-only shared `max_length` selection and a post-selection exposure report. Tested on hand-made arrays and development-seed requests only; no model run, no confirmatory data; no production, generator, adapter, audit or split change; nothing is frozen. |
| `1.2.0-draft` (D4-3) | **Arm C encoder `armc-char3-v1` implemented** (`arm_c.py`, section 12.10.1): deterministic serialization of a masked and normalized request into hashed character 3-grams as a `ModelInput` for the unchanged `LightweightAnomalyModel`. Encoding only: no model is fitted or scored, no Arm C builder or runner exists, no confirmatory data; no production, adapter, budget, generator, audit or split change; nothing is frozen. |
| `1.2.0-draft` (D4-4) | **Paired A/B/C arm builders implemented** (`arms.py`, `arms-1`, section 12.12.1): the three representations of the same requests, one train-only shared vocabulary, the shared `budget-rule-1` budget with fail-closed handling of unrepresentable records, a fresh-model factory and manifest provenance. Preparation only: no model is fitted, predicted with or scored, no metric, no runner, no confirmatory data; no production, adapter, encoder, budget, generator, policy, audit or split change; nothing is frozen. |
| `1.2.0-draft` (D4-5) | **Detection metrics and paired group bootstrap implemented** (`detection_metrics.py`, `detection-metrics-1`, section 12.11.1): step-wise Average Precision, ROC-AUC, threshold metrics, validation-only F1 threshold selection and a paired normalized-template-group bootstrap of AP(B) - AP(A). Tested on hand-made arrays only: no model is fitted or scored, no real detection metric exists, no development or confirmatory scoring, no success margin; no arm, encoder, generator, policy, audit, split or production change; nothing is frozen. |
| `1.2.0-draft` (D4-6C1) | **Documentation only: optimization and score-selection decisions recorded** (section 12.11.2): arm-relative learning rates `lr = c / L` with `c` in {0.1, 0.3, 0.8}, the 27-candidate shared grid, validation-AP selection with a fixed tie-break order, raw-logit ranking scores for the held-out evaluation, and numerical-provenance requirements. Informed by the label-free D4-6B diagnostic on development seed 6101. No code, model, dataset, manifest or confirmatory data was created or changed; several items remain open; nothing is frozen. |

This protocol is **separate from** the archived KPI 1-4 evaluation (`../evaluation/`, its protocols,
fixtures, manifests and reports). Nothing here re-runs, replaces or reinterprets that evidence, which
stays tied to its original protocol. Nothing in this folder changes the production pipeline, the
prediction API or the dashboard.

> **Scope of this step:** the protocol text, the leakage-safe splitting code, the synthetic scenario
> catalog, the deterministic generator (`synthgen-2`), the non-model audit and the simulated rule catalog
> exist. There is **no model training, metrics runner or A/B/C runner yet, and no accuracy, PR-AUC or other
> detection-performance figure exists.** `synthgen-3` (G1 and G2) is implemented and has been exercised only on
> the development seed 6101 (section 12.2.1); the Arm C code, the runner, the budget-rule code and the
> manifest tooling are **not implemented**. The draft becomes a frozen protocol only after the generator revision,
> runner and arm code, hashes, the final decision rule and the pre-run manifest exist and are approved
> (sections 11 and 12.12); until then every statement here is a plan, not a frozen commitment.
>
> **Reading order:** sections 1-10 are the working text as revised through D3b. Section 12 records the D3c-1 decisions
> and **takes precedence over any earlier statement it contradicts**; where an earlier section is superseded it carries
> a pointer to section 12.

## 1. Purpose

Evaluate Member 4's lightweight predictor on **held-out synthetic data** and test whether adding
structure-aware rule tokens helps compared with request-only input, with leakage controls strong enough
that the result is defensible *as a synthetic held-out result*. The proposal permits synthetic or
simulated API requests, specifies no dataset size, split or classifier, and sets no detection target
(proposal section 3.2, 3.3, Appendix 1); this protocol defines them and **introduces no
detection-accuracy success target.**

## 2. Data

### 2.1 Primary source: synthetic API requests

Requests are produced by `generator.py` from the documented families in `synthetic_catalog.py`. They are
fabricated and harmless: no live traffic, no real users, credentials or private data, and no external
datasets. The generator is deterministic for a recorded seed (default dataset seed `5101`, fixed in
advance) and is versioned. Only single-request scenarios are in scope; stateful or multi-request
scenarios are excluded in this phase.

Scenario families (details and labelling rules are in the catalog):

- **Normal:** routine CRUD (including nested resource paths), search and filtering, normal authentication,
  JSON create/update (including nested bodies of 2-7 fields), legitimate administration, health and status
  calls. **Hard negatives:** ordinary text that merely contains SQL-ish, markup-ish or shell-ish words;
  ordinary dotted file paths in nested folders; legitimately large bodies; **legitimate text/plain and XML
  bodies with a matching declared type**; **administrators setting privileged-looking fields on admin-only
  paths**; legitimately nested configuration bodies (3-5 levels).
- **Suspicious:** SQL-injection-like text, path-traversal-like paths, command-injection-like suffixes,
  markup-style text, mass assignment, unauthorized role/action combinations, and suspicious payload
  structure. **Hard positives:** obfuscated or encoded variants, injection text inside large bodies and
  unauthenticated login-style endpoints, keyword-free mass assignment, and low-privilege roles that look
  plausible. Mass assignment is defined by context: privileged-looking fields sent by a **non-admin role to
  a non-admin endpoint**.
- **Mixed:** one shared template whose *values* (or one header) decide the label: role/action mismatch,
  content-type mismatch, and an endpoint whose text slot holds either ordinary or suspicious text. These
  produce deliberate **mixed-label templates**, which the splitter keeps together. A **content-type
  mismatch is incompatibility between the declared type and the actual body format** (JSON, XML, form or
  plain text), never the declared value alone: every declared type is legitimate for its own format.

Suspicious payloads are short, generic, illustrative patterns; none is a working exploit.

### 2.2 How labels are defined, and what that does and does not mean

A request is labelled suspicious **only** by the documented construction of its scenario family (the
rule is written on each family). Labels are metadata (`SplitMetadata`, `GenerationMetadata`); they are
never request fields and never model features.

**Same-author limitation.** The same author writes the families, the payloads and the labels. Therefore
a score on this data measures whether a model learns the *author's* distinction on templates it has not
seen. It does not measure detection of real attackers, real traffic or real attack techniques, and
some association between request content and label is expected by construction. Results must be
reported as **synthetic held-out performance and feasibility**.

### 2.3 SR-BH 2020 (deferred)

SR-BH 2020 (Harvard Dataverse, DOI `10.7910/DVN/OGOIXX`) is **deferred, not downloaded, and not locally
verified.** Nothing in this protocol relies on it. If it is considered later, its use is subject to
source-file verification and applicable supervisor and institutional approval, and it would be added by
a new protocol version.

### 2.4 Template-level versus record-level prevalence

The splitter groups by normalized template (`tmpl-1`), and variants of a template differ only in values,
so **the number of distinct templates, not the number of records, is the effective sample size.** Every
report must state both:

- record-level prevalence: share of records labelled suspicious;
- template-level composition: all-normal, all-suspicious and mixed templates, and the share of
  templates containing at least one suspicious record (the design target is roughly 30-35%).

The requested template count is **never silently forced**: if normalization collapses candidates, the
actual count is reported with the shortfall.

### 2.5 Construction audit versus detection-quality evaluation

`audit.py` is a **construction audit**: it checks that the generator followed its own rules (no class words in
request fields, labels and ids independent of each other, groups and exact duplicates isolated, labels
consistent with each family's documented rule) and describes how request properties associate with labels.
It is **not** a detection-quality evaluation: it trains no model and produces no accuracy, PR-AUC, ROC-AUC,
precision, recall or F1. A clean audit does not make a later model score meaningful by itself.

### 2.6 Refinement after the first pilot (D3a-R): what changed and why

The first pilot (`synthgen-1`) raised six descriptive warnings. They were investigated before this milestone:

| Warning | Finding | Action |
|---|---|---|
| Body-shape association with class (34.9% of records) | **Largely a measurement artifact.** One template contributes 4-8 near-identical records, so "a shape seen in >= 5 records in one class" mostly flagged a single template's own variants. | Audit now also reports exclusivity supported by >= 3 *distinct templates* (`class_exclusive_template_share`) and warns on that; the legacy record-level figure is still reported. On the first pilot the template-supported figure for body shape was 0.0%. |
| Class-exclusive content types (`text/plain`, `application/xml` only in suspicious) | **Accidental construction artifact.** Legitimate plain-text and XML requests did not exist, so the value alone marked the class. | Added legitimate plain-text and XML requests (matching declared type) and redefined the mismatch family by incompatibility with the actual body; at pilot scale every declared type occurs in both classes (a very small dataset can miss a type in one class by chance). A semantics check verifies this independently of the generator. |
| Path depth (7.9%) | **Mixed.** Depths 5-7 came only from traversal paths (partly the real signal), depth 4 only from dotted-file paths. | Added deeper legitimate resource paths, deeper dotted-file folders, and traversal tails of different lengths so depths overlap. Traversal paths still contain `..` by design. |
| Body size (5.5%) | **Accidental.** Large bodies existed only in the normal hard-negative family. | Injection and mass-assignment scenarios now also appear in large bodies. |
| Top-level field count (8.0%) | **Accidental.** 5-7 fields occurred only in mass assignment; 8+ only in large normal bodies. | Normal bodies now span 2-7 fields; mass assignment spans 2-7 base fields; large suspicious bodies exist. |
| Normal-only sub-resources (3 values, 67 records) | **Accidental sampling imbalance.** | Resources and sub-resources are now drawn round-robin per family group (normal versus suspicious/mixed) so every value is used by both. |

A further accidental marker found and removed during the refinement: suspicious text endpoints had fewer
query parameters than normal search endpoints (query-parameter count became class-exclusive); they now carry
the same optional paging and filter parameters.

Where a property **is** the documented anomaly it is kept (section 2.7). No suspicious scenario was relabelled
benign to balance a statistic, and the held-out split was not touched.

### 2.7 Intentional associations that remain (by design)

These are security-relevant properties of the scenarios, so some association with the label is expected and is
not a defect; they limit what a score can mean:

- **Injection-like, markup-like and command-like strings** occur only in suspicious scenarios (and, as ordinary
  words, in hard negatives). Detecting them is the point of the study, so a model that learns them has learned
  the author's definition, not necessarily real attacks.
- **Path traversal** paths contain `..` segments or encoded separators.
- **Nesting deeper than the legitimate depth** (6+ levels) and **a list or object in a field whose name implies
  a number** are the structural anomalies; legitimate nested bodies are bounded at 5 levels.
- **Unauthorized role/action combinations and non-admin mass assignment** are defined by header and endpoint
  context; the same privileged-looking field names appear in legitimate administrator updates.
- **Content-type mismatch** is defined by declared type versus actual body, so the mismatching examples are
  rarer than ordinary requests (few per type at pilot scale).
- **Small residual imbalances** remain below the attention-flag level (for example, at the pilot a few exact path
  depths and header-name sets occur in only one class, each for a few percent of records). They are reported, not
  hidden, and were not engineered away.
- **Body shape** remains the strongest descriptive association (high normalized mutual information), partly
  because most shapes are specific to one template; this normalized measure is biased upward for many distinct
  values, so use it with the shuffled reference and the cross-split overlap statistics, not alone.

## 3. Comparisons (identical examples, identical splits)

| Arm | Input | Notes |
|---|---|---|
| **A** | Request-only features | The existing pipeline with an empty rule list. This path is supported today: `service.analyze` accepts `rules=[]` and emits `[RULES_START] [RULES_END]` framing (checked by a test). |
| **B** | Request + **simulated** rule tokens | Rules come from a deterministic function of request-observable fields only (for example method, path prefix, content type, header presence), from a catalog written and frozen **before** labels are examined, with neutral policy ids. Labels, scenario families and detector output never enter rule construction. **The catalog `simpol-1` exists (section 3.1); the runner that uses it is not implemented yet.** |
| **C** | Conventional HTTP preprocessing baseline | A standard, non-structure-aware preprocessing of the same requests (for example hashed character n-grams), same classifier and bucket sizes. **Not implemented yet. Superseded by section 12.10:** a comparable logistic-regression model family with a matched feature-budget policy, not necessarily the identical class; compatibility is checked before coding. |

All arms use the same model family and hyperparameter grid with an equal tuning budget. The vocabulary
is fitted on the training split only, separately per arm. A record that any arm rejects (for example its
framing exceeds `max_length`) is removed from **all** arms and reported, so comparisons stay paired.

**Simulated rules do not prove the value of genuine Member 2 verified rules.** A rule derived
deterministically from the request adds no information beyond the request; a difference between A and B
can only show a representation effect on a hand-built catalog, and "no difference" is an acceptable
outcome. No Member 2 integration exists, and Arm B must be labelled *synthetic or simulated rules*
wherever reported. The model scores hashed unigrams and bigrams over one linear token sequence, so it
cannot represent interactions between distant segments; this is a property of the model, not of rule
fusion in general.

Known confounds to control: rule tokens use part of the fixed token budget (report truncation rates per
arm; use the same `max_length` for all arms); request framing (method, path, headers) is mandatory in
the tensor, and a request whose framing alone exceeds `max_length` is rejected, not truncated.

### 3.1 Simulated policy rule catalog `simpol-1` (Arm B input)

**All rules are simulated.** They are not Member 2 output, not formally verified, and carry no verification status.
Nothing in `policy_sim.py` imports or calls Member 2, makes a network call, or connects to the API, dashboard or
pipeline.

- **Version / provenance:** `simpol-1`; catalog SHA-256
  `79273ba6dac5cfa453e258024c2a9e4510e041b68c8db8b79781c52b4de1ba79` (covers every id, attribute and applicability
  condition; pinned in `test_policy_sim.py`). Every result is a `SimulatedRuleSet` with `provenance="simulated"`,
  `verification_status="not_verified_simulated"`, the catalog version and hash.
- **Catalog:** 13 policies with neutral sequential ids `SIM-P01`...`SIM-P13` and three allowlisted attributes
  (`action`, `resource`, `effect`) from fixed vocabularies; `subject` and `interpretation` are deliberately omitted to
  keep each rule at 20 tokens in the existing tokenizer. Ids and attributes never name a label, attack, scenario or
  family (checked at import and in tests).
- **Allowed inputs (the only things that may decide applicability):** HTTP method; a coarse path category from the
  first path segment (admin, auth, files, documents, status, other resource); the *declared* content-type class
  (json, xml, form, plain, other, none) from the header value only; presence (not value) of an authorization header;
  presence of a body; presence of a query string.
- **Never used:** labels, scenario families, template ids, source ids, split assignments, record ids, timestamps,
  generated metadata, role-header values, body or query content, model predictions. `simulate_rules` accepts only an
  event id and `RequestFeatures`, so none of these can be passed in.
- **Assignment:** every policy whose conditions hold is assigned, in catalog order, with no duplicates and no
  selection by label. An empty list is valid and means only that no simulated policy applies.

| Id | Applies when | Action / resource / effect |
|---|---|---|
| SIM-P01 | admin path | manage / admin-area / Allow |
| SIM-P02 | other path, GET, authorization present | read / resources / Allow |
| SIM-P03 | other path, POST/PUT/PATCH, authorization present | write / resources / Allow |
| SIM-P04 | other path, DELETE | delete / resources / Allow |
| SIM-P05 | auth path, POST | authenticate / sessions / Allow |
| SIM-P06 | status path, GET | read / status / Allow |
| SIM-P07 | files path, authorization present | read / files / Allow |
| SIM-P08 | documents path | read / documents / Allow |
| SIM-P09 | has body, declared JSON | submit / json-payloads / Allow |
| SIM-P10 | has body, declared XML or plain text | submit / text-payloads / Allow |
| SIM-P11 | has body, declared form | submit / form-payloads / Allow |
| SIM-P12 | no authorization header, admin / files / documents / other path | access / protected-resources / Deny |
| SIM-P13 | GET with a query string | search / query-parameters / Allow |

**Rule-source label.** The existing API `RuleSource` has only `none`, `mock_fixture` and `verified`. There is no
`simulated` value, `verified` would be false, and `mock_fixture` is only the nearest honest label. The pipeline's
input type is named `VerifiedRuleInfo`; that name must not be read as a claim about these rules. The simulator is
therefore kept isolated from the API rather than misrepresenting the rules, and no API or dashboard file was changed.

**Why Arm B results will not prove the value of real verified rules.** Every rule is a deterministic function of the
request, so the rules add no independent security information. A difference between Arm A and Arm B can only show
whether the rule-token *representation* changes results on synthetic requests. It cannot show the effectiveness of
genuine, independently verified Member 2 rules, and fusion is neither expected nor assumed to help. The catalog was
authored by the same author who wrote the generator, with knowledge of its path conventions (but not of labels), and
was written before any rule-specific diagnostics were run (the pilot's labels had already been seen in the earlier
generator audits, but no condition in the catalog refers to a label); the diagnostics below describe it and were **not**
used to choose, add or drop a rule.

**Pilot diagnostics (`synthgen-2` pilot, 1,501 requests; descriptive only; no model).**

| Item | Result |
|---|---|
| Rules per request | mean 1.88, range 1-2 (177 requests with 1 rule, 1,324 with 2) |
| Requests with no rule | 0% |
| Duplicate policy ids within a request | 0 |
| Policy with zero coverage | `SIM-P12` (no anonymous request reaches a protected path in this synthetic data) |
| Policies applied to one class only (3+ templates) | `SIM-P04` (DELETE on resource paths: normal only), `SIM-P06` (status paths: normal only), `SIM-P07` (files paths: suspicious only) |
| Rule-set feature, normalized association with label | 0.115 (label-shuffled reference maximum 0.020); 11.0% of records have a rule set seen in 3+ templates and one class |
| Tokens per rule / per request | 20 per rule; +37.6 per request on average (max +40) |
| Request tokens, Arm A / Arm B | mean 50.1 / 87.8, max 102 / 142 |
| At `max_length` 128 | Arm A: 0 requests longer; Arm B: 18 longer (payload truncated), 0 rejected |
| At `max_length` 256 | none longer in either arm, none rejected |

Reading the diagnostics: the class-aligned policies mirror structure of the generator rather than new information.
The first-segment `files` path occurs only in traversal scenarios, DELETE on ordinary resource paths occurs only in
normal scenarios, and status paths are normal health calls, so each rule restates a path or method association the
request tokens already carry. These are attention flags, not proof of leakage absence or presence. They show a
property of the synthetic data (some structural combinations exist in only one class) that the earlier audit did not
surface because it does not summarize path category; any generator revision is a separate, reviewed step and must not
be driven by labels in the catalog. Because rule tokens are 20 per rule, Arm B uses more of a fixed token budget:
`max_length` must be identical across arms and chosen from training-split length statistics, or truncation rates must
be reported per arm (at 128 only Arm B is truncated for 1.2% of requests).

## 4. Pilot stage (before the full study)

1. **Pilot generation:** about **250 distinct normalized template groups** and about **1,500 requests**
   (default config), subject to what normalization actually allows.
2. **Non-model checks only** (`audit.py`): actual group and request counts, class proportions at record
   *and* template level, mixed-label templates, group-size distribution, duplicate counts, feasibility of
   the group-aware split, label/family-term scan, feature/metadata separation, shared pools, determinism,
   split isolation, and descriptive shortcut statistics (section 6).
3. **Review.** No model is trained, tuned or scored during the pilot, and no detection-performance
   number is computed.
4. **Full study** (about 2,000 templates and about 12,000 records, subject to feasibility; section 12.7) only
   after the pilot is reviewed, the generator is revised (`synthgen-3`, section 12.2), and this protocol
   is frozen with a pre-run manifest (section 10). A generator change after freezing requires a new
   generator and protocol version. The `synthgen-2` seed-5101 pilot is exploratory and burned (section 12.4).

### 4.1 Pilot record (non-model; counts and hashes only)

*Status of this pilot: exploratory and burned for confirmatory purposes (section 12.4). Its labels, audits, policy
diagnostics and held-out split were inspected while the design was being refined.*

| Item | `synthgen-1` pilot | `synthgen-2` pilot (after D3a-R) |
|---|---|---|
| Generator fingerprint | `dcd042946fc3571b...` | `d24fbec1688beb5e...` (full value pinned in the generator tests) |
| Templates (requested / actual) | 250 / 250 | 250 / 250 |
| Records | 1,487 | 1,501 |
| Suspicious records | 404 (27.2%) | 395 (26.3%) |
| Templates: normal / suspicious / mixed | 169 / 57 / 24 | 166 / 53 / 31 |
| Templates containing suspicious records | 32.4% | 33.6% |
| Exact duplicates | 0 | 0 |
| Audit errors / warnings | 0 / 5 (template-supported measure; 6 under the original record-level measure) | 0 / 0 |

The pilot is a feasibility and quality check only: the held-out splits hold about 45 templates each, too few for
stable estimates. The fingerprint change is expected: the generator and catalog versions changed. The audit
measure changed too (section 2.6), so before/after warning counts must be read with that in mind: under the
original record-level measure the body-shape flag would still fire, because that measure counts a template's own
variants.

## 5. Splitting (implemented in `splits.py`; unchanged from 1.0.0-draft)

**Primary: group-aware train / validation / test** (default 0.70 / 0.15 / 0.15).
**Secondary robustness check: time-forward**, where timestamps are meaningful. Synthetic timestamps carry
no real drift, so for synthetic data it is a mechanical check only; it is not expected to be informative
and is not a primary or required result.

- **Unit of grouping:** the normalized request template (`tmpl-1`): upper-cased method; decoded,
  lower-cased path with repeated slashes and trailing slash removed; digits, hex strings and UUIDs replaced
  by placeholders; query parameters sorted with values reduced to a class; body reduced to its structural
  shape. Requests that differ only in values - including suspicious-payload variants - share a template and
  therefore a split. Group ids are label-independent.
- **Whole groups only**, assigned by a seeded greedy fit to the requested ratios.
- **Exact duplicates** always share a split (verified independently); optionally dropped.
- **Label conflicts:** exact duplicates with different labels are an error by default (or excluded and
  reported). Templates holding both classes are reported as `mixed_label_templates` and kept together.
- **Source identity** is optional and never the only key (`merge_by_source`).
- **Infeasible data fails explicitly** (too few groups, a split missing a class, achieved sizes far from the
  ratios); `strict=False` reports the problems instead. Isolation is never relaxed.
- **Reproducibility:** primary split seed `4401`; robustness seeds `4402, 4403, 4404, 4405`; arbitrary
  values fixed before any data existed and not tuned. These split seeds are unchanged by D3c-1; the confirmatory
  study applies the primary split seed to a different, newly generated dataset (section 12.5 for dataset seeds).
  The group-aware train/validation/test split and the sealed-test rule are restated in section 12.6.

## 6. Leakage controls and non-model audit

1. Labels, scenario families, response-like fields and detector decisions are **never** model features and
   never enter templates, group ids, fingerprints, record ids, source ids, timestamps, header names,
   endpoint names or rule identifiers. Model-facing content (`RequestFeatures`) and metadata
   (`SplitMetadata`, `GenerationMetadata`) are separate types.
2. Record ids, source ids and timestamps are opaque and assigned after a shuffle, independent of label.
   Resources, methods, headers, user agents, roles and word lists are shared pools used by every class.
3. `audit.py` reports **errors** (construction-rule violations: label or family terms in request fields,
   feature/metadata mixing, non-determinism, group or exact-duplicate overlap across splits) and
   **warnings**: *descriptive* shortcut statistics from request length, method, header sets, roles,
   content types and payload shapes: the normalized association with the label next to label-shuffled
   references, the share of records with class-exclusive values (both the legacy record-level measure and the
   template-supported measure, which raises the warnings), and how much held-out feature vocabulary also
   occurs in training. An independent **content-type semantics check** verifies that every declared type matches
   the actual body format unless the request is a labelled suspicious member of the mismatch family. Warnings are attention flags, **not pass/fail criteria**, and no
   numeric success threshold is set. Some association is expected by construction (for example a
   mass-assignment body shape). **Passing every check does not show that the generator is free of leakage
   or shortcuts.**
4. The vocabulary, hyperparameters, `num_buckets`, epochs, learning rate and the decision threshold are
   fitted or chosen on **training and validation data only**. The held-out test split is used **only for
   final scoring, once**, after the pre-run manifest; any change requires a new protocol version.
5. Provenance holds caller-supplied opaque `record_id`s, hashed group ids, counts and hashes and no raw
   request text.

## 7. Metrics (defined; metric and bootstrap functions implemented in D4-5, section 12.11.1; the runner is not implemented yet)

- **Primary: PR-AUC** (threshold-free; the score is uncalibrated, and positives are a minority).
- **Secondary:** ROC-AUC; precision, recall, F1, false-positive rate, false-negative rate and the confusion
  matrix at one threshold **chosen on validation only** (rule to be pre-registered, for example maximum F1);
  recall by scenario family with small families flagged as unreliable; false-positive and false-negative
  review by family and template group, including hard negatives and hard positives separately.
- **Uncertainty:** cluster (template group) bootstrap, with paired differences between arms on the same
  test items.
- **Latency:** secondary only, under its own new protocol; the archived KPI 4 result is not re-run or
  compared.
- **Score semantics:** the model score is uncalibrated and is not an attack probability.
- **No numeric success target** is set. A decision rule for "B improves on A" (for example a paired
  grouped-bootstrap interval for the PR-AUC difference excluding zero, and its margin) must be pre-registered
  before the scored run. The comparison plan and what is still undecided are in section 12.11.

## 8. Retraining

The shipped demo model and vocabulary were fitted on two hand-written examples and are **not** used. Each
arm is fitted from scratch on the training split; the artifacts live outside the served path, and the API
and dashboard keep their current demo model.

## 9. What results can and cannot support

- **Can support:** synthetic held-out performance of each arm on unseen templates drawn from the author's
  scenario families, with group-aware splits, validation-only selection and intervals; and the *feasibility*
  of the pipeline and the A/B/C comparison.
- **Cannot be claimed:** real-world attack-detection accuracy; performance on real traffic or real attackers;
  the value of genuine Member 2 verified rules; Member 2 or Member 3 integration; KPI 5; calibrated
  probabilities.

## 10. Provenance and freezing

Before the scored run a pre-run manifest must be written once and never overwritten, recording: protocol
version and the hash of this file; hashes of `splits.py`, `synthetic_catalog.py`, `generator.py` and
`audit.py`; generator, catalog and template versions; dataset seed and split seeds; generator
`fingerprint`, split `manifest_sha256` and `dataset_fingerprint`; the simulated rule catalog version and SHA-256 (`simpol-1`, section 3.1) and the hash of `policy_sim.py`; and the
SHA-256 of every production source file used. The new modules are outside the archived KPI source-hash
inventory, so the archived reports do not cover them. Archived KPI 1-4 fixtures, protocols, hashes and
reports must not be modified or regenerated. The manifest plan and its additional contents are in section 12.12.

## 11. Open items before this draft can be frozen

Status after D3c-1. "Decided" means the plan is recorded in section 12; it does not mean the code exists or the
protocol is frozen.

| # | Item | Status |
|---|---|---|
| 1 | Review of the pilot statistics and audit warnings; revise the generator under a new version | **Implemented** (D3c-2): `synthgen-3` with corrections G1 and G2, design note and development-seed results in 12.2.1. The confirmatory dataset is not generated. |
| 2 | Decision rule and margin for A versus B; shortcut-flag handling | **Open.** Comparison plan recorded (12.11); the rule and any margin are fixed before the confirmatory scored run. |
| 3 | Arm C | **Included** (12.10). Model-class compatibility and the feature-budget policy are **open** until checked before coding. A flat-versus-structure-aware tokenization ablation is **not included** and remains open. |
| 4 | Leave-family-out split | **Open**, not decided. |
| 5 | `max_length` identical across arms from training-split statistics only | **Rule decided and implemented** (12.8, 12.8.1); the value is computed later from the confirmatory training split. Header handling is identical across arms by sharing `request_to_context-v1` (12.9). |
| 6 | Review of `simpol-1` and its diagnostics | **Resolved:** `simpol-1` is unchanged; SIM-P06 and SIM-P12 are documented limitations (12.3); SIM-P04 and SIM-P07 are addressed in the generator (12.2), not in the catalog. |
| 7 | Final scale and prevalence sensitivities | Scale **proposed** (12.7, subject to feasibility). Prevalence sensitivities are **open**. |
| 8 | Arm C feature-budget policy, n-gram settings and equal hyperparameter-search effort | Encoder, n-gram settings and serialization **implemented** (12.10.1: character 3-grams, same model class, no truncation). The equal-effort search grid and the `num_buckets` values are recorded in D4-6C1 (12.11.2) as an accepted draft decision (27 candidates per arm); not frozen. |
| 9 | Hyperparameter grids, selection criterion, threshold rule, bootstrap settings (resamples, seed) | Threshold rule (maximum validation F1) and bootstrap settings (2,000 resamples, seed 8201, 95% percentile interval, minimum-valid rule) **implemented as draft** in D4-5 (12.11.1). The hyperparameter grid, the model selection criterion with its tie-break order, arm-relative learning rates and the ranking score are **recorded as accepted draft design decisions** in D4-6C1 (12.11.2); the bootstrap minimum-valid rule stays open pending final freeze; nothing is fixed until the manifest (12.12). |
| 10 | Whether `audit.py` gains path-root and method-by-path-category summaries | **Implemented** (D4-0, section 12.2.2) as a separate descriptive field; audit thresholds, errors and warnings unchanged. |
| 11 | Scope of label-conditioned audits relative to the sealed test split | **Open** (12.6). |
| 12 | Whether development-seed (6101) dry runs report any detection metric | **Open.** Until decided, none is reported (12.5). |
| 13 | Runner, arm, budget-rule and manifest code; the actual manifest | **Not implemented** (12.12). Only `request_to_context-v1` (12.9.1), `budget-rule-1` (12.8.1) the Arm C encoder (12.10.1), the paired arm builders (12.12.1) and the metric and bootstrap functions (12.11.1) exist so far; the runner and manifest tool do not. |
| 14 | Any generator correction beyond G1 and G2 | **Not approved.** Needs a documented reason and approval before the generator is fixed for the confirmatory study (12.2). |

## 12. Design decisions recorded in D3c-1 (documentation only)

These decisions were approved after the D3c read-only design review. They are recorded **before** the generator is
changed and before any model evaluation code exists. They record intent; none is implemented by this step, and none
of them freezes the protocol (section 12.12).

### 12.1 Version and history

The protocol is `heldout-eval-1.2.0-draft`. The version history above is retained unchanged; this step adds one row.

### 12.2 Generator `synthgen-3`: two planned corrections

`synthgen-3` was approved with exactly these initial corrections and implemented in D3c-2 (section 12.2.1). Their
aim is to remove two *generator artifacts* seen in the exploratory pilot diagnostics (section 3.1). They reduce those
associations; they are not claimed to remove them.

- **G1 - normal files-path scenarios.** In `synthgen-2` the first path segment `files` occurs only in the
  traversal-style suspicious scenarios, so it is a path token that marks the class (and is what makes SIM-P07
  class-aligned). `synthgen-3` adds normal (label 0) requests under a files path, so that this structure occurs in
  both classes.
- **G2 - suspicious DELETE scenarios on ordinary resource paths, with comparable normal DELETE structures.** In
  `synthgen-2`, DELETE on an ordinary resource path occurs only in normal scenarios (which is what makes SIM-P04
  class-aligned). `synthgen-3` adds suspicious (label 1) scenarios that use DELETE on ordinary resource paths. Any new
  suspicious DELETE structure that carries a query string or body needs a comparable normal DELETE with the same
  kind of query or body, so that one marker is not traded for another.

Rules for these corrections:

- **Labels are never changed to balance statistics.** The label of every existing scenario family is unchanged. A new
  suspicious scenario is suspicious only by a documented construction rule of its family, never by its HTTP method
  or path category alone, and a new normal scenario is normal by the same standard.
- The exact mechanism, rates and expected template counts per structure are set in a short design note **before** any
  generator code is changed. The expected counts are computed from the family weights (a design-level check), not from
  labels of generated confirmatory data.
- The corrections follow the exploratory inspection of the seed-5101 pilot's labels and diagnostics. They are
  therefore **post hoc relative to that exploratory data** and are disclosed as such. Their purpose is to remove
  generator artifacts, not to improve any score.
- `tmpl-1`, `splits.py` and `simpol-1` do not change. `synthgen-3` has its own generator fingerprint, and the
  catalog version is bumped if the scenario catalog changes (expected: `synthetic-catalog-3`).
- No other correction is approved. Any further correction needs a documented reason and approval **before** the
  generator is fixed for the confirmatory study. After the confirmatory dataset (seed 7101) has been generated, the
  generator does not change for that study; a change creates a new generator version, a new protocol version and a
  new dataset (see 12.5 for the only permitted recovery path).

#### 12.2.1 G1/G2 implementation note (D3c-2, recorded before any generator code was changed)

*Design constants below were fixed before any `synthgen-3` data was generated. They are not tuned on labels, audit
output or any model result. Expected counts are design-level arithmetic from the unchanged family weights (total 106),
at 2,000 requested templates, before template collapse; they are approximate.*

**Families that change** (only these; every other family, weight, pool and label rule is unchanged):
`hn_dotted_paths` and `crud_read` (G1); `crud_write`, `traversal_like`, `sqli_like`, `cmdi_like` and `markup_like` (G2).

**G1 - normal files paths (label 0 by the existing "always 0" rules).**
- `hn_dotted_paths` (hard negative) draws its first segment from `{documents, files}` with equal probability: an
  ordinary dotted file in nested folders, e.g. `/files/<resource>/<id>/<folder>/<name>.final.<ext>`, GET.
- `crud_read` prefixes its existing nested read path with `files` with probability 0.12: a routine read of a nested
  item under a files path, GET.
- Permitted: method GET, path category `files_area`, authorization header present (the existing default for these
  families), optional paging/filter query as these families already have.
- Expected at 2,000 templates: about 75 normal files-path templates (about 57 + 18) versus about 40 suspicious
  files-path templates, i.e. about 35% suspicious, close to the roughly one-third template prevalence.

**G2 - comparable DELETE on ordinary resource paths (path category `resource_area`).** Four DELETE structures, two
per class, share the same method, path category, authorization and role-header handling, no body and no content type:

| Structure | Class | Family | Path | Query |
|---|---|---|---|---|
| N1 (existing) | normal | `crud_write` | `/<resource>/<id>` | none |
| N2 (new) | normal | `crud_write` (DELETE, probability 0.5) | `/<resource>/<id>` | one ordinary text parameter plus the optional paging/filter parameters (`_text_query`), the same builder the suspicious text families use |
| S1 (new) | suspicious | `traversal_like` (probability 0.3) | `/<resource>/` + traversal segments | none |
| S2 (new) | suspicious | `sqli_like`, `cmdi_like`, `markup_like` (query-slot branch, DELETE with probability 0.2) | `/<resource>/<id>` | the same `_text_query` builder with a suspicious value in the text parameter |

- Comparability: every DELETE cell (no query / text query) holds both classes by design. A DELETE has no body and no
  content type in either class. Query presence, authorization-header presence and body shape therefore do not mark a
  class for DELETE requests. S2 and N2 differ only in the value of the text parameter, as the existing GET text
  scenarios do.
- Traversal on a resource root uses only DELETE and carries no query, so it adds no new paging-only DELETE cell.
  The remaining 0.7 of traversal templates keep the existing `files` root with GET and their existing paging query.
- Expected at 2,000 templates: about 38 normal DELETE templates (about 19 N1 + 19 N2) versus about 37 suspicious
  (about 17 S1 + 20 S2).

**Labels keep their existing security meaning.** No family's label rule changes. S1 is suspicious because its path
contains a traversal-like construction; S2 because its text parameter holds an injection-, command- or markup-like
value. The HTTP method is never the reason for a label. N1 and N2 are normal under `crud_write`'s existing rule.

**Versions and fingerprints.** `GENERATOR_VERSION` becomes `synthgen-3` and `CATALOG_VERSION` becomes
`synthetic-catalog-3`. Record ids are derived from the generator version, so every record id changes, and the
generator fingerprint and the pinned regression values in the tests change. The default dataset seed becomes the
development seed 6101 (12.5), so the default configuration no longer reproduces the burned seed-5101 data. `tmpl-1`,
`splits.py`, `policy_sim.py`, `policy_diagnostics.py` and `simpol-1` (SHA-256 unchanged) are not modified. No
additional correction is made; the audit methodology is not changed in this step.

**Implementation result (D3c-2; development seed 6101 only; descriptive; no model, no metric).** The confirmatory seed
7101 and the reserve seeds 7102 and 7103 were not generated or inspected. Counts below use a template once per class
it holds; the unchanged `synthgen-2` code was run on the same seed 6101 for the "before" columns.

| Structure (templates, normal / suspicious) | `synthgen-2`, 250 | `synthgen-3`, 250 | `synthgen-2`, 2,000 | `synthgen-3`, 2,000 |
|---|---|---|---|---|
| First segment `files` | 0 / 7 | 7 / 6 | 0 / 73 | 76 / 39 |
| DELETE, resource path, no query | 6 / 0 | 0 / 3 | 19 / 0 | 12 / 19 |
| DELETE, resource path, text query | 0 / 0 | 3 / 1 | 0 / 0 | 19 / 11 |
| SIM-P04 (DELETE on resource path) | 6 / 0 | 3 / 4 | 19 / 0 | 31 / 30 |
| SIM-P07 (files path, authorization present) | 0 / 7 | 7 / 6 | 0 / 73 | 76 / 39 |
| SIM-P06 (GET on status path) | 3 / 0 | 3 / 0 | 26 / 0 | 26 / 0 |
| SIM-P12 (anonymous, protected path) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

At 2,000 templates the only (method, path-category) cell with at least 3 templates in one class and none in the other
is GET on status paths (the documented SIM-P06 limitation); in `synthgen-2` the cells for DELETE on resource paths and GET
on files paths were also one-class-only.

Development pilot (250 templates, seed 6101, split seed 4401): 1,466 records, 369 suspicious (25.2%); templates
168 normal / 49 suspicious / 33 mixed, 32.8% containing a suspicious record; 455 hard-negative and 75 hard-positive
records; 13 collapsed specs, no shortfall, no exact duplicates; generator fingerprint
`908627770c1c07cabc3832a7e3720e80c0c60b6fa97607cd18b71b035388ec00`; train / validation / test 1,024 / 225 / 217
records and 158 / 47 / 45 templates. Planned-scale development run (2,000 templates): 12,068 records, 3,253 suspicious
(27.0%); templates 1,348 / 444 / 208; train / validation / test 8,450 / 1,811 / 1,807 records and 1,269 / 366 / 365
templates. The non-model audit reports 0 errors and 0 warnings at both scales with the audit thresholds unchanged.
Pilot token lengths (all records, descriptive only, not used to choose `max_length`; 12.8): Arm A mean 48.5, maximum 92;
Arm B mean 85.5, maximum 132; at `max_length` 128, 5 Arm B requests are longer and none of Arm A.

Remaining limitations, reported and not adjusted:
- The design constants were not changed after the development data was read. At 2,000 templates the normal
  no-query DELETE cell (12 templates) is smaller than the arithmetic estimate in this note (about 19), because
  `/<resource>/<id>` has only 25 distinct normalized templates and collapses; the suspicious side of that cell has
  19. Both classes are present, but not in equal numbers. At 250 templates the cell has no normal template (0 / 3), a
  one-class cell at small support. Small cells such as DELETE on a `documents` path with a text query (0 / 3 at 2,000)
  also remain. Whether to rebalance would be a further correction needing separate approval (section 11, item 14).
- `..` and encoded separators remain the documented traversal anomaly, including on DELETE. Injection-like values
  remain the documented anomaly in the text parameter.
- GET on status paths stays one-class by construction (SIM-P06); SIM-P12 stays inert; `simpol-1` is unchanged.
- `audit.py` has no path-category or method-by-category summary (section 11, item 10). The zero-warning audit result does
  not show that the generator is free of leakage or shortcuts, and the cell counts above are a separate descriptive
  check, not part of the audit. The audit report's `protocol` label still reads `heldout-eval-1.1.0-draft`; it was
  not changed in this step.
- The tokenizer limitation in 12.9 is unchanged: query-borne text remains inside one atomic path token for Arms A and B.

#### 12.2.2 Structural coverage diagnostics (D4-0; descriptive)

`audit.structural_coverage(dataset)` reports, by class, the distribution of requests over path category, over method by path
category, and over method by path category by query presence (present/absent). Categories, methods and query presence come
from `policy_sim.observe`, the definition the simulated rule catalog uses.

- **Record level versus template level.** Each cell reports record counts per class (normal / suspicious) and distinct
  normalized-template counts. A template is counted **once per cell**; a mixed-label template stays one template and is
  reported separately from `normal_only` and `suspicious_only` templates (`with_normal` and `with_suspicious` overlap on
  mixed templates and are not additive).
- **Class-exclusive cells** (records of one class only, supported by at least 3 distinct templates) and **sparse cells**
  (fewer than 3 distinct templates) are listed as attention output.
- **Nothing else changes.** The report is a separate `structural_coverage` field. The audit's errors, warnings,
  thresholds (record support 5, template support 3, exclusive share 0.05), `construction_checks_passed`, fingerprints and
  duplicate and group-overlap checks are exactly as before; exclusive or sparse cells are never promoted to warnings.
  Output is deterministic, JSON-safe and holds no raw request content.
- **Construction diagnostics are not detection-quality evaluation.** This output trains no model and computes no PR-AUC, ROC-AUC,
  F1 or accuracy. A zero-warning audit, with or without this field, does not show that shortcut features or leakage are absent.

Development-seed observations (seed 6101 only; confirmatory and reserve seeds not used). At 2,000 templates (12,068 records):
the only class-exclusive (method, path category) cell is GET on status paths (0 / 26 suspicious / normal templates; SIM-P06 is
accepted as a one-class structural association); GET on files paths holds 76 normal-only, 39 suspicious-only and no mixed
templates (SIM-P07, both classes); DELETE on resource paths holds 31 normal-only and 30 suspicious-only (SIM-P04, both
classes); and SIM-P12 remains unused (no anonymous request reaches a protected path). The accepted no-query DELETE
imbalance is visible as 12 normal-only versus 19 suspicious-only templates with query absent (30 templates, 19 normal-only and 11
suspicious-only, with a text query present). Sparse cells at that scale: DELETE and PATCH on documents paths without a query
(1 template each); DELETE on documents paths with a query is exclusive to suspicious (3 templates). At 250 templates the
no-query DELETE cell holds only suspicious templates (3), and several further small cells are exclusive or sparse; such small-scale
cells are expected sampling variation, are reported, and are not repaired.

### 12.3 Simulated rule catalog `simpol-1`: unchanged

`simpol-1` (13 policies, SHA-256 in section 3.1) is **kept unchanged**. No rule is added, edited or dropped.

- **SIM-P06 (GET on status paths)** applies only to health and status calls, which are normal by construction. This is
  a **structural limitation** of the synthetic scenarios: the rule restates a path association that the request
  tokens already carry, and it stays class-aligned. It is documented, not engineered away.
- **SIM-P12 (Deny for a missing authorization header on protected paths)** has zero coverage in the current generator
  because no anonymous request reaches a protected path category. It is documented as an **inert, zero-coverage
  policy**. No scenario is added merely to give it coverage; such a scenario would tie the rule to a label. G1 and G2
  are not designed to create anonymous protected-path requests. Whatever P12 coverage the confirmatory data actually
  has is reported as observed and is not adjusted.
- SIM-P04 and SIM-P07 are addressed through the generator (12.2), not by editing the catalog. After `synthgen-3`,
  per-policy coverage and class association are reported descriptively again; they are not used to change the catalog.

### 12.4 Status of the seed-5101 pilot

The `synthgen-2` pilot (dataset seed 5101, primary split seed 4401) and its held-out split are **exploratory and
burned**. Its labels, construction audits, policy diagnostics and split were inspected while the design was refined.

- **Allowed:** debugging and construction checks (for example that code runs and that construction rules hold).
- **Not allowed:** use as untouched final test evidence; selection of final hyperparameters, thresholds, `max_length`
  or any other final model choice from it; reporting any score from it as evidence of detection quality.

### 12.5 Seed plan and reserve-seed rule

| Purpose | Dataset seed |
|---|---|
| Exploratory pilot (`synthgen-2`), burned | 5101 |
| Development and debugging (`synthgen-3`) | **6101** |
| Confirmatory study | **7101** |
| Reserved recovery seeds, used in this order | **7102**, then **7103** |

Seeds are arbitrary values fixed here before any dataset exists under them. **No seed is chosen, kept or replaced
because of performance, and a seed is never replaced because a result was unexpected or unfavourable.**

A reserve seed may be used **only** when all of the following hold:

1. The confirmatory dataset (7101, or 7102 if it was already replaced) failed for a **construction or
   infrastructure reason**: a construction error of the kind listed in section 6.3 (class or family terms in request
   fields, feature/metadata mixing, non-determinism, group or exact-duplicate overlap across splits, content-type
   semantics violation), an infeasible group-aware split, or a tooling failure that makes the dataset unusable.
2. The failure was found **before any model was trained or scored on that dataset**. After any model has seen the
   confirmatory dataset, no reserve seed may replace it.
3. The cause is fixed in code and the fix gets a new generator (patch) version and is recorded in the manifest.
4. A **log entry** is written before the reserve seed is generated: date, the failed seed, its generator
   fingerprint, the exact error or failure, the code fix, and the reserve seed that replaces it. The log is part of
   the provenance (12.12).

The failed seed's dataset is burned and is not used. Association warnings, shortcut statistics, unfavourable
distributions and any model output are **not** reasons to use a reserve seed. If 7103 also fails, work stops and the
protocol is revised before any further generation. Development-seed (6101) runs are for debugging and construction
checks; they are not reported as results and do not select final hyperparameters. Whether such runs may print any
detection metric is an open item (section 11, item 12): until decided, none is reported.

### 12.6 Splitting and the sealed test split

The primary split is **group-aware train / validation / test** (0.70 / 0.15 / 0.15; section 5), with primary split
seed 4401 applied to the new dataset. The held-out **test split stays sealed** until every model choice has been
finalized on training and validation data only: vocabulary settings, `num_buckets`, epochs, learning rate, any other
hyperparameter, the Arm C settings, `max_length`, and the decision threshold. The test split is scored **once**, after
the pre-run manifest and after the selections are recorded (12.12). The scope of label-conditioned descriptive audits
on the test split (for example whether they are computed on training and validation only) is open (section 11,
item 11); in any case no audit result on the confirmatory test split may be used to change the generator, the
catalog, the arms or the selection.

### 12.7 Scale of the full study

The study targets **approximately 2,000 normalized templates and approximately 12,000 synthetic requests**, subject
to feasibility checks. These are **proposed study settings, not sample sizes mandated by the supervisor or the
original proposal** (the proposal specifies none; section 1). The requested count is never silently forced
(section 2.4): the actual template and record counts and any shortfall are reported. Feasibility checks include
template collapse, a feasible group-aware split with both classes in every split, expected template counts per
structure (12.2), and run time. As before, the number of distinct templates, not the number of records, is the
effective sample size.

### 12.8 Equal `max_length` for Arms A and B

One `max_length` is used for **both** Arm A and Arm B. It is the **smallest value in {128, 192, 256, 384, 512}** that
accommodates the **confirmatory training split's Arm B token sequence lengths without truncation**, that is, with every
training-split Arm B sequence no longer than the chosen value.

- Only the training split is used. Validation results, test results, validation and test lengths, and any model
  output are **not** used. Arm B is used because rule tokens make it the longer arm.
- 256 is a candidate, not a pre-approved value, and 128 (the demo and archived-KPI value) is not assumed. The lengths
  seen in the burned pilot do not determine the value.
- If no value in the grid accommodates the training split, nothing is truncated silently: work stops and the rule is
  revised under a new protocol version.
- A record that any arm rejects (for example because its framing does not fit) is removed from **all** arms and
  reported (section 3). Truncation rates per arm on validation and test are reported, because validation and test
  sequences can exceed the training maximum.
- The value is computed after the confirmatory dataset exists and recorded in the manifest (12.12). It is not
  computed in this step.

#### 12.8.1 `budget-rule-1` (implemented in D4-2)

`tokenization_ai/heldout_eval/token_budget.py` implements the rule of 12.8. It is evaluation infrastructure: it trains nothing, reads no
label or metadata, and has not been run on any confirmatory data.

- **Candidate grid and algorithm.** The grid is `(128, 192, 256, 384, 512)` (strictly ascending positive integers; validated).
  `select_shared_max_length(train_arm_b_lengths)` takes one length per **training-split Arm B** sequence, keyed by opaque record id, and
  returns the smallest grid value that is greater than or equal to the largest length (a length equal to the value fits). The same value is
  then used for Arm A and Arm B. The result does not depend on input order.
- **Training-only restriction.** The function's only data parameter is the training Arm B lengths; it cannot see validation or test
  sequences, labels, scores or any metadata. Validation and test lengths are reported only afterwards by the exposure report against the
  already fixed value, and nothing that report finds can change the budget: a finding triggers protocol review, not a larger
  `max_length`, and no record is discarded.
- **Input validation and failure.** Empty input, non-integer, boolean or non-positive lengths, empty or whitespace-padded record ids,
  duplicate ids and an invalid grid raise `BudgetInputError` (messages never echo record content). If the longest training length exceeds the
  largest candidate, the result has `status="no_candidate_fits"` and no selected value; `require_selected()` raises
  `BudgetNotFeasibleError`. Nothing is truncated or chosen silently, and work stops until the rule is revised under a new protocol version.
- **Sequence and framing semantics.** A length is `len(TokenSequence.tokens)` exactly as the tokenizer emits it and the tensorizer receives
  it (token-to-id encoding is one-to-one). It already includes `[SEQ_START]`/`[SEQ_END]`, method, path and header tokens, payload tokens and
  the whole rules segment (`[RULES_START] [RULES_END]` even with no rules, plus 20 tokens per simulated rule in Arm B). No framing overhead
  is estimated. The real tensorizer never cuts framing or rules: a longer sequence has whole payload fields dropped (*truncated*), and a
  sequence whose framing plus rules cannot fit even with an empty payload is *rejected* (`ValueError`). `tensorizer_outcome` reports
  `fits` / `truncated` / `rejected` by running the tensorizer's own payload-truncation routine on the real sequence.
- **`BudgetDecision` (recorded in the manifest).** `status`, `selected_max_length`, `rule_id` (`budget-rule-1`), `candidate_grid`, `basis`,
  `train_records`, `length_statistics` (`min`, `mean` = sum/n rounded to 6 decimals, `median` = middle value or mean of the two middle values,
  `p95` and `p99` by nearest rank = value at 1-based rank ceil(p * n) of the sorted lengths, `max`), `headroom` (selected value minus the
  longest length), `records_over_candidate` (training records longer than each candidate), and `lengths_sha256`, the SHA-256 of the
  compact ASCII JSON `[[record_id, length], ...]` sorted by record id (not Python `hash()`).
- **Exposure report.** `exposure_report(decision, split, arm_a_lengths, arm_b_lengths, ...)` accepts only a decision with a selected value.
  For one split it reports per arm the record count, the number and share over budget, the longest length and largest excess, and a
  truncation-risk flag; optionally the tensorizer outcome counts and rejected ids; and the pairing (records in both arms, records present in
  only one arm, ids capped at 20) and paired differences (over budget in either, only in Arm B, only in Arm A, both; mean and maximum B-minus-A
  length; records where Arm A is longer, which is unexpected). `findings` lists, explicitly: pairing mismatches; Arm A longer than Arm B; an
  over-budget training sequence in either arm (unexpected, since the rule fits Arm B by construction and does not cover Arm A); validation or test
  records over the frozen budget; and records the tensorizer would reject (protocol section 3 requires removing those from all arms; the
  report only reports them). `budget_changed_by_this_report` is always false and `records_excluded_by_this_report` is always 0.
- **Manifest provenance.** The future pre-run manifest records the whole `BudgetDecision` dict, the SHA-256 of `token_budget.py`, and the
  training-split Arm B length fingerprint. The exposure reports for validation and test are produced only after the manifest is written and
  belong to the results, not to the manifest.
- **Limitations.** The rule fits the longest training Arm B sequence, so a single outlier sets the budget; validation and test sequences
  can still exceed it (payload truncation, reported per arm); Arm A is not part of the selection; the rule says nothing about whether a
  budget is *sufficient* for detection. At this milestone it was exercised on hand-made arrays and development-seed (6101) requests only.

### 12.9 `request_to_context-v1`: production-faithful A/B mapping

Arms A and B map a synthetic request to the pipeline's input context through one explicit, versioned function named
`request_to_context-v1`. It is **production-faithful**: it supplies the request as the production pipeline receives
it, and it does not reshape the request to help any arm. The mapping is hashed and recorded in the manifest. The
existing diagnostic mapping in `policy_diagnostics.py` is only a starting point and is not `request_to_context-v1`
until it has been reviewed, fixed as a dedicated function and hashed.

**Information-visibility limitation (documented, not corrected).** The path, including any query string, is passed
as supplied. The production tokenizer splits a path on `/`, and `UnifiedApiContext` has no query field, so a query
string stays inside the last path segment and becomes **one atomic path token** (for example
`search?q=word+word&page=3`). Literal values are atomic tokens too, and a literal not seen in the training vocabulary
maps to the unknown token. Therefore text carried in a query string, or in an unseen value, can be **invisible** to
Arms A and B, while a character-n-gram arm (Arm C) can see its substrings. Consequences:

- Comparisons of A or B with C measure the *deployed pipeline* against a *conventional baseline* with a different
  visibility of the request. They are **not** an isolated test of structure-aware tokenization.
- Recall by scenario family (section 7) is reported so that families whose signal sits in a query string or an
  unseen value can be read with this limitation in mind.
- The mapping is implemented as `request_to_context-v1` (section 12.9.1).
- Splitting the query out of the path would be a different mapping. It would need a new mapping version
  (`request_to_context-v2`) and a new protocol version; it is not done in this study.

#### 12.9.1 `request_to_context-v1` (implemented in D4-1)

`tokenization_ai/heldout_eval/request_context.py` provides `request_to_context(record) -> UnifiedApiContext` and the id
`MAPPING_ID = "request_to_context-v1"`. It is a pure, deterministic function of the request content. It reads only
`record.record_id` and `record.features`; it never reads the label, scenario family, template id, split, source id, timestamp or any
prediction, and it does not mutate the record. It attaches **no rules**: the arm builders attach an empty rule list for Arm A and the
**simulated** `simpol-1` rules (not Member 2 verified) for Arm B, so the same mapped context serves both arms.

| Context field | `request_to_context-v1` |
|---|---|
| `event_id` | the opaque `record_id` (a context identifier only; the tokenizer never emits it) |
| `method` | as given (`Normalizer` upper-cases it later) |
| `path` | the complete string **including `?query`**, unsplit and not percent-decoded |
| `headers` | every request header copied as given, including `content-type` and `authorization`; duplicate names (after lower-casing) are rejected, not merged |
| `content_type` | the `content-type` header value (name matched case-insensitively), else `None`; the tokenizer does not read this field |
| `payload` | decided by the **body content**, never by the declared type or a label: empty body gives `{}`; a body that parses as a JSON object gives that dict; a form-like body (contains `=`, no space once stripped) gives `parse_qsl(keep_blank_values=True)` as a string dict (a repeated name keeps its last value); anything else (plain text, XML, a JSON non-object) gives `{"text": body}` |
| `client_ip`, `threat_score`, `timestamp` | not set |

- **Production-faithful:** the field names and types of `UnifiedApiContext`; the whole path with its query as one string (as
  `adapters._build_context` receives `metadata["path"]`); the header dict; and the stage order that follows (fusion, privacy masking,
  normalization, structure-aware tokenization, as in `service._front`). Masking and normalization are not reimplemented or pre-empted in the
  adapter: credentials stay raw in the context and are masked by the existing `PrivacyMasker`.
- **Evaluation conventions (not production facts):** production only guarantees `payload: dict`, so the two non-JSON body rules (form-like
  and `{"text": body}`), the content-based choice between them, and the rejection of duplicate header names are conventions of this study.
  They are the same body rules as the diagnostic mapping `policy_diagnostics.to_context`; a test asserts the two mappings are identical for
  every generated request on the development seed, and that the result equals the production adapter's output for a JSON request.
- **Limitations (kept, not fixed):** a query string stays inside the last path segment as one atomic literal and is not decoded, so query
  text is almost invisible to Arms A and B (12.9); a body is classified by a heuristic on its content, so an unusual body can fall under a
  different rule than a real upstream parser would choose; a declared-type mismatch is visible only as a header literal beside a payload shape.
- **Provenance for the pre-run manifest:** record `MAPPING_ID` and `source_sha256()` of `request_context.py` (SHA-256 of the source with
  line endings normalized to LF, stable across checkouts), together with the hashes of the production sources it feeds (`models.py`,
  `fusion.py`, `masking.py`, `normalization.py`, `tokenizer.py`).

### 12.10 Arm C: conventional hashed character-n-gram baseline

Arm C is a conventional, non-structure-aware preprocessing of the same requests: the request is flattened to text and
represented by hashed character n-grams, and scored by a **comparable logistic-regression model family** with a
**matched feature-budget policy** and **equal hyperparameter-search effort** (the same number of grid points, the same
selection criterion, the same use of training and validation data only).

- **Model compatibility is not assumed.** The existing `LightweightAnomalyModel` consumes integer token ids with an
  attention mask and applies its own multiplicative hashing to ids and id pairs; it is not known to be suitable for raw
  character features. Before any coding, a compatibility check decides whether Arm C reuses it (for example by
  encoding hashed n-grams as ids) or uses a separate logistic regression with a matching loss, optimizer and
  regularization. If a different implementation is used, every difference is disclosed.
- **Privacy masking is not bypassed.** The same privacy masking applies to Arm C as to Arms A and B.
- **No extra label-derived information.** Arm C receives the same request fields as Arm A. It receives no label,
  scenario family, template id, split assignment, generated metadata or simulated policy id, and it has no rule tokens.
- **Differences in visibility and interpretation are documented.** Arm C sees substrings of query strings and values
  that Arms A and B treat as atomic tokens (12.9); it does not use the structure-aware tokenizer, vocabulary or
  tensorizer; and `max_length` does not apply to it unless the feature-budget policy says otherwise. A difference
  between C and A or B cannot be attributed to structure-awareness alone.
- The n-gram range, the hashing function, the text flattening and normalization steps and the feature-budget policy are
  **open** (section 11, item 8). They are written down before the manifest and never tuned on the test split.

#### 12.10.1 Arm C encoder `armc-char3-v1` (implemented in D4-3)

`tokenization_ai/heldout_eval/arm_c.py` provides `encode_request(NormalizationResult) -> ModelInput`, `serialize_masked_request`, `char_ngram_ids`
and `encoder_parameters()` (the fixed configuration, for the manifest). It builds, fits and scores no model and imports none.

- **Privacy-stage contract.** The only accepted input is a `NormalizationResult`, the output of the existing `PrivacyMasker.mask_with_provenance`
  followed by `Normalizer.normalize_with_provenance`. An `EvalRecord`, a raw `UnifiedApiContext`, a `FusedInput` or any label or metadata is
  refused. The existing stages give no tamper-proof stage proof (the result is a plain frozen dataclass, and no flag is trusted), so the boundary is
  **content-based**: the encoder re-runs the existing masker and normalizer and refuses the record if either would still change it, so a raw
  credential, token, e-mail or card number is refused. A forged provenance set does not help. Records carrying rules are refused (Arm C is
  request-only). *Limit:* this shows the existing masker has nothing left to mask; it cannot show the stages were executed, and it does not
  protect values the masker does not classify as sensitive. Masking is not reimplemented. The later Arm C builder must pass the real stage outputs.
- **Serialization.** `<METHOD> <path-with-query>` line; one `<name>: <value>` line per header sorted by header name (code-point order); an empty
  separator line; then `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`. The path keeps its query string unsplit and
  undecoded (as in `request_to_context-v1`). CR and LF inside the method, path, header names and header values are written as the two characters
  `\r` and `\n`, so content cannot forge a line or the separator. The result is case-folded with **ASCII-only** lowercasing (A-Z to a-z),
  independent of Python's Unicode tables; non-ASCII characters in the path and headers are left unchanged and non-ASCII payload characters are
  JSON-escaped. No percent-decoding. No rule tokens, labels, family names, record ids, timestamps or source ids enter the text; the event id is
  only `ModelInput.event_id`.
- **Character 3-grams and hashing.** All overlapping windows of 3 code points in order (`len(text) - 2` of them; none for text shorter than 3). Each is
  UTF-8 encoded and hashed with `hashlib.blake2b(digest_size=8)` (no key, salt or personalization); the digest is read as a big-endian integer and
  reduced to its low 31 bits (`& 0x7FFFFFFF`), so ids are non-negative integers below 2**31. Python's `hash()` is never used. There is no
  vocabulary, unknown token, padding or truncation; `attention_mask` is all ones, so the id count equals `len(text) - 2`.
- **`ModelInput` and the unchanged model.** The existing validation accepts the result. The model counts one hashed unigram per id (here, one
  character 3-gram) and one hashed bigram per adjacent pair of ids (two overlapping 3-grams, together a 4-character span) in one shared
  `num_buckets` space, so feature mass is `2n - 1`. The same model class and `num_buckets` setting can serve Arms A, B and C; each hyperparameter
  setting must use a fresh instance because `fit` warm-starts from existing weights. Encoding creates and fits nothing.
- **Differences from Arms A and B (not equivalence).** Arm C reads the characters of query strings and of values that Arms A and B treat as atomic
  tokens (mapped to the unknown token when unseen); it uses no structure tags and no vocabulary; it has no `max_length`; and it never includes rules.
  Arm C produces many more feature positions than A and B, so feature mass differs substantially; an equal `num_buckets` does not mean equal
  information or equal collision rates. Arm C is a conventional preprocessing baseline, not an isolated test of structure-awareness, and a
  difference between arms cannot be attributed to structure alone. Lowercasing and the absence of decoding are conventions of this study.
- **Development-seed preprocessing diagnostics** (seed 6101, 250 templates, 1,466 requests; label-free; no model fitted; confirmatory and reserve
  seeds not used). Serialized length: mean 225.7 characters, median 208, p95 415, maximum 613. Trigrams per request: mean 223.7, maximum 611. Feature
  mass: mean 446.5, against about 96 for Arm A and about 170 for Arm B (from their mean token counts of 48.5 and 85.5), i.e. roughly 4.6 and 2.6
  times more. Encoding is deterministic; ids lie in [40962, 2146963750]; the mask is all ones. All 1,341 requests with an authorization header show
  the masked placeholder; no credential pattern and no event id appears in any serialized text. The first 400 requests touch all 1,024 buckets at
  `num_buckets` 1024 and 3,654 of 4,096 at 4096, so heavy collisions are expected at small bucket counts. These are preprocessing observations, not detection results.
- **Manifest provenance.** Record `encoder_parameters()` (encoder id `armc-char3-v1`), the SHA-256 of `arm_c.py`, and the hashes of the existing
  `masking.py`, `normalization.py`, `models.py` and `model.py` sources that Arm C relies on.

### 12.11 Metric and comparison plan

- **Primary metric: PR-AUC** (threshold-free; the score is uncalibrated and positives are a minority).
- **Primary contrast: Arm A versus Arm B**, paired on the identical test items and the identical splits.
- **Uncertainty:** a **grouped (template-group) bootstrap** that resamples whole template groups and computes the
  paired PR-AUC difference on the same resample for both arms. The interval, number of resamples and bootstrap seed are
  fixed before the confirmatory scored run.
- Comparisons involving Arm C, ROC-AUC, precision, recall, F1, the confusion matrix at one validation-chosen
  threshold, recall by family and false-positive and false-negative review (section 7) are **secondary**. They use
  the same machinery and are reported descriptively, not as the primary claim.
- **No success margin is set.** The final decision rule for "B improves on A", and any margin, are **fixed before the
  confirmatory scored run** and recorded in the manifest. No improvement is claimed before results exist. "No
  difference" and "B worse than A" are acceptable outcomes.
- Score semantics are unchanged: the score is uncalibrated and is not an attack probability.

#### 12.11.1 Detection metrics `detection-metrics-1` (implemented in D4-5)

`tokenization_ai/heldout_eval/detection_metrics.py` computes metrics from **already-computed scores and labels**. It imports nothing outside
the standard library, trains, fits and scores no model, reads no request, and has been exercised only on small hand-made arrays. No
development-seed or confirmatory score exists, and the confirmatory seeds were not touched.

- **Inputs.** Scores are finite real numbers (`bool`, NaN and infinities are rejected); labels are the integers 0 or 1. Invalid input raises
  `DetectionMetricsError` (messages hold positions and counts only). A metric that is undefined for valid input is returned as `None`, never as 0.
- **Primary: Average Precision (the PR-AUC estimator).** Scores sorted descending, tied scores grouped into one block, step-wise sum without
  interpolation: `AP = (1/P) * sum_k pos_k * TP_k / n_k` over tied blocks (cumulative positives `TP_k` and records `n_k` after block `k`). The value is
  independent of input order and of the order inside a tie. All scores equal gives the prevalence. **Undefined** (`None`) with no positive record.
- **Secondary.** ROC-AUC (Mann-Whitney, ties worth one half, exact integer arithmetic; undefined unless both classes occur). TP, FP, TN, FN; precision
  (undefined without predicted positives), recall (undefined without positives), F1 = 2TP / (2TP + FP + FN) (undefined only when that denominator is 0;
  with positives present and nothing predicted it is 0), false-positive rate (undefined without negatives), false-negative rate (undefined without positives).
- **Validation-only threshold (`select_threshold`).** Predict positive when `score >= threshold`. Candidates: every distinct validation score plus the
  all-negative prediction (`threshold = None`, treated as the highest candidate). Maximum validation F1 (compared exactly as fractions); a tie goes to the
  highest threshold. Both classes are required in validation. The function's only parameters are validation scores and labels; the test split is
  used only afterwards, through `evaluate_at_threshold` with the already fixed threshold.
- **Paired group bootstrap (`paired_group_bootstrap`).** Statistic `AP(B) - AP(A)`. Whole normalized-template groups are drawn with replacement
  (`G` draws from `G` groups); a drawn group contributes all its records (mixed-label templates stay intact) and a group drawn `m` times contributes
  each record `m` times. One draw serves both arms, and the arms are joined by record id (id sets of both score maps, labels and groups must match).
  2,000 replicates; local `random.Random(8201)` (global random state is neither read nor changed); 95% percentile interval by linear interpolation
  between order statistics, from the valid replicates.
- **Undefined replicates.** A resample without a positive record has an undefined AP for both arms (labels are shared). Such replicates are counted
  (`replicates_undefined`, `undefined_reasons`), never redrawn and never replaced. **Draft minimum-valid rule:** at least 95% of the replicates must be
  valid (1,900 of 2,000), otherwise `status = "insufficient_valid_replicates"` and no interval is reported (the full-sample estimate still is). If the
  full-sample AP is undefined the status is `"undefined_point_estimate"` and no replicate is run. The 95% share is a draft convention chosen to
  match the interval level, not tuned on any result; because invalid replicates are dropped, the interval is conditional on the resample containing
  a positive.
- **No success margin.** Nothing here decides whether B improves on A. The decision rule and any margin are still fixed before the confirmatory
  scored run (section 12.11).
- **Scope of the interval.** It describes only the sampling variability of the synthetic generator under group resampling of one generated dataset
  and one split. It ignores model-fitting, hyperparameter, threshold and dataset-seed variability, is rough with few groups, and supports no
  real-world detection claim. Scores are uncalibrated ranking scores, and Arm B's simulated rules are not Member 2 verified rules.
- **Manifest provenance.** Record `METRICS_ID`, the SHA-256 of `detection_metrics.py`, the replicate count, seed, confidence level and
  `MIN_VALID_PERCENT`.

#### 12.11.2 Optimization and score-selection decisions (recorded in D4-6C1; documentation only)

**Status: accepted draft design decisions, not a protocol freeze.** No code, estimator, logit extraction, runner, manifest, model fit or
score exists for any of this, and no confirmatory data was generated (seeds 7101-7103 unused). These are planned specifications, not
measured performance.

**Informing evidence (D4-6B; label-free; development seed 6101 only; training split only).** On the development seed (primary split seed
4401; 250 and 2,000 templates) the feature matrices of the unchanged `LightweightAnomalyModel` were built from the training records of
each arm and the largest eigenvalue of `[X | 1]^T [X | 1] / n` was estimated for `num_buckets` 1024, 2048 and 4096. No label was read by the
diagnostic, no model was fitted, `predict` was not called, and no validation or test record was prepared. Findings: the curvature differs
strongly by arm (at 2,000 templates, `lambda_max` about 112-117 for Arm A, 270-283 for Arm B and 308-476 for Arm C; Arm C falls about 35%
from 1024 to 4096 buckets, A and B change by 4-5%); power iteration converged in 6-8 iterations and matched an exact Gram-matrix eigenvalue
to about 1e-11 (relative) at 250 templates; estimates at the two scales differed by under about 4%. Feature extraction is cheap (0.05-0.26 ms
per record) and the dense training matrix needs 66-264 MB (peak about twice that inside `fit`), so the grid below is feasible. These
diagnostics informed the choices below; they were not used to tune anything against labels or performance, and the actual values for the study
are computed only from the confirmatory training features.

**Decision 1 - arm-relative learning rates (supersedes the common absolute grid).** For each arm and `num_buckets`, `L = lambda_max / 4`,
where `lambda_max` is the largest eigenvalue of `[X | 1]^T [X | 1] / n`; `X` is built from the **training split only** with the model's actual
feature representation and the appended column of ones stands for the bias. Each candidate uses `learning_rate = c / L` with the same relative
step set for A, B and C: **`c` in {0.1, 0.3, 0.8}**. `L` bounds the curvature of the logistic loss (curvature at most one quarter of the
feature Gram spectrum), so `c <= 1` is the conservative range and `c < 2` the monotone-descent range; the bound is a numerical-stability
statement, **not** evidence of predictive quality.
- *Why not common absolute rates.* The proposed common values {0.002, 0.005, 0.012} gave every arm a stable candidate, but progress along the
  dominant curvature direction scales with `lr * L`, and `L` is about 2.4-2.5 times larger for Arm B and about 2.7-4.2 times larger for Arm C
  than for Arm A. With a fixed epoch count, a common absolute rate would have trained Arm A less than Arm B, confounding the primary A-versus-B
  contrast. The relative grid gives every arm the same fraction of its own stability limit. Larger absolute rates (for example 0.03) would also
  exceed the limit for B and C.
- *Estimator (proposed implementation settings, subject to compatibility tests; to be versioned, for example `lambda-power-1`).* Deterministic
  power iteration on `[X | 1]^T [X | 1] / n` without forming the matrix, in float64; initial vector uniform with entries `1/sqrt(d)`
  (`d = num_buckets + 1`); the Rayleigh quotient is the estimate; convergence when the relative change of the Rayleigh quotient between
  successive iterations is at most `1e-10`; at most `5,000` iterations. No random start, no seed.
- *Failure handling.* If the estimate is not finite, not positive, or does not converge within the maximum, the run **stops**. There is no
  fallback learning rate and no silent substitution.
- *Timing and provenance.* All learning rates are computed from the confirmatory **training** features after the protocol and runner are
  final, **before any model is fitted**, and recorded in the pre-run manifest. No label, validation AP, test datum or observed model
  performance may influence them.
- *Limits.* The Rayleigh quotient is a lower bound on the true eigenvalue (small if convergence is tight); a different seed changes `lambda_max`
  by a few percent; dominant-direction progress says nothing about slow directions, and the model has no regularization, so epochs act as an
  implicit stopping rule.

**Decision 2 - shared hyperparameter grid (27 candidates per arm).** `num_buckets` in {1024, 2048, 4096}; `epochs` in {50, 150, 400};
`relative_step_c` in {0.1, 0.3, 0.8}. Fixed: `min_frequency = 2` and `max_literal_vocab_size = None` for the shared, training-only A/B vocabulary
(not tuned, which avoids a tuning dimension Arm C lacks); the same model class for all arms; a **fresh** model instance for every candidate;
training on the training split only; validation-only selection. 8,192 buckets are excluded (dense-matrix memory).

**Decision 3 - validation model selection.** Per arm, select the candidate with the highest validation Average Precision (`detection-metrics-1`).
APs are compared after rounding to 10 decimal places (full-precision APs are kept in the diagnostic output). Ties are broken in this fixed order:
(1) fewer epochs, (2) smaller `num_buckets`, (3) smaller `relative_step_c`, (4) the stable lexicographic candidate identifier. Test scores and
labels are never used to choose a candidate.

**Decision 4 - raw-logit ranking score (held-out evaluation only).** The evaluation ranks by the raw linear-model logit `w . features + bias`, not
by `predict()`, because `predict()` returns a float64 sigmoid that becomes exactly 1.0 (or 0.0) for large logits and would create artificial ties.
- Applies **only** to this held-out evaluation. `LightweightAnomalyModel`, the production API and its score semantics are unchanged.
- Future evaluation code must verify, on non-saturated examples, that the extracted logit is mathematically compatible with `predict()`
  (`sigmoid(logit)` equals `predict()` within floating-point tolerance). If faithful extraction cannot be done safely without changing production
  code, work **stops** and approval is requested; production code is not changed to make it work.
- Validation AP, validation threshold selection, test AP and ROC-AUC use this same representation. Logits are uncalibrated ranking scores, not
  probabilities; a threshold chosen on logits is a **logit-scale threshold** and must not be presented or compared as a probability threshold.
  Section 12.11.1 applies with "score" read as this logit.
- The number of distinct scores per arm and split is to be reported, as a tie diagnostic.

**Decision 5 - numerical reproducibility.** The future pre-run manifest (not created now) must record: the eigenvalue method and its version; the
actual training-set eigenvalue estimates (with iteration counts and convergence flags) per arm and `num_buckets`; the selected-grid learning rates
per arm and bucket count; the shared relative-step grid; Python and numpy versions; relevant BLAS information and thread configuration; the full
hyperparameter grid and tie-breaking rules; and the SHA-256 of the model source (`model.py`) and of the other sources listed in 12.12.

**Decision 6 - matters deliberately kept open (nothing here is silently frozen).**
- The confirmatory A-versus-B success rule and any margin (12.11).
- The paired exclusion and truncation policy for records the frozen budget cannot represent (12.8.1, 12.12.1).
- The bootstrap minimum-valid-replicate rule (12.11.1) pending final freeze.
- The test-scoring crash and recovery policy (including what happens after an interrupted scoring run).
- Whether development-seed (6101) scoring may report any detection metric (section 11, item 12).
- The runner, estimator, logit extraction, manifest and selection-record implementations, and the final protocol freeze.

No confirmatory experiment may start until these are resolved and documented.

**Supersession.** These decisions replace the open grid and learning-rate placeholders of section 11 items 8-9 and the "common learning-rate
grid" idea from the D4-6A/D4-6B review (a common absolute grid {0.002, 0.005, 0.012} was proposed there and not adopted). Where section 12.12.1
says every hyperparameter setting needs a fresh model, that still holds.

### 12.12 Evaluation provenance and the write-once pre-run manifest (plan)

**This step creates no manifest.** The pre-run manifest is written **once**, only after all relevant runner, arm,
budget-rule and mapping code exists and has been reviewed, and **before any model training or scoring**. It is never
overwritten (as for the archived KPI manifest, whose writer refuses to overwrite an existing file). The protocol is
**not fully frozen** until the runner code, hashes, final decision rule and manifest are approved.

Planned order:

1. Code complete and reviewed: `synthgen-3`, `request_to_context-v1`, Arms A, B and C, the budget rule, the metrics
   and bootstrap code, and the manifest tool.
2. Generate the confirmatory dataset (seed 7101) once. Run construction checks. Split. Compute the training-split
   Arm B lengths and the `max_length` (12.8). No model is trained, no score is computed and no test label statistic is
   used.
3. Write the manifest.
4. Train and select using training and validation data only. Record the selected hyperparameters, threshold and the
   hashes of the fitted artifacts in a second write-once record **before** the test split is scored (planned; to be
   confirmed when the runner is reviewed).
5. Score the test split once.

The manifest records at least:

- the protocol version and the SHA-256 of this file;
- the SHA-256 of `splits.py`, `synthetic_catalog.py`, `generator.py`, `audit.py`, `policy_sim.py` and
  `policy_diagnostics.py`, of the mapping `request_to_context-v1`, of the arm, runner, budget-rule, metrics, bootstrap
  and manifest code, and of every production source file used;
- generator, catalog, template and simulated-catalog versions (`simpol-1` and its SHA-256), and the dataset,
  split and bootstrap seeds, including the reserve-seed log (12.5);
- the generator `fingerprint`, the split `manifest_sha256`, the `dataset_fingerprint`, the actual template and record
  counts and the exclusion counts;
- the training-split Arm B length statistics and the selected `max_length` (12.8);
- the hyperparameter grids, selection criterion, threshold rule, Arm C settings (12.10), the decision rule and any
  margin (12.11);
- the Python and numpy versions.

#### 12.12.1 Arm builders `arms-1` (implemented in D4-4)

`tokenization_ai/heldout_eval/arms.py` prepares, for the same requests, the inputs of the three arms. It fits, predicts with and scores no
detection model, reads no label (`record.meta` is never touched; the future runner aligns labels to `record_ids`) and has not been run on any
confirmatory data.

- **Stage flow (real stages only).** For each record: `request_to_context-v1`; `InputFusion`; `PrivacyMasker.mask_with_provenance`;
  `Normalizer.normalize_with_provenance`; `HybridTokenizer.tokenize`. **Arm A** fuses an *empty* rule list. **Arm B** fuses the `rule_info` of the
  unchanged `simpol-1` simulator; the `SimulatedRuleSet` wrapper is kept beside the sequence and re-checked (`provenance="simulated"`,
  `verification_status="not_verified_simulated"`, catalog version `simpol-1`, catalog SHA-256), and a mismatch stops the build. **Arm C** passes the
  actual `NormalizationResult` of the rule-free Arm A front end to `armc-char3-v1`; no result object is built by hand and the encoder's privacy
  boundary is never bypassed. Arm C never receives rules and has no vocabulary. A record's three representations always describe the same
  request.
- **Split identity and pairing.** A split (`train`, `validation` or `test`) is named by the caller, never inferred; a split is a non-empty list of
  `EvalRecord` with unique ids and is processed sorted by record id, so input order cannot change a value. `ArmInputs` aligns the three arms by
  `record_ids` and refuses (`PairingError`) missing, extra, duplicate or reordered records and Arm A or B inputs not at the shared `max_length`;
  `assert_disjoint` refuses a record id in two splits. Any construction failure raises (`ArmBuildError`) for the whole split; no record is skipped,
  so class balance and sample counts cannot change silently. Errors and diagnostics hold record ids, stage names and counts only, never request content.
- **Train-only shared vocabulary.** One `Vocabulary`, fitted once, on the **training** split only, on the training **Arm B** sequences. Arm B contains
  every Arm A token (Arm A differs only by lacking rule tokens), so a single fit counts each request once and serves both arms; fitting on A and B
  together would count every shared literal twice and make `min_frequency` ineffective. Validation and test are encoded with the frozen
  vocabulary and never refit. Reserved ids and the order of literal ids are deterministic (by count, then token). A literal absent from the frozen
  vocabulary encodes to the unknown id; `unknown_token_report` gives per-arm counts. With `min_frequency=2` the training split itself contains
  unknown tokens, whereas with `min_frequency=1` it contains none (so the unknown id would never be trained). **`min_frequency=2` is the proposed
  experimental setting and is not frozen** (`MIN_FREQUENCY_PROPOSED`); the vocabulary record (size, SHA-256 of the id-ordered token list,
  parameters, training record count) is for the manifest.
- **Shared budget and exclusions.** The caller selects one `BudgetDecision` with `budget-rule-1` from `training_arm_b_lengths`; the builder never
  chooses a length itself. The same `max_length` is applied to Arm A and Arm B with the real `Tensorizer`. Records that need payload truncation are
  allowed and counted per arm (`ArmInputs.summary`). A record whose framing plus rules cannot fit the frozen budget makes `build_arm_inputs` fail
  closed with `PairedExclusionRequired`; its `ExclusionProposal` lists record ids, arms and a reason only. `propose_exclusions` returns the same
  proposal without raising. Nothing is excluded, truncated or re-budgeted automatically: removing such records from all arms, and every
  truncation rule, must be decided and frozen in the protocol before the final evaluation.
- **Model preparation.** `new_untrained_model(num_buckets)` returns a fresh, untrained `LightweightAnomalyModel`; every hyperparameter setting needs
  its own instance (`fit` warm-starts from existing weights). No search or runner exists.
- **Provenance (`build_provenance`).** JSON-safe: arm identifiers, `request_to_context-v1`, `armc-char3-v1` parameters, `synthgen-3`,
  `synthetic-catalog-3`, the `simpol-1` version, SHA-256 and explicit not-verified status, the budget decision, the vocabulary record, paired
  record counts per split and the SHA-256 (LF-normalized) of the stage and arm sources it relies on (`fusion`, `masking`, `normalization`,
  `tokenizer`, `vocabulary`, `tensorizer`, `models`, `model`, `request_context`, `arm_c`, `token_budget`, `policy_sim`, `arms`). Arm B rules are
  never described as verified.
- **Limitations (no performance result exists).** Arm B adds deterministic, request-derived simulated representations, not independently verified
  information. Arm C observes query and value characters that Arms A and B lose through atomic tokens and the unknown token. Feature mass and
  collisions differ by arm, and a shared model class and `num_buckets` do not give equal information. The vocabulary, the budget, the exclusion and
  truncation rules and `min_frequency` are not frozen until the manifest exists. On the development seed the builders were exercised only for
  preparation (preprocessing counts), never for scoring.

### 12.13 What simulated rules and these results cannot show

Simulated policy tokens cannot prove the effectiveness of genuine Member 2 verified rules. Every `simpol-1` rule is a
deterministic function of the request, so a difference between Arm A and Arm B can only show a *representation*
effect on a hand-built catalog; "no difference" is acceptable. Arm B must be labelled **synthetic or simulated rules**
wherever it is reported. Results are synthetic held-out performance on the author's scenario families. They cannot
support claims of real-world attack-detection quality, performance on real traffic, Member 2 or Member 3 integration,
KPI 5, or calibrated probabilities (section 9).

## 13. Chronological record

This record separates what was found in exploratory data, what was revised in the design, what was decided before
the confirmatory run, and what confirmatory evidence exists. It holds no performance figures.

| Order | Stage | Kind | Record |
|---|---|---|---|
| 1 | Before D3a | Pre-run decision | Protocol draft `1.0.0-draft`: leakage-safe group-aware splitting (`splits.py`); SR-BH later deferred (`1.1.0-draft`). |
| 2 | D3a | Exploratory finding | `synthgen-1` pilot (1,487 requests) produced six descriptive audit warnings. No model was run. |
| 3 | D3a-R | Design revision | The warnings were investigated and the generator was refined to `synthgen-2` (section 2.6). Post hoc relative to the first pilot. |
| 4 | D3b | Pre-run decision | Simulated catalog `simpol-1` written; no condition refers to a label (section 3.1). |
| 5 | D3b | Exploratory finding | Diagnostics on the `synthgen-2` pilot (seed 5101): SIM-P04, SIM-P06 and SIM-P07 aligned with one class; SIM-P12 had zero coverage; Arm B was truncated at `max_length` 128 and Arm A was not (section 3.1). No model was run. |
| 6 | D3c | Design finding (code review, no data labels) | Review of the generator and tokenizer: a query string stays inside one atomic path token (12.9). |
| 7 | D3c-1 | Design revision (recorded) | `synthgen-3` with G1 and G2 approved (12.2); `simpol-1` kept (12.3); pilot marked exploratory and burned (12.4). |
| 8 | D3c-1 | Pre-run decision (recorded) | Seed plan (12.5), splitting and sealed test (12.6), scale (12.7), `max_length` rule (12.8), `request_to_context-v1` (12.9), Arm C definition (12.10), metric plan (12.11), manifest plan (12.12). |
| 9 | D3c-2 | Design revision (implemented) | G1/G2 implementation note written before any generator code was changed; `synthgen-3` / `synthetic-catalog-3` implemented; default dataset seed set to the development seed 6101; checked on the development seed only, structure and construction checks, no model (12.2.1). Structural before/after figures were read from the development seed after the design constants were fixed; the constants were not adjusted. |
| 9a | D4-0 | Design revision (diagnostics only) | `structural_coverage` added to the audit report and the audit protocol label corrected (12.2.2). Development seed 6101 only. |
| 9b | D4-5 | Design revision (implemented) | Metric and paired group-bootstrap functions, threshold rule and minimum-valid-replicate rule written (12.11.1) and tested on hand-made arrays. No model score, no data, no result. |
| 9c | D4-6B / D4-6C1 | Exploratory finding, then pre-run decision (recorded) | Label-free feature-scale diagnostic on development seed 6101 (training split, no model); arm-relative learning rates, 27-candidate grid, validation-AP selection with tie-breaks and raw-logit ranking recorded as accepted draft decisions (12.11.2). No code, no score. |
| 10 | Pending | Pre-run decision | Arm C compatibility check and feature-budget policy; decision rule and any margin; exclusion and truncation policy; recovery policy; runner, estimator, logit extraction and manifest code. |
| 11 | Pending | Pre-run decision | Write-once pre-run manifest, before any training or scoring. |
| 12 | Pending | Confirmatory evidence | Confirmatory dataset (seed 7101), validation-only selection, one scored test run. **No confirmatory evidence exists yet.** |
