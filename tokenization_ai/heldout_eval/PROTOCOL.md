# Held-out evaluation protocol (Member 4)

**Protocol version:** `heldout-eval-1.2.0-draft`  **Status: DRAFT, not frozen.**
**Template version:** `tmpl-1`  **Generator (implemented):** `synthgen-2`  **Catalog (implemented):** `synthetic-catalog-2`
**Generator (approved, not implemented):** `synthgen-3` (section 12.2)
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

This protocol is **separate from** the archived KPI 1-4 evaluation (`../evaluation/`, its protocols,
fixtures, manifests and reports). Nothing here re-runs, replaces or reinterprets that evidence, which
stays tied to its original protocol. Nothing in this folder changes the production pipeline, the
prediction API or the dashboard.

> **Scope of this step:** the protocol text, the leakage-safe splitting code, the synthetic scenario
> catalog, the deterministic generator (`synthgen-2`), the non-model audit and the simulated rule catalog
> exist. There is **no model training, metrics runner or A/B/C runner yet, and no accuracy, PR-AUC or other
> detection-performance figure exists.** `synthgen-3`, the Arm C code, the runner, the budget-rule code and the
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

## 7. Metrics (defined; the metrics runner is not implemented yet)

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
| 1 | Review of the pilot statistics and audit warnings; revise the generator under a new version | **Decided in principle** (section 12.2: `synthgen-3`, corrections G1 and G2). Design note and implementation pending. |
| 2 | Decision rule and margin for A versus B; shortcut-flag handling | **Open.** Comparison plan recorded (12.11); the rule and any margin are fixed before the confirmatory scored run. |
| 3 | Arm C | **Included** (12.10). Model-class compatibility and the feature-budget policy are **open** until checked before coding. A flat-versus-structure-aware tokenization ablation is **not included** and remains open. |
| 4 | Leave-family-out split | **Open**, not decided. |
| 5 | `max_length` identical across arms from training-split statistics only | **Rule decided** (12.8); the value is computed later from the confirmatory training split. Header handling is identical across arms by sharing `request_to_context-v1` (12.9). |
| 6 | Review of `simpol-1` and its diagnostics | **Resolved:** `simpol-1` is unchanged; SIM-P06 and SIM-P12 are documented limitations (12.3); SIM-P04 and SIM-P07 are addressed in the generator (12.2), not in the catalog. |
| 7 | Final scale and prevalence sensitivities | Scale **proposed** (12.7, subject to feasibility). Prevalence sensitivities are **open**. |
| 8 | Arm C feature-budget policy, n-gram settings and equal hyperparameter-search effort | **Open** (12.10). |
| 9 | Hyperparameter grids, selection criterion, threshold rule, bootstrap settings (resamples, seed) | **Open**; to be fixed before the manifest (12.11, 12.12). |
| 10 | Whether `audit.py` gains path-root and method-by-path-category summaries | **Proposed, not approved.** Needs a separate approval before any code change. |
| 11 | Scope of label-conditioned audits relative to the sealed test split | **Open** (12.6). |
| 12 | Whether development-seed (6101) dry runs report any detection metric | **Open.** Until decided, none is reported (12.5). |
| 13 | Runner, arm, budget-rule and manifest code; the actual manifest | **Not implemented** (12.12). |
| 14 | Any generator correction beyond G1 and G2 | **Not approved.** Needs a documented reason and approval before the generator is fixed for the confirmatory study (12.2). |

## 12. Design decisions recorded in D3c-1 (documentation only)

These decisions were approved after the D3c read-only design review. They are recorded **before** the generator is
changed and before any model evaluation code exists. They record intent; none is implemented by this step, and none
of them freezes the protocol (section 12.12).

### 12.1 Version and history

The protocol is `heldout-eval-1.2.0-draft`. The version history above is retained unchanged; this step adds one row.

### 12.2 Generator `synthgen-3`: two planned corrections

`synthgen-3` is approved for implementation in a later step, with exactly these initial planned corrections. Their
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
- Splitting the query out of the path would be a different mapping. It would need a new mapping version
  (`request_to_context-v2`) and a new protocol version; it is not done in this study.

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
| 7 | D3c-1 | Design revision (recorded, not yet implemented) | `synthgen-3` with G1 and G2 (12.2); `simpol-1` kept (12.3); pilot marked exploratory and burned (12.4). |
| 8 | D3c-1 | Pre-run decision (recorded) | Seed plan (12.5), splitting and sealed test (12.6), scale (12.7), `max_length` rule (12.8), `request_to_context-v1` (12.9), Arm C definition (12.10), metric plan (12.11), manifest plan (12.12). |
| 9 | Pending | Design revision | `synthgen-3` design note, then implementation. |
| 10 | Pending | Pre-run decision | Arm C compatibility check and feature-budget policy; hyperparameter grids; decision rule and any margin; runner, budget-rule and manifest code. |
| 11 | Pending | Pre-run decision | Write-once pre-run manifest, before any training or scoring. |
| 12 | Pending | Confirmatory evidence | Confirmatory dataset (seed 7101), validation-only selection, one scored test run. **No confirmatory evidence exists yet.** |
