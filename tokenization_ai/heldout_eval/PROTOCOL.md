# Held-out evaluation protocol (Member 4)

**Protocol version:** `heldout-eval-1.1.0-draft`  **Status: DRAFT, not frozen.**
**Template version:** `tmpl-1`  **Generator:** `synthgen-2`  **Catalog:** `synthetic-catalog-2`
**Code (this folder):** `splits.py`, `synthetic_catalog.py`, `generator.py`, `audit.py`
**Tests:** `tests/tokenization_ai/test_heldout_splits.py`, `test_synthetic_catalog.py`,
`test_synthetic_generator.py`, `test_heldout_audit.py`

## Change history

| Version | Change |
|---|---|
| `1.0.0-draft` | First draft: SR-BH as the candidate dataset; leakage-safe group splitting (`splits.py`). |
| `1.1.0-draft` | **Synthetic API requests are the primary evaluation source.** SR-BH is deferred and not downloaded. Added the scenario catalog, deterministic generator and non-model audit, and a pilot stage. The comparison arms, primary metric, splitting rules and leakage controls are unchanged. The `1.0.0-draft` text remains in git history. |
| `1.1.0-draft` (D3a-R refinement, same draft version) | **Dataset quality refinement after reviewing the first pilot's six descriptive warnings** (generator `synthgen-1` -> `synthgen-2`, catalog `-1` -> `-2`). See section 2.6 for what was changed and why, section 2.7 for the intentional associations that remain, and section 4.1 for the pilot record. The audit gained a template-supported exclusivity measure and an independent content-type semantics check. Splitting code is unchanged. |

This protocol is **separate from** the archived KPI 1-4 evaluation (`../evaluation/`, its protocols,
fixtures, manifests and reports). Nothing here re-runs, replaces or reinterprets that evidence, which
stays tied to its original protocol. Nothing in this folder changes the production pipeline, the
prediction API or the dashboard.

> **Scope of this step:** the protocol text, the leakage-safe splitting code, the synthetic scenario
> catalog, the deterministic generator and the non-model audit exist. There is **no model training,
> metrics runner, rule simulator or A/B/C runner yet, and no accuracy, PR-AUC or other
> detection-performance figure exists.** The draft becomes a frozen protocol only after the pilot
> (section 4) is reviewed and the open items in section 11 are decided.

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
| **B** | Request + **simulated** rule tokens | Rules come from a deterministic function of request-observable fields only (for example method, path prefix, content type, header presence), from a catalog written and frozen **before** labels are examined, with neutral policy ids. Labels, scenario families and detector output never enter rule construction. **Not implemented yet.** |
| **C** | Conventional HTTP preprocessing baseline | A standard, non-structure-aware preprocessing of the same requests (for example hashed character n-grams), same classifier and bucket sizes. **Not implemented yet.** |

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

## 4. Pilot stage (before the full study)

1. **Pilot generation:** about **250 distinct normalized template groups** and about **1,500 requests**
   (default config), subject to what normalization actually allows.
2. **Non-model checks only** (`audit.py`): actual group and request counts, class proportions at record
   *and* template level, mixed-label templates, group-size distribution, duplicate counts, feasibility of
   the group-aware split, label/family-term scan, feature/metadata separation, shared pools, determinism,
   split isolation, and descriptive shortcut statistics (section 6).
3. **Review.** No model is trained, tuned or scored during the pilot, and no detection-performance
   number is computed.
4. **Full study** (proposed about 2,000 templates and about 12,000 records, to be confirmed by review) only
   after the pilot is reviewed, the generator is revised if the audit shows problems, and this protocol
   is frozen with a pre-run manifest (section 10). A generator change after freezing requires a new
   generator and protocol version.

### 4.1 Pilot record (non-model; counts and hashes only)

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
  values fixed before any data existed and not tuned.

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
  before the scored run.

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
`fingerprint`, split `manifest_sha256` and `dataset_fingerprint`; the frozen rule catalog hash; and the
SHA-256 of every production source file used. The new modules are outside the archived KPI source-hash
inventory, so the archived reports do not cover them. Archived KPI 1-4 fixtures, protocols, hashes and
reports must not be modified or regenerated.

## 11. Open items before this draft can be frozen

1. Review of the pilot statistics and audit warnings; revise the generator under a new version if needed.
2. The decision rule and margin for A versus B (section 7), and the shortcut-flag handling.
3. Whether Arm C and a flat-versus-structure-aware tokenization ablation are included.
4. Whether to add a leave-family-out split (an explicit group key would be needed in `splits.py`).
5. `max_length` and header handling identical across arms, fixed from training-split length statistics only.
6. Approval of the simulated-rule catalog before labels are examined.
7. Final scale of the full study (templates and records) and prevalence sensitivities.
