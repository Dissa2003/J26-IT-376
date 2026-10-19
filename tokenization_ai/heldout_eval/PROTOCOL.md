# Held-out evaluation protocol (Member 4)

**Protocol version:** `heldout-eval-1.0.0-draft`  **Status: DRAFT, not frozen.**
**Template version:** `tmpl-1`  **Code:** `splits.py` (this folder)  **Tests:** `tests/tokenization_ai/test_heldout_splits.py`

This protocol is **separate from** the archived KPI 1-4 evaluation (`../evaluation/`, its protocols,
fixtures, manifests and reports). Nothing here re-runs, replaces or reinterprets that evidence, which
stays tied to its original protocol. Nothing in this folder changes the production pipeline, the
prediction API or the dashboard.

> **Scope of this step:** only the protocol text and the leakage-safe splitting code exist. There is
> **no model training, metrics runner, rule simulator or comparison experiment yet, no dataset has been
> downloaded, and no accuracy or performance figure exists.** The draft becomes a frozen protocol only
> after the open items in section 10 are decided and a pre-run manifest is written (section 9).

## 1. Purpose

Evaluate Member 4's lightweight predictor on **held-out data** and test whether adding structure-aware
rule tokens helps compared with request-only input, with leakage controls strong enough that a result
can be defended. The proposal specifies no dataset size, split or classifier and sets no detection
target (proposal section 3.2, 3.3, Appendix 1); this protocol defines them and **introduces no
detection-accuracy success target**.

## 2. Dataset

- **Candidate:** SR-BH 2020, Harvard Dataverse, DOI `10.7910/DVN/OGOIXX` (real HTTP requests from one
  WordPress honeypot, ModSecurity-logged and manually CAPEC-labelled, per the adapter documentation in
  `../dataset_adapters.py`).
- **Availability: NOT yet downloaded and NOT locally verified.** The licence, schema, row counts and
  label semantics quoted in the adapter docstring are unverified claims until checked against the
  source file.
- **Use is conditional** on (a) verification of the source file (header, licence, hashes, row and class
  counts) and (b) applicable supervisor and institutional approval. If either fails, the study cannot
  claim real-data results and any run is feasibility-only (section 8).
- **Provenance to record before the scored run:** DOI, retrieval date, file name and SHA-256, row count,
  class counts, and the `dataset_fingerprint` produced by `splits.py`.
- **Scope limit:** one honeypot over a short period. Results do not transfer to generic API traffic.
  Label noise is unknown until inspected.

## 3. Comparisons (identical examples, identical splits)

| Arm | Input | Notes |
|---|---|---|
| **A** | Request-only features | The existing pipeline with an empty rule list. This path is supported today: `service.analyze` accepts `rules=[]` and emits `[RULES_START] [RULES_END]` framing (checked by a test; no blocker found). |
| **B** | Request + **simulated** rule tokens | Rules are produced by a deterministic function of request-observable fields only, from a catalog written and frozen **before** anyone looks at labels. Labels, response fields and detector output never enter rule construction. Not implemented yet. |
| **C** | Conventional HTTP preprocessing baseline | A standard, non-structure-aware preprocessing of the same requests (for example hashed character n-grams), per proposal section 3.3. Not implemented yet. |

All arms use the same model family and the same hyperparameter grid with an equal tuning budget. The
vocabulary is fitted on the training split only, separately per arm.

**Simulated rules do not prove the value of genuine Member 2 verified rules.** A rule derived
deterministically from the request adds no information beyond the request; any difference between A and
B can only show a representation effect on a hand-built catalog. No Member 2 integration exists, and
Arm B must be labelled *synthetic or simulated rules* wherever it is reported.

Known confound to control: rule tokens use part of the fixed token budget, so Arm B has less payload
space. Truncation rates per arm must be reported, and `max_length` must be treated identically across
arms. Request framing (method, path, headers) is mandatory in the tensor and a request whose framing
alone exceeds `max_length` is rejected, not truncated; real requests with many headers need a
pre-registered `max_length` or header policy.

## 4. Splitting (implemented in `splits.py`)

**Primary: group-aware train / validation / test** (default ratios 0.70 / 0.15 / 0.15).
**Secondary robustness check: time-forward** split, where timestamps exist: train precedes validation
precedes test.

- **Unit of grouping:** the normalized request *template* (`tmpl-1`): upper-cased method; decoded,
  lower-cased path with repeated slashes and trailing slash removed; digits, hex strings and UUIDs
  replaced by placeholders; query parameters sorted with values reduced to a class (`<empty>`, `<n>`,
  `<v>`); body reduced to its structural shape (JSON key/type shape, form key/class shape, or a length
  bucket for raw text). Requests that differ only in values - including attack-payload variants - share
  a template and therefore a split. The group id is the first 16 hex digits of the SHA-256 of
  `tmpl-1` plus the template. Templates are label-independent.
- **Whole groups only:** a group never spans splits. The assignment is a seeded greedy fit of group
  sizes and class counts to the requested ratios.
- **Exact duplicates** (same method, path, body and headers): always in one split (they share a
  template); the isolation check verifies this independently. Optionally dropped (first by `record_id`
  kept, others listed as excluded).
- **Label conflicts:** exact duplicates with different labels are label noise and are an **error** by
  default (or excluded and reported with `exact_conflict_policy="exclude"`). Templates that legitimately
  contain both classes (same shape, different values) are **reported** as `mixed_label_templates` and
  kept together.
- **Source identity is optional and never the only key.** `merge_by_source=True` additionally merges
  groups that share a source id. SR-BH's collection environment may obscure original sources, so the
  default grouping does not rely on them.
- **Time-forward:** groups ordered by their latest timestamp and cut by cumulative record count; a group
  spanning a boundary goes wholly to the later split (reported as `straddling_groups`). Every record
  needs a timestamp. If timestamps are missing or unusable the check is not run and that is stated.
- **Infeasible data fails explicitly** (`SplitInfeasibleError`): fewer than three groups, a split missing
  a class (default at least one record of each class per split, configurable), or achieved sizes
  deviating from the requested ratios by more than the tolerance (default 0.10) because of large
  groups. `strict=False` returns the result with the problems listed instead; isolation is never
  relaxed in either mode.
- **Reproducibility:** fixed seeds, recorded here. Primary split seed `4401`; robustness seeds
  `4402, 4403, 4404, 4405`. These are arbitrary values fixed before any data exists and are not tuned.
  The manifest hash is identical for identical records, parameters and seed.

## 5. Leakage controls

1. Labels, response status, ModSecurity decisions and attack annotations are **never** model features
   and never enter templates, group ids or fingerprints. Model-facing content (`RequestFeatures`:
   method, path, body, headers) and split metadata (`SplitMetadata`: label, timestamp, source id) are
   separate types; the feature type has no field for any of them. Labels are used only to balance the
   split and to check classes.
2. Splitting metadata (timestamps, sources) is read from the raw rows by the splitting code itself,
   not through the model-facing adapter, which deliberately denylists them.
3. The vocabulary, hyperparameters, `num_buckets`, epochs, learning rate, class weights and the decision
   threshold are fitted or chosen on **training and validation data only**. Model and threshold
   selection use validation data only.
4. **The held-out test split is used only for final scoring, once**, after the pre-run manifest. No
   re-tuning after seeing test results; any change requires a new protocol version.
5. Any subsampling for size happens before splitting, under a recorded seed.
6. Provenance records hold caller-supplied opaque `record_id`s, hashed group ids, counts and hashes -
   no raw request text. Record ids must not contain request content.

## 6. Metrics (defined; the metrics runner is not implemented yet)

- **Primary: PR-AUC** (threshold-free; the score is uncalibrated, and positives are expected to be a
  minority in honeypot data).
- **Secondary:** ROC-AUC; precision, recall, F1, false-positive rate, false-negative rate and the
  confusion matrix at one threshold **chosen on validation only**; per-CAPEC-category recall with small
  categories flagged as unreliable; false-positive and false-negative review by template group.
- **Uncertainty:** cluster (group) bootstrap, with paired differences between arms on the same test
  items.
- **Latency:** secondary only, under its own new protocol; the archived KPI 4 result is not re-run or
  compared.
- **Score semantics:** the model score is uncalibrated and is not an attack probability. A reliability
  analysis would be needed before any calibration claim.
- **No numeric success target** is set. A decision rule for "B improves on A" (for example a paired
  grouped-bootstrap interval for the PR-AUC difference excluding zero, and its margin) must be
  pre-registered by the author before the scored run.

## 7. Retraining

The shipped demo model and vocabulary were fitted on two hand-written examples and are **not** used.
Each arm is fitted from scratch on the training split. These artifacts live outside the served path;
the API and dashboard keep their current demo model.

## 8. What each outcome can and cannot support

- **Can support a research claim** (limited to the data's scope): held-out detection quality of the
  request-only model on verified real data, with group-aware splits, validation-only selection and
  intervals.
- **Feasibility only:** anything on synthetic data; the A-versus-B comparison with simulated rules.
- **Cannot be claimed from this study:** the value of genuine Member 2 verified rules; Member 2 or
  Member 3 integration; KPI 5; calibrated probabilities; performance on general API traffic.

## 9. Provenance and freezing

Before the scored run a pre-run manifest must be written, once and never overwritten, recording: protocol
version and the hash of this file, `splits.py` hash, template version, seeds, split `manifest_sha256`
and `dataset_fingerprint`, dataset file hash, and SHA-256 of every production source file used (including
`adapters.py`, `anomaly_signal.py` or `policy_report_adapter.py` if used). The new modules are outside
the archived KPI source-hash inventory, so the archived reports do not cover them. Archived KPI 1-4
fixtures, protocols, hashes and reports must not be modified or regenerated.

## 10. Open items before this draft can be frozen

1. Obtain and verify SR-BH (or choose another dataset); confirm licence and supervisor/institutional approval.
2. The decision rule and margin for A versus B (section 6).
3. Whether Arm C and a flat-versus-structure-aware tokenization ablation are included.
4. Whether the time-forward check is possible (timestamps usable and class-balanced across time).
5. `max_length` and header handling identical across arms.
6. Approval of the simulated-rule catalog before labels are examined.
7. Whether the `tmpl-1` normalization is too coarse or too fine on the real data; change only by a new
   template version, decided before looking at model results.
