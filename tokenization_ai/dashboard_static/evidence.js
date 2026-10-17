/* Documented LogiShield research evaluation evidence (KPIs 1-5).
 *
 * Static, frozen facts. Nothing here is measured by the dashboard and nothing
 * is derived from the live session. Each value is copied from a named
 * evidence file or protocol, never rounded up and never invented:
 *
 *   KPI 1-3  KPI_Evaluation/kpi_report.json (+ kpi_report.md), protocol 1.0.0
 *   KPI 4    KPI_Evaluation/Post_D3_Latency/latency_report.json (latest),
 *            KPI_Evaluation/latency_report.json (pre-D3 baseline), latency protocol 1.0.0
 *   KPI 5    tokenization_ai/INTEGRATION_STATUS.md section 4 (not measured)
 *
 * The evidence folder lives outside the repository on the author's machine
 * (see tokenization_ai/README.md section 8); the repository copies of the
 * numbers are in README.md. If a value here ever disagrees with those
 * sources, the tests fail rather than the page quietly showing it. */

export const EVIDENCE_FOLDER = "LogiShield_PP1_Coding_Evidence\\KPI_Evaluation\\";

export const EVIDENCE_INTEGRITY = [
  ["KPI 1–3 protocol", "1.0.0 (tokenization_ai/evaluation/PROTOCOL.md), fixtures frozen by a pre-run manifest before scoring"],
  ["KPI 1–3 results hash", "f353779dcad0da96… (identical in the original run and in a re-run on the current pipeline code)"],
  ["KPI 1–3 code version", "Original report: masking.py sha256 d23ae1f9… (before the privacy-masking fix). Re-run on masking.py 8eb0f66a… gave the same results hash; this shows reproducibility, not a measurement of the new masking rules."],
  ["KPI 4 protocol", "1.0.0 (tokenization_ai/evaluation/LATENCY_PROTOCOL.md), protocol-conformant runs"],
  ["KPI 4 code version", "Latest run: git commit ea50a49 (masking.py 8eb0f66a…), clean tree. Earlier baseline: pre-fix code (masking.py d23ae1f9…), kept for comparison."],
  ["Evidence status", "No missing or contradictory source was found when this page was written. KPI 5 has no result to show."]
];

/* Kinds: "interval" (point estimate and Wilson interval vs target), "latency" (mean vs ms target), "pending". */
export const KPIS = [
  {
    id: "kpi1",
    kind: "interval",
    title: "KPI 1 · Input alignment",
    question: "Are the API context and the rule information fused into one aligned record for the same event?",
    target: "≥ 90%",
    targetValue: 90,
    result: "31 / 31 valid events (100.00%)",
    point: 100,
    interval: [88.97, 100.0],
    sample: "31 valid synthetic events (17 curated, 12 seeded, 2 designed). A further 5 invalid events were all rejected (5 / 5); that is reported separately and is not part of the percentage.",
    verdict: ["Point estimate meets the target on this synthetic set; target not statistically demonstrated", "mock"],
    context: [
      "Method: output compared with independently written expected outputs (hand-written fixtures plus a separate reference encoder).",
      "Scope: specification coverage of hand-designed scenarios, protocol 1.0.0."
    ],
    limits: [
      "With N = 31 even a perfect score has a Wilson 95% lower bound of 89.0%, below the 90% target.",
      "Synthetic, small-N and same-author oracle: it detects regressions and deviations from the written design, not a flawed design.",
      "Not real-world traffic, not detection accuracy, not Formal Verification or Runtime Verification integration; rules are mock fixtures."
    ],
    refs: ["kpi_report.json → results.kpi1_input_alignment", "tokenization_ai/evaluation/PROTOCOL.md §4", "tokenization_ai/README.md §8"]
  },
  {
    id: "kpi2",
    kind: "interval",
    title: "KPI 2 · Structure-aware token mapping",
    question: "Does tokenization emit the specified sequence, with the correct authoritative role for each token?",
    target: "≥ 90%",
    targetValue: 90,
    result: "31 / 31 sequences matched exactly (100.00%)",
    point: 100,
    interval: [88.97, 100.0],
    sample: "31 valid synthetic events. Token level: 3719 / 3719 positions matched in text and role; masking provenance exact for 31 / 31 events.",
    verdict: ["Point estimate meets the target on this synthetic set; target not statistically demonstrated", "mock"],
    context: [
      "Method: exact comparison with expected token sequences; role correctness, masking provenance and grammar conformance are reported as separate secondary measures.",
      "Masking provenance is measured against hand-curated sensitive locations (21 of 21 paths, no false positives)."
    ],
    limits: [
      "Same N = 31 limitation: Wilson 95% lower bound 89.0% is below the 90% target.",
      "Masking provenance says nothing about unlisted sensitive data; masking is a name- and shape-based blocklist.",
      "Synthetic fixtures only; the anomaly model is not exercised by KPIs 1–3."
    ],
    refs: ["kpi_report.json → results.kpi2_token_mapping", "tokenization_ai/evaluation/PROTOCOL.md §4", "tokenization_ai/README.md §8"]
  },
  {
    id: "kpi3",
    kind: "interval",
    title: "KPI 3 · Aligned tensor correctness",
    question: "Are fixed-length tensors, attention masks and rule-preserving truncation produced as specified?",
    target: "≥ 95%",
    targetValue: 95,
    result: "30 / 30 tensors correct (100.00%)",
    point: 100,
    interval: [88.65, 100.0],
    sample: "30 tensors (21 padded, 0 exact-fit, 9 truncated), fixed length 128, vocabulary 395 fitted on a development set only. One over-capacity case was rejected as specified (1 / 1).",
    verdict: ["Point estimate meets the target on this synthetic set; target not statistically demonstrated", "mock"],
    context: [
      "Method: independent expected-token sequences fed to the tensorizer, plus a secondary pipeline-consistency check (30 / 30).",
      "Unknown-token rate on non-truncated literals is 40.41% (99 / 245), reported separately: it describes the small development vocabulary, not real traffic."
    ],
    limits: [
      "N = 30: Wilson 95% lower bound 88.7% is below the 95% target.",
      "No exact-fit cases were included, and only one tensor configuration (128) was tested.",
      "Synthetic fixtures; not a validated population result."
    ],
    refs: ["kpi_report.json → results.kpi3_aligned_tensor_correctness", "tokenization_ai/evaluation/PROTOCOL.md §4", "tokenization_ai/README.md §8"]
  },
  {
    id: "kpi4",
    kind: "latency",
    title: "KPI 4 · Average pipeline prediction time",
    question: "How long does the in-process LogiShield pipeline take, on average, from fusion to anomaly score?",
    target: "≤ 5 ms",
    targetValue: 5,
    result: "0.3413 ms pooled mean (worst batch mean 0.3481 ms)",
    mean: 0.3413,
    worst: 0.3481,
    baseline: "Earlier baseline before the masking fix: 0.3200 ms (worst batch 0.3371 ms).",
    sample: "5 batches × 1000 calls = 5000 timed calls on the mixed synthetic workload S6 (30 valid cases, 14–655 tokens before truncation), after warm-up.",
    verdict: ["Within the 5 ms target in this run on this machine and workload (single run)", "mock"],
    context: [
      "Boundary B1: in-process pipeline (fusion, masking, normalization, tokenization, tensorization, model scoring), no tracing, no HTTP.",
      "Machine: Windows 10, 8 logical CPUs, Python 3.14.7, 'Ultimate Performance' power scheme; GC disabled in timed loops; commit ea50a49."
    ],
    limits: [
      "Not production API latency, not an SLA, and not isolated model-inference time (model scoring alone is about 0.075 ms mean in the informational breakdown).",
      "Single machine and single run on a synthetic workload; other processes were not controlled.",
      "Not comparable with the per-request timings shown for the current browser session."
    ],
    refs: ["Post_D3_Latency/latency_report.json → primary", "latency_report.json (pre-fix baseline)", "tokenization_ai/evaluation/LATENCY_PROTOCOL.md", "tokenization_ai/README.md §8"]
  },
  {
    id: "kpi5",
    kind: "pending",
    title: "KPI 5 · Downstream usability",
    question: "Is the anomaly signal usable by the next verification component (Runtime Verification)?",
    target: "≥ 90%",
    result: "Not measured",
    sample: "No measurement exists.",
    verdict: ["Pending genuine Runtime Verification integration", "pending"],
    context: [
      "It needs a documented Runtime Verification input contract and a running consumer. Neither exists in the repository."
    ],
    limits: [
      "Any percentage computed today would be measured against a contract LogiShield invented, i.e. a self-assessment, so none is reported.",
      "The adapter to Runtime Verification raises NotImplementedError by design."
    ],
    refs: ["tokenization_ai/INTEGRATION_STATUS.md §3–§4", "tokenization_ai/CONTRACT_PROPOSAL.md"]
  }
];

/* KPI 4, other boundaries, S6 mixed workload pooled means in ms (informational, latest run). */
export const LATENCY_BOUNDARIES = [
  ["B1", "In-process pipeline (the KPI)", "0.3413", "primary"],
  ["B1T", "B1 with the production tracer", "0.3605", ""],
  ["B2", "Adapters + pipeline", "0.3580", ""],
  ["B3", "Real HTTP POST /v1/analyze on loopback (client, JSON and projection included)", "2.7007", "p95 4.4582, p99 4.8902"]
];

export const LATENCY_NOTES = [
  "In the B3 large-payload stratum (S4) the p95 was 5.1237 ms and one 37.2 ms outlier occurred, so the 5 ms figure is a mean for B1 only, not a tail-latency guarantee.",
  "B3 runs client and server in one process on loopback and is reported for transparency; the pre-registered KPI is B1."
];

export const METHODOLOGY = [
  "The development and evaluation fixtures are synthetic and hand-designed; no real traffic, users or credentials were used.",
  "KPI 1–3 percentages are the share of designed scenarios handled as specified. They are not detection accuracy, precision, recall or F1.",
  "A perfect score on 30–31 cases cannot statistically demonstrate a 90–95% population target; the Wilson 95% intervals are shown for that reason.",
  "The lightweight model is fitted on two synthetic fixtures. Its anomaly score is uncalibrated: not a probability, not a verdict, not an Allow or Block decision.",
  "KPI 4 is an in-process timing benchmark on one machine and workload. It is not production latency or an SLA.",
  "Live session figures above are single observations from this browser session and are not part of any KPI.",
  "Formal Verification and Runtime Verification integration is incomplete, there is no final Runtime Verification hand-off, and KPI 5 has not been measured."
];
