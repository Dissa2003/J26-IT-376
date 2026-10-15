/* Route table. "anomaly" is the one functional page in Phase 1; every other
 * page is an honest pending state that says what is required before it can
 * show real data. No numbers, rows or events are invented here. */

export const DEFAULT_ROUTE = "anomaly";

export const ROUTES = [
  {
    id: "dashboard", label: "Dashboard", icon: "home", kind: "pending",
    title: "Dashboard",
    subtitle: "Consolidated monitoring overview across all LogiShield modules.",
    owner: "All modules",
    shows: [
      "System-wide request and anomaly trend",
      "Pipeline and component health summary",
      "Recent requests with their downstream verdicts"
    ],
    requires: [
      "Aggregated data from Member 1 (ingestion), Member 2 (verification) and Member 3 (runtime verification)",
      "An agreed cross-member data contract and a way to collect it",
      "Stored history: the Member 4 service is stateless and keeps none"
    ],
    status: [
      "Member 4 analysis is functional; see Anomaly Detection.",
      "No other module is connected to this dashboard."
    ],
    placeholders: [
      { title: "API request & anomaly trend", icon: "traffic", needs: "Needs stored traffic history" },
      { title: "Processing pipeline status", icon: "layers", needs: "Needs Member 1–3 status feeds" },
      { title: "Recent API requests", icon: "list", needs: "Needs a request log with verdicts" }
    ]
  },
  {
    id: "traffic", label: "API Traffic", icon: "traffic", kind: "pending",
    title: "API Traffic",
    subtitle: "Request volume, methods, endpoints and status codes (planned).",
    owner: "Member 1 · Data Processing",
    shows: ["Request volume over time", "Request methods and status codes", "Top endpoints"],
    requires: [
      "Aggregate traffic metrics from Member 1's ingestion service (it currently handles one event per request and exposes no aggregate endpoint)",
      "A stored request history to chart"
    ],
    status: [
      "Member 4 does not observe API traffic; it only analyzes requests posted to it.",
      "Member 1 → Member 4 is tested in-process only, not as a deployed link."
    ],
    placeholders: [
      { title: "Request volume", icon: "traffic", needs: "Needs aggregate traffic metrics" },
      { title: "Request methods", icon: "layers", needs: "Needs a request history" },
      { title: "Top endpoints", icon: "list", needs: "Needs a request history" }
    ]
  },
  {
    id: "anomaly", label: "Anomaly Detection", icon: "alert", kind: "anomaly",
    title: "Anomaly Detection",
    subtitle: "Run a synthetic request through the real Member 4 pipeline and inspect every stage."
  },
  {
    id: "rules", label: "Verified Rules", icon: "shield", kind: "pending",
    title: "Verified Rules",
    subtitle: "Formally verified security rules and policies (Member 2).",
    owner: "Member 2 · Formal Verification",
    shows: ["Rule catalogue with verification status", "Verification coverage by category", "Recent rule-validation activity"],
    requires: [
      "Member 2's real verified-rule output and an agreed contract: the current Member 2 report has a different shape and no event linkage",
      "A decision on how a rule set is associated with a request"
    ],
    status: [
      "Rules shown on the Anomaly Detection page are MOCK fixtures, not Member 2 output.",
      "Member 4 never verifies rules; it only tokenizes rule information it is given."
    ],
    placeholders: [
      { title: "Security rules", icon: "decisions", needs: "Needs Member 2 verified-rule output" },
      { title: "Verification coverage", icon: "shield", needs: "Needs Member 2 verification results" },
      { title: "Recent validation activity", icon: "history", needs: "Needs Member 2 activity feed" }
    ]
  },
  {
    id: "decisions", label: "Allow / Block Decisions", icon: "decisions", kind: "pending",
    title: "Allow / Block Decisions",
    subtitle: "Runtime decisions with explanations (Member 3).",
    owner: "Member 3 · Runtime Verification",
    shows: ["Allow, block and challenge decisions", "Decision reasons and rule outcomes", "Explainable verdict details"],
    requires: [
      "Member 3's runtime verification and an agreed input contract",
      "A Member 4 → Member 3 output adapter (not implemented; it deliberately raises NotImplementedError)"
    ],
    status: [
      "No decision exists anywhere in this system yet.",
      "Member 4 emits only an uncalibrated score; a score is not an Allow or Block verdict.",
      "KPI 5 (downstream usability) cannot be measured until Member 3 integration exists."
    ],
    placeholders: [
      { title: "Decision trend", icon: "activity", needs: "Needs Member 3 decisions" },
      { title: "Recent decisions", icon: "decisions", needs: "Needs Member 3 decisions" },
      { title: "Explainable verdict details", icon: "info", needs: "Needs Member 3 explanations" }
    ]
  },
  {
    id: "logs", label: "Logs & Traces", icon: "logs", kind: "pending",
    title: "Logs & Traces",
    subtitle: "Searchable logs and per-request traces across components (planned).",
    owner: "All modules",
    shows: ["Searchable log events", "End-to-end trace timeline", "Per-request component flow"],
    requires: [
      "Persisted logs and trace ids from each module",
      "A cross-module correlation id convention"
    ],
    status: [
      "Member 4 returns per-stage timings with each analysis (see Anomaly Detection) but stores no logs.",
      "Trace and timing summaries contain counts and durations only, never raw values."
    ],
    placeholders: [
      { title: "Log events", icon: "logs", needs: "Needs persisted logs" },
      { title: "Trace timeline", icon: "history", needs: "Needs cross-module traces" },
      { title: "Component event flow", icon: "components", needs: "Needs cross-module traces" }
    ]
  },
  {
    id: "components", label: "System Components", icon: "components", kind: "pending",
    title: "System Components",
    subtitle: "Architecture, health and performance of LogiShield modules (planned).",
    owner: "All modules",
    shows: ["Health of each module", "Component architecture and data flow", "Inter-component events"],
    requires: [
      "Health endpoints from all modules and a way to collect them in one place",
      "Member 4 added to the shared deployment configuration (it currently runs standalone on 127.0.0.1:8004)"
    ],
    status: [
      "Only the Member 4 service is observable here; its readiness chip is in the page header.",
      "Health, CPU, throughput and uptime for other modules are not available."
    ],
    placeholders: [
      { title: "Component health", icon: "components", needs: "Needs health feeds from Members 1–3" },
      { title: "Architecture & data flow", icon: "layers", needs: "Needs agreed contracts" },
      { title: "Inter-component events", icon: "list", needs: "Needs an event feed" }
    ]
  },
  {
    id: "reports", label: "Reports", icon: "reports", kind: "pending",
    title: "Reports",
    subtitle: "Security analytics and exportable reports (planned).",
    owner: "All modules",
    shows: ["Trend and comparison reports", "Threat and component summaries", "Exportable report output"],
    requires: [
      "Stored analysis and decision history to report on",
      "Integrated data from Members 1–3"
    ],
    status: [
      "The Member 4 KPI 1–4 evaluation evidence lives in research artefacts and the README, not in this dashboard.",
      "KPI 5 is not measured and is not reported here."
    ],
    placeholders: [
      { title: "Request & threat trend", icon: "traffic", needs: "Needs stored history" },
      { title: "Component performance", icon: "components", needs: "Needs module feeds" },
      { title: "Recent incidents", icon: "list", needs: "Needs integrated decisions" }
    ]
  },
  {
    id: "settings", label: "Settings", icon: "settings", kind: "pending", tag: "Soon", separated: true,
    title: "Settings",
    subtitle: "Reserved for a later phase.",
    owner: "Dashboard",
    shows: ["Nothing yet"],
    requires: ["No configuration features are defined; none are provided to avoid unsupported controls"],
    status: ["This is a placeholder only."],
    placeholders: []
  }
];

export function routeById(id) {
  return ROUTES.filter((r) => r.id === id)[0] || null;
}
