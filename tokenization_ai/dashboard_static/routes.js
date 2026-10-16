/* Route table. Dashboard, Anomaly Detection, Logs & Traces and System Components
 * are functional; every other page is an honest pending state that says what is required before it can
 * show real data. No numbers, rows or events are invented here. */

export const DEFAULT_ROUTE = "dashboard";

export const ROUTES = [
  {
    id: "dashboard", label: "Dashboard", icon: "home", kind: "overview",
    title: "LogiShield Dashboard",
    subtitle: "Member 4 Research Prototype Overview"
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
    id: "logs", label: "Logs & Traces", icon: "logs", kind: "logs",
    title: "Logs & Traces",
    subtitle: "Member 4 Session Traces: the analyses run in this dashboard session, with per-stage timing and provenance."
  },
  {
    id: "components", label: "System Components", icon: "components", kind: "components",
    title: "System Components",
    subtitle: "Architecture and integration status of the LogiShield modules. Only Member 4 has a live status."
  },
  {
    id: "reports", label: "Reports", icon: "reports", kind: "reports",
    title: "Reports",
    subtitle: "Current-session analytics and documented Member 4 research evaluation evidence."
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
