/* Route table. Dashboard, Policy Verification, Anomaly Detection, Logs & Traces and System Components
 * are functional; every other page is an honest pending state that says what is required before it can
 * show real data. No numbers, rows or events are invented here. */

export const DEFAULT_ROUTE = "dashboard";

export const ROUTES = [
  {
    id: "dashboard", label: "Dashboard", icon: "home", kind: "overview",
    title: "LogiShield Dashboard",
    subtitle: "LogiShield Research Implementation Overview"
  },
  {
    id: "policy-verification", label: "Policy Verification", icon: "shield", kind: "policy",
    title: "Policy Verification",
    subtitle: "Translate a supported security policy into auditable logic and a Z3-backed verification result."
  },
  {
    id: "policy-management", label: "Policy Management", icon: "list", kind: "policy-data",
    title: "Policy Management",
    subtitle: "Review policies verified by this running backend process."
  },
  {
    id: "conflicts", label: "Conflict Detection", icon: "alert", kind: "conflicts",
    title: "Conflict Detection",
    subtitle: "Conflicts returned by the scoped policy conflict detector."
  },
  {
    id: "rule-cache", label: "Rule Cache", icon: "database", kind: "cache",
    title: "Rule Cache",
    subtitle: "Rules accepted by the local in-memory eligibility gate."
  },
  {
    id: "explainability", label: "Explainability", icon: "code", kind: "explainability",
    title: "Explainability",
    subtitle: "Research-readable evidence from the policy verification pipeline."
  },
  {
    id: "evaluation", label: "Research Evaluation", icon: "reports", kind: "evaluation",
    title: "Research Evaluation",
    subtitle: "Measured evaluation results only; no fabricated benchmark metrics."
  },
  {
    id: "runtime", label: "Runtime Monitoring", icon: "activity", kind: "implementation",
    title: "Runtime Monitoring",
    subtitle: "Implementation surface reserved for a future live runtime integration."
  },
  {
    id: "cicd", label: "CI/CD", icon: "cube", kind: "implementation",
    title: "CI/CD",
    subtitle: "Implementation surface reserved for a future deployment integration."
  },
  {
    id: "traffic", label: "API Traffic", icon: "traffic", kind: "pending",
    title: "API Traffic",
    subtitle: "Request volume, methods, endpoints and status codes (planned).",
    owner: "Ingestion · Data Processing",
    shows: ["Request volume over time", "Request methods and status codes", "Top endpoints"],
    requires: [
      "Aggregate traffic metrics from Ingestion's ingestion service (it currently handles one event per request and exposes no aggregate endpoint)",
      "A stored request history to chart"
    ],
    status: [
      "LogiShield does not observe API traffic; it only analyzes requests posted to it.",
      "Ingestion → LogiShield is tested in-process only, not as a deployed link."
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
    subtitle: "Run a synthetic request through the real LogiShield pipeline and inspect every stage."
  },
  {
    id: "rules", label: "Verified Rules", icon: "shield", kind: "pending",
    title: "Verified Rules",
    subtitle: "Formally verified security rules and policies (Formal Verification).",
    owner: "Formal Verification · Formal Verification",
    shows: ["Rule catalogue with verification status", "Verification coverage by category", "Recent rule-validation activity"],
    requires: [
      "Formal Verification's real verified-rule output and an agreed contract: the current Formal Verification report has a different shape and no event linkage",
      "A decision on how a rule set is associated with a request"
    ],
    status: [
      "Rules shown on the Anomaly Detection page are MOCK fixtures, not Formal Verification output.",
      "LogiShield never verifies rules; it only tokenizes rule information it is given."
    ],
    placeholders: [
      { title: "Security rules", icon: "decisions", needs: "Needs Formal Verification verified-rule output" },
      { title: "Verification coverage", icon: "shield", needs: "Needs Formal Verification verification results" },
      { title: "Recent validation activity", icon: "history", needs: "Needs Formal Verification activity feed" }
    ]
  },
  {
    id: "decisions", label: "Allow / Block Decisions", icon: "decisions", kind: "pending",
    title: "Allow / Block Decisions",
    subtitle: "Runtime decisions with explanations (Runtime Verification).",
    owner: "Runtime Verification · Runtime Verification",
    shows: ["Allow, block and challenge decisions", "Decision reasons and rule outcomes", "Explainable verdict details"],
    requires: [
      "Runtime Verification's runtime verification and an agreed input contract",
      "LogiShield's score wired into the runtime path (the handoff runtime-verification-input-1 exists but is not called at runtime)"
    ],
    status: [
      "This dashboard shows no decision; Runtime Verification's own decisions do not use LogiShield's score yet.",
      "LogiShield emits only an uncalibrated score; a score is not an Allow or Block verdict.",
      "KPI 5 (downstream usability) cannot be measured until Runtime Verification integration exists."
    ],
    placeholders: [
      { title: "Decision trend", icon: "activity", needs: "Needs Runtime Verification decisions" },
      { title: "Recent decisions", icon: "decisions", needs: "Needs Runtime Verification decisions" },
      { title: "Explainable verdict details", icon: "info", needs: "Needs Runtime Verification explanations" }
    ]
  },
  {
    id: "logs", label: "Logs & Traces", icon: "logs", kind: "logs",
    title: "Logs & Traces",
    subtitle: "LogiShield Session Traces: the analyses run in this dashboard session, with per-stage timing and provenance."
  },
  {
    id: "components", label: "System Components", icon: "components", kind: "components",
    title: "System Components",
    subtitle: "Architecture and integration status of the LogiShield modules. Only LogiShield has a live status."
  },
  {
    id: "reports", label: "Reports", icon: "reports", kind: "reports",
    title: "Reports",
    subtitle: "Current-session analytics and documented LogiShield research evaluation evidence."
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
