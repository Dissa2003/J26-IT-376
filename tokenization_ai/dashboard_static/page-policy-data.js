import { el, clear, card, badge, emptyNote } from "./ui.js";
import { getVerificationRecords, getRuleCache, getConflicts } from "./api.js";

let selectedPolicyId = null;

function codeBlock(value) {
  const pre = el("pre", "policy-code");
  pre.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return pre;
}

function linkFor(report) {
  const link = el("a", "table-link", report.policy_id || "unnamed policy");
  link.href = "#/explainability";
  link.addEventListener("click", () => { selectedPolicyId = report.policy_id; });
  return link;
}

function serviceUnavailable(root, label) {
  clear(root);
  root.appendChild(el("div", "policy-error", "The " + label + " endpoint is not connected."));
  root.appendChild(emptyNote("No local data was fabricated. Submit a policy from Policy Verification, then refresh this view."));
}

function reportList(root, records, query = "") {
  clear(root);
  const normalized = query.trim().toLowerCase();
  const filtered = normalized
    ? records.filter((report) => JSON.stringify(report).toLowerCase().includes(normalized))
    : records;
  if (!filtered.length) {
    root.appendChild(emptyNote("No policies have been verified in this backend process yet."));
    return;
  }
  const table = el("div", "data-list");
  filtered.forEach((report) => {
    const row = el("article", "data-row");
    const main = el("div");
    main.appendChild(linkFor(report));
    main.appendChild(el("p", "card-note", report.original_policy || "No original policy returned."));
    row.appendChild(main);
    const meta = el("div", "data-row-meta");
    meta.appendChild(report.rule_status === "contradictory" ? badge("Contradictory", "bad")
      : report.rule_status === "conflicted" ? badge("Conflicted", "bad")
      : badge(report.rule_status || "unknown", report.rule_status === "eligible" ? "live" : "pending"));
    meta.appendChild(el("span", "muted", report.status || "UNKNOWN"));
    row.appendChild(meta);
    table.appendChild(row);
  });
  root.appendChild(table);
}

function conflictsList(root, conflicts) {
  clear(root);
  if (!conflicts.length) {
    root.appendChild(emptyNote("No conflicts have been returned by the backend."));
    root.appendChild(el("p", "card-note", "This is not evidence that all possible policies are conflict-free; it reflects only recorded submissions."));
    return;
  }
  conflicts.forEach((conflict) => {
    const item = el("article", "conflict-item");
    item.appendChild(el("h3", null, conflict.conflict_id || "Conflict"));
    item.appendChild(el("p", null, conflict.explanation || "The backend returned a conflict without an explanation."));
    item.appendChild(codeBlock(conflict));
    root.appendChild(item);
  });
}

function cacheList(root, reports) {
  clear(root);
  if (!reports.length) {
    root.appendChild(emptyNote("No rules are currently eligible in the local cache."));
    root.appendChild(el("p", "card-note", "Cached means accepted by the in-memory eligibility gate only. Durable storage and deployment are not connected."));
    return;
  }
  reports.forEach((report) => {
    const item = el("article", "data-row");
    item.appendChild(el("div", null, report.policy_id || "unnamed policy"));
    const state = el("div", "data-row-meta");
    state.appendChild(badge("eligible", "live"));
    state.appendChild(badge("cached", "info"));
    state.appendChild(badge("not deployed", "pending"));
    item.appendChild(state);
    root.appendChild(item);
  });
}

function explainReport(root, report) {
  clear(root);
  if (!report) {
    root.appendChild(emptyNote("No verification report is available. Run a policy verification first."));
    return;
  }
  root.appendChild(el("p", "card-note", "Latest process-local report returned by the backend."));
  [["Structured policy", report.structured_policy], ["Predicate logic", report.fol_formula],
    ["Constraints", report.constraints], ["Solver result", report.solver_result],
    ["Explanation", report.explanation]].forEach((entry) => {
    const section = el("section", "explain-section");
    section.appendChild(el("h3", null, entry[0]));
    section.appendChild(codeBlock(entry[1] || "Not returned."));
    root.appendChild(section);
  });
  const trace = el("section", "explain-section");
  trace.appendChild(el("h3", null, "Formal calculus"));
  if (Array.isArray(report.inference_trace) && report.inference_trace.length) {
    report.inference_trace.forEach((step) => trace.appendChild(codeBlock(step)));
  } else {
    trace.appendChild(emptyNote("No inference steps were returned."));
  }
  root.appendChild(trace);
}

function implementationView(root, route) {
  clear(root);
  const box = el("div", "info-banner");
  box.appendChild(badge("Not connected", "pending"));
  box.appendChild(el("p", null, route.id === "runtime"
    ? "Runtime telemetry, alerts and live decisions are not connected to this application."
    : "No deployment provider or CI/CD service is connected. This view intentionally shows no simulated runs."));
  root.appendChild(box);
}

async function load(root, route) {
  if (route.kind === "implementation" || route.kind === "evaluation") {
    if (route.kind === "implementation") { implementationView(root, route); return; }
    clear(root);
    root.appendChild(emptyNote("No evaluation results available."));
    root.appendChild(el("p", "card-note", "Accuracy, precision, recall, F1, latency and baselines will appear only after a real evaluation run produces measured data."));
    return;
  }
  const request = route.kind === "cache" ? getRuleCache()
    : route.kind === "conflicts" ? getConflicts() : getVerificationRecords();
  const result = await request;
  if (!result.ok || !Array.isArray(result.payload)) {
    serviceUnavailable(root, route.title.toLowerCase());
    return;
  }
  if (route.kind === "cache") { cacheList(root, result.payload); }
  else if (route.kind === "conflicts") { conflictsList(root, result.payload); }
  else if (route.kind === "explainability") {
    const selected = result.payload.find((report) => report.policy_id === selectedPolicyId) || result.payload[0];
    explainReport(root, selected);
  } else { reportList(root, result.payload); }
}

export function mount(root, route) {
  const head = el("div", "page-head");
  head.appendChild(el("h1", "page-title", route.title));
  head.appendChild(el("p", "page-sub", route.subtitle));
  root.appendChild(head);
  const panel = card({ title: route.title, icon: route.kind === "cache" ? "database" : "shield", badge: badge("Backend data", "info") });
  if (route.kind === "policy-data") {
    const search = el("input", "text-input");
    search.type = "search";
    search.placeholder = "Search recorded policy text, ID, status…";
    panel.body.appendChild(search);
    const list = el("div");
    panel.body.appendChild(list);
    getVerificationRecords().then((result) => {
      if (!result.ok || !Array.isArray(result.payload)) { serviceUnavailable(list, "policy records"); return; }
      reportList(list, result.payload);
      search.addEventListener("input", () => reportList(list, result.payload, search.value));
    });
  } else {
    panel.body.appendChild(emptyNote("Loading…"));
    load(panel.body, route);
  }
  root.appendChild(panel.root);
}

export function unmount() {}
