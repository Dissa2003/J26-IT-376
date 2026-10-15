/* Dashboard overview: Member 4 research prototype.
 *
 * Reads only (a) the shared in-memory session store of allowlisted analysis
 * summaries and (b) Member 4's own readiness from shell.js (GET /health).
 * It makes no requests of its own and shows no gateway traffic, rule
 * verification, decision or other-module status. With no analysis yet every
 * metric is an empty state, never a zero standing in for history. */

import { el, clear, icon, card, kv, badge, emptyNote, fmtMs, fmtTime, kpi } from "./ui.js";
import { donut, barChart } from "./charts.js";
import { shell, healthLabel } from "./shell.js";
import { session } from "./session.js";
import { buildOverviewModel } from "./overview-model.js";
import { COMPONENTS } from "./page-components.js";
import { STAGE_INFO } from "./page-logs.js";

const ROLES = [
  ["control", "Control tag"],
  ["literal_api", "API literal"],
  ["masked", "Masked"],
  ["literal_rule", "Rule literal"]
];

const TRANSPARENCY = [
  "Synthetic demo fixtures only: no real traffic, users or credentials are involved.",
  "A lightweight prototype model fitted on two synthetic fixtures; its score is not detection accuracy.",
  "Scores are uncalibrated: not a probability, not a verdict and not an Allow or Block decision.",
  "Session-local data: held in this page's memory and lost on reload; nothing is persisted.",
  "No end-to-end runtime verification: the Member 4 → Member 3 hand-off is not implemented.",
  "KPI 5 (downstream usability) is not measured.",
  "Cross-member integration is incomplete; no connected or deployed system is demonstrated."
];

const ACTIONS = [
  ["#/anomaly", "alert", "Anomaly Detection", "Run a synthetic request through the real Member 4 pipeline", true],
  ["#/logs", "logs", "Logs & Traces", "Inspect this session's analysis traces, timings and provenance", false],
  ["#/components", "components", "System Components", "Architecture, integration status and documented limitations", false]
];

let ui = null;
let unsubscribers = [];

function link(href, className, text) {
  const a = el("a", className, text);
  a.href = href;
  return a;
}

function healthBadge(health) {
  const map = {
    ready: ["Ready", "live"], notready: ["Not ready", "bad"], down: ["Unreachable", "bad"], checking: ["Checking…", "info"]
  };
  const m = map[health.state] || ["Unknown", "pending"];
  return badge(m[0], m[1]);
}

/* ---------- build ---------- */

function build(root, route) {
  const head = el("div", "overview-head");
  const titles = el("div");
  titles.appendChild(el("h1", "page-title", route.title));
  titles.appendChild(el("p", "page-sub", route.subtitle));
  head.appendChild(titles);
  const side = el("div", "overview-side");
  side.appendChild(badge("Member 4 prototype scope", "info"));
  ui.ready = el("button", "health-chip");
  ui.ready.type = "button";
  ui.ready.title = "Re-check the Member 4 service (GET /health)";
  ui.ready.addEventListener("click", () => { shell.refreshHealth(); });
  ui.readyDot = el("span", "dot dot-idle");
  ui.readyText = el("span", null, "Checking service…");
  ui.ready.appendChild(ui.readyDot);
  ui.ready.appendChild(ui.readyText);
  side.appendChild(ui.ready);
  head.appendChild(side);
  root.appendChild(head);

  const scope = el("div", "pending-banner info-banner");
  scope.appendChild(badge("Scope", "info"));
  const scopeText = el("div");
  scopeText.appendChild(el("strong", null, "Member 4 research prototype overview"));
  scopeText.appendChild(el("p", null,
    "This page summarizes only what the Member 4 service and this browser session have actually produced. " +
    "It does not show gateway traffic, rule verification, Allow/Block decisions or any other module's status."));
  scope.appendChild(scopeText);
  root.appendChild(scope);

  ui.degraded = el("div", "error-box");
  ui.degraded.setAttribute("role", "status");
  ui.degraded.hidden = true;
  root.appendChild(ui.degraded);

  const tiles = el("div", "grid g-4");
  ui.kCount = kpi("blue", "history", "Analyses this session");
  ui.kScore = kpi("purple", "activity", "Latest anomaly score");
  ui.kTokens = kpi("green", "code", "Latest token count");
  ui.kMasked = kpi("pink", "lock", "Latest masked fields");
  [ui.kCount, ui.kScore, ui.kTokens, ui.kMasked].forEach((k) => tiles.appendChild(k.root));
  root.appendChild(tiles);
  root.appendChild(el("p", "card-note trace-note",
    "Every figure comes from successful POST /v1/analyze responses received by this page since it loaded (the in-memory session store), or from GET /health."));

  const rowA = el("div", "grid g-wide");
  const activity = card({ title: "Session analysis activity", icon: "traffic", badge: badge("Session-local", "info") });
  ui.activity = activity.body;
  const latest = card({ title: "Latest analysis", icon: "file" });
  ui.latest = latest.body;
  rowA.appendChild(activity.root);
  rowA.appendChild(latest.root);
  root.appendChild(rowA);

  const rowB = el("div", "grid g-wide");
  const perf = card({ title: "Pipeline performance overview", icon: "clock", badge: badge("Latest analysis", "info") });
  ui.perf = perf.body;
  const roles = card({ title: "Token roles", icon: "code", badge: badge("Latest analysis", "info") });
  ui.roles = roles.body;
  rowB.appendChild(perf.root);
  rowB.appendChild(roles.root);
  root.appendChild(rowB);

  const comps = card({ title: "Research component status", icon: "components", badge: badge("Documented", "pending") });
  ui.comps = comps.body;
  root.appendChild(comps.root);

  const rowC = el("div", "grid g-2");
  const actions = card({ title: "Quick actions", icon: "play" });
  const list = el("div", "actions");
  ACTIONS.forEach((a) => {
    const item = link(a[0], "action" + (a[4] ? " action-primary" : ""));
    const ico = el("span", "action-ico");
    ico.appendChild(icon(a[1], 22));
    item.appendChild(ico);
    const text = el("span", "action-text");
    text.appendChild(el("strong", null, a[2]));
    text.appendChild(el("span", null, a[3]));
    item.appendChild(text);
    list.appendChild(item);
  });
  actions.body.appendChild(list);
  const trans = card({ title: "Scientific transparency", icon: "warn", badge: badge("Read me", "mock") });
  const ul = el("ul", "limits");
  TRANSPARENCY.forEach((t) => ul.appendChild(el("li", null, t)));
  trans.body.appendChild(ul);
  rowC.appendChild(actions.root);
  rowC.appendChild(trans.root);
  root.appendChild(rowC);
}

/* ---------- paint ---------- */

function paintHealth(health) {
  if (!ui) { return; }
  ui.readyDot.className = "dot " + (health.state === "ready" ? "dot-ok" : (health.state === "checking" ? "dot-idle" : "dot-bad"));
  ui.readyText.textContent = healthLabel(health);
  if (health.state === "notready" || health.state === "down") {
    ui.degraded.hidden = false;
    ui.degraded.textContent = health.state === "down"
      ? "The Member 4 service could not be reached from this page, so new analyses will fail. Metrics below are from analyses already completed in this session."
      : "The Member 4 service did not report ready, so new analyses may fail. Metrics below are from analyses already completed in this session.";
  } else {
    ui.degraded.hidden = true;
    ui.degraded.textContent = "";
  }
  paintComponents(health);
}

function paintKpis(model) {
  const k = model.kpis;
  ui.kCount.value.textContent = k.analyses === null ? "—" : String(k.analyses);
  ui.kCount.sub.textContent = k.analyses === null ? "No analysis in this session yet" : "Successful analyses since this page loaded";
  ui.kScore.value.textContent = k.score === null ? "—" : k.score.toFixed(4);
  ui.kScore.sub.textContent = k.score === null ? "Run an analysis to see a score" : "Uncalibrated · model " + model.latest.modelVersion;
  ui.kTokens.value.textContent = k.tokens === null ? "—" : String(k.tokens);
  ui.kTokens.sub.textContent = k.tokens === null ? "No analysis yet"
    : "of " + k.maxLength + " fixed length · " + k.padded + " padded";
  ui.kMasked.value.textContent = k.masked === null ? "—" : String(k.masked);
  ui.kMasked.sub.textContent = k.masked === null ? "No analysis yet" : "Path-based provenance";
}

function paintActivity(model) {
  const body = ui.activity;
  clear(body);
  if (!model.hasData) {
    const empty = el("div", "empty-state");
    empty.appendChild(icon("traffic", 28));
    empty.appendChild(el("strong", null, "No analyses in this session yet"));
    empty.appendChild(el("p", null,
      "This chart draws one bar per successful analysis observed in this browser session. Nothing is shown until you run one."));
    empty.appendChild(link("#/anomaly", "btn link-btn", "Open Anomaly Detection"));
    body.appendChild(empty);
    return;
  }
  const wrap = el("div", "chart-wrap");
  wrap.appendChild(barChart(model.chart, "Uncalibrated anomaly score of each analysis observed in this session"));
  body.appendChild(wrap);
  body.appendChild(el("p", "card-note",
    "One bar per successful analysis (latest outlined)" + (model.count > model.shown ? ", the most recent " + model.shown + " of " + model.count : "") +
    ". Heights are uncalibrated scores from a model fitted on two synthetic fixtures; they are not risk levels or detection accuracy." +
    (model.count === 1 ? " Run another scenario to compare." : "")));
}

function paintLatest(model) {
  const body = ui.latest;
  clear(body);
  if (!model.hasData) {
    body.appendChild(emptyNote("The latest successful analysis will be summarized here."));
    return;
  }
  const e = model.latest;
  body.appendChild(kv([
    ["Scenario", e.scenarioTitle],
    ["Event ID", e.eventId],
    ["Model version", e.modelVersion],
    ["Anomaly score", e.score.toFixed(4) + " (uncalibrated)"],
    ["Tokens", e.tokens.total + " of " + e.tensor.maxLength + " · " + e.tensor.paddedPositions + " padded" +
      (e.tensor.payloadTruncated ? " · truncated" : "")],
    ["Masked fields", e.maskedCount],
    ["Rule source", e.ruleLabel],
    ["Pipeline time", fmtMs(e.totalMs) + " ms"],
    ["Observed", fmtTime(e.observedAt) + " (dashboard local time)"]
  ]));
  body.appendChild(el("p", "caveat",
    "The score is an uncalibrated output of a lightweight prototype model. It is not detection accuracy, a probability of attack, or an Allow/Block verdict."));
  body.appendChild(link("#/logs", "card-link", "View this event in Logs & Traces"));
}

function paintPerformance(model) {
  const body = ui.perf;
  clear(body);
  if (!model.hasData) {
    body.appendChild(emptyNote("Per-stage timings appear here after the first analysis."));
    return;
  }
  const bars = el("div", "bars");
  model.stages.forEach((s) => {
    const info = STAGE_INFO[s.stage] || [s.stage];
    const row = el("div", "bar-row");
    row.appendChild(el("span", "bar-name", info[0]));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = Math.max(2, (s.ms / (model.maxStageMs || 1)) * 100).toFixed(1) + "%";
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("span", "bar-val", fmtMs(s.ms) + " ms"));
    bars.appendChild(row);
  });
  body.appendChild(bars);
  const total = el("div", "bar-total");
  total.appendChild(el("span", null, "Total reported pipeline time"));
  total.appendChild(el("strong", null, fmtMs(model.totalMs) + " ms"));
  body.appendChild(total);
  body.appendChild(el("p", "card-note",
    "Observed timings of one analysis on this machine (the latest). They are not the KPI 4 benchmark, not deployed gateway latency and not isolated model inference."));
}

function paintRoles(model) {
  const body = ui.roles;
  clear(body);
  if (!model.hasData) {
    body.appendChild(emptyNote("Token role counts appear here after the first analysis."));
    return;
  }
  const split = el("div", "split");
  const wrap = el("div", "donut-wrap");
  wrap.appendChild(donut(ROLES.map((r) => ({ key: r[0], value: model.roles[r[0]] }))));
  const center = el("div", "donut-center");
  center.appendChild(el("strong", null, model.latest.tokens.total));
  center.appendChild(el("span", null, "tokens"));
  wrap.appendChild(center);
  split.appendChild(wrap);
  const legend = el("ul", "legend");
  ROLES.forEach((r) => {
    const li = el("li");
    li.appendChild(el("span", "sw sw-" + r[0]));
    li.appendChild(el("span", null, r[1]));
    li.appendChild(el("span", "cnt", model.roles[r[0]]));
    legend.appendChild(li);
  });
  split.appendChild(legend);
  body.appendChild(split);
  body.appendChild(el("p", "card-note", "Roles are assigned by the tokenizer, not inferred from text."));
}

function paintComponents(health) {
  const body = ui.comps;
  clear(body);
  const strip = el("ol", "proc-strip");
  COMPONENTS.forEach((c) => {
    const li = el("li", "proc-node" + (c.live ? " proc-m4" : ""));
    const top = el("div", "proc-top");
    const ico = el("span", "proc-ico");
    ico.appendChild(icon(c.icon, 22));
    top.appendChild(ico);
    const names = el("div");
    names.appendChild(el("strong", null, c.order + " · " + c.name));
    names.appendChild(el("span", "proc-member", c.member));
    top.appendChild(names);
    li.appendChild(top);
    const chips = el("div", "proc-chips");
    if (c.live) {
      chips.appendChild(healthBadge(health));
      chips.appendChild(badge(c.link[0], c.link[1]));
    } else {
      chips.appendChild(badge(c.link[0], c.link[1]));
    }
    li.appendChild(chips);
    li.appendChild(el("p", "proc-out", c.live ? "Live readiness from GET /health" : "No live status available"));
    strip.appendChild(li);
  });
  body.appendChild(strip);
  body.appendChild(el("p", "card-note",
    "Only Member 4 has a live status. Order follows the research pipeline; apart from the in-process-tested Member 1 context link, the connections are proposed, not integrated. " +
    "No connected, deployed, verified or operational system is claimed."));
  body.appendChild(link("#/components", "card-link", "Architecture and integration details in System Components"));
}

function paintAll() {
  if (!ui) { return; }
  const model = buildOverviewModel(session.list());
  paintKpis(model);
  paintActivity(model);
  paintLatest(model);
  paintPerformance(model);
  paintRoles(model);
}

/* ---------- lifecycle ---------- */

export function mount(root, route) {
  ui = {};
  build(root, route);
  paintHealth(shell.health);
  paintAll();
  unsubscribers = [
    shell.subscribe((health) => { if (ui) { paintHealth(health); } }),
    session.subscribe(() => { if (ui) { paintAll(); } })
  ];
}

export function unmount() {
  unsubscribers.forEach((off) => off());
  unsubscribers = [];
  ui = null;
}
