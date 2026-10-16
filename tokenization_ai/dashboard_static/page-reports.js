/* Reports: (A) the current browser session, from the shared in-memory store,
 * and (B) documented Member 4 research evaluation evidence, from evidence.js.
 *
 * The two halves are deliberately separate: the session half is live,
 * single-observation and session-local; the evidence half is frozen,
 * pre-registered and synthetic. Neither is production traffic, a security
 * incident record, detection accuracy or an Allow/Block outcome. The page
 * makes no requests; exports are produced only on an explicit click, from the
 * allowlisted store, and are never persisted. */

import { el, clear, icon, card, kv, badge, emptyNote, fmtMs, fmtTime, kpi, button } from "./ui.js";
import { donut, barChart, intervalChart, rangeBar } from "./charts.js";
import { session } from "./session.js";
import { buildOverviewModel } from "./overview-model.js";
import { buildSessionReport } from "./reports-model.js";
import { STAGE_INFO } from "./page-logs.js";
import { KPIS, LATENCY_BOUNDARIES, LATENCY_NOTES, METHODOLOGY, EVIDENCE_INTEGRITY, EVIDENCE_FOLDER } from "./evidence.js";
import { buildJsonReport, buildCsv, exportFilename } from "./export-format.js";

const ROLES = [
  ["control", "Control tag"],
  ["literal_api", "API literal"],
  ["masked", "Masked"],
  ["literal_rule", "Rule literal"]
];

let ui = null;
let unsubscribe = null;
let lastExport = "";

function link(href, className, text) {
  const a = el("a", className, text);
  a.href = href;
  return a;
}

function sectionHead(title, badgeText, badgeTone) {
  const head = el("div", "section-head");
  head.appendChild(el("h2", "section-title", title));
  head.appendChild(badge(badgeText, badgeTone));
  return head;
}

function list(items, className) {
  const ul = el("ul", className || "limits");
  items.forEach((t) => ul.appendChild(el("li", null, t)));
  return ul;
}

/* A user-initiated download of an in-memory string; nothing is stored or sent. */
function download(filename, mime, text) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function doExport(kind) {
  const entries = session.list();
  if (!entries.length) { return; }
  const now = new Date().toISOString();
  const filename = exportFilename(kind, now);
  if (kind === "csv") {
    download(filename, "text/csv;charset=utf-8", buildCsv(entries));
  } else {
    download(filename, "application/json", JSON.stringify(buildJsonReport(entries, now), null, 2) + "\n");
  }
  lastExport = "Exported " + filename + " (" + entries.length + " " + (entries.length === 1 ? "analysis" : "analyses") + ").";
  paintExport(entries.length);
}

/* ---------- build ---------- */

function buildEvidenceCard(k) {
  const pending = k.kind === "pending";
  const c = card({
    title: k.title,
    icon: pending ? "warn" : (k.kind === "latency" ? "clock" : "rules"),
    badge: badge(pending ? "Pending" : (k.kind === "latency" ? "Benchmark" : "Synthetic evaluation"), pending ? "pending" : "info"),
    className: "ev-card" + (pending ? " dashed" : "")
  });
  c.body.appendChild(el("p", "card-note", k.question));
  c.body.appendChild(kv([
    ["Target", k.target],
    ["Observed", k.result],
    ["Sample", k.sample]
  ]));

  if (k.kind === "interval") {
    const viz = el("div", "iv-wrap");
    viz.appendChild(intervalChart({ point: k.point, low: k.interval[0], high: k.interval[1], target: k.targetValue }));
    c.body.appendChild(viz);
    c.body.appendChild(el("p", "card-note",
      "Dot: observed point estimate. Bar: Wilson 95% interval (" + k.interval[0].toFixed(1) + "–" + k.interval[1].toFixed(1) +
      "%). Dashed line: proposal target."));
  } else if (k.kind === "latency") {
    const viz = el("div", "iv-wrap");
    viz.appendChild(rangeBar({ mean: k.mean, worst: k.worst, target: k.targetValue, max: k.targetValue }));
    c.body.appendChild(viz);
    c.body.appendChild(el("p", "card-note", "Bar: pooled mean. Tick: worst batch mean. Dashed line: proposal target. " + k.baseline));
  }

  const verdict = el("div", "verdict");
  verdict.appendChild(badge(k.verdict[0], k.verdict[1]));
  c.body.appendChild(verdict);

  const details = el("details", "ev-details");
  details.appendChild(el("summary", null, "Context, limitations and evidence references"));
  details.appendChild(el("p", "label", "Measurement context"));
  details.appendChild(list(k.context));
  details.appendChild(el("p", "label", "Limitations"));
  details.appendChild(list(k.limits));
  details.appendChild(el("p", "label", "Evidence"));
  const refs = el("ul", "refs");
  k.refs.forEach((r) => refs.appendChild(el("li", null, r)));
  details.appendChild(refs);
  c.body.appendChild(details);
  return c.root;
}

function build(root, route) {
  const head = el("div", "page-head");
  head.appendChild(el("h1", "page-title", route.title));
  head.appendChild(el("p", "page-sub", route.subtitle));
  root.appendChild(head);

  // ---------- A. current browser session ----------
  root.appendChild(sectionHead("Current Browser Session — Member 4 Prototype", "Session-local", "info"));
  const scope = el("div", "pending-banner info-banner");
  const scopeText = el("div");
  scopeText.appendChild(el("strong", null, "Live observations from this page load only"));
  scopeText.appendChild(el("p", null,
    "Built from the successful analyses this browser session received from POST /v1/analyze, held in memory and lost on reload. " +
    "These are not production traffic, security incidents, detection accuracy or Allow/Block outcomes, and scores are uncalibrated."));
  scope.appendChild(scopeText);
  root.appendChild(scope);

  const tiles = el("div", "grid g-4");
  ui.kCount = kpi("blue", "history", "Analyses this session");
  ui.kScore = kpi("purple", "activity", "Latest anomaly score");
  ui.kTokens = kpi("green", "code", "Latest token count");
  ui.kMasking = kpi("pink", "lock", "Analyses with masking");
  [ui.kCount, ui.kScore, ui.kTokens, ui.kMasking].forEach((k) => tiles.appendChild(k.root));
  root.appendChild(tiles);
  root.appendChild(el("p", "card-note trace-note",
    "Sources: session store entries built from POST /v1/analyze responses. \"Analyses with masking\" counts entries whose response reported at least one masked field."));

  const rowA = el("div", "grid g-wide");
  const activity = card({ title: "Per-analysis activity", icon: "traffic", badge: badge("Session-local", "info") });
  ui.activity = activity.body;
  const timing = card({ title: "Latest pipeline timing breakdown", icon: "clock", badge: badge("Latest analysis", "info") });
  ui.timing = timing.body;
  rowA.appendChild(activity.root);
  rowA.appendChild(timing.root);
  root.appendChild(rowA);

  const rowB = el("div", "grid g-wide");
  const summary = card({ title: "Tokenization & masking summary", icon: "code", badge: badge("All session analyses", "info") });
  ui.summary = summary.body;
  const exp = card({ title: "Export session summaries", icon: "file" });
  ui.exportBody = exp.body;
  rowB.appendChild(summary.root);
  rowB.appendChild(exp.root);
  root.appendChild(rowB);

  const table = card({ title: "Analysis summary", icon: "logs", badge: badge("Session-local", "info") });
  ui.table = table.body;
  root.appendChild(table.root);

  const links = el("div", "grid g-2");
  [["#/logs", "logs", "Logs & Traces", "Open per-stage traces, filters and the stored summary for each analysis"],
   ["#/anomaly", "alert", "Anomaly Detection", "Run another synthetic request through the real Member 4 pipeline"]].forEach((a) => {
    const item = link(a[0], "action");
    const ico = el("span", "action-ico");
    ico.appendChild(icon(a[1], 22));
    item.appendChild(ico);
    const text = el("span", "action-text");
    text.appendChild(el("strong", null, a[2]));
    text.appendChild(el("span", null, a[3]));
    item.appendChild(text);
    links.appendChild(item);
  });
  root.appendChild(links);

  // ---------- B. documented research evidence ----------
  root.appendChild(el("div", "section-gap"));
  root.appendChild(sectionHead("Member 4 Research Evaluation Evidence", "Documented · frozen", "pending"));
  const ev = el("div", "pending-banner info-banner");
  const evText = el("div");
  evText.appendChild(el("strong", null, "Separate from the live session: documented, pre-registered, synthetic evaluation results"));
  evText.appendChild(el("p", null,
    "These figures are copied from archived evaluation reports; this page does not measure or recompute them. " +
    "They come from controlled synthetic test cases, not a validated real-world population, and they are not the per-request timings shown above."));
  ev.appendChild(evText);
  root.appendChild(ev);

  const grid = el("div", "grid g-2");
  KPIS.forEach((k) => grid.appendChild(buildEvidenceCard(k)));
  root.appendChild(grid);

  const bounds = card({ title: "KPI 4 measurement boundaries (informational)", icon: "layers", badge: badge("Same run", "info") });
  const wrap = el("div", "table-wrap");
  const t = el("table", "table");
  const headRow = t.createTHead().insertRow();
  ["Boundary", "What it times", "Pooled mean, mixed workload (ms)", "Note"].forEach((h) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = h;
    headRow.appendChild(th);
  });
  const tbody = t.createTBody();
  LATENCY_BOUNDARIES.forEach((b) => {
    const row = tbody.insertRow();
    row.insertCell().textContent = b[0];
    row.insertCell().textContent = b[1];
    const num = row.insertCell();
    num.className = "num";
    num.textContent = b[2];
    row.insertCell().textContent = b[3] === "primary" ? "The pre-registered KPI" : b[3];
  });
  wrap.appendChild(t);
  bounds.body.appendChild(wrap);
  bounds.body.appendChild(list(LATENCY_NOTES));
  root.appendChild(bounds.root);

  const integ = card({ title: "Evidence integrity", icon: "shield", badge: badge("Traceable", "info") });
  integ.body.appendChild(kv(EVIDENCE_INTEGRITY));
  integ.body.appendChild(el("p", "card-note",
    "Archived reports live outside the repository in " + EVIDENCE_FOLDER + " (see tokenization_ai/README.md §8). Frozen protocols and fixtures are not edited by this dashboard."));
  root.appendChild(integ.root);

  // ---------- C. methodology ----------
  const meth = card({ title: "Methodology and limitations", icon: "warn", badge: badge("Read me", "mock") });
  meth.body.appendChild(list(METHODOLOGY));
  root.appendChild(meth.root);
}

/* ---------- paint ---------- */

function paintExport(count) {
  const body = ui.exportBody;
  clear(body);
  const empty = count === 0;
  body.appendChild(el("p", "card-note",
    "Download the current session's allowlisted summaries. The export is created in your browser only when you click; " +
    "nothing is stored or sent anywhere."));
  const row = el("div", "export-row");
  const json = button("Export JSON", "btn", () => doExport("json"), "file");
  const csv = button("Export CSV", "btn", () => doExport("csv"), "list");
  json.disabled = empty;
  csv.disabled = empty;
  row.appendChild(json);
  row.appendChild(csv);
  body.appendChild(row);
  body.appendChild(el("p", "status", empty
    ? "Nothing to export: no analysis has been completed in this session."
    : (lastExport || count + " " + (count === 1 ? "analysis" : "analyses") + " ready to export.")));
  body.appendChild(el("p", "label", "Included fields"));
  body.appendChild(el("p", "card-note",
    "Sequence, observation time, event id, scenario, model version, uncalibrated score, request method and projected path, " +
    "rule source and count, masked-field paths and counts, token counts by role, tensor counts, and per-stage timings."));
  body.appendChild(el("p", "label", "Never included"));
  body.appendChild(el("p", "card-note",
    "Raw requests, passwords, bearer tokens, token text, the masked representation, unrecognised values or any other response field. " +
    "CSV cells are quoted and neutralised against spreadsheet formula injection."));
}

function paintAll() {
  if (!ui) { return; }
  const entries = session.list();
  const overview = buildOverviewModel(entries);
  const model = buildSessionReport(entries, overview);
  const k = overview.kpis;

  ui.kCount.value.textContent = k.analyses === null ? "—" : String(k.analyses);
  ui.kCount.sub.textContent = k.analyses === null ? "No analysis in this session yet" : "Successful analyses since this page loaded";
  ui.kScore.value.textContent = k.score === null ? "—" : k.score.toFixed(4);
  ui.kScore.sub.textContent = k.score === null ? "Run an analysis to see a score" : "Uncalibrated · model " + overview.latest.modelVersion;
  ui.kTokens.value.textContent = k.tokens === null ? "—" : String(k.tokens);
  ui.kTokens.sub.textContent = k.tokens === null ? "No analysis yet" : "of " + k.maxLength + " fixed length · " + k.padded + " padded";
  ui.kMasking.value.textContent = model.withMasking === null ? "—" : model.withMasking + " of " + model.count;
  ui.kMasking.sub.textContent = model.withMasking === null ? "No analysis yet"
    : model.maskedFieldsTotal + " masked field" + (model.maskedFieldsTotal === 1 ? "" : "s") + " in total";

  // activity chart
  const act = ui.activity;
  clear(act);
  if (!model.hasData) {
    const empty = el("div", "empty-state");
    empty.appendChild(icon("traffic", 28));
    empty.appendChild(el("strong", null, "No analyses in this session yet"));
    empty.appendChild(el("p", null, "This chart draws one bar per successful analysis observed in this browser session. Nothing is shown until you run one."));
    empty.appendChild(link("#/anomaly", "btn link-btn", "Open Anomaly Detection"));
    act.appendChild(empty);
  } else {
    const wrap = el("div", "chart-wrap");
    wrap.appendChild(barChart(overview.chart, "Uncalibrated anomaly score of each analysis observed in this session"));
    act.appendChild(wrap);
    act.appendChild(el("p", "card-note",
      "One bar per successful analysis (latest outlined)" + (overview.count > overview.shown ? ", the most recent " + overview.shown + " of " + overview.count : "") +
      ". Heights are uncalibrated scores, not risk levels or detection accuracy."));
  }

  // latest timing
  const tm = ui.timing;
  clear(tm);
  if (!model.hasData) {
    tm.appendChild(emptyNote("The per-stage timings of the latest analysis appear here."));
  } else {
    const bars = el("div", "bars");
    overview.stages.forEach((s) => {
      const info = STAGE_INFO[s.stage] || [s.stage];
      const row = el("div", "bar-row");
      row.appendChild(el("span", "bar-name", info[0]));
      const track = el("div", "bar-track");
      const fill = el("div", "bar-fill");
      fill.style.width = Math.max(2, (s.ms / (overview.maxStageMs || 1)) * 100).toFixed(1) + "%";
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("span", "bar-val", fmtMs(s.ms) + " ms"));
      bars.appendChild(row);
    });
    tm.appendChild(bars);
    const total = el("div", "bar-total");
    total.appendChild(el("span", null, "Total reported pipeline time"));
    total.appendChild(el("strong", null, fmtMs(overview.totalMs) + " ms"));
    tm.appendChild(total);
    tm.appendChild(el("p", "card-note",
      "One observed analysis (#" + overview.latest.seq + ") on this machine. Not the KPI 4 benchmark below, not deployed gateway latency and not isolated model inference."));
  }

  // tokenization & masking summary
  const sm = ui.summary;
  clear(sm);
  if (!model.hasData) {
    sm.appendChild(emptyNote("Tokenization and masking totals appear here after the first analysis."));
  } else {
    const split = el("div", "split");
    const dw = el("div", "donut-wrap");
    dw.appendChild(donut(ROLES.map((r) => ({ key: r[0], value: model.roleTotals[r[0]] }))));
    const center = el("div", "donut-center");
    center.appendChild(el("strong", null, model.tokensTotal));
    center.appendChild(el("span", null, "tokens"));
    dw.appendChild(center);
    split.appendChild(dw);
    const legend = el("ul", "legend");
    ROLES.forEach((r) => {
      const li = el("li");
      li.appendChild(el("span", "sw sw-" + r[0]));
      li.appendChild(el("span", null, r[1]));
      li.appendChild(el("span", "cnt", model.roleTotals[r[0]]));
      legend.appendChild(li);
    });
    split.appendChild(legend);
    sm.appendChild(split);
    sm.appendChild(kv([
      ["Analyses", model.count],
      ["Tokens (all analyses)", model.tokensTotal],
      ["Masked fields (all analyses)", model.maskedFieldsTotal],
      ["Analyses with masking", model.withMasking + " of " + model.count],
      ["Unknown tokens", model.unknownTotal],
      ["Payloads truncated", model.truncated],
      ["Score range (uncalibrated)", model.scoreMin.toFixed(4) + " – " + model.scoreMax.toFixed(4)]
    ]));
    sm.appendChild(el("p", "card-note", "Totals add up the allowlisted per-analysis summaries in the session store."));
  }

  // analysis summary table
  const tb = ui.table;
  clear(tb);
  if (!model.hasData) {
    tb.appendChild(emptyNote("No analyses in this session yet. Nothing is listed until an analysis succeeds."));
  } else {
    const wrap = el("div", "table-wrap");
    const table = el("table", "table events");
    const headRow = table.createTHead().insertRow();
    ["#", "Observed (local)", "Event ID", "Scenario", "Rule source", "Tokens", "Padded", "Masked", "Score", "Pipeline"].forEach((h) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = h;
      headRow.appendChild(th);
    });
    const body = table.createTBody();
    model.rows.forEach((e) => {
      const row = body.insertRow();
      [["num", e.seq], ["num", fmtTime(e.observedAt)], ["mono", e.eventId], ["cap", e.scenarioId], ["", e.ruleLabel],
       ["num", e.tokens.total], ["num", e.tensor.paddedPositions], ["num", e.maskedCount], ["num", e.score.toFixed(4)],
       ["num", fmtMs(e.totalMs) + " ms"]].forEach((c) => {
        const cell = row.insertCell();
        if (c[0]) { cell.className = c[0]; }
        cell.textContent = String(c[1]);
      });
    });
    wrap.appendChild(table);
    tb.appendChild(wrap);
    tb.appendChild(el("p", "card-note",
      "Observed = when this dashboard received the response (its local clock). Scores are uncalibrated model outputs, not verdicts."));
  }

  paintExport(model.count);
}

/* ---------- lifecycle ---------- */

export function mount(root, route) {
  ui = {};
  lastExport = "";
  build(root, route);
  paintAll();
  unsubscribe = session.subscribe(() => { if (ui) { paintAll(); } });
}

export function unmount() {
  if (unsubscribe) { unsubscribe(); }
  unsubscribe = null;
  ui = null;
}
