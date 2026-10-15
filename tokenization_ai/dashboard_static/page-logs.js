/* Logs & Traces: "Member 4 Session Traces".
 *
 * Shows only the allowlisted summaries that the Anomaly Detection page put
 * into the in-memory session store from successful POST /v1/analyze
 * responses. It makes no request of its own. These are session-local
 * analysis traces, not persisted gateway logs and not distributed tracing. */

import { el, clear, icon, card, kv, badge, emptyNote, fmtMs, button, field } from "./ui.js";
import { session } from "./session.js";

const STAGE_INFO = {
  fusion: ["Input fusion", "merge", "Join API context and rule information by event id"],
  masking: ["Privacy masking", "lock", "Replace sensitive values with placeholders and record the masked paths"],
  normalization: ["Normalization", "filter", "Canonical method, path, header names and key order"],
  tokenization: ["Tokenization", "code", "Emit control tags and literals, each with an authoritative role"],
  tensorization: ["Tensor preparation", "grid", "Pad or truncate to a fixed length with an attention mask"],
  model_inference: ["AI prediction", "activity", "Hashed n-gram features into a logistic-regression score"]
};

const ROLE_LABELS = [
  ["control", "control"],
  ["literal_api", "API literal"],
  ["masked", "masked"],
  ["literal_rule", "rule literal"]
];

/* Filter and selection state outlives navigation within this page load. */
const view = { scenario: "all", rule: "all", masked: "all", query: "", newestFirst: true, selected: null };

let ui = null;
let unsubscribe = null;

function observedTime(iso) {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? "unknown" : d.toLocaleTimeString([], { hour12: false });
}

function matches(entry) {
  if (view.scenario !== "all" && entry.scenarioId !== view.scenario) { return false; }
  if (view.rule !== "all" && entry.ruleSource !== view.rule) { return false; }
  if (view.masked === "with" && entry.maskedCount === 0) { return false; }
  if (view.masked === "none" && entry.maskedCount > 0) { return false; }
  if (view.query && entry.eventId.toLowerCase().indexOf(view.query.toLowerCase()) < 0) { return false; }
  return true;
}

function setOptions(select, options, current) {
  const key = options.map((o) => o[0] + "=" + o[1]).join("|");
  if (select.getAttribute("data-key") !== key) {
    clear(select);
    options.forEach((o) => {
      const option = document.createElement("option");
      option.value = o[0];
      option.textContent = o[1];
      select.appendChild(option);
    });
    select.setAttribute("data-key", key);
  }
  select.value = current;
}

/* ---------- build ---------- */

function build(root, route) {
  const head = el("div", "page-head");
  head.appendChild(el("h1", "page-title", route.title));
  head.appendChild(el("p", "page-sub", route.subtitle));
  root.appendChild(head);

  const banner = el("div", "pending-banner info-banner");
  banner.appendChild(badge("Session-local", "info"));
  const text = el("div");
  text.appendChild(el("strong", null, "Member 4 Session Traces"));
  text.appendChild(el("p", null,
    "These are session-local analysis traces built from the responses of POST /v1/analyze that this dashboard received since the page loaded. " +
    "They are not persisted gateway logs and not distributed tracing across the LogiShield modules. They are held in memory only and disappear on reload."));
  banner.appendChild(text);
  root.appendChild(banner);

  // Filters.
  const filters = el("section", "card filters-card");
  const bar = el("div", "filters");
  ui.fScenario = el("select", "control");
  ui.fRule = el("select", "control");
  ui.fMasked = el("select", "control");
  ui.fQuery = el("input", "control");
  ui.fQuery.type = "search";
  ui.fQuery.placeholder = "Event id contains…";
  ui.fQuery.maxLength = 64;
  ui.fScenario.addEventListener("change", () => { view.scenario = ui.fScenario.value; paint(); });
  ui.fRule.addEventListener("change", () => { view.rule = ui.fRule.value; paint(); });
  ui.fMasked.addEventListener("change", () => { view.masked = ui.fMasked.value; paint(); });
  ui.fQuery.addEventListener("input", () => { view.query = ui.fQuery.value; paint(); });
  bar.appendChild(field("Scenario", ui.fScenario));
  bar.appendChild(field("Rule source", ui.fRule));
  bar.appendChild(field("Masked fields", ui.fMasked));
  bar.appendChild(field("Search", ui.fQuery));
  ui.reset = button("Reset filters", "btn-ghost", () => {
    view.scenario = "all"; view.rule = "all"; view.masked = "all"; view.query = "";
    ui.fQuery.value = "";
    paint();
  }, "refresh");
  bar.appendChild(ui.reset);
  filters.appendChild(bar);
  root.appendChild(filters);

  // Events table.
  const events = card({ title: "Session events", icon: "logs" });
  ui.count = el("span", "count-note", "");
  events.head.insertBefore(ui.count, events.head.querySelector(".badge"));
  ui.sort = button("Newest first", "btn-ghost", () => { view.newestFirst = !view.newestFirst; paint(); }, "sort");
  ui.clearAll = button("Clear session", "btn-ghost", () => { view.selected = null; session.clear(); }, "trash");
  events.head.appendChild(ui.sort);
  events.head.appendChild(ui.clearAll);
  ui.tableBody = events.body;
  root.appendChild(events.root);

  // Detail row: timeline | flow | details.
  const low = el("div", "grid g-3");
  const timeline = card({ title: "Trace timeline", icon: "history", badge: badge("Selected event", "info") });
  ui.timelineBody = timeline.body;
  const flow = card({ title: "Member 4 component flow", icon: "layers", badge: badge("Internal stages", "info") });
  ui.flowBody = flow.body;
  const detail = card({ title: "Event details & correlation", icon: "info" });
  ui.detailBody = detail.body;
  low.appendChild(timeline.root);
  low.appendChild(flow.root);
  low.appendChild(detail.root);
  root.appendChild(low);
}

/* ---------- paint ---------- */

function paintFilters(all) {
  const scenarios = [["all", "All scenarios"]];
  const rules = [["all", "All rule sources"]];
  all.slice().reverse().forEach((e) => {
    if (!scenarios.some((o) => o[0] === e.scenarioId)) { scenarios.push([e.scenarioId, e.scenarioTitle]); }
    if (!rules.some((o) => o[0] === e.ruleSource)) { rules.push([e.ruleSource, e.ruleLabel]); }
  });
  if (!scenarios.some((o) => o[0] === view.scenario)) { view.scenario = "all"; }
  if (!rules.some((o) => o[0] === view.rule)) { view.rule = "all"; }
  setOptions(ui.fScenario, scenarios, view.scenario);
  setOptions(ui.fRule, rules, view.rule);
  setOptions(ui.fMasked, [["all", "All events"], ["with", "With masked fields"], ["none", "No masked fields"]], view.masked);
  const empty = all.length === 0;
  [ui.fScenario, ui.fRule, ui.fMasked, ui.fQuery, ui.reset, ui.sort, ui.clearAll].forEach((c) => { c.disabled = empty; });
  ui.sort.lastChild.textContent = view.newestFirst ? "Newest first" : "Oldest first";
}

function paintTable(shown, total) {
  const body = ui.tableBody;
  clear(body);
  ui.count.textContent = total === 0 ? "0 events" : "Showing " + shown.length + " of " + total;

  if (total === 0) {
    const empty = el("div", "empty-state");
    empty.appendChild(icon("logs", 28));
    empty.appendChild(el("strong", null, "No analyses in this session yet"));
    empty.appendChild(el("p", null, "Nothing is shown here until an analysis succeeds. Run one on the Anomaly Detection page and it appears here without another backend request."));
    const link = el("a", "btn link-btn", "Open Anomaly Detection");
    link.href = "#/anomaly";
    empty.appendChild(link);
    body.appendChild(empty);
    return;
  }
  if (!shown.length) {
    body.appendChild(emptyNote("No events match the current filters."));
    return;
  }

  const wrap = el("div", "table-wrap");
  const table = el("table", "table events");
  const headRow = table.createTHead().insertRow();
  ["#", "Observed (local)", "Event ID", "Scenario", "Request", "Rule source", "Masked", "Tokens", "Score", "Pipeline"].forEach((h) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = h;
    headRow.appendChild(th);
  });
  const tbody = table.createTBody();
  shown.forEach((entry) => {
    const row = tbody.insertRow();
    row.tabIndex = 0;
    row.className = "event-row" + (entry.seq === view.selected ? " is-selected" : "");
    if (entry.seq === view.selected) { row.setAttribute("aria-current", "true"); }
    const choose = () => { view.selected = entry.seq; paint(); };
    row.addEventListener("click", choose);
    row.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); choose(); }
    });
    const cells = [
      ["num", String(entry.seq)],
      ["num", observedTime(entry.observedAt)],
      ["mono", entry.eventId],
      ["cap", entry.scenarioId],
      ["", (entry.method + " " + entry.path).trim()],
      ["", entry.ruleLabel],
      ["num", String(entry.maskedCount)],
      ["num", String(entry.tokens.total)],
      ["num", entry.score.toFixed(4)],
      ["num", fmtMs(entry.totalMs) + " ms"]
    ];
    cells.forEach((c) => {
      const cell = row.insertCell();
      if (c[0]) { cell.className = c[0]; }
      cell.textContent = c[1];
    });
  });
  wrap.appendChild(table);
  body.appendChild(wrap);
  body.appendChild(el("p", "card-note",
    "Observed = when this dashboard received the response (its local clock), not when the request was made. Score is an uncalibrated model output."));
}

function paintTimeline(entry) {
  const body = ui.timelineBody;
  clear(body);
  if (!entry) {
    body.appendChild(emptyNote("Select an event to see its six-stage execution timeline."));
    return;
  }
  body.appendChild(el("p", "card-note",
    "Event " + entry.eventId + " · observed " + observedTime(entry.observedAt) + " (dashboard local time)."));
  const max = entry.totalMs || 1;
  let offset = 0;
  const list = el("ol", "timeline");
  entry.stages.forEach((stage, index) => {
    const info = STAGE_INFO[stage.stage] || [stage.stage, "info", ""];
    const li = el("li", "tl-item" + (stage.ok ? "" : " tl-fail"));
    li.appendChild(el("span", "tl-dot", String(index + 1)));
    const main = el("div", "tl-main");
    const top = el("div", "tl-top");
    top.appendChild(el("strong", null, info[0]));
    top.appendChild(el("span", "tl-ms", fmtMs(stage.ms) + " ms"));
    main.appendChild(top);
    main.appendChild(el("p", "tl-desc", info[2]));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = Math.max(2, (stage.ms / max) * 100).toFixed(1) + "%";
    track.appendChild(fill);
    main.appendChild(track);
    const counts = Object.keys(stage.counts).map((k) => k + ": " + stage.counts[k]).join(", ");
    main.appendChild(el("p", "tl-meta", "starts +" + fmtMs(offset) + " ms" + (counts ? " · " + counts : "") + (stage.ok ? "" : " · failed")));
    li.appendChild(main);
    list.appendChild(li);
    offset += stage.ms;
  });
  body.appendChild(list);
  body.appendChild(el("p", "card-note",
    "Start offsets are derived by adding the durations of the preceding stages (the stages run one after another). Total " + fmtMs(entry.totalMs) + " ms."));
}

function paintFlow(entry) {
  const body = ui.flowBody;
  clear(body);
  if (!entry) {
    body.appendChild(emptyNote("Select an event to see its processing flow."));
    return;
  }
  const chain = el("ol", "flow");
  const node = (iconName, title, meta, cls) => {
    const li = el("li", "flow-node" + (cls ? " " + cls : ""));
    li.appendChild(icon(iconName, 20));
    const text = el("div", "flow-text");
    text.appendChild(el("strong", null, title));
    if (meta) { text.appendChild(el("span", null, meta)); }
    li.appendChild(text);
    return li;
  };
  chain.appendChild(node("globe", "Unified API context + rule information", "Input as supplied", "flow-io"));
  entry.stages.forEach((stage) => {
    const info = STAGE_INFO[stage.stage] || [stage.stage, "info", ""];
    chain.appendChild(node(info[1], info[0], fmtMs(stage.ms) + " ms", stage.ok ? "" : "flow-fail"));
  });
  chain.appendChild(node("activity", "Anomaly score " + entry.score.toFixed(4), "Uncalibrated · " + entry.modelVersion, "flow-io"));
  chain.appendChild(node("decisions", "Member 3 hand-off", "Not implemented; no agreed contract", "flow-pending"));
  body.appendChild(chain);
  body.appendChild(el("p", "card-note",
    "Only Member 4's internal stages are shown. No other module takes part in this trace."));
}

function paintDetails(entry) {
  const body = ui.detailBody;
  clear(body);
  if (!entry) {
    body.appendChild(emptyNote("Select an event to see its correlation details and stored summary."));
    return;
  }
  const allOk = entry.stages.length > 0 && entry.stages.every((s) => s.ok);
  const roles = ROLE_LABELS.map((r) => r[1] + " " + entry.tokens.byRole[r[0]]).join(", ");
  body.appendChild(kv([
    ["Event ID", entry.eventId],
    ["Sequence", "#" + entry.seq],
    ["Observed", observedTime(entry.observedAt) + " (dashboard local time)"],
    ["Scenario", entry.scenarioTitle],
    ["Request", (entry.method + " " + entry.path).trim()],
    ["Result", allOk ? "All six stages completed" : "A stage reported failure"],
    ["Model version", entry.modelVersion],
    ["Anomaly score", entry.score.toFixed(4) + " (uncalibrated)"],
    ["Rule source", entry.ruleLabel],
    ["Rules fused", entry.ruleCount + (entry.ruleIds.length ? " (" + entry.ruleIds.join(", ") + ")" : "")],
    ["Tokens", entry.tokens.total + " (" + roles + ")"],
    ["Tensor", entry.tensor.tokensInTensor + " of " + entry.tensor.maxLength + ", " + entry.tensor.paddedPositions +
      " padded" + (entry.tensor.payloadTruncated ? ", truncated" : "")],
    ["Unknown tokens", entry.tensor.unknownTokens],
    ["Hidden by projection", entry.hidden.inRepresentation + " in representation, " + entry.hidden.inTokens + " in tokens"],
    ["Accuracy claim", entry.accuracyClaim || "none"]
  ]));

  body.appendChild(el("p", "label", "Masked fields (" + entry.maskedCount + ")"));
  if (entry.maskedFields.length) {
    const chips = el("ul", "chips");
    entry.maskedFields.forEach((m) => {
      chips.appendChild(el("li", "tok role-masked", m.field + (m.placeholder ? " → " + m.placeholder : "")));
    });
    body.appendChild(chips);
  } else {
    body.appendChild(el("p", "card-note", "No fields were masked for this request."));
  }

  body.appendChild(el("p", "label", "Stored session summary (allowlisted JSON)"));
  const pre = el("pre", "code code-short", JSON.stringify(entry, null, 2));
  pre.tabIndex = 0;
  body.appendChild(pre);
  body.appendChild(el("p", "card-note",
    "This is everything the dashboard keeps for this event. The request, token text and masked representation are not stored."));
}

function paint() {
  if (!ui) { return; }
  const all = session.list();
  paintFilters(all);
  let shown = all.filter(matches);
  if (!view.newestFirst) { shown = shown.slice().reverse(); }
  if (!shown.some((e) => e.seq === view.selected)) {
    view.selected = shown.length ? shown[0].seq : null;
  }
  const selected = shown.filter((e) => e.seq === view.selected)[0] || null;
  paintTable(shown, all.length);
  paintTimeline(selected);
  paintFlow(selected);
  paintDetails(selected);
}

/* ---------- lifecycle ---------- */

export function mount(root, route) {
  ui = {};
  build(root, route);
  unsubscribe = session.subscribe(paint);
  paint();
}

export function unmount() {
  if (unsubscribe) { unsubscribe(); }
  unsubscribe = null;
  ui = null;
}
