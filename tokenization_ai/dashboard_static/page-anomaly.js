/* Anomaly Detection page: the one functional page in Phase 1.
 *
 * It posts a synthetic preset to the real POST /v1/analyze and renders what
 * the backend returned. It does no masking, tokenizing or scoring. The score
 * is shown as an uncalibrated model output, never as a verdict. The in-memory
 * session list keeps response summaries only, never the request. */

import { el, clear, icon, card, kv, badge, emptyNote, fmtMs, kpi } from "./ui.js";
import { gauge, donut } from "./charts.js";
import * as api from "./api.js";
import { shell } from "./shell.js";
import { session, valueAtPath } from "./session.js";

const PIPELINE = [
  ["fusion", "Input fusion", "merge"],
  ["masking", "Privacy masking", "lock"],
  ["normalization", "Normalization", "filter"],
  ["tokenization", "Tokenization", "code"],
  ["tensorization", "Tensor preparation", "grid"],
  ["model_inference", "AI prediction", "activity"]
];

const ROLES = [
  ["control", "Control tag"],
  ["literal_api", "API literal"],
  ["masked", "Masked"],
  ["literal_rule", "Rule literal"]
];

const LIMITS = [
  "Synthetic demonstration data only. The rules shown are MOCK fixtures, not Formal Verification output.",
  "The model is fitted on two hand-written synthetic fixtures. Its score separates those fixtures; it is not detection accuracy.",
  "The score is uncalibrated: not a probability of attack, not a verdict, and not an Allow or Block decision.",
  "Masking is a name- and shape-based blocklist, not a guarantee. Unlisted field names are not masked.",
  "Formal Verification and Runtime Verification integration and KPI 5 are pending; nothing on this page represents them.",
  "Stage timings are single-request, in-process measurements from this run. They are not the KPI 4 benchmark or HTTP latency."
];

const HISTORY_ROWS = 8;

const state = {
  scenarios: [],
  loaded: false,
  scenariosFailed: false,
  selectedId: null,
  renderedId: null,
  busy: false,
  result: null,
  error: null
};

let ui = null;
let unsubscribe = null;

/* ---------- helpers ---------- */

function currentScenario() {
  return state.scenarios.filter((s) => s.id === state.selectedId)[0] || null;
}

function stageOf(data, name) {
  return data.timings.stages.filter((s) => s.stage === name)[0] || null;
}

function stageMs(data, name) {
  const stage = stageOf(data, name);
  return stage ? fmtMs(stage.elapsed_ms) + " ms" : "not recorded";
}

/* ---------- build (static structure, built once per mount) ---------- */

function build(root, route) {
  const head = el("div", "page-head");
  head.appendChild(el("h1", "page-title", route.title));
  head.appendChild(el("p", "page-sub", route.subtitle));
  root.appendChild(head);

  ui.error = el("div", "error-box");
  ui.error.setAttribute("role", "alert");
  ui.error.hidden = true;
  root.appendChild(ui.error);

  // Real-time-free KPI tiles: every value is read from the backend response.
  const tiles = el("div", "grid g-4");
  ui.kReady = kpi("green", "shield", "Service readiness");
  ui.kScore = kpi("blue", "activity", "Anomaly score");
  ui.kTokens = kpi("purple", "code", "Tokens in sequence");
  ui.kMasked = kpi("pink", "lock", "Fields masked");
  [ui.kReady, ui.kScore, ui.kTokens, ui.kMasked].forEach((k) => tiles.appendChild(k.root));
  root.appendChild(tiles);

  // Pipeline strip.
  const strip = el("ol", "pipeline");
  strip.setAttribute("aria-label", "LogiShield pipeline stages");
  ui.steps = {};
  PIPELINE.forEach((p) => {
    const li = el("li", "step");
    const ico = el("span", "step-ico");
    ico.appendChild(icon(p[2], 22));
    const text = el("div", "step-text");
    const meta = el("div", "step-meta", "—");
    text.appendChild(el("div", "step-name", p[1]));
    text.appendChild(meta);
    li.appendChild(ico);
    li.appendChild(text);
    strip.appendChild(li);
    ui.steps[p[0]] = { root: li, meta };
  });
  root.appendChild(strip);

  // Row: controls | score | timings.
  const main = el("div", "grid g-main");

  const ctl = card({ title: "Scenario & request", icon: "play", badge: badge("Synthetic", "synth") });
  ui.scenarioList = el("div", "scenario-list");
  ui.scenarioList.setAttribute("role", "radiogroup");
  ui.scenarioList.setAttribute("aria-label", "Synthetic scenario");
  ui.scenarioButtons = [];
  ctl.body.appendChild(ui.scenarioList);
  const reqLabel = el("p", "label");
  reqLabel.appendChild(el("span", null, "Request sent to the API"));
  reqLabel.appendChild(badge("Before masking", "synth"));
  ctl.body.appendChild(reqLabel);
  ui.requestView = el("pre", "code code-short", "Select a scenario.");
  ui.requestView.tabIndex = 0;
  ui.requestView.setAttribute("aria-label", "Synthetic request body, shown before masking");
  ctl.body.appendChild(ui.requestView);
  ctl.body.appendChild(el("p", "card-note",
    "Synthetic fixture shown as sent. The backend masks it; the response is never used to rebuild it."));
  ui.ruleBox = el("div", "rule-box");
  ui.ruleBox.hidden = true;
  ui.ruleChip = badge("", "mock");
  ui.ruleText = el("p", null, "");
  ui.ruleBox.appendChild(ui.ruleChip);
  ui.ruleBox.appendChild(ui.ruleText);
  ctl.body.appendChild(ui.ruleBox);
  ui.analyze = el("button", "btn");
  ui.analyze.type = "button";
  ui.analyzeIcon = el("span");
  ui.analyzeLabel = el("span", null, "Analyze");
  ui.analyze.appendChild(ui.analyzeIcon);
  ui.analyze.appendChild(ui.analyzeLabel);
  ui.analyze.addEventListener("click", analyze);
  ctl.body.appendChild(ui.analyze);
  ui.status = el("p", "status");
  ui.status.setAttribute("role", "status");
  ui.status.setAttribute("aria-live", "polite");
  ctl.body.appendChild(ui.status);
  main.appendChild(ctl.root);

  const stack = el("div", "stack");
  const score = card({ title: "Lightweight AI prediction", icon: "activity", badge: badge("Uncalibrated", "info") });
  ui.scoreBody = score.body;
  stack.appendChild(score.root);

  const timing = card({ title: "Per-stage timing", icon: "clock", badge: badge("This run", "info") });
  ui.timingBody = timing.body;
  stack.appendChild(timing.root);
  main.appendChild(stack);
  root.appendChild(main);

  // Row: masking | tokenization.
  const mid = el("div", "grid g-wide");
  const masking = card({ title: "Privacy masking & normalization", icon: "lock" });
  ui.maskingBody = masking.body;
  const tokens = card({ title: "Structure-aware tokenization", icon: "code" });
  ui.tokensBody = tokens.body;
  mid.appendChild(masking.root);
  mid.appendChild(tokens.root);
  root.appendChild(mid);

  // Row: context & fusion | tensor | session.
  const low = el("div", "grid g-3");
  const context = card({ title: "Input context & fusion", icon: "merge" });
  ui.contextBody = context.body;
  ui.contextBadgeHost = context.head;
  const tensor = card({ title: "Tensor preparation", icon: "grid" });
  ui.tensorBody = tensor.body;
  const sessionCard = card({ title: "This session", icon: "history", badge: badge("In memory", "info") });
  ui.historyBody = sessionCard.body;
  low.appendChild(context.root);
  low.appendChild(tensor.root);
  low.appendChild(sessionCard.root);
  root.appendChild(low);

  // Row: provenance | limitations.
  const foot = el("div", "grid g-2");
  const prov = card({ title: "Provenance & privacy projection", icon: "shield" });
  ui.provBody = prov.body;
  const limits = card({ title: "Research limitations", icon: "warn", badge: badge("Read me", "mock") });
  const ul = el("ul", "limits");
  LIMITS.forEach((t) => ul.appendChild(el("li", null, t)));
  limits.body.appendChild(ul);
  foot.appendChild(prov.root);
  foot.appendChild(limits.root);
  root.appendChild(foot);
}

/* ---------- paint ---------- */

function paintHealth(health) {
  if (!ui) { return; }
  const tone = health.state === "ready" ? "green" : (health.state === "checking" ? "blue" : "red");
  ui.kReady.root.className = "kpi kpi-" + tone;
  const word = { ready: "Ready", notready: "Not ready", down: "Unreachable", checking: "Checking…" };
  ui.kReady.value.textContent = word[health.state] || "Unknown";
  ui.kReady.sub.textContent = "GET /health" + (health.service ? " · " + health.service : "");
}

function paintControls() {
  if (!ui) { return; }
  const scenario = currentScenario();

  if (ui.scenarioButtons.length !== state.scenarios.length) {
    clear(ui.scenarioList);
    ui.scenarioButtons = state.scenarios.map((s) => {
      const button = el("button", "scenario");
      button.type = "button";
      button.setAttribute("role", "radio");
      button.appendChild(el("strong", null, s.title));
      button.appendChild(el("span", null, s.description));
      button.addEventListener("click", () => select(s.id));
      ui.scenarioList.appendChild(button);
      return button;
    });
    if (!state.scenarios.length) {
      ui.scenarioList.appendChild(emptyNote(state.loaded ? "Could not load the demo scenarios." : "Loading scenarios…"));
    }
  }
  state.scenarios.forEach((s, i) => {
    ui.scenarioButtons[i].setAttribute("aria-checked", String(s.id === state.selectedId));
    ui.scenarioButtons[i].disabled = state.busy;
  });

  ui.requestView.textContent = scenario ? JSON.stringify(scenario.request, null, 2) : "Select a scenario.";

  if (scenario) {
    const source = scenario.request.rule_source;
    ui.ruleBox.hidden = false;
    if (source === "mock_fixture") {
      ui.ruleChip.textContent = "MOCK rule fixture";
      ui.ruleText.textContent = "Illustrative mock data, not genuine Formal Verification verification output. This service does not verify rules.";
    } else {
      ui.ruleChip.textContent = "No verified-rule input";
      ui.ruleText.textContent = "No rules supplied. This is not a verification result.";
    }
  } else {
    ui.ruleBox.hidden = true;
  }

  ui.analyze.disabled = state.busy || !scenario;
  ui.analyze.className = "btn" + (state.busy ? " is-loading" : "");
  clear(ui.analyzeIcon);
  ui.analyzeIcon.appendChild(icon(state.busy ? "spinner" : "play", 18));
  ui.analyzeLabel.textContent = state.busy ? "Analyzing…" : "Analyze";

  let kind = "";
  let text = "Loading scenarios…";
  if (state.busy) { kind = "loading"; text = "Sending the synthetic request to POST /v1/analyze…"; }
  else if (state.error) { kind = "fail"; text = "Analysis failed."; }
  else if (state.result) { kind = "ok"; text = "Analysis complete. These values came from the LogiShield service."; }
  else if (state.scenariosFailed) { kind = "fail"; text = "Demo scenarios unavailable."; }
  else if (scenario) { text = "Ready. Press Analyze to send this synthetic request to the backend."; }
  ui.status.className = "status" + (kind ? " " + kind : "");
  ui.status.textContent = text;
}

function paintKpisAndPipeline(data) {
  ui.kScore.value.textContent = data ? data.prediction.score.toFixed(4) : "–";
  ui.kScore.sub.textContent = data ? "Model " + data.prediction.model_version : "Uncalibrated · not a verdict";
  ui.kTokens.value.textContent = data ? String(data.tokens.total) : "–";
  ui.kTokens.sub.textContent = data
    ? "of " + data.tensor.max_length + " fixed length · " + data.tensor.padded_positions + " padded"
    : "Fixed-length tensor";
  ui.kMasked.value.textContent = data ? String(data.masking.masked_field_count) : "–";
  ui.kMasked.sub.textContent = data ? "Path-based provenance" : "Privacy masking";

  PIPELINE.forEach((p) => {
    const step = ui.steps[p[0]];
    const stage = data ? stageOf(data, p[0]) : null;
    step.root.className = "step" + (stage ? " done" : (state.busy ? " busy" : ""));
    step.meta.textContent = stage ? fmtMs(stage.elapsed_ms) + " ms" : (state.busy ? "…" : "—");
  });
}

function paintScore(data) {
  const body = ui.scoreBody;
  clear(body);
  if (!data) {
    body.appendChild(emptyNote("Run an analysis to see the model output."));
    return;
  }
  const grid = el("div", "score-grid");
  const wrap = el("div", "score-wrap");
  wrap.appendChild(gauge(data.prediction.score));
  wrap.appendChild(el("p", "gauge-num", data.prediction.score.toFixed(4)));
  wrap.appendChild(el("p", "gauge-cap", "Uncalibrated model output (0 to 1)"));
  grid.appendChild(wrap);
  const side = el("div", "score-side");
  side.appendChild(kv([
    ["Model version", data.prediction.model_version],
    ["Inference time", stageMs(data, "model_inference")],
    ["Total pipeline time", fmtMs(data.timings.total_elapsed_ms) + " ms"]
  ]));
  side.appendChild(el("p", "caveat",
    data.provenance.model.note +
    " This score is not a verdict, not a probability, not a threshold decision and not detection accuracy."));
  grid.appendChild(side);
  body.appendChild(grid);
}

function paintTiming(data) {
  const body = ui.timingBody;
  clear(body);
  if (!data) {
    body.appendChild(emptyNote("Run an analysis to see per-stage timings."));
    return;
  }
  const stages = data.timings.stages;
  const max = stages.reduce((m, s) => Math.max(m, s.elapsed_ms), 0) || 1;
  const bars = el("div", "bars");
  stages.forEach((stage) => {
    const row = el("div", "bar-row");
    row.appendChild(el("span", "bar-name", stage.stage));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = Math.max(2, (stage.elapsed_ms / max) * 100).toFixed(1) + "%";
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("span", "bar-val", fmtMs(stage.elapsed_ms) + " ms"));
    bars.appendChild(row);
  });
  body.appendChild(bars);
  const total = el("div", "bar-total");
  total.appendChild(el("span", null, "Total pipeline time"));
  total.appendChild(el("strong", null, fmtMs(data.timings.total_elapsed_ms) + " ms"));
  body.appendChild(total);
  body.appendChild(el("p", "card-note",
    "One request, in-process, this run only. Not the KPI 4 benchmark and not HTTP latency."));
}

function paintMasking(data) {
  const body = ui.maskingBody;
  clear(body);
  if (!data) {
    body.appendChild(emptyNote("Run an analysis to see what was masked and how the request was normalized."));
    return;
  }
  const rep = data.masked_representation;
  body.appendChild(kv([
    ["Fields masked", data.masking.masked_field_count],
    ["Values hidden in preview", data.privacy.hidden_in_representation],
    ["Masking time", stageMs(data, "masking")],
    ["Normalization time", stageMs(data, "normalization")]
  ]));
  body.appendChild(el("p", "label", "Masked / normalized representation"));
  const pre = el("pre", "code", JSON.stringify({ method: rep.method, path: rep.path, headers: rep.headers, payload: rep.payload }, null, 2));
  pre.tabIndex = 0;
  body.appendChild(pre);

  body.appendChild(el("p", "label", "Masked-field provenance"));
  const fields = data.masking.masked_fields;
  if (!fields.length) {
    body.appendChild(emptyNote("No fields were masked for this request."));
    return;
  }
  const wrap = el("div", "table-wrap");
  const table = el("table", "table");
  const headRow = table.createTHead().insertRow();
  ["Field path", "Replaced with", "Basis"].forEach((h) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = h;
    headRow.appendChild(th);
  });
  const tbody = table.createTBody();
  fields.forEach((field) => {
    const row = tbody.insertRow();
    row.insertCell().textContent = field;
    const placeholder = valueAtPath(rep, field);
    const cell = row.insertCell();
    if (placeholder) {
      const chip = el("span", "tok role-masked", placeholder);
      cell.appendChild(chip);
    } else {
      cell.textContent = "not shown";
    }
    row.insertCell().textContent = "Masker path record";
  });
  wrap.appendChild(table);
  body.appendChild(wrap);
  if (data.masking.masked_fields_truncated) {
    body.appendChild(el("p", "card-note", "Field list truncated by the backend."));
  }
  body.appendChild(el("p", "card-note",
    "A token is marked masked because the trusted masker recorded its path, not because its text looks like a placeholder."));
}

function paintTokens(data) {
  const body = ui.tokensBody;
  clear(body);
  if (!data) {
    body.appendChild(emptyNote("Run an analysis to see the token sequence and the role of every token."));
    return;
  }
  const items = data.tokens.items;
  const counts = {};
  ROLES.forEach((r) => { counts[r[0]] = 0; });
  items.forEach((item) => { counts[item.role] = (counts[item.role] || 0) + 1; });

  body.appendChild(kv([
    ["Tokens produced", data.tokens.total],
    ["Tokens displayed", data.tokens.returned + (data.tokens.truncated ? " (truncated)" : "")],
    ["Literals hidden in preview", data.privacy.hidden_in_tokens],
    ["Stage time", stageMs(data, "tokenization")]
  ]));

  const split = el("div", "split");
  const donutWrap = el("div", "donut-wrap");
  donutWrap.appendChild(donut(ROLES.map((r) => ({ key: r[0], value: counts[r[0]] }))));
  const center = el("div", "donut-center");
  center.appendChild(el("strong", null, items.length));
  center.appendChild(el("span", null, "tokens"));
  donutWrap.appendChild(center);
  split.appendChild(donutWrap);
  const legend = el("ul", "legend");
  ROLES.forEach((r) => {
    const li = el("li");
    li.appendChild(el("span", "sw sw-" + r[0]));
    li.appendChild(el("span", null, r[1]));
    li.appendChild(el("span", "cnt", counts[r[0]]));
    legend.appendChild(li);
  });
  split.appendChild(legend);
  body.appendChild(split);

  body.appendChild(el("p", "label", "Token sequence"));
  const list = el("ul", "chips tokens");
  list.setAttribute("aria-label", "Token sequence");
  items.forEach((item) => {
    const chip = el("li", "tok role-" + item.role, item.token);
    chip.title = item.role;
    list.appendChild(chip);
  });
  body.appendChild(list);
  body.appendChild(el("p", "card-note",
    "Roles are assigned by the tokenizer when each token is emitted, not inferred from text."));
}

function paintContext(data) {
  const body = ui.contextBody;
  clear(body);
  if (!data) {
    body.appendChild(emptyNote("Run an analysis to see the input context and the fusion result."));
    return;
  }
  const rule = data.provenance.rule_status;
  const rep = data.masked_representation;
  body.appendChild(el("p", "label", "Input context"));
  body.appendChild(kv([
    ["Event id", data.event_id],
    ["Method", rep.method],
    ["Path", rep.path],
    ["Rule source", rule.label],
    ["Contract", data.contract_version]
  ]));
  const fusionLabel = el("p", "label");
  fusionLabel.appendChild(el("span", null, "Input fusion"));
  if (rule.source === "mock_fixture") { fusionLabel.appendChild(badge("Mock rules", "mock")); }
  body.appendChild(fusionLabel);
  body.appendChild(kv([
    ["Rules fused", rule.rule_count],
    ["Rule ids shown", rule.rule_ids.length ? rule.rule_ids.join(", ") : "none"],
    ["Verified by this service", rule.verification_performed_by_this_service ? "yes" : "no"],
    ["Stage time", stageMs(data, "fusion")]
  ]));
  body.appendChild(el("p", "card-note", rule.note));
}

function paintTensor(data) {
  const body = ui.tensorBody;
  clear(body);
  if (!data) {
    body.appendChild(emptyNote("Run an analysis to see the fixed-length tensor preparation."));
    return;
  }
  const t = data.tensor;
  const used = t.max_length ? (t.tokens_in_tensor / t.max_length) * 100 : 0;
  const bar = el("div", "occupancy");
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", t.tokens_in_tensor + " of " + t.max_length + " positions used");
  const usedPart = el("div", "occ-used");
  usedPart.style.width = used.toFixed(2) + "%";
  const pad = el("div", "occ-pad");
  pad.style.width = (100 - used).toFixed(2) + "%";
  bar.appendChild(usedPart);
  bar.appendChild(pad);
  body.appendChild(bar);
  const legend = el("div", "occ-legend");
  legend.appendChild(el("span", null, t.tokens_in_tensor + " tokens"));
  legend.appendChild(el("span", null, t.padded_positions + " padding"));
  body.appendChild(legend);
  body.appendChild(kv([
    ["Fixed length", t.max_length],
    ["Tokens before truncation", t.token_count_before_truncation],
    ["Tokens in tensor", t.tokens_in_tensor],
    ["Padded positions", t.padded_positions],
    ["Payload truncated", t.payload_truncated ? "yes" : "no"],
    ["Unknown-token count", t.unknown_token_count],
    ["Vocabulary size", t.vocabulary_size],
    ["Stage time", stageMs(data, "tensorization")]
  ]));
  body.appendChild(el("p", "card-note", "Token ids are not exposed by the API."));
}

function paintHistory() {
  const body = ui.historyBody;
  clear(body);
  body.appendChild(el("p", "card-note",
    "Summaries of analyses run on this page since it loaded. Held in memory only; the request is never stored."));
  const entries = session.list().slice(0, HISTORY_ROWS);
  if (!entries.length) {
    body.appendChild(emptyNote("No analyses yet."));
    return;
  }
  const wrap = el("div", "table-wrap");
  const table = el("table", "table");
  const headRow = table.createTHead().insertRow();
  ["#", "Scenario", "Score", "Tokens"].forEach((h) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = h;
    headRow.appendChild(th);
  });
  const tbody = table.createTBody();
  entries.forEach((entry, index) => {
    const row = tbody.insertRow();
    if (index === 0) { row.className = "is-current"; }
    row.insertCell().textContent = String(entry.seq);
    const nameCell = row.insertCell();
    nameCell.className = "cap";
    nameCell.textContent = entry.scenarioId;
    [entry.score.toFixed(4), String(entry.tokens.total)].forEach((v) => {
      const cell = row.insertCell();
      cell.className = "num";
      cell.textContent = v;
    });
  });
  wrap.appendChild(table);
  body.appendChild(wrap);
  body.appendChild(el("p", "card-note",
    "Scores are uncalibrated outputs of a model fitted on these two synthetic fixtures; the comparison is not evidence of detection accuracy."));
  const link = el("a", "card-link", "Open full session traces in Logs & Traces");
  link.href = "#/logs";
  body.appendChild(link);
}

function paintProvenance(data) {
  const body = ui.provBody;
  clear(body);
  if (!data) {
    body.appendChild(emptyNote("Run an analysis to see provenance and the privacy projection."));
    return;
  }
  const p = data.provenance;
  body.appendChild(kv([
    ["Input origin", p.input_origin],
    ["Rule source", p.rule_status.source],
    ["Rule status", p.rule_status.label],
    ["Model version", p.model.model_version],
    ["Detection accuracy claim", p.detection_accuracy_claim],
    ["Output kind", data.privacy.output_kind],
    ["Policy", data.privacy.policy]
  ]));
  body.appendChild(el("p", "card-note", data.privacy.note));
}

function paintResults() {
  if (!ui) { return; }
  const data = state.result;
  ui.error.hidden = !state.error;
  ui.error.textContent = state.error || "";
  paintKpisAndPipeline(data);
  paintScore(data);
  paintTiming(data);
  paintMasking(data);
  paintTokens(data);
  paintContext(data);
  paintTensor(data);
  paintHistory();
  paintProvenance(data);
}

/* ---------- actions ---------- */

function select(id) {
  if (state.busy) { return; }
  if (state.renderedId !== null && state.renderedId !== id) {
    // Never leave the previous scenario's result under a different request.
    state.result = null;
    state.renderedId = null;
  }
  state.error = null;
  state.selectedId = id;
  paintControls();
  paintResults();
}

async function analyze() {
  if (state.busy) { return; }
  const scenario = currentScenario();
  if (!scenario) { return; }

  state.busy = true;
  state.error = null;
  paintControls();
  paintResults();

  const outcome = await api.analyze(scenario.request);

  state.busy = false;
  if (outcome.ok) {
    const d = outcome.data;
    state.result = d;
    state.renderedId = scenario.id;
    state.error = null;
    session.add(d, scenario);
  } else {
    state.result = null;
    state.renderedId = null;
    state.error = outcome.message;
  }
  paintControls();
  paintResults();
  shell.refreshHealth(true);
}

async function loadScenarios() {
  try {
    state.scenarios = await api.getScenarios();
    state.scenariosFailed = state.scenarios.length === 0;
    if (state.selectedId === null && state.scenarios.length) { state.selectedId = state.scenarios[0].id; }
  } catch (error) {
    state.scenarios = [];
    state.scenariosFailed = true;
  }
  state.loaded = true;
  paintControls();
}

/* ---------- lifecycle ---------- */

export function mount(root, route) {
  ui = {};
  build(root, route);
  unsubscribe = shell.subscribe(paintHealth);
  paintHealth(shell.health);
  paintControls();
  paintResults();
  if (!state.loaded) { loadScenarios(); }
}

export function unmount() {
  if (unsubscribe) { unsubscribe(); }
  unsubscribe = null;
  ui = null;
}
