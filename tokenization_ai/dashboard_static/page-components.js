/* System Components: architecture and integration status.
 *
 * Everything here is either (a) documented architecture/integration fact
 * taken from the repository README and tokenization_ai/INTEGRATION_STATUS.md,
 * or (b) Member 4's own live readiness from GET /health (via shell.js) and
 * the last observed stage timings from this session. No status, latency,
 * CPU, memory, throughput or uptime is shown for any other module. */

import { el, clear, svg, icon, card, badge, emptyNote, fmtMs, button } from "./ui.js";
import { shell } from "./shell.js";
import { session } from "./session.js";

export const COMPONENTS = [
  {
    order: 1, name: "Data Processing", member: "Member 1", icon: "database", module: "ingestion/",
    role: "Ingress interception, buffering and serialization, data masking and context dispatching.",
    output: "Structured Unified API Context",
    link: ["Tested in-process", "info"],
    evidence: "Member 1's real output is accepted by Member 4's adapter in tests. No network hop, broker or deployment is involved."
  },
  {
    order: 2, name: "Formal Verification", member: "Member 2", icon: "rules", module: "processing/",
    role: "LTL/SMT rule encoding, neuro-symbolic constraint layer and formal verification engine.",
    output: "Rule cache, LTL formula, JSON vector / verified rule outputs",
    link: ["Not integrated", "pending"],
    evidence: "Member 2's report has a different shape and no event id. Member 4's demos use MOCK rule fixtures."
  },
  {
    order: 3, name: "Tokenization & AI", member: "Member 4", icon: "cpu", module: "tokenization_ai/", live: true,
    role: "Input fusion, privacy-aware masking, structure-aware tokenization, tensor preparation and lightweight AI prediction.",
    output: "Anomaly score, internal AI-ready tensor, per-stage trace",
    link: ["Standalone", "info"],
    evidence: "Runs standalone on 127.0.0.1:8004 and is not part of docker-compose. The rule-aligned signal is not implemented."
  },
  {
    order: 4, name: "Runtime Verification", member: "Member 3", icon: "shield", module: "engine/, api_gateway/",
    role: "Z3 runtime rule engine, hybrid circuit breaker and mock API gateway router.",
    output: "Allow/Block verdict and XAI trace",
    link: ["Pending contract", "pending"],
    evidence: "No Member 3 input contract exists, and the repository holds only placeholder code for this module."
  }
];

const PIPELINE = [
  ["fusion", "Input fusion", "merge", "fusion.py", "InputFusion"],
  ["masking", "Privacy masking", "lock", "masking.py", "PrivacyMasker"],
  ["normalization", "Normalization", "filter", "normalization.py", "Normalizer"],
  ["tokenization", "Tokenization", "code", "tokenizer.py", "HybridTokenizer"],
  ["tensorization", "Tensor preparation", "grid", "tensorizer.py", "Tensorizer"],
  ["model_inference", "AI prediction", "activity", "model.py", "LightweightAnomalyModel"]
];

const INTEGRATION = [
  {
    link: "Member 1 → Member 4", what: "Unified API context",
    status: ["Tested in-process", "info"],
    evidence: "tests/tokenization_ai/test_member1_integration.py passes Member 1's real app output through Member 4's adapter, service and API.",
    limit: "Not a network, broker or deployment integration. Standard (non-interception) ingest events cannot be consumed by Member 4."
  },
  {
    link: "Member 2 → Member 4", what: "Verified rule information",
    status: ["Not integrated", "pending"],
    evidence: "Documented schema mismatch: Member 2's verification report has no event id and a different shape from Member 4's rule input.",
    limit: "Every rule in the demos is a MOCK fixture. Member 4 never verifies rules; it only tokenizes the rule information it is given."
  },
  {
    link: "Member 4 → Member 3", what: "Anomaly-aware signal",
    status: ["Pending contract", "pending"],
    evidence: "adapters.to_member3_output raises NotImplementedError by design. No Member 3 input contract exists.",
    limit: "Nothing is sent to Member 3, and no Allow/Block decision exists anywhere in this system."
  },
  {
    link: "KPI 5", what: "Downstream usability",
    status: ["Not measured", "pending"],
    evidence: "It cannot be measured honestly without a genuine Member 3 contract and consumer.",
    limit: "No percentage is reported or implied."
  },
  {
    link: "Deployment", what: "Packaging",
    status: ["Standalone", "info"],
    evidence: "Member 4 is not listed in docker-compose.yml; it is started with python -m tokenization_ai.",
    limit: "A complete deployed system is not demonstrated."
  },
  {
    link: "engine/ on port 8003", what: "Generic placeholder",
    status: ["Different component", "pending"],
    evidence: "engine/ holds a generic StubModel (score = the largest feature value, clipped to 0–1) in the Member 3 module slot.",
    limit: "It is not Member 4's trained model on port 8004, and this dashboard never calls it."
  }
];

const LIMITATIONS = [
  "The Member 4 service is stateless. The dashboard keeps only allowlisted analysis summaries in memory for the current page load.",
  "The model scores hashed n-gram features of token ids with a logistic regression. There is no learned embedding table, and it is fitted on two synthetic fixtures.",
  "The rule-aligned signal and the next-component input named in the proposal are not defined or implemented.",
  "Masking is a name- and shape-based blocklist, not a guarantee.",
  "Arrows marked proposed or not integrated describe the intended architecture from the research README. They are not agreed contracts.",
  "No health, CPU, memory, throughput, uptime or instance data exists for any module other than Member 4's own readiness check."
];

let ui = null;
let unsubscribers = [];

/* ---------- diagram ---------- */

function text(parent, x, y, className, value) {
  const t = svg("text", { x: x, y: y, class: className });
  t.textContent = value;
  parent.appendChild(t);
  return t;
}

function node(parent, spec) {
  const g = svg("g", { class: "dg-node" + (spec.cls ? " " + spec.cls : "") });
  g.appendChild(svg("rect", { x: spec.x, y: spec.y, width: spec.w, height: spec.h, rx: 12 }));
  const ico = icon(spec.icon, 26);
  ico.setAttribute("x", spec.x + 12);
  ico.setAttribute("y", spec.y + 12);
  g.appendChild(ico);
  text(g, spec.x + 48, spec.y + 31, "dg-title", spec.title);
  spec.lines.forEach((line, i) => text(g, spec.x + 14, spec.y + 58 + i * 17, "dg-sub", line));
  parent.appendChild(g);
  return g;
}

function marker(defs, id, color) {
  const m = svg("marker", { id: id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" });
  m.appendChild(svg("path", { d: "M0 0L10 5L0 10z", fill: color }));
  defs.appendChild(m);
}

function edge(parent, d, kind) {
  parent.appendChild(svg("path", {
    d: d, class: "dg-edge dg-" + kind, fill: "none",
    "marker-end": "url(#" + (kind === "tested" ? "arr-tested" : "arr-pending") + ")"
  }));
}

function buildDiagram(health) {
  const root = svg("svg", { class: "diagram", viewBox: "0 0 1160 440", role: "img",
    "aria-label": "LogiShield component architecture: API requests, Member 1, Member 2, Member 4, Member 3 and the decision" });
  const defs = svg("defs");
  marker(defs, "arr-tested", "#22d3ee");
  marker(defs, "arr-pending", "#8ea5cf");
  root.appendChild(defs);

  edge(root, "M150 215 C 178 215 172 90 200 90", "pending");
  edge(root, "M410 90 C 446 90 444 190 480 190", "tested");
  edge(root, "M410 350 C 446 350 444 250 480 250", "pending");
  edge(root, "M720 220 L 788 220", "pending");
  edge(root, "M1020 220 L 1046 220", "pending");

  text(root, 424, 54, "dg-edge-label dg-label-tested", "tested in-process");
  text(root, 424, 388, "dg-edge-label", "not integrated");
  text(root, 424, 404, "dg-edge-label", "schemas differ");
  text(root, 726, 208, "dg-edge-label", "no contract");

  node(root, { x: 10, y: 165, w: 140, h: 100, icon: "globe", title: "API requests",
    lines: ["Ingress", "interception"], cls: "dg-ghost" });
  node(root, { x: 200, y: 30, w: 210, h: 120, icon: "database", title: "1 · Data Processing",
    lines: ["Member 1", "Unified API context"] });
  node(root, { x: 200, y: 290, w: 210, h: 120, icon: "rules", title: "2 · Formal Verification",
    lines: ["Member 2", "Verified rule information"], cls: "dg-ghost" });
  const m4 = node(root, { x: 480, y: 130, w: 240, h: 180, icon: "cpu", title: "3 · Tokenization & AI",
    lines: ["Member 4 · this service", "Fusion, masking, normalization,", "tokenization, tensor, model", "Anomaly score (uncalibrated)"], cls: "dg-m4" });
  const dotClass = health.state === "ready" ? "dg-dot-ok" : (health.state === "checking" ? "dg-dot-idle" : "dg-dot-bad");
  const label = { ready: "Ready (GET /health)", notready: "Not ready (GET /health)", down: "Unreachable", checking: "Checking…" }[health.state] || "Unknown";
  m4.appendChild(svg("circle", { cx: 494, cy: 289, r: 5, class: dotClass }));
  text(m4, 507, 293, "dg-sub", label);
  node(root, { x: 790, y: 165, w: 230, h: 110, icon: "shield", title: "4 · Runtime Verification",
    lines: ["Member 3", "Verdict + XAI trace"], cls: "dg-ghost" });
  node(root, { x: 1050, y: 165, w: 100, h: 110, icon: "decisions", title: "Decision",
    lines: ["Allow / Block", "not produced"], cls: "dg-ghost" });
  return root;
}

/* ---------- sections ---------- */

function healthBadge(health) {
  const map = {
    ready: ["Ready", "live"], notready: ["Not ready", "bad"], down: ["Unreachable", "bad"], checking: ["Checking…", "info"]
  };
  const m = map[health.state] || ["Unknown", "pending"];
  return badge(m[0], m[1]);
}

function paintCards(health) {
  const grid = ui.cards;
  clear(grid);
  COMPONENTS.forEach((c) => {
    const box = el("section", "card comp-card" + (c.live ? " comp-m4" : ""));
    const head = el("header", "comp-head");
    const ico = el("span", "comp-ico");
    ico.appendChild(icon(c.icon, 26));
    head.appendChild(ico);
    const title = el("div", "comp-title");
    title.appendChild(el("h2", null, c.name));
    title.appendChild(el("p", null, c.member + " · " + c.module));
    head.appendChild(title);
    box.appendChild(head);

    const body = el("div", "comp-body");
    body.appendChild(el("p", "comp-role", c.role));
    const rows = el("dl", "kv comp-kv");
    rows.appendChild(el("dt", null, "Output"));
    rows.appendChild(el("dd", null, c.output));
    rows.appendChild(el("dt", null, "Live status"));
    const status = el("dd");
    if (c.live) {
      status.appendChild(healthBadge(health));
      status.appendChild(el("span", "comp-note", " GET /health"));
    } else {
      status.appendChild(badge("Not connected", "pending"));
    }
    rows.appendChild(status);
    rows.appendChild(el("dt", null, "Integration"));
    const link = el("dd");
    link.appendChild(badge(c.link[0], c.link[1]));
    rows.appendChild(link);
    body.appendChild(rows);
    body.appendChild(el("p", "card-note", c.evidence));
    if (c.live) {
      body.appendChild(button("Re-check readiness", "btn-ghost", () => { shell.refreshHealth(); }, "refresh"));
    }
    box.appendChild(body);
    grid.appendChild(box);
  });
}

function paintDiagram(health) {
  clear(ui.diagram);
  ui.diagram.appendChild(buildDiagram(health));
}

function paintPipeline() {
  const grid = ui.pipeline;
  clear(grid);
  const last = session.list()[0] || null;
  PIPELINE.forEach((p, index) => {
    const box = el("div", "stage-card");
    const top = el("div", "stage-top");
    top.appendChild(el("span", "tl-dot", String(index + 1)));
    top.appendChild(icon(p[2], 20));
    box.appendChild(top);
    box.appendChild(el("strong", null, p[1]));
    box.appendChild(el("code", null, p[3]));
    box.appendChild(el("span", "stage-class", p[4]));
    const stage = last ? last.stages.filter((s) => s.stage === p[0])[0] : null;
    box.appendChild(el("span", "stage-ms", stage ? "last: " + fmtMs(stage.ms) + " ms" : "last: —"));
    grid.appendChild(box);
  });
  ui.pipelineNote.textContent = last
    ? "Timings are from the latest analysis in this session (event " + last.eventId + "), single request, not a benchmark."
    : "No analysis in this session yet, so no timings are shown. Run one on the Anomaly Detection page.";
}

function paintAll(health) {
  paintCards(health);
  paintDiagram(health);
  paintPipeline();
}

/* ---------- lifecycle ---------- */

export function mount(root, route) {
  ui = {};
  const head = el("div", "page-head");
  head.appendChild(el("h1", "page-title", route.title));
  head.appendChild(el("p", "page-sub", route.subtitle));
  root.appendChild(head);

  const banner = el("div", "pending-banner info-banner");
  banner.appendChild(badge("Documented", "info"));
  const textBox = el("div");
  textBox.appendChild(el("strong", null, "Architecture and integration status, not live telemetry"));
  textBox.appendChild(el("p", null,
    "Only Member 4's readiness is checked live (GET /health). Everything about the other modules is documented repository fact; " +
    "no operational status, load or performance figure is shown for them."));
  banner.appendChild(textBox);
  root.appendChild(banner);

  ui.cards = el("div", "grid comp-grid");
  root.appendChild(ui.cards);

  const arch = card({ title: "Component architecture & data flow", icon: "components", badge: badge("Proposed flow", "pending") });
  ui.diagram = el("div", "diagram-wrap");
  arch.body.appendChild(ui.diagram);
  const legend = el("ul", "edge-legend");
  [["tested", "Tested in-process (data contract only, not deployed)"],
   ["pending", "Proposed or pending: not integrated, no agreed contract"],
   ["ghost", "Module not connected to this dashboard"]].forEach((item) => {
    const li = el("li");
    li.appendChild(el("span", "lg-swatch lg-" + item[0]));
    li.appendChild(el("span", null, item[1]));
    legend.appendChild(li);
  });
  arch.body.appendChild(legend);
  arch.body.appendChild(el("p", "card-note",
    "Order follows the research README pipeline. Member 4 consumes the unified API context and verified rule information, then hands an anomaly-aware signal to Member 3; only the first link has test evidence."));
  root.appendChild(arch.root);

  const impl = card({ title: "Member 4 pipeline (implemented)", icon: "layers", badge: badge("Implemented", "live") });
  ui.pipeline = el("div", "stage-cards");
  impl.body.appendChild(ui.pipeline);
  ui.pipelineNote = el("p", "card-note", "");
  impl.body.appendChild(ui.pipelineNote);
  impl.body.appendChild(el("p", "card-note",
    "Orchestrated by service.py (TokenizationAIService) and exposed through app.py as POST /v1/analyze on 127.0.0.1:8004."));
  root.appendChild(impl.root);

  const integ = card({ title: "Integration status", icon: "merge" });
  const wrap = el("div", "table-wrap");
  const table = el("table", "table integ");
  const headRow = table.createTHead().insertRow();
  ["Link", "Status", "Evidence", "What this does not show"].forEach((h) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = h;
    headRow.appendChild(th);
  });
  const tbody = table.createTBody();
  INTEGRATION.forEach((r) => {
    const row = tbody.insertRow();
    const first = row.insertCell();
    first.appendChild(el("strong", null, r.link));
    first.appendChild(el("span", "cell-sub", r.what));
    row.insertCell().appendChild(badge(r.status[0], r.status[1]));
    row.insertCell().textContent = r.evidence;
    row.insertCell().textContent = r.limit;
  });
  wrap.appendChild(table);
  integ.body.appendChild(wrap);
  integ.body.appendChild(el("p", "card-note",
    "Sources: README.md, tokenization_ai/INTEGRATION_STATUS.md, tokenization_ai/CONTRACT_PROPOSAL.md and docker-compose.yml."));
  root.appendChild(integ.root);

  const lim = card({ title: "Documented limitations", icon: "warn", badge: badge("Read me", "mock") });
  const ul = el("ul", "limits");
  LIMITATIONS.forEach((t) => ul.appendChild(el("li", null, t)));
  lim.body.appendChild(ul);
  root.appendChild(lim.root);

  paintAll(shell.health);
  unsubscribers = [
    shell.subscribe((health) => { if (ui) { paintCards(health); paintDiagram(health); } }),
    session.subscribe(() => { if (ui) { paintPipeline(); } })
  ];
}

export function unmount() {
  unsubscribers.forEach((off) => off());
  unsubscribers = [];
  ui = null;
}
