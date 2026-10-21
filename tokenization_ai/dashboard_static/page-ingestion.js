/* Ingestion Component Page: Live HTTP Interception, AST PII Masking,
 * Zero-Copy Protobuf Serialization, and Unified Context Vector Generation. */

import { el, clear, card, badge, button, icon, kv, emptyNote, fmtMs } from "./ui.js";
import { interceptRequest, getIngestionHealth, getContextStream } from "./api.js";

let pollTimer = null;

const PRESETS = [
  {
    id: "orders",
    name: "E-Commerce Order Checkout",
    subtitle: "High-value purchase with credit card, email, password & shipping address",
    method: "POST",
    path: "/v1/orders",
    clientIp: "198.51.100.42",
    clientRole: "customer",
    threatScore: 0.25,
    headers: {
      "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "X-Forwarded-For": "198.51.100.42"
    },
    payload: {
      "order_id": "ord-88392",
      "customer_email": "jane.doe@example.com",
      "account_password": "superSecretPassword123!",
      "credit_card": "4532-1234-5678-9010",
      "cvv": "842",
      "amount": 149.95,
      "items_count": 3,
      "shipping_address": "742 Evergreen Terrace, Springfield"
    }
  },
  {
    id: "auth",
    name: "Credential Abuse & High-Risk Auth",
    subtitle: "Suspicious login attempt with high AI threat score and admin targets",
    method: "POST",
    path: "/v1/auth/token",
    clientIp: "203.0.113.195",
    clientRole: "anonymous",
    threatScore: 0.92,
    headers: {
      "X-Real-IP": "203.0.113.195"
    },
    payload: {
      "username": "root_admin",
      "password": "Password999!",
      "client_secret": "sec_live_9f81a7b6c5d4e3f2",
      "attempt_count": 14,
      "risk_flag": 1
    }
  },
  {
    id: "healthcare",
    name: "Healthcare Patient Record Access",
    subtitle: "Clinical request with patient SSN, diagnosis codes and medical notes",
    method: "POST",
    path: "/api/v2/patient-records",
    clientIp: "10.240.0.15",
    clientRole: "doctor",
    threatScore: 0.45,
    headers: {
      "Authorization": "Bearer clin-token-771"
    },
    payload: {
      "patient_name": "John Smith",
      "patient_ssn": "987-65-4321",
      "dob": "1984-07-15",
      "diagnosis_code": "E11.9",
      "patient_notes": "Prescription renewal request for insulin."
    }
  },
  {
    id: "transfer",
    name: "Financial Wire Transfer",
    subtitle: "Cross-border banking transfer with IBAN, currency, and amount",
    method: "POST",
    path: "/api/v1/transfers/wire",
    clientIp: "192.0.2.77",
    clientRole: "authenticated_user",
    threatScore: 0.78,
    headers: {
      "X-Idempotency-Key": "idemp-wire-991823"
    },
    payload: {
      "sender_account": "ACC-5829104",
      "recipient_iban": "DE89370400440532013000",
      "amount": 12500.00,
      "currency": "EUR",
      "memo": "Property payment invoice #412"
    }
  }
];

function codeBlock(value) {
  const pre = el("pre", "policy-code");
  pre.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return pre;
}

function stage(title, iconName, content, tone = "info") {
  const item = el("article", "policy-stage");
  const head = el("div", "policy-stage-head");
  const marker = el("span", "policy-stage-marker");
  marker.appendChild(icon(iconName, 18));
  head.appendChild(marker);
  head.appendChild(el("h3", null, title));
  head.appendChild(badge(tone === "live" ? "Executed" : (tone === "bad" ? "Failed" : "Captured"), tone));
  item.appendChild(head);
  item.appendChild(content);
  return item;
}

export function mount(container, route) {
  clear(container);

  const head = el("header", "page-head");
  head.appendChild(el("h1", "page-title", route ? route.title : "Data Ingestion & Interception"));
  head.appendChild(el("p", "page-sub",
    route ? route.subtitle : "Intercept HTTP traffic, apply AST PII data masking, serialize to Protobuf, and generate Unified Context Vectors."));
  container.appendChild(head);

  // Status Banner
  const statusBanner = el("div", "pending-banner info-banner");
  const bannerIcon = el("span", "card-ico");
  bannerIcon.appendChild(icon("database", 24));
  statusBanner.appendChild(bannerIcon);
  const bannerText = el("div");
  const bannerTitle = el("strong", null, "Ingestion Service: Probing status...");
  const bannerSub = el("p", null, "Connecting to ingestion service on port 8001 via LogiShield proxy...");
  bannerText.appendChild(bannerTitle);
  bannerText.appendChild(bannerSub);
  statusBanner.appendChild(bannerText);
  container.appendChild(statusBanner);

  async function checkHealth() {
    const res = await getIngestionHealth();
    if (res.ok && res.payload) {
      statusBanner.className = "pending-banner";
      statusBanner.style.borderLeftColor = "var(--green)";
      bannerTitle.textContent = "Ingestion Service: Live & Connected";
      bannerSub.textContent = "Service: ingestion:8001 | Broker: " + (res.payload.broker_backend || "InMemoryBroker")
        + (res.payload.is_fallback ? " (Fallback mode)" : "")
        + " | Hotpath Cache Latency: " + (res.payload.hotpath_cache_latency_ms || "< 0.1") + " ms";
    } else {
      statusBanner.className = "pending-banner";
      statusBanner.style.borderLeftColor = "var(--amber)";
      bannerTitle.textContent = "Ingestion Service: Not reachable on :8001";
      bannerSub.textContent = "Start the ingestion component via: python -m ingestion (or docker compose up ingestion).";
    }
  }
  checkHealth();

  // Layout Grid: Left = Presets & Input Form, Right = Execution Outputs & Stream
  const grid = el("div", "grid g-main");
  container.appendChild(grid);

  const leftCol = el("div", "stack");
  const rightCol = el("div", "stack");
  grid.appendChild(leftCol);
  grid.appendChild(rightCol);

  // Card 1: Presets
  const presetsCard = card({
    title: "Scenario Presets",
    icon: "layers",
    badge: badge("Sample Traffic", "info")
  });
  const presetList = el("div", "scenario-list");
  presetsCard.body.appendChild(presetList);
  leftCol.appendChild(presetsCard.root);

  // Card 2: Interactive Input Form
  const formCard = card({
    title: "HTTP Interception Input",
    icon: "traffic",
    badge: badge("Live Execution", "live")
  });
  const form = el("form", "policy-form");
  form.addEventListener("submit", (e) => e.preventDefault());

  // Method & Path Row
  const row1 = el("div", "g-2 grid", "");
  row1.style.marginTop = "0";

  const methodWrap = el("label", "field");
  methodWrap.appendChild(el("span", "label", "HTTP Verb"));
  const methodSelect = el("select", "text-input");
  ["POST", "PUT", "GET", "DELETE", "PATCH"].forEach((m) => {
    const opt = el("option", null, m);
    opt.value = m;
    methodSelect.appendChild(opt);
  });
  methodWrap.appendChild(methodSelect);
  row1.appendChild(methodWrap);

  const pathWrap = el("label", "field");
  pathWrap.appendChild(el("span", "label", "Endpoint Path"));
  const pathInput = el("input", "text-input");
  pathInput.type = "text";
  pathInput.value = "/v1/orders";
  pathWrap.appendChild(pathInput);
  row1.appendChild(pathWrap);
  form.appendChild(row1);

  // IP & Role Row
  const row2 = el("div", "g-2 grid", "");
  row2.style.marginTop = "0";

  const ipWrap = el("label", "field");
  ipWrap.appendChild(el("span", "label", "Client IP"));
  const ipInput = el("input", "text-input");
  ipInput.type = "text";
  ipInput.value = "198.51.100.42";
  ipWrap.appendChild(ipInput);
  row2.appendChild(ipWrap);

  const roleWrap = el("label", "field");
  roleWrap.appendChild(el("span", "label", "Client Role"));
  const roleInput = el("input", "text-input");
  roleInput.type = "text";
  roleInput.value = "customer";
  roleWrap.appendChild(roleInput);
  row2.appendChild(roleWrap);
  form.appendChild(row2);

  // Headers JSON
  const headersWrap = el("label", "field");
  headersWrap.appendChild(el("span", "label", "HTTP Headers (JSON)"));
  const headersTextarea = el("textarea", "policy-input");
  headersTextarea.style.minHeight = "64px";
  headersWrap.appendChild(headersTextarea);
  form.appendChild(headersWrap);

  // Request Payload JSON
  const payloadWrap = el("label", "field");
  payloadWrap.appendChild(el("span", "label", "Request Payload Attributes (x) (JSON)"));
  const payloadTextarea = el("textarea", "policy-input");
  payloadTextarea.style.minHeight = "160px";
  payloadWrap.appendChild(payloadTextarea);
  form.appendChild(payloadWrap);

  // AI Threat Score Input & Slider
  const scoreWrap = el("div", "field");
  const scoreLabelRow = el("div", "label");
  scoreLabelRow.appendChild(el("span", null, "AI Threat Score (y)"));
  const scoreValBadge = badge("0.25", "info");
  scoreLabelRow.appendChild(scoreValBadge);
  scoreWrap.appendChild(scoreLabelRow);

  const sliderRow = el("div", "grid g-2", "");
  sliderRow.style.marginTop = "4px";
  sliderRow.style.alignItems = "center";

  const scoreSlider = el("input", "text-input");
  scoreSlider.type = "range";
  scoreSlider.min = "0.0";
  scoreSlider.max = "1.0";
  scoreSlider.step = "0.01";
  scoreSlider.value = "0.25";

  const scoreNumInput = el("input", "text-input");
  scoreNumInput.type = "number";
  scoreNumInput.min = "0.0";
  scoreNumInput.max = "1.0";
  scoreNumInput.step = "0.01";
  scoreNumInput.value = "0.25";

  scoreSlider.addEventListener("input", () => {
    scoreNumInput.value = scoreSlider.value;
    scoreValBadge.textContent = Number(scoreSlider.value).toFixed(2);
  });
  scoreNumInput.addEventListener("input", () => {
    scoreSlider.value = scoreNumInput.value;
    scoreValBadge.textContent = Number(scoreNumInput.value).toFixed(2);
  });

  sliderRow.appendChild(scoreSlider);
  sliderRow.appendChild(scoreNumInput);
  scoreWrap.appendChild(sliderRow);
  form.appendChild(scoreWrap);

  // Action Buttons
  const btnRow = el("div", "export-row");
  const submitBtn = button("Intercept & Execute Ingestion Pipeline", "btn-primary", null, "play");
  const refreshStreamBtn = button("Refresh Context Stream", "btn-ghost", null, "refresh");
  btnRow.appendChild(submitBtn);
  btnRow.appendChild(refreshStreamBtn);
  form.appendChild(btnRow);

  formCard.body.appendChild(form);
  leftCol.appendChild(formCard.root);

  // Right Column: Output Card
  const outputCard = card({
    title: "System Execution Output",
    icon: "activity",
    badge: badge("Awaiting Input", "pending")
  });
  const outputBody = el("div", "policy-output");
  outputBody.appendChild(emptyNote("Select a scenario preset and click 'Intercept & Execute Ingestion Pipeline' to run the ingestion processing stage."));
  outputCard.body.appendChild(outputBody);
  rightCol.appendChild(outputCard.root);

  // Right Column: Stream Card
  const streamCard = card({
    title: "Buffered Context Stream (/v1/context-stream)",
    icon: "database",
    badge: badge("Live Cache", "info")
  });
  const streamBody = el("div", "policy-output");
  streamCard.body.appendChild(streamBody);
  rightCol.appendChild(streamCard.root);

  // Load Preset Function
  function loadPreset(preset) {
    methodSelect.value = preset.method;
    pathInput.value = preset.path;
    ipInput.value = preset.clientIp;
    roleInput.value = preset.clientRole;
    headersTextarea.value = JSON.stringify(preset.headers, null, 2);
    payloadTextarea.value = JSON.stringify(preset.payload, null, 2);
    scoreSlider.value = String(preset.threatScore);
    scoreNumInput.value = String(preset.threatScore);
    scoreValBadge.textContent = Number(preset.threatScore).toFixed(2);

    Array.from(presetList.children).forEach((btn) => {
      btn.setAttribute("aria-checked", btn.getAttribute("data-id") === preset.id ? "true" : "false");
    });
  }

  // Populate Presets
  PRESETS.forEach((preset, idx) => {
    const pBtn = el("button", "scenario");
    pBtn.type = "button";
    pBtn.setAttribute("data-id", preset.id);
    pBtn.setAttribute("aria-checked", idx === 0 ? "true" : "false");
    pBtn.appendChild(el("strong", null, preset.name));
    pBtn.appendChild(el("span", null, preset.subtitle));
    pBtn.addEventListener("click", () => loadPreset(preset));
    presetList.appendChild(pBtn);
  });

  loadPreset(PRESETS[0]);

  // Execute Ingestion
  async function runIngestion() {
    let parsedHeaders = {};
    let parsedPayload = {};
    try {
      parsedHeaders = headersTextarea.value.trim() ? JSON.parse(headersTextarea.value) : {};
    } catch (err) {
      alert("Invalid JSON in HTTP Headers: " + err.message);
      return;
    }
    try {
      parsedPayload = payloadTextarea.value.trim() ? JSON.parse(payloadTextarea.value) : {};
    } catch (err) {
      alert("Invalid JSON in Request Payload: " + err.message);
      return;
    }

    const threatScore = parseFloat(scoreNumInput.value) || 0.0;
    const requestData = {
      metadata: {
        method: methodSelect.value,
        path: pathInput.value,
        content_type: "application/json",
        client_ip: ipInput.value || null,
        client_role: roleInput.value || null,
        headers: parsedHeaders
      },
      payload: parsedPayload,
      threat_score: threatScore
    };

    submitBtn.disabled = true;
    clear(outputBody);
    const loading = el("div", "loading-note");
    loading.appendChild(el("p", null, "Executing line-rate interception, AST PII masking & Protobuf serialization..."));
    outputBody.appendChild(loading);

    const startTime = performance.now();
    const result = await interceptRequest(requestData);
    const totalMs = (performance.now() - startTime).toFixed(2);
    submitBtn.disabled = false;
    clear(outputBody);

    if (!result.ok) {
      outputCard.head.querySelector(".badge").className = "badge badge-bad";
      outputCard.head.querySelector(".badge").textContent = "Execution Failed";

      const errBox = el("div", "policy-error");
      errBox.appendChild(el("strong", null, "Ingestion Failed (HTTP " + result.status + ")"));
      const errDetail = result.payload && result.payload.detail
        ? (typeof result.payload.detail === "string" ? result.payload.detail : JSON.stringify(result.payload.detail))
        : (result.networkError ? "Could not reach Ingestion service on port 8001. Ensure python -m ingestion is running." : "Unknown error");
      errBox.appendChild(el("p", null, errDetail));
      outputBody.appendChild(errBox);
      return;
    }

    const ack = result.payload;
    outputCard.head.querySelector(".badge").className = "badge badge-live";
    outputCard.head.querySelector(".badge").textContent = "Pipeline Succeeded (" + totalMs + " ms)";

    // Verdict Banner
    const verdict = el("div", "policy-result live");
    verdict.appendChild(el("div", "policy-result-status", "Structured Unified API Context Formed"));
    verdict.appendChild(el("p", null, "Event ID: " + ack.event_id + " | Ingestion Execution Time: "
      + fmtMs(ack.pipeline_metrics ? ack.pipeline_metrics.execution_time_ms : 0) + " ms | Protobuf Size: " + ack.protobuf_bytes_size + " bytes"));
    outputBody.appendChild(verdict);

    const stages = el("div", "policy-stages");

    // Stage 1: Captured Metadata
    stages.appendChild(stage("1. Ingress Interception & Metadata", "traffic", codeBlock({
      event_id: ack.event_id,
      method: requestData.metadata.method,
      path: requestData.metadata.path,
      client_ip: requestData.metadata.client_ip,
      client_role: requestData.metadata.client_role,
      headers: requestData.metadata.headers
    }), "live"));

    // Stage 2: AST PII Masking Comparison Table
    const maskingSection = el("div");
    maskingSection.appendChild(el("p", "card-note", "AST syntax-tree analysis recursively identifies and redacts sensitive PII keys (passwords, emails, card numbers, secrets)."));

    const diffTableWrap = el("div", "table-wrap");
    diffTableWrap.style.marginTop = "8px";
    const diffTable = el("table", "table");
    const thead = el("thead");
    const trh = el("tr");
    ["Payload Attribute", "Raw Inbound Input", "Masked Outbound State", "Privacy Status"].forEach((h) => {
      trh.appendChild(el("th", null, h));
    });
    thead.appendChild(trh);
    diffTable.appendChild(thead);

    const tbody = el("tbody");
    const rawKeys = Object.keys(requestData.payload);
    rawKeys.forEach((key) => {
      const rawVal = String(requestData.payload[key]);
      const maskedVal = String(ack.masked_payload[key] !== undefined ? ack.masked_payload[key] : rawVal);
      const isRedacted = maskedVal.indexOf("[REDACTED") !== -1 || maskedVal !== rawVal;

      const row = el("tr");
      row.appendChild(el("td", "nowrap", key));
      row.appendChild(el("td", null, rawVal));
      const maskedTd = el("td");
      maskedTd.textContent = maskedVal;
      if (isRedacted) {
        maskedTd.style.color = "var(--cyan)";
        maskedTd.style.fontWeight = "600";
      }
      row.appendChild(maskedTd);
      row.appendChild(el("td", "nowrap", isRedacted ? badge("Redacted PII", "synth") : badge("Clear", "pending")));
      tbody.appendChild(row);
    });
    diffTable.appendChild(tbody);
    diffTableWrap.appendChild(diffTable);
    maskingSection.appendChild(diffTableWrap);

    maskingSection.appendChild(el("div", "label", "Full Masked Payload"));
    maskingSection.appendChild(codeBlock(ack.masked_payload));
    stages.appendChild(stage("2. AST Privacy-Preserving PII Redaction", "lock", maskingSection, "live"));

    // Stage 3: Zero-Copy Protobuf Wire Serialization
    const protoSection = el("div");
    protoSection.appendChild(el("p", "card-note", "Serialized with Google Protobuf (ingestion_context.proto) into efficient byte sequence for inter-module transit."));
    const protoKv = kv([
      ["Protobuf Binary Size", ack.protobuf_bytes_size + " bytes"],
      ["Encoding Format", "IngestionContext (Protocol Buffers v3)"],
      ["Zero-Copy Buffer", "Memory-mapped slice ready for hotpath routing"]
    ]);
    protoSection.appendChild(protoKv);
    stages.appendChild(stage("3. Zero-Copy Protobuf Wire Serialization", "cube", protoSection, "live"));

    // Stage 4: Unified Context Vector
    const vectorSection = el("div");
    vectorSection.appendChild(el("p", "card-note", "Unified numerical vector combining threat score y and normalized numeric attributes x: [y, x_1, x_2, ...]"));
    const vecWrap = el("div", "proc-chips");
    vecWrap.style.margin = "8px 0";
    if (Array.isArray(ack.unified_context_vector)) {
      ack.unified_context_vector.forEach((val, i) => {
        const chip = badge((i === 0 ? "y (threat): " : "x_" + i + ": ") + Number(val).toFixed(3), i === 0 ? "info" : "live");
        chip.style.fontSize = "12px";
        chip.style.height = "26px";
        vecWrap.appendChild(chip);
      });
    }
    vectorSection.appendChild(vecWrap);
    vectorSection.appendChild(codeBlock(ack.unified_context_vector));
    stages.appendChild(stage("4. Unified Context Vector Formation", "grid", vectorSection, "live"));

    // Stage 5: Downstream Handoff Action to Anomaly Detection
    const handoffSection = el("div", "actions");
    const handoffBtn = button("Forward Context to LogiShield Tokenization & AI Prediction", "btn-primary", () => {
      // Preload into session storage or navigate to anomaly
      window.location.hash = "#/anomaly";
    }, "play");
    handoffBtn.style.width = "100%";
    handoffSection.appendChild(handoffBtn);
    stages.appendChild(stage("5. Downstream Component Dispatch", "activity", handoffSection, "info"));

    outputBody.appendChild(stages);

    // Refresh context stream
    loadContextStream();
  }

  submitBtn.addEventListener("click", runIngestion);
  refreshStreamBtn.addEventListener("click", loadContextStream);

  // Load Context Stream
  async function loadContextStream() {
    clear(streamBody);
    const loading = el("p", "card-note", "Fetching recently buffered context envelopes...");
    streamBody.appendChild(loading);

    const res = await getContextStream(10);
    clear(streamBody);

    if (!res.ok || !res.payload) {
      streamBody.appendChild(emptyNote("Context stream is currently unavailable or empty. Run an interception above to populate."));
      return;
    }

    const envelopes = Array.isArray(res.payload.envelopes) ? res.payload.envelopes : [];
    if (envelopes.length === 0) {
      streamBody.appendChild(emptyNote("No buffered context envelopes yet. Execute an interception to see live buffering."));
      return;
    }

    const countNote = el("p", "card-note", "Showing " + envelopes.length + " most recently buffered context envelopes in broker memory.");
    streamBody.appendChild(countNote);

    const tblWrap = el("div", "table-wrap");
    const tbl = el("table", "table");
    const thd = el("thead");
    const trh = el("tr");
    ["Event ID", "Threat Score (y)", "Masked Keys", "Protobuf Size", "Vector Dims"].forEach((h) => trh.appendChild(el("th", null, h)));
    thd.appendChild(trh);
    tbl.appendChild(thd);

    const tbd = el("tbody");
    envelopes.slice().reverse().forEach((env) => {
      const tr = el("tr");
      tr.appendChild(el("td", "nowrap", env.event_id || "–"));
      const tdScore = el("td", "num", Number(env.threat_score || 0).toFixed(2));
      tdScore.style.fontWeight = "600";
      tdScore.style.color = (env.threat_score || 0) > 0.7 ? "var(--red)" : "var(--green)";
      tr.appendChild(tdScore);

      const keys = env.masked_payload ? Object.keys(env.masked_payload).slice(0, 4).join(", ") + "..." : "–";
      tr.appendChild(el("td", null, keys));
      tr.appendChild(el("td", "num", (env.protobuf_bytes_size || 0) + " B"));
      tr.appendChild(el("td", "num", Array.isArray(env.unified_context_vector) ? env.unified_context_vector.length : 0));
      tbd.appendChild(tr);
    });
    tbl.appendChild(tbd);
    tblWrap.appendChild(tbl);
    streamBody.appendChild(tblWrap);
  }

  loadContextStream();
  pollTimer = setInterval(checkHealth, 8000);
}

export function unmount() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}
