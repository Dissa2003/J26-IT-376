/* Report export formatting: JSON and CSV from allowlisted session entries.
 *
 * Pure module: no DOM, no network, no storage. Both formats are rebuilt field
 * by field from typed values, so even an entry carrying extra properties
 * cannot leak them. CSV cells are RFC 4180 quoted and neutralised against
 * spreadsheet formula injection. The caller performs the actual download. */

export const STAGE_ORDER = ["fusion", "masking", "normalization", "tokenization", "tensorization", "model_inference"];
const ROLE_KEYS = ["control", "literal_api", "masked", "literal_rule"];

export const SCOPE_STATEMENT =
  "Current browser session of the LogiShield research implementation. Synthetic demonstration fixtures only; " +
  "scores are uncalibrated outputs of a model fitted on two synthetic fixtures and are not detection accuracy, " +
  "probabilities or Allow/Block verdicts. Data is session-local and contains no raw requests or secrets.";

function str(value, max) {
  return typeof value === "string" ? value.slice(0, max) : "";
}

function num(value) {
  return typeof value === "number" && isFinite(value) ? value : 0;
}

function stageMsFor(entry, name) {
  const stages = Array.isArray(entry.stages) ? entry.stages : [];
  const hit = stages.filter((s) => s && s.stage === name)[0];
  return hit ? num(hit.ms) : null;
}

/* One entry, rebuilt from allowlisted, typed fields only. */
export function pickEntry(entry) {
  const tokens = entry.tokens || {};
  const byRole = tokens.byRole || {};
  const tensor = entry.tensor || {};
  const roles = {};
  ROLE_KEYS.forEach((k) => { roles[k] = num(byRole[k]); });
  return {
    sequence: num(entry.seq),
    observed_at_utc: str(entry.observedAt, 40),
    scenario_id: str(entry.scenarioId, 40),
    scenario_title: str(entry.scenarioTitle, 80),
    event_id: str(entry.eventId, 64),
    model_version: str(entry.modelVersion, 64),
    anomaly_score_uncalibrated: num(entry.score),
    method: str(entry.method, 16),
    path: str(entry.path, 128),
    rule_source: str(entry.ruleSource, 24),
    rule_count: num(entry.ruleCount),
    masked_field_count: num(entry.maskedCount),
    masked_fields: (Array.isArray(entry.maskedFields) ? entry.maskedFields : []).slice(0, 20)
      .map((m) => str(m && m.field, 64)),
    tokens_total: num(tokens.total),
    tokens_by_role: roles,
    tensor_max_length: num(tensor.maxLength),
    tokens_in_tensor: num(tensor.tokensInTensor),
    padded_positions: num(tensor.paddedPositions),
    payload_truncated: tensor.payloadTruncated === true,
    unknown_tokens: num(tensor.unknownTokens),
    pipeline_total_ms: num(entry.totalMs),
    stage_ms: STAGE_ORDER.reduce((acc, name) => { acc[name] = stageMsFor(entry, name); return acc; }, {})
  };
}

export function buildJsonReport(entries, generatedAtIso) {
  const list = Array.isArray(entries) ? entries : [];
  return {
    report_type: "logishield-session-summaries",
    schema_version: 1,
    generated_at_utc: str(generatedAtIso, 40),
    scope: SCOPE_STATEMENT,
    event_count: list.length,
    events: list.map(pickEntry)
  };
}

/* ---- CSV ---- */

export const CSV_COLUMNS = [
  "sequence", "observed_at_utc", "scenario_id", "scenario_title", "event_id", "model_version",
  "anomaly_score_uncalibrated", "method", "path", "rule_source", "rule_count", "masked_field_count", "masked_fields",
  "tokens_total", "tokens_control", "tokens_api_literal", "tokens_masked", "tokens_rule_literal",
  "tensor_max_length", "tokens_in_tensor", "padded_positions", "payload_truncated", "unknown_tokens",
  "pipeline_total_ms",
  "stage_fusion_ms", "stage_masking_ms", "stage_normalization_ms", "stage_tokenization_ms",
  "stage_tensorization_ms", "stage_model_inference_ms"
];

const FORMULA_TRIGGERS = ["=", "+", "-", "@", "\t", "\r"];

/* Quote per RFC 4180 and defuse spreadsheet formulas. Numbers and booleans are
 * written as-is; text beginning with a formula trigger (also after leading
 * spaces) is prefixed with an apostrophe so it is read as plain text. */
export function csvCell(value) {
  let s;
  if (typeof value === "number") {
    s = isFinite(value) ? String(value) : "";
  } else if (typeof value === "boolean") {
    s = value ? "true" : "false";
  } else {
    s = value === null || value === undefined ? "" : String(value);
    const first = s.charAt(0);
    const firstVisible = s.replace(/^ +/, "").charAt(0);
    if (FORMULA_TRIGGERS.indexOf(first) >= 0 || FORMULA_TRIGGERS.indexOf(firstVisible) >= 0) {
      s = "'" + s;
    }
  }
  if (/[",\r\n]/.test(s)) {
    s = '"' + s.split('"').join('""') + '"';
  }
  return s;
}

export function csvRow(entry) {
  const p = pickEntry(entry);
  const values = [
    p.sequence, p.observed_at_utc, p.scenario_id, p.scenario_title, p.event_id, p.model_version,
    p.anomaly_score_uncalibrated, p.method, p.path, p.rule_source, p.rule_count, p.masked_field_count,
    p.masked_fields.join(";"),
    p.tokens_total, p.tokens_by_role.control, p.tokens_by_role.literal_api, p.tokens_by_role.masked,
    p.tokens_by_role.literal_rule,
    p.tensor_max_length, p.tokens_in_tensor, p.padded_positions, p.payload_truncated, p.unknown_tokens,
    p.pipeline_total_ms
  ].concat(STAGE_ORDER.map((name) => p.stage_ms[name]));
  return values.map(csvCell).join(",");
}

export function buildCsv(entries) {
  const list = Array.isArray(entries) ? entries : [];
  return [CSV_COLUMNS.join(",")].concat(list.map(csvRow)).join("\r\n") + "\r\n";
}

/* Filenames always name the component and the scope; the stamp is UTC. */
export function exportFilename(kind, generatedAtIso) {
  const d = new Date(generatedAtIso);
  const stamp = isNaN(d.getTime())
    ? "unknown-time"
    : d.toISOString().split("-").join("").split(":").join("").split(".")[0] + "Z";
  return "logishield-session-summaries-" + stamp + "." + (kind === "csv" ? "csv" : "json");
}
