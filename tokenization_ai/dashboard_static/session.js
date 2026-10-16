/* Session-local store of privacy-safe analysis summaries.
 *
 * Pure module: no DOM, no network, no browser storage. Entries live in this
 * page's memory only and vanish on reload. They are built from a successful
 * POST /v1/analyze response by explicit allowlist: every stored value is
 * rebuilt from a typed, length-limited field, so anything not named here
 * (the original request, token text, the masked representation, unknown
 * keys) cannot enter the store even if a response were to contain it.
 */

export const MAX_EVENTS = 50;

const ROLE_KEYS = ["control", "literal_api", "masked", "literal_rule"];
const STAGE_NAMES = ["fusion", "masking", "normalization", "tokenization", "tensorization", "model_inference"];
const SAFE_KEY = /^[a-z_]{1,40}$/;

function text(value, max) {
  return typeof value === "string" ? value.slice(0, max) : "";
}

function count(value) {
  return typeof value === "number" && isFinite(value) && value >= 0 ? value : 0;
}

/* Count-only summaries: numeric or boolean values under short snake_case keys. */
function countsOnly(summary) {
  const out = {};
  if (!summary || typeof summary !== "object") { return out; }
  Object.keys(summary).slice(0, 8).forEach((key) => {
    const value = summary[key];
    if (SAFE_KEY.test(key) && (typeof value === "boolean" || (typeof value === "number" && isFinite(value)))) {
      out[key] = value;
    }
  });
  return out;
}

/* Read a value from the returned (already projected) representation by the
 * dotted field path the backend reported. Display and allowlisting only. */
export function valueAtPath(rep, dotted) {
  let cur = rep;
  const parts = String(dotted).split(".");
  for (let i = 0; i < parts.length; i += 1) {
    if (cur !== null && typeof cur === "object" && parts[i] in cur) { cur = cur[parts[i]]; } else { return null; }
  }
  return typeof cur === "string" ? cur : null;
}

/* A masked value is kept only if it is a short bracketed placeholder. */
function placeholderOrNull(value) {
  if (typeof value !== "string" || value.length > 24) { return null; }
  return value.charAt(0) === "[" && value.charAt(value.length - 1) === "]" ? value : null;
}

export function summarize(data, scenario, observedAt, sequence) {
  const rep = data && typeof data.masked_representation === "object" && data.masked_representation ? data.masked_representation : {};
  const prov = data && data.provenance ? data.provenance : {};
  const rule = prov.rule_status || {};
  const tensor = data && data.tensor ? data.tensor : {};
  const tokens = data && data.tokens ? data.tokens : {};
  const masking = data && data.masking ? data.masking : {};
  const privacy = data && data.privacy ? data.privacy : {};
  const timings = data && data.timings ? data.timings : {};
  const prediction = data && data.prediction ? data.prediction : {};

  const byRole = {};
  ROLE_KEYS.forEach((key) => { byRole[key] = 0; });
  (Array.isArray(tokens.items) ? tokens.items : []).forEach((item) => {
    if (item && ROLE_KEYS.indexOf(item.role) >= 0) { byRole[item.role] += 1; }
  });

  const maskedFields = (Array.isArray(masking.masked_fields) ? masking.masked_fields : []).slice(0, 20).map((field) => ({
    field: text(field, 64),
    placeholder: placeholderOrNull(valueAtPath(rep, field))
  }));

  const stages = (Array.isArray(timings.stages) ? timings.stages : [])
    .filter((s) => s && STAGE_NAMES.indexOf(s.stage) >= 0)
    .map((s) => ({ stage: s.stage, ms: count(s.elapsed_ms), ok: s.success === true, counts: countsOnly(s.summary) }));

  const score = Number(prediction.score);

  return Object.freeze({
    seq: sequence,
    observedAt: text(observedAt, 40),
    scenarioId: text(scenario && scenario.id, 40),
    scenarioTitle: text(scenario && scenario.title, 80),
    eventId: text(data && data.event_id, 64),
    contract: text(data && data.contract_version, 40),
    modelVersion: text(prediction.model_version, 64),
    score: isFinite(score) ? Math.max(0, Math.min(1, score)) : 0,
    method: text(rep.method, 16),
    path: text(rep.path, 128),
    ruleSource: text(rule.source, 24),
    ruleLabel: text(rule.label, 80),
    ruleCount: count(rule.rule_count),
    ruleIds: (Array.isArray(rule.rule_ids) ? rule.rule_ids : []).slice(0, 20).map((id) => text(id, 64)),
    maskedCount: count(masking.masked_field_count),
    maskedFields: maskedFields,
    tokens: { total: count(tokens.total), byRole: byRole },
    tensor: {
      maxLength: count(tensor.max_length),
      beforeTruncation: count(tensor.token_count_before_truncation),
      tokensInTensor: count(tensor.tokens_in_tensor),
      paddedPositions: count(tensor.padded_positions),
      payloadTruncated: tensor.payload_truncated === true,
      unknownTokens: count(tensor.unknown_token_count),
      vocabularySize: count(tensor.vocabulary_size)
    },
    hidden: { inRepresentation: count(privacy.hidden_in_representation), inTokens: count(privacy.hidden_in_tokens) },
    totalMs: count(timings.total_elapsed_ms),
    stages: stages,
    accuracyClaim: text(prov.detection_accuracy_claim, 16)
  });
}

let items = [];
let counter = 0;
const listeners = new Set();

function notify() {
  listeners.forEach((fn) => fn());
}

export const session = {
  /* Add the summary of one successful response. Returns the stored entry. */
  add(data, scenario) {
    counter += 1;
    const entry = summarize(data, scenario, new Date().toISOString(), counter);
    items.unshift(entry);
    if (items.length > MAX_EVENTS) { items.length = MAX_EVENTS; }
    notify();
    return entry;
  },
  /* Newest first. */
  list() { return items.slice(); },
  size() { return items.length; },
  clear() {
    items = [];
    counter = 0;
    notify();
  },
  subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }
};
