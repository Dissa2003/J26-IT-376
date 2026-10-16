/* Pure view-model for the session half of the Reports page.
 *
 * Takes the allowlisted session entries (newest first) and the overview model
 * built from them (overview-model.js) and adds session-wide aggregates. No
 * DOM, no network, no storage. Empty session: every metric is null / empty. */

const ROLE_KEYS = ["control", "literal_api", "masked", "literal_rule"];

export function buildSessionReport(entries, overview) {
  const list = Array.isArray(entries) ? entries : [];
  const roleTotals = {};
  ROLE_KEYS.forEach((key) => { roleTotals[key] = 0; });
  let tokensTotal = 0;
  let maskedFieldsTotal = 0;
  let withMasking = 0;
  let unknownTotal = 0;
  let truncated = 0;
  let scoreMin = null;
  let scoreMax = null;

  list.forEach((e) => {
    tokensTotal += e.tokens.total;
    ROLE_KEYS.forEach((key) => { roleTotals[key] += e.tokens.byRole[key] || 0; });
    maskedFieldsTotal += e.maskedCount;
    if (e.maskedCount > 0) { withMasking += 1; }
    unknownTotal += e.tensor.unknownTokens;
    if (e.tensor.payloadTruncated) { truncated += 1; }
    scoreMin = scoreMin === null ? e.score : Math.min(scoreMin, e.score);
    scoreMax = scoreMax === null ? e.score : Math.max(scoreMax, e.score);
  });

  return {
    hasData: list.length > 0,
    count: list.length,
    overview: overview,
    withMasking: list.length ? withMasking : null,
    maskedFieldsTotal: list.length ? maskedFieldsTotal : null,
    tokensTotal: list.length ? tokensTotal : null,
    roleTotals: list.length ? roleTotals : null,
    unknownTotal: list.length ? unknownTotal : null,
    truncated: list.length ? truncated : null,
    scoreMin: scoreMin,
    scoreMax: scoreMax,
    rows: list.slice()
  };
}
