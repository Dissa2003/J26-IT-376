/* Pure view-model for the Dashboard overview. No DOM, no network, no storage.
 *
 * Every figure is derived from the shared session store's allowlisted entries
 * (newest first). With no entries every value is null / empty: there is no
 * default, historical or simulated data. */

export const CHART_LIMIT = 12;

export function buildOverviewModel(entries) {
  const list = Array.isArray(entries) ? entries : [];
  const latest = list.length ? list[0] : null;

  const chart = list.slice(0, CHART_LIMIT).reverse().map((e) => ({
    seq: e.seq,
    value: e.score,
    label: e.scenarioId,
    latest: latest !== null && e.seq === latest.seq,
    title: "Analysis #" + e.seq + " · " + e.scenarioId + " · score " + e.score.toFixed(4) + " (uncalibrated)"
  }));

  const stages = latest ? latest.stages.map((s) => ({ stage: s.stage, ms: s.ms, ok: s.ok })) : [];

  return {
    hasData: latest !== null,
    count: list.length,
    shown: chart.length,
    latest: latest,
    kpis: {
      analyses: list.length > 0 ? list.length : null,
      score: latest ? latest.score : null,
      tokens: latest ? latest.tokens.total : null,
      maxLength: latest ? latest.tensor.maxLength : null,
      padded: latest ? latest.tensor.paddedPositions : null,
      masked: latest ? latest.maskedCount : null
    },
    chart: chart,
    roles: latest ? latest.tokens.byRole : null,
    stages: stages,
    maxStageMs: stages.reduce((m, s) => Math.max(m, s.ms), 0),
    totalMs: latest ? latest.totalMs : null
  };
}
