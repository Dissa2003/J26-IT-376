/* The only module that talks to the network. Same-origin calls to the
 * existing LogiShield endpoints; nothing here scores, masks or tokenizes. */

const ANALYZE_URL = "/v1/analyze";
const HEALTH_URL = "/health";
const SCENARIOS_URL = "/demo/assets/scenarios.json";
export const CONTRACT = "logishield-analysis-v2";
const TIMEOUT_MS = 15000;

export async function verifyPolicyText(policyText, policyId) {
  const response = await fetch("/api/v1/verify-text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ policy_text: policyText, policy_id: policyId || undefined }),
    credentials: "same-origin",
    cache: "no-store"
  });
  let payload = null;
  try { payload = await response.json(); } catch (error) { payload = null; }
  return { ok: response.ok, status: response.status, payload };
}

async function getPolicyData(path) {
  try {
    const response = await fetch(path, { cache: "no-store", credentials: "same-origin" });
    let payload = null;
    try { payload = await response.json(); } catch (error) { payload = null; }
    return { ok: response.ok, status: response.status, payload };
  } catch (error) {
    return { ok: false, status: 0, payload: null };
  }
}

export function getVerificationRecords() {
  return getPolicyData("/api/v1/verification-records");
}

export function getRuleCache() {
  return getPolicyData("/api/v1/rule-cache");
}

export function getConflicts() {
  return getPolicyData("/api/v1/conflicts");
}

export async function getHealth() {
  try {
    const response = await fetch(HEALTH_URL, { cache: "no-store", credentials: "same-origin" });
    let body = null;
    try { body = await response.json(); } catch (error) { body = null; }
    if (response.ok && body && body.ready === true) {
      return { state: "ready", service: typeof body.service === "string" ? body.service : "" };
    }
    return { state: "notready", service: "" };
  } catch (error) {
    return { state: "down", service: "" };
  }
}

/* Presets are read from the existing /demo scenarios file (single source). */
export async function getScenarios() {
  const response = await fetch(SCENARIOS_URL, { cache: "no-store", credentials: "same-origin" });
  if (!response.ok) { throw new Error("scenarios"); }
  const data = await response.json();
  return Array.isArray(data.scenarios) ? data.scenarios : [];
}

/* Resolves to { ok: true, data } or { ok: false, message }. Never rejects. */
export async function analyze(request) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(ANALYZE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
      credentials: "same-origin",
      cache: "no-store"
    });
    let payload = null;
    try { payload = await response.json(); } catch (error) { payload = null; }
    if (!response.ok) {
      const code = payload && typeof payload.error === "string" ? payload.error : "error";
      const detail = payload && typeof payload.detail === "string" ? payload.detail : "";
      return {
        ok: false,
        message: "The backend rejected the request (HTTP " + response.status + ", " + code + ")." + (detail ? " " + detail : "")
      };
    }
    if (!payload || payload.contract_version !== CONTRACT) {
      return { ok: false, message: "Unexpected response contract; expected " + CONTRACT + "." };
    }
    return { ok: true, data: payload };
  } catch (error) {
    const timedOut = error && error.name === "AbortError";
    return {
      ok: false,
      message: timedOut
        ? "The backend did not respond within " + (TIMEOUT_MS / 1000) + " seconds."
        : "Could not reach the LogiShield service. Is it running on this machine?"
    };
  } finally {
    clearTimeout(timer);
  }
}
