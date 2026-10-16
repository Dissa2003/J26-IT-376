/* Shared shell state: the Member 4 service readiness shown in the header and
 * on the Anomaly Detection page. Only GET /health is used. */

import { getHealth } from "./api.js";

const listeners = new Set();

export const shell = {
  health: { state: "checking", service: "" },

  subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },

  /* quiet = keep showing the previous state until the new one arrives. */
  async refreshHealth(quiet) {
    if (!quiet) {
      shell.health = { state: "checking", service: "" };
      listeners.forEach((fn) => fn(shell.health));
    }
    shell.health = await getHealth();
    listeners.forEach((fn) => fn(shell.health));
    return shell.health;
  }
};

export function healthLabel(health) {
  if (health.state === "ready") { return "Member 4 service ready"; }
  if (health.state === "notready") { return "Member 4 service not ready"; }
  if (health.state === "down") { return "Member 4 service unreachable"; }
  return "Checking service…";
}
