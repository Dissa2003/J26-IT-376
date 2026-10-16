/* LogiShield dashboard shell: sidebar, header readiness chip and hash router.
 * Presentation only. */

import { el, clear, icon, hydrateIcons } from "./ui.js";
import { shell, healthLabel } from "./shell.js";
import { ROUTES, DEFAULT_ROUTE, routeById } from "./routes.js";
import * as anomaly from "./page-anomaly.js";
import * as logs from "./page-logs.js";
import * as components from "./page-components.js";
import * as overview from "./page-overview.js";
import * as reports from "./page-reports.js";
import { mountPending } from "./page-pending.js";

let activeUnmount = null;

function buildNav() {
  const nav = document.getElementById("nav");
  clear(nav);
  ROUTES.forEach((route) => {
    if (route.separated) { nav.appendChild(el("div", "nav-sep")); }
    const link = el("a", "nav-item");
    link.href = "#/" + route.id;
    link.setAttribute("data-route", route.id);
    link.title = route.label;
    link.appendChild(icon(route.icon, 20));
    link.appendChild(el("span", "nav-label", route.label));
    if (route.tag) { link.appendChild(el("span", "nav-tag", route.tag)); }
    nav.appendChild(link);
  });
}

function currentRouteId() {
  const hash = window.location.hash || "";
  return hash.indexOf("#/") === 0 ? hash.slice(2) : "";
}

function render() {
  const id = currentRouteId();
  const route = routeById(id);
  if (!route) {
    window.location.hash = "#/" + DEFAULT_ROUTE;
    return;
  }

  if (activeUnmount) { activeUnmount(); activeUnmount = null; }
  const view = document.getElementById("view");
  clear(view);

  const pages = { overview: overview, anomaly: anomaly, logs: logs, components: components, reports: reports };
  if (pages[route.kind]) {
    pages[route.kind].mount(view, route);
    activeUnmount = pages[route.kind].unmount;
  } else {
    mountPending(view, route);
  }

  document.querySelectorAll("#nav .nav-item").forEach((link) => {
    if (link.getAttribute("data-route") === route.id) { link.setAttribute("aria-current", "page"); }
    else { link.removeAttribute("aria-current"); }
  });
  document.title = route.title.indexOf("LogiShield") === 0 ? route.title : route.title + " · LogiShield";
  view.scrollTop = 0;
  window.scrollTo(0, 0);
}

function paintHeaderHealth(health) {
  const dot = document.getElementById("health-dot");
  dot.className = "dot " + (health.state === "ready" ? "dot-ok" : (health.state === "checking" ? "dot-idle" : "dot-bad"));
  document.getElementById("health-text").textContent = healthLabel(health);
}

function boot() {
  hydrateIcons(document);
  buildNav();
  shell.subscribe(paintHeaderHealth);
  document.getElementById("health-chip").addEventListener("click", () => { shell.refreshHealth(); });
  window.addEventListener("hashchange", render);
  render();
  shell.refreshHealth();
}

boot();
