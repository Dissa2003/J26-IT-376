/* Pending-state page: says plainly what the page will show, what must exist
 * first, and what is true today. It renders no data, charts or tables. */

import { el, icon, card, badge } from "./ui.js";

function list(items) {
  const ul = el("ul", "limits");
  items.forEach((text) => ul.appendChild(el("li", null, text)));
  return ul;
}

export function mountPending(root, route) {
  const head = el("div", "page-head");
  head.appendChild(el("h1", "page-title", route.title));
  head.appendChild(el("p", "page-sub", route.subtitle));
  root.appendChild(head);

  const banner = el("div", "pending-banner");
  banner.appendChild(badge("Pending", "pending"));
  const text = el("div");
  text.appendChild(el("strong", null, "Not available in this phase"));
  text.appendChild(el("p", null,
    "This page shows no data. Nothing here is live, sampled or simulated. Source: " + route.owner + "."));
  banner.appendChild(text);
  root.appendChild(banner);

  const info = el("div", "grid g-3");
  const shows = card({ title: "What this page will show", icon: "info" });
  shows.body.appendChild(list(route.shows));
  const needs = card({ title: "Required before it works", icon: "warn" });
  needs.body.appendChild(list(route.requires));
  const status = card({ title: "Current status", icon: "shield" });
  status.body.appendChild(list(route.status));
  info.appendChild(shows.root);
  info.appendChild(needs.root);
  info.appendChild(status.root);
  root.appendChild(info);

  if (route.placeholders.length) {
    const grid = el("div", "grid g-3");
    route.placeholders.forEach((ph) => {
      const box = el("section", "card dashed ph");
      const body = el("div", "ph-body");
      const title = el("div", "ph-title");
      title.appendChild(icon(ph.icon, 20));
      title.appendChild(el("span", null, ph.title));
      body.appendChild(title);
      body.appendChild(el("p", "ph-need", ph.needs));
      body.appendChild(el("p", "ph-none", "No data"));
      box.appendChild(body);
      grid.appendChild(box);
    });
    root.appendChild(grid);
  }

  const link = el("a", "btn link-btn", "Open Anomaly Detection (functional)");
  link.href = "#/anomaly";
  const wrap = el("div", "grid");
  wrap.appendChild(link);
  root.appendChild(wrap);
}
