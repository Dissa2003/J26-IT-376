/* Small DOM helpers and the icon set. Everything is built with createElement /
 * createElementNS and textContent; there is no HTML string injection. */

const SVG_NS = "http://www.w3.org/2000/svg";

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) { node.className = className; }
  if (text !== undefined && text !== null) { node.textContent = String(text); }
  return node;
}

export function svg(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.keys(attrs || {}).forEach((key) => node.setAttribute(key, String(attrs[key])));
  return node;
}

export function clear(node) {
  while (node.firstChild) { node.removeChild(node.firstChild); }
}

export function fmtMs(value) { return Number(value).toFixed(3); }

/* 24x24 outline icons. Each entry is a list of [tag, attributes]. */
const ICONS = {
  home: [["path", { d: "M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z" }]],
  traffic: [["path", { d: "M6 20V11M12 20V5M18 20v-6" }]],
  alert: [["path", { d: "M12 3.5 2.5 20h19z" }], ["path", { d: "M12 10v5M12 17.8v.2" }]],
  shield: [["path", { d: "M12 2.5 4 6v6c0 5 3.4 8.4 8 9.5 4.6-1.1 8-4.5 8-9.5V6z" }], ["path", { d: "M8.5 12l2.5 2.5 4.5-5" }]],
  decisions: [["path", { d: "M6 3h9l4 4v14H6z" }], ["path", { d: "M9 12h7M9 16h7M9 8h3" }]],
  logs: [["path", { d: "M6 3h9l4 4v14H6z" }], ["path", { d: "M15 3v4h4M9 12h6M9 16h6" }]],
  components: [
    ["circle", { cx: 12, cy: 6, r: 2.5 }], ["circle", { cx: 6, cy: 18, r: 2.5 }], ["circle", { cx: 18, cy: 18, r: 2.5 }],
    ["path", { d: "M12 8.5V12M10.6 12.6 7.4 15.8M13.4 12.6l3.2 3.2" }]
  ],
  reports: [["path", { d: "M5 20V12M10 20V6M15 20v-9M20 20V3" }]],
  settings: [
    ["circle", { cx: 12, cy: 12, r: 3 }],
    ["path", { d: "M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1" }]
  ],
  search: [["circle", { cx: 11, cy: 11, r: 7 }], ["path", { d: "M20 20l-4-4" }]],
  play: [["path", { d: "M8 5v14l11-7z", fill: "currentColor", stroke: "none" }]],
  spinner: [["path", { d: "M12 3a9 9 0 1 0 9 9" }]],
  database: [
    ["ellipse", { cx: 12, cy: 5.5, rx: 8, ry: 3 }],
    ["path", { d: "M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" }]
  ],
  clock: [["circle", { cx: 12, cy: 12, r: 9 }], ["path", { d: "M12 7v5l3 2" }]],
  layers: [["path", { d: "M12 3 3 8l9 5 9-5z" }], ["path", { d: "M3 13l9 5 9-5" }]],
  lock: [["rect", { x: 5, y: 11, width: 14, height: 9, rx: 2 }], ["path", { d: "M8 11V8a4 4 0 0 1 8 0v3" }]],
  code: [["path", { d: "M8 8l-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14" }]],
  grid: [["rect", { x: 4, y: 4, width: 16, height: 16, rx: 2 }], ["path", { d: "M4 9.3h16M4 14.7h16M9.3 4v16M14.7 4v16" }]],
  activity: [["path", { d: "M3 12h4l3-8 4 16 3-8h4" }]],
  info: [["circle", { cx: 12, cy: 12, r: 9 }], ["path", { d: "M12 11v6M12 7.5v.2" }]],
  filter: [["path", { d: "M4 5h16l-6 8v6l-4-2v-4z" }]],
  merge: [["path", { d: "M6 4v6a4 4 0 0 0 4 4h4a4 4 0 0 1 4 4v2M18 4v6a4 4 0 0 1-4 4" }]],
  cube: [["path", { d: "M12 3 4 7.5v9L12 21l8-4.5v-9z" }], ["path", { d: "M4 7.5 12 12l8-4.5M12 12v9" }]],
  history: [["path", { d: "M4 12a8 8 0 1 0 2.5-5.8L4 8.5" }], ["path", { d: "M4 4v4.5h4.5M12 8v4l3 2" }]],
  list: [["path", { d: "M9 6h11M9 12h11M9 18h11M4.5 6h.1M4.5 12h.1M4.5 18h.1" }]],
  file: [["path", { d: "M6 3h9l4 4v14H6z" }], ["path", { d: "M15 3v4h4" }]],
  warn: [["circle", { cx: 12, cy: 12, r: 9 }], ["path", { d: "M12 7.5v5.5M12 16.3v.2" }]],
  globe: [["circle", { cx: 12, cy: 12, r: 9 }], ["ellipse", { cx: 12, cy: 12, rx: 4, ry: 9 }], ["path", { d: "M3 12h18" }]],
  cpu: [["rect", { x: 7, y: 7, width: 10, height: 10, rx: 1.5 }], ["path", { d: "M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" }]],
  rules: [["path", { d: "M6 3h9l4 4v14H6z" }], ["path", { d: "M15 3v4h4M9 14l2 2 4-4" }]],
  arrowDown: [["path", { d: "M12 4v15M6 13l6 6 6-6" }]],
  refresh: [["path", { d: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7" }]],
  trash: [["path", { d: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" }]],
  sort: [["path", { d: "M7 4v16M3 16l4 4 4-4M17 20V4M13 8l4-4 4 4" }]]
};

export function icon(name, size) {
  const px = size || 20;
  const root = svg("svg", {
    viewBox: "0 0 24 24", width: px, height: px, fill: "none", stroke: "currentColor",
    "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round",
    "aria-hidden": "true", focusable: "false"
  });
  (ICONS[name] || ICONS.info).forEach((part) => root.appendChild(svg(part[0], part[1])));
  return root;
}

/* Replace every <span data-icon="name"> in static HTML with the SVG icon. */
export function hydrateIcons(scope) {
  scope.querySelectorAll("[data-icon]").forEach((node) => {
    node.appendChild(icon(node.getAttribute("data-icon"), Number(node.getAttribute("data-size")) || 20));
    node.removeAttribute("data-icon");
  });
}

export function badge(text, tone) {
  return el("span", "badge badge-" + (tone || "info"), text);
}

/* A titled card. Returns { root, body }. */
export function card(opts) {
  const root = el("section", "card" + (opts.className ? " " + opts.className : ""));
  const head = el("header", "card-head");
  if (opts.icon) {
    const ico = el("span", "card-ico");
    ico.appendChild(icon(opts.icon, 20));
    head.appendChild(ico);
  }
  head.appendChild(el("h2", null, opts.title));
  if (opts.badge) { head.appendChild(opts.badge); }
  const body = el("div", "card-body");
  root.appendChild(head);
  root.appendChild(body);
  return { root, body, head };
}

/* Definition list from [[label, value], ...]. */
export function kv(rows) {
  const dl = el("dl", "kv");
  rows.forEach((row) => {
    dl.appendChild(el("dt", null, row[0]));
    dl.appendChild(el("dd", null, row[1]));
  });
  return dl;
}

export function emptyNote(text) { return el("p", "empty-note", text); }

/* A button with an optional leading icon. */
export function button(label, className, onClick, iconName) {
  const node = el("button", className || "btn-ghost");
  node.type = "button";
  if (iconName) { node.appendChild(icon(iconName, 16)); }
  node.appendChild(el("span", null, label));
  if (onClick) { node.addEventListener("click", onClick); }
  return node;
}

/* A labelled form control wrapper: <label><span>text</span>control</label>. */
export function field(labelText, control) {
  const wrap = el("label", "field");
  wrap.appendChild(el("span", "field-label", labelText));
  wrap.appendChild(control);
  return wrap;
}
