/* Minimal SVG charts. They only draw numbers they are given; nothing here
 * computes, estimates or invents data. Styling comes from dashboard.css
 * classes and SVG presentation attributes (no inline style attributes). */

import { svg } from "./ui.js";

/* Half-circle gauge for a value in [0, 1]. Neutral blue-to-cyan colouring on
 * purpose: the score is an uncalibrated model output, not a verdict. */
export function gauge(value) {
  const v = Math.max(0, Math.min(1, Number(value)));
  const root = svg("svg", { class: "gauge", viewBox: "0 0 200 112", role: "img", "aria-label": "Anomaly score " + v.toFixed(4) + " out of 1" });
  const defs = svg("defs");
  const grad = svg("linearGradient", { id: "gauge-grad", x1: 0, y1: 0, x2: 1, y2: 0 });
  grad.appendChild(svg("stop", { offset: 0, "stop-color": "#1f6bff" }));
  grad.appendChild(svg("stop", { offset: 1, "stop-color": "#22d3ee" }));
  defs.appendChild(grad);
  root.appendChild(defs);
  const arc = "M20 100 A80 80 0 0 1 180 100";
  root.appendChild(svg("path", { class: "gauge-track", d: arc, fill: "none", "stroke-width": 16, "stroke-linecap": "round", pathLength: 100 }));
  if (v > 0) {
    root.appendChild(svg("path", {
      d: arc, fill: "none", stroke: "url(#gauge-grad)", "stroke-width": 16, "stroke-linecap": "round",
      pathLength: 100, "stroke-dasharray": (v * 100).toFixed(3) + " 100"
    }));
  }
  [["20", "0"], ["180", "1"]].forEach((t) => {
    const label = svg("text", { x: t[0], y: 111, "text-anchor": "middle", fill: "#6a82ad", "font-size": 10 });
    label.textContent = t[1];
    root.appendChild(label);
  });
  return root;
}

/* Donut from [{ key, value }]; each segment gets class "seg-<key>". */
export function donut(segments) {
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  const root = svg("svg", { viewBox: "0 0 42 42", role: "img", "aria-label": "Token role distribution" });
  root.appendChild(svg("circle", { class: "seg-track", cx: 21, cy: 21, r: 15.9155, fill: "none", "stroke-width": 5 }));
  if (total > 0) {
    let offset = 0;
    segments.forEach((s) => {
      if (s.value <= 0) { return; }
      const len = (s.value / total) * 100;
      root.appendChild(svg("circle", {
        class: "seg-" + s.key, cx: 21, cy: 21, r: 15.9155, fill: "none", "stroke-width": 5, pathLength: 100,
        "stroke-dasharray": len.toFixed(3) + " " + (100 - len).toFixed(3), "stroke-dashoffset": (-offset).toFixed(3),
        transform: "rotate(-90 21 21)"
      }));
      offset += len;
    });
  }
  return root;
}

/* Bar chart for values in [0, 1], one bar per item { seq, value, label, latest, title }.
 * Draws only the numbers it is given. */
export function barChart(items, ariaLabel) {
  const slot = 58;
  const left = 40;
  const top = 18;
  const plotH = 120;
  const slots = Math.max(items.length, 8); // keep bar width stable when there are few events
  const width = left + slots * slot + 8;
  const height = top + plotH + 44;
  const root = svg("svg", { class: "bar-chart", viewBox: "0 0 " + width + " " + height, role: "img", "aria-label": ariaLabel });
  const defs = svg("defs");
  const grad = svg("linearGradient", { id: "bc-grad", x1: 0, y1: 1, x2: 0, y2: 0 });
  grad.appendChild(svg("stop", { offset: 0, "stop-color": "#1f6bff" }));
  grad.appendChild(svg("stop", { offset: 1, "stop-color": "#22d3ee" }));
  defs.appendChild(grad);
  root.appendChild(defs);

  [0, 0.5, 1].forEach((v) => {
    const y = top + plotH - v * plotH;
    root.appendChild(svg("line", { class: "bc-grid", x1: left, x2: width - 4, y1: y, y2: y }));
    const t = svg("text", { class: "bc-axis", x: left - 6, y: y + 4, "text-anchor": "end" });
    t.textContent = v === 0 ? "0" : (v === 1 ? "1" : "0.5");
    root.appendChild(t);
  });

  items.forEach((item, i) => {
    const x = left + i * slot + 9;
    const w = slot - 18;
    const h = Math.max(2, Math.max(0, Math.min(1, item.value)) * plotH);
    const y = top + plotH - h;
    const bar = svg("rect", { class: "bc-bar" + (item.latest ? " bc-latest" : ""), x: x, y: y, width: w, height: h, rx: 4 });
    const tip = svg("title");
    tip.textContent = item.title;
    bar.appendChild(tip);
    root.appendChild(bar);
    const val = svg("text", { class: "bc-val", x: x + w / 2, y: y - 5, "text-anchor": "middle" });
    val.textContent = item.value.toFixed(4);
    root.appendChild(val);
    const seq = svg("text", { class: "bc-lab", x: x + w / 2, y: top + plotH + 16, "text-anchor": "middle" });
    seq.textContent = "#" + item.seq;
    root.appendChild(seq);
    const name = svg("text", { class: "bc-lab bc-sub", x: x + w / 2, y: top + plotH + 31, "text-anchor": "middle" });
    name.textContent = String(item.label).slice(0, 10);
    root.appendChild(name);
  });
  return root;
}
