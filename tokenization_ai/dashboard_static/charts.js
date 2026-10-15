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
