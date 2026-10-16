"""Regression tests for the two defects found in the final dashboard QA audit.

1. Small secondary text (the --dim token) was 4.08:1 on the card panel colour,
   below WCAG AA (4.5:1) for small text.
2. On windows shorter than 720px (for example a 768px-high laptop screen has
   about 650px of browser viewport) the sticky sidebar clipped its footer with
   no way to reach it.

Both were confirmed in a real browser; these tests keep the fixes in place.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "tokenization_ai" / "dashboard_static"
CSS = (ROOT / "dashboard.css").read_text(encoding="utf-8")
CHARTS = (ROOT / "charts.js").read_text(encoding="utf-8")


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _ratio(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _token(name: str) -> str:
    marker = f"--{name}:"
    start = CSS.index(marker) + len(marker)
    return CSS[start: CSS.index(";", start)].strip()


def test_secondary_text_colours_meet_wcag_aa_on_the_lightest_card_surface():
    surface = "#0d2147"  # the brightest stop of the card gradient (worst case)
    for name in ("ink", "muted", "dim"):
        assert _ratio(_token(name), surface) >= 4.5, name


def test_chart_axis_labels_use_the_accessible_dim_colour():
    dim = _token("dim")
    assert f"fill: {dim}" in CSS  # .bc-axis
    assert f'fill: "{dim}"' in CHARTS  # gauge end labels
    assert "#6a82ad" not in CSS and "#6a82ad" not in CHARTS


def test_sidebar_scrolls_instead_of_clipping_its_footer():
    block = CSS[CSS.index(".sidebar {"): CSS.index("}", CSS.index(".sidebar {"))]
    assert "overflow-y: auto" in block
    assert "position: sticky" in block and "height: 100vh" in block


def test_sidebar_has_a_compact_variant_for_short_windows():
    assert "@media (max-height: 760px) and (min-width: 1101px)" in CSS
    short = CSS[CSS.index("@media (max-height: 760px)"):]
    for rule in (".nav-item { min-height: 40px; }", ".side-foot-text { display: none; }"):
        assert rule in short
    # The icon rail below 1100px keeps working: the compact rules never apply there.
    assert "min-width: 1101px" in short
