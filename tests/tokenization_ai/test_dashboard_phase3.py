"""Phase 3 regression tests: the Dashboard overview.

The overview's figures come from a pure view-model (overview-model.js) over the
shared session store, so the model is executed for real in Node. The page
source is checked for sources, wording and discipline. Browser behaviour
(landing page, navigation, live updates, clear-session reset, health states,
layout, console errors) is verified by a separate real-Chrome run.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tokenization_ai.app import create_app
from tokenization_ai.dashboard import DASHBOARD_ASSETS
from tokenization_ai.demo import _build_demo_service

ROOT = Path(__file__).resolve().parents[2] / "tokenization_ai"
DASH = ROOT / "dashboard_static"
DEMO_STATIC = ROOT / "static"
NODE = shutil.which("node")


def _read(name):
    return (DASH / name).read_text(encoding="utf-8")


@pytest.fixture
def client():
    return TestClient(create_app(_build_demo_service()))


# --- hosting / routing ---------------------------------------------------------


@pytest.mark.parametrize("name", ["overview-model.js", "page-overview.js"])
def test_overview_assets_are_allowlisted_and_served(client, name):
    assert name in DASHBOARD_ASSETS
    response = client.get(f"/dashboard/assets/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")


def test_dashboard_overview_is_the_default_landing_page():
    routes = _read("routes.js")
    assert 'export const DEFAULT_ROUTE = "dashboard";' in routes
    assert 'id: "dashboard", label: "Dashboard", icon: "home", kind: "overview"' in routes
    assert 'title: "LogiShield Dashboard"' in routes
    assert 'subtitle: "Member 4 Research Prototype Overview"' in routes
    assert 'href="#/dashboard"' in _read("index.html")
    app = _read("app.js")
    assert "overview: overview" in app and 'from "./page-overview.js"' in app


def test_original_demo_and_other_members_are_untouched(client):
    assert sorted(p.name for p in DEMO_STATIC.iterdir()) == [
        "demo.css", "demo.js", "favicon.svg", "index.html", "scenarios.json",
    ]
    assert client.get("/demo").status_code == 200
    assert client.get("/demo/assets/page-overview.js").status_code == 404


# --- the pure model, executed in Node -------------------------------------------


def _run_model(driver: str):
    source = _read("overview-model.js")
    result = subprocess.run(
        [NODE, "--input-type=module", "-"],
        input=source + "\n" + driver,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


ENTRY_JS = r"""
function entry(seq, scenarioId, score, tokens, masked, ms) {
  return { seq, observedAt: "2026-01-01T00:00:00.000Z", scenarioId, scenarioTitle: scenarioId + " title", eventId: "evt-" + seq,
    modelVersion: "m-1", score, method: "POST", path: "/x", ruleSource: "none", ruleLabel: "none", ruleCount: 0, ruleIds: [],
    maskedCount: masked, maskedFields: [], tokens: { total: tokens, byRole: { control: 1, literal_api: 2, masked: 3, literal_rule: 4 } },
    tensor: { maxLength: 128, beforeTruncation: tokens, tokensInTensor: tokens, paddedPositions: 128 - tokens, payloadTruncated: false,
      unknownTokens: 0, vocabularySize: 65 },
    hidden: { inRepresentation: 0, inTokens: 0 }, totalMs: ms,
    stages: [{ stage: "fusion", ms: 0.1, ok: true, counts: {} }, { stage: "masking", ms: 0.4, ok: true, counts: {} }],
    accuracyClaim: "none" };
}
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_model_is_all_empty_before_the_first_analysis():
    out = _run_model(ENTRY_JS + "console.log(JSON.stringify(buildOverviewModel([])));")
    assert out["hasData"] is False and out["count"] == 0 and out["latest"] is None
    # No fabricated zeros: every metric is null, every collection empty.
    assert out["kpis"] == {
        "analyses": None, "score": None, "tokens": None, "maxLength": None, "padded": None, "masked": None,
    }
    assert out["chart"] == [] and out["stages"] == [] and out["roles"] is None and out["totalMs"] is None


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_model_reports_only_what_the_session_entries_contain():
    out = _run_model(ENTRY_JS + r"""
const newestFirst = [entry(3, "suspicious", 0.9991, 44, 2, 0.25), entry(2, "benign", 0.0016, 26, 0, 0.2), entry(1, "benign", 0.0016, 26, 0, 0.21)];
console.log(JSON.stringify(buildOverviewModel(newestFirst)));
""")
    assert out["hasData"] and out["count"] == 3
    assert out["kpis"] == {"analyses": 3, "score": 0.9991, "tokens": 44, "maxLength": 128, "padded": 84, "masked": 2}
    assert [c["seq"] for c in out["chart"]] == [1, 2, 3]  # chronological, oldest first
    assert [c["latest"] for c in out["chart"]] == [False, False, True]
    assert [c["value"] for c in out["chart"]] == [0.0016, 0.0016, 0.9991]
    assert out["roles"] == {"control": 1, "literal_api": 2, "masked": 3, "literal_rule": 4}
    assert [s["stage"] for s in out["stages"]] == ["fusion", "masking"] and out["maxStageMs"] == 0.4
    assert out["totalMs"] == 0.25


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_model_chart_is_bounded_and_keeps_the_newest_events():
    out = _run_model(ENTRY_JS + r"""
const list = [];
for (let seq = 30; seq >= 1; seq -= 1) { list.push(entry(seq, "benign", 0.5, 26, 0, 0.2)); }
const m = buildOverviewModel(list);
console.log(JSON.stringify({ shown: m.shown, count: m.count, first: m.chart[0].seq, last: m.chart[m.chart.length - 1].seq }));
""")
    assert out == {"shown": 12, "count": 30, "first": 19, "last": 30}


def test_model_module_is_pure():
    js = _read("overview-model.js")
    for forbidden in ("fetch(", "document.", "window.", "localStorage", "sessionStorage", "Math.random", "import "):
        assert forbidden not in js, forbidden


# --- the page: sources, wording, discipline ---------------------------------------


def test_overview_reads_only_the_shared_store_and_health():
    js = _read("page-overview.js")
    assert "fetch(" not in js
    for needle in ('from "./session.js"', 'from "./shell.js"', 'from "./overview-model.js"', "session.list()", "shell.health"):
        assert needle in js
    assert "session.add(" not in js and "session.clear(" not in js  # the overview never writes to the store


def test_overview_shows_empty_states_not_zeros_before_any_analysis():
    js = _read("page-overview.js")
    assert 'k.analyses === null ? "' + chr(0x2014) + '"' in js  # an em dash, never a 0
    assert "No analysis in this session yet" in js
    assert "No analyses in this session yet" in js
    assert 'link("#/anomaly", "btn link-btn", "Open Anomaly Detection")' in js  # a working call to action


def test_overview_never_claims_system_health_accuracy_or_verdicts():
    js = _read("page-overview.js")
    for forbidden in (
        "All systems", "Operational", "Healthy", "Running", "Uptime", "req/s", "Critical", "Incidents", "Allowed Requests",
        "Blocked Requests", "Math.random", "setInterval", "setTimeout", "99.",
    ):
        assert forbidden not in js, forbidden
    for needle in (
        "not detection accuracy", "Uncalibrated", "not the KPI 4 benchmark", "not deployed gateway latency",
        "Only Member 4 has a live status", "KPI 5", "Session-local", "lost on reload", "Synthetic demo fixtures",
    ):
        assert needle in js, needle


def test_overview_component_facts_are_reused_not_duplicated():
    page = _read("page-overview.js")
    comps = _read("page-components.js")
    assert 'import { COMPONENTS } from "./page-components.js";' in page
    assert "export const COMPONENTS = [" in comps
    # Statuses are read from the shared facts, never restated here.
    for restated in ("Tested in-process", "Not integrated", "Pending contract"):
        assert restated not in page
    assert page.count("c.live") >= 2  # only the flagged Member 4 component gets a live badge


def test_overview_quick_actions_are_real_links():
    js = _read("page-overview.js")
    for target in ('"#/anomaly"', '"#/logs"', '"#/components"'):
        assert target in js
    routes = _read("routes.js")
    for rid in ("anomaly", "logs", "components"):
        assert f'id: "{rid}"' in routes


def test_dashboard_source_discipline_still_holds_for_every_script():
    for path in sorted(DASH.glob("*.js")):
        js = path.read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")
        for forbidden in (
            "localStorage", "sessionStorage", "indexedDB", "document.cookie", "console.", "innerHTML", "outerHTML",
            "insertAdjacentHTML", "document.write", "eval(", "new Function", "XMLHttpRequest", "sendBeacon",
            "WebSocket", "http://", "https://",
        ):
            assert forbidden not in js, (path.name, forbidden)
        if path.name != "api.js":
            assert "fetch(" not in js, path.name


def test_session_store_privacy_allowlist_is_unchanged_by_phase3():
    # Phase 3 must not widen what is stored: the entry shape is still the Phase 2 allowlist.
    js = _read("session.js")
    start = js.index("return Object.freeze({")
    block = js[start: js.index("});", start)]
    keys = [line.strip().split(":")[0] for line in block.splitlines()[1:] if ":" in line and not line.startswith("      ")]
    assert keys == [
        "seq", "observedAt", "scenarioId", "scenarioTitle", "eventId", "contract", "modelVersion", "score", "method",
        "path", "ruleSource", "ruleLabel", "ruleCount", "ruleIds", "maskedCount", "maskedFields", "tokens", "tensor",
        "hidden", "totalMs", "stages", "accuracyClaim",
    ]


def test_scenarios_still_return_the_documented_real_backend_values(client):
    scenarios = {
        s["id"]: s["request"]
        for s in json.loads((DEMO_STATIC / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    }
    benign = client.post("/v1/analyze", json=scenarios["benign"]).json()
    suspicious = client.post("/v1/analyze", json=scenarios["suspicious"]).json()
    assert (benign["tokens"]["total"], benign["tensor"]["padded_positions"], benign["masking"]["masked_field_count"]) == (26, 102, 0)
    assert (suspicious["tokens"]["total"], suspicious["tensor"]["padded_positions"], suspicious["masking"]["masked_field_count"]) == (44, 84, 2)
