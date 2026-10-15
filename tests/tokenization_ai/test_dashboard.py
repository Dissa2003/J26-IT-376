"""Tests for the Member 4 ``/dashboard`` host and its static client.

The dashboard is presentation only. These tests check that it is served with a
strict CSP from its own allowlisted directory, that the original ``/demo`` is
unaffected, that the preset scenarios still produce the documented real backend
results, and that the client source contains no storage, logging, unsafe DOM,
external origin, scoring logic or fabricated figures. They do not execute
JavaScript; the in-browser journey is checked by a separate browser run.
"""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tokenization_ai.app import CONTRACT_VERSION, create_app
from tokenization_ai.dashboard import DASHBOARD_ASSETS
from tokenization_ai.demo import _build_demo_service

ROOT = Path(__file__).resolve().parents[2] / "tokenization_ai"
DASH = ROOT / "dashboard_static"
DEMO_STATIC = ROOT / "static"
RAW_SECRETS = ["hunter2", "demo-not-a-real-token"]


@pytest.fixture
def client():
    return TestClient(create_app(_build_demo_service()))


def _js_files():
    return sorted(DASH.glob("*.js"))


def _read(name):
    return (DASH / name).read_text(encoding="utf-8")


# --- hosting -----------------------------------------------------------------


def test_dashboard_page_is_served_with_strict_headers(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>LogiShield Dashboard</title>" in response.text
    csp = response.headers["content-security-policy"]
    for directive in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "connect-src 'self'"):
        assert directive in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("name,media_type", sorted(DASHBOARD_ASSETS.items()))
def test_every_allowlisted_asset_is_served(client, name, media_type):
    response = client.get(f"/dashboard/assets/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type.split(";")[0])
    assert response.content


def test_allowlist_matches_the_files_on_disk():
    on_disk = {p.name for p in DASH.iterdir() if p.is_file() and p.name != "index.html"}
    assert on_disk == set(DASHBOARD_ASSETS)


@pytest.mark.parametrize(
    "name", ["missing.js", "index.html", "..%2Fapp.py", "%2e%2e/app.py", "%2e%2e%2fstatic%2fdemo.js", "scenarios.json"]
)
def test_unlisted_dashboard_assets_and_traversal_are_rejected(client, name):
    assert client.get(f"/dashboard/assets/{name}").status_code == 404


def test_dashboard_adds_no_post_routes_or_schema_entries(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert not [p for p in paths if p.startswith("/dashboard")]
    assert client.post("/dashboard", json={}).status_code == 405


# --- original demo preserved -------------------------------------------------


def test_demo_and_existing_routes_still_work(client):
    assert client.get("/demo").status_code == 200
    assert client.get("/health").json()["ready"] is True
    assert client.get("/docs").status_code == 200
    for name in ("demo.css", "demo.js", "scenarios.json", "favicon.svg"):
        assert client.get(f"/demo/assets/{name}").status_code == 200
    # The two asset namespaces stay separate.
    assert client.get("/demo/assets/app.js").status_code == 404
    assert client.get("/dashboard/assets/demo.js").status_code == 404


def test_demo_static_directory_is_untouched_by_the_dashboard():
    assert sorted(p.name for p in DEMO_STATIC.iterdir()) == [
        "demo.css", "demo.js", "favicon.svg", "index.html", "scenarios.json",
    ]


# --- real backend behaviour behind the Anomaly Detection page ----------------


def _scenarios():
    return {
        s["id"]: s["request"]
        for s in json.loads((DEMO_STATIC / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    }


def test_benign_scenario_returns_the_documented_real_result(client):
    response = client.post("/v1/analyze", json=_scenarios()["benign"])
    assert response.status_code == 200
    body = response.json()
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["tokens"]["total"] == 26
    assert body["tensor"]["padded_positions"] == 102
    assert body["masking"]["masked_field_count"] == 0
    assert body["prediction"]["score"] < 0.01


def test_suspicious_scenario_returns_the_documented_real_result(client):
    response = client.post("/v1/analyze", json=_scenarios()["suspicious"])
    assert response.status_code == 200
    body = response.json()
    assert body["tokens"]["total"] == 44
    assert body["tensor"]["padded_positions"] == 84
    assert body["masking"]["masked_field_count"] == 2
    assert body["provenance"]["rule_status"]["source"] == "mock_fixture"
    assert body["provenance"]["rule_status"]["verification_performed_by_this_service"] is False
    assert body["provenance"]["detection_accuracy_claim"] == "none"
    assert body["prediction"]["score"] > 0.9
    for secret in RAW_SECRETS:
        assert secret not in response.text


def test_response_has_every_field_the_dashboard_reads(client):
    body = client.post("/v1/analyze", json=_scenarios()["suspicious"]).json()
    stages = {s["stage"] for s in body["timings"]["stages"]}
    assert stages == {"fusion", "masking", "normalization", "tokenization", "tensorization", "model_inference"}
    js = _read("page-anomaly.js")
    for stage in stages:
        assert f'"{stage}"' in js
    for key in ("masked_fields", "padded_positions", "unknown_token_count", "vocabulary_size", "model_version"):
        assert key in js


# --- client source discipline --------------------------------------------------


def test_navigation_has_every_required_page_and_a_placeholder_settings():
    routes = _read("routes.js")
    for label in (
        "Dashboard", "API Traffic", "Anomaly Detection", "Verified Rules",
        "Allow / Block Decisions", "Logs & Traces", "System Components", "Reports", "Settings",
    ):
        assert f'label: "{label}"' in routes
    assert routes.count('kind: "anomaly"') == 1
    assert routes.count('kind: "pending"') == 8


def test_pending_pages_show_no_data_and_say_so():
    pending = _read("page-pending.js")
    assert "Not available in this phase" in pending
    assert "No data" in pending
    for forbidden in ("fetch(", "gauge(", "donut(", "Math."):
        assert forbidden not in pending


def test_only_the_api_module_uses_the_network_and_only_known_endpoints():
    api = _read("api.js")
    assert 'ANALYZE_URL = "/v1/analyze"' in api
    assert 'HEALTH_URL = "/health"' in api
    assert 'SCENARIOS_URL = "/demo/assets/scenarios.json"' in api
    assert re.findall(r"fetch\(\s*([A-Za-z_]+)", api) == ["HEALTH_URL", "SCENARIOS_URL", "ANALYZE_URL"]
    assert 'CONTRACT = "member4-analysis-v2"' in api
    for path in _js_files():
        if path.name != "api.js":
            assert "fetch(" not in path.read_text(encoding="utf-8"), path.name


def test_client_source_has_no_storage_logging_or_unsafe_dom():
    for path in _js_files():
        # The SVG XML namespace is an identifier, not a network origin.
        js = path.read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")
        for forbidden in (
            "localStorage", "sessionStorage", "indexedDB", "document.cookie", "console.",
            "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(",
            "new Function", "XMLHttpRequest", "sendBeacon", "WebSocket", "setAttribute(\"style\"",
            "http://", "https://",
        ):
            assert forbidden not in js, (path.name, forbidden)


def test_client_does_no_scoring_masking_or_fabrication():
    for path in _js_files():
        js = path.read_text(encoding="utf-8")
        for forbidden in ("Math.random", "Math.exp", "RegExp", "crypto.", "tokenize(", "[MASKED"):
            assert forbidden not in js, (path.name, forbidden)
    anomaly = _read("page-anomaly.js")
    assert "data.prediction.score" in anomaly
    assert anomaly.count("state.result = d;") == 1  # a result only ever comes from a successful response


def test_no_fabricated_figures_or_verdict_vocabulary_in_the_client():
    text = "\n".join(p.read_text(encoding="utf-8") for p in DASH.iterdir() if p.is_file())
    for forbidden in (
        "326", "94.8", "96.2", "12,426", "10,842", "Critical", "Monitor", "Throttle", "Challenge",
        "Allowed Requests", "Blocked Requests", "Total Anomalies", "Operational", "Healthy",
        "SAMPLE DATA", "sample_data",
    ):
        assert forbidden not in text, forbidden


def test_session_history_stores_response_summaries_never_the_request():
    js = _read("page-anomaly.js")
    block = js[js.index("state.history.unshift({"):]
    block = block[: block.index("});")]
    assert "request" not in block
    assert "HISTORY_LIMIT" in js


def test_scenario_change_clears_the_previous_result():
    js = _read("page-anomaly.js")
    select = js[js.index("function select("): js.index("async function analyze")]
    assert "state.renderedId !== id" in select
    assert "state.result = null" in select


def test_analyze_ignores_repeated_clicks_while_busy():
    js = _read("page-anomaly.js")
    analyze = js[js.index("async function analyze"): js.index("async function loadScenarios")]
    assert "if (state.busy) { return; }" in analyze
    assert "state.busy = true" in analyze
    assert "ui.analyze.disabled = state.busy" in js


def test_html_and_css_load_nothing_from_other_origins():
    html = _read("index.html")
    css = _read("dashboard.css")
    for text in (html, css):
        assert not re.search(r"(src|href)=\"https?://", text)
        assert "url(http" not in text and "@import" not in text
    assert "<script>" not in html
    assert " style=" not in html
    assert '<script type="module" src="/dashboard/assets/app.js"></script>' in html
