"""Route and static-asset tests for the LogiShield ``/demo`` client.

The client is presentation only. These tests check that it is served, that
its fixtures are accepted by the real backend, and that the client source
contains no scoring/processing logic, storage, console logging or external
origins. They do not execute JavaScript; the in-browser journey is checked
separately by a manual/headless-browser smoke test.
"""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tokenization_ai.app import CONTRACT_VERSION, create_app
from tokenization_ai.demo import _build_demo_service

STATIC = Path(__file__).resolve().parents[2] / "tokenization_ai" / "static"
RAW_SECRETS = ["hunter2", "demo-not-a-real-token"]


@pytest.fixture
def client():
    return TestClient(create_app(_build_demo_service()))


def _scenarios():
    return json.loads((STATIC / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]


def test_demo_page_loads_with_required_content(client):
    response = client.get("/demo")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "<title>LogiShield LogiShield Demo</title>" in html
    assert "/demo/assets/demo.css" in html
    assert "/demo/assets/demo.js" in html
    # Declared explicitly so browsers do not request /favicon.ico (a 404 console error).
    assert 'rel="icon"' in html and "/demo/assets/favicon.svg" in html
    assert 'id="analyze-btn"' in html
    for stage in (
        "Input context",
        "Input fusion",
        "Privacy masking",
        "Structure-aware tokenization",
        "Tensor preparation",
        "Lightweight AI prediction",
    ):
        assert stage in html
    # Research status: implemented vs not yet integrated, no overclaiming.
    assert "Implemented" in html and "Not yet integrated" in html
    assert "Genuine Formal Verification verified-rule output" in html
    assert "Runtime Verification runtime handoff" in html
    assert "not detection accuracy" in html


def test_demo_page_has_restrictive_security_headers(client):
    headers = client.get("/demo").headers
    csp = headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "script-src 'self'" in csp
    assert "connect-src 'self'" in csp
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "name,content_type",
    [
        ("demo.css", "text/css"),
        ("demo.js", "text/javascript"),
        ("scenarios.json", "application/json"),
        ("favicon.svg", "image/svg+xml"),
    ],
)
def test_required_assets_are_served(client, name, content_type):
    response = client.get(f"/demo/assets/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)
    assert response.content


@pytest.mark.parametrize("name", ["missing.js", "index.html", "..%2Fapp.py", "%2e%2e/app.py"])
def test_unlisted_assets_and_traversal_are_rejected(client, name):
    response = client.get(f"/demo/assets/{name}")
    assert response.status_code == 404


def test_existing_routes_still_work(client):
    assert client.get("/health").json()["ready"] is True
    assert client.get("/docs").status_code == 200
    scenario = _scenarios()[0]
    assert client.post("/v1/analyze", json=scenario["request"]).status_code == 200


def test_both_scenarios_exist_and_are_labelled_synthetic():
    scenarios = {s["id"]: s for s in _scenarios()}
    assert set(scenarios) == {"benign", "suspicious"}
    assert scenarios["benign"]["request"]["rule_source"] == "none"
    assert scenarios["suspicious"]["request"]["rule_source"] == "mock_fixture"
    rules = scenarios["suspicious"]["request"]["rule_info"]["rules"]
    assert all(r["rule_id"].startswith("MOCK-") for r in rules)
    served = json.loads((STATIC / "scenarios.json").read_text(encoding="utf-8"))
    assert "MOCK" in served["disclaimer"] and "not genuine Formal Verification" in served["disclaimer"]


def test_scenarios_call_the_real_backend_with_v2_contract(client):
    scores = {}
    for scenario in _scenarios():
        response = client.post("/v1/analyze", json=scenario["request"])
        assert response.status_code == 200, scenario["id"]
        body = response.json()
        assert body["contract_version"] == CONTRACT_VERSION == "logishield-analysis-v2"
        assert body["event_id"] == scenario["request"]["context"]["event_id"]
        assert body["privacy"]["output_kind"] == "privacy_safe_projection"
        assert body["provenance"]["rule_status"]["verification_performed_by_this_service"] is False
        for secret in RAW_SECRETS:
            assert secret not in response.text
        scores[scenario["id"]] = body["prediction"]["score"]
    # Wiring check on the model's own synthetic training fixtures only.
    assert scores["suspicious"] > scores["benign"]


def test_backend_error_states_the_client_must_handle(client):
    bad = {"context": _scenarios()[0]["request"]["context"], "rule_source": "mock_fixture"}
    response = client.post("/v1/analyze", json=bad)
    assert response.status_code == 422
    # The client reads these two string fields to build its failure message.
    assert isinstance(response.json()["error"], str)
    assert isinstance(response.json()["detail"], str)


def test_client_source_posts_only_to_the_analyze_endpoint():
    js = (STATIC / "demo.js").read_text(encoding="utf-8")
    assert 'API_URL = "/v1/analyze"' in js
    assert re.findall(r"fetch\(\s*([A-Za-z_]+)", js) == ["API_URL", "SCENARIOS_URL"]
    assert "https://" not in js and "http://" not in js
    assert "CONTRACT = \"logishield-analysis-v2\"" in js


def test_client_source_has_no_storage_logging_or_unsafe_dom():
    js = (STATIC / "demo.js").read_text(encoding="utf-8")
    for forbidden in (
        "localStorage",
        "sessionStorage",
        "indexedDB",
        "document.cookie",
        "console.",
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "XMLHttpRequest",
        "sendBeacon",
        "WebSocket",
    ):
        assert forbidden not in js, forbidden


def test_client_source_does_no_scoring_masking_or_response_fabrication():
    js = (STATIC / "demo.js").read_text(encoding="utf-8")
    for forbidden in ("Math.random", "setTimeout(function () { render", "MASKED_", "[MASKED"):
        assert forbidden not in js, forbidden
    # No local masking/tokenizing/scoring machinery: no regexes, no hashing, no sigmoid.
    for forbidden in ("RegExp", ".replace(", "Math.exp", "crypto.", "tokenize("):
        assert forbidden not in js, forbidden
    # The score shown is read from the response, never computed.
    assert "data.prediction.score" in js
    # Every analysis render is reached only from a successful fetch result.
    assert js.count("render(result.payload)") == 1


def test_html_and_css_load_nothing_from_other_origins():
    for name in ("index.html", "demo.css"):
        text = (STATIC / name).read_text(encoding="utf-8")
        assert not re.search(r"(src|href)=\"https?://", text)
        assert "url(http" not in text and "@import" not in text
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "<script src=\"/demo/assets/demo.js\"></script>" in html
    assert "<script>" not in html
    assert " style=" not in html
