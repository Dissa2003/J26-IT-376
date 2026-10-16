"""Phase 2 regression tests: Logs & Traces and System Components.

Source-level checks plus a Node-executed test of the session store, which is a
pure module (no DOM, no network, no storage) so its privacy allowlist can be
exercised for real. Browser behaviour (navigation, filters, selection,
timeline, console errors, layout) is verified by a separate real-Chrome run.
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


# --- hosting -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["session.js", "page-logs.js", "page-components.js"])
def test_new_phase2_assets_are_allowlisted_and_served(client, name):
    assert name in DASHBOARD_ASSETS
    response = client.get(f"/dashboard/assets/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.headers["content-security-policy"].startswith("default-src 'none'")


def test_original_demo_assets_are_unchanged_in_name_and_still_served(client):
    assert sorted(p.name for p in DEMO_STATIC.iterdir()) == [
        "demo.css", "demo.js", "favicon.svg", "index.html", "scenarios.json",
    ]
    assert client.get("/demo").status_code == 200
    assert client.get("/demo/assets/page-logs.js").status_code == 404  # namespaces stay separate


# --- session store: privacy allowlist (executed in Node) -----------------------


def _run_session_js(driver: str) -> dict:
    source = _read("session.js")
    result = subprocess.run(
        [NODE, "--input-type=module", "-"],
        input=source + "\n" + driver,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


HOSTILE_RESPONSE = r"""
const data = {
  contract_version: "member4-analysis-v2", event_id: "evt-1",
  // fields a response should never carry, to prove the store cannot keep them
  raw_request: { password: "HUNTER2-RAW", headers: { Authorization: "Bearer RAW-TOKEN" } },
  context: { payload: { password: "HUNTER2-RAW" } },
  provenance: { detection_accuracy_claim: "none", leaked: "RAW-TOKEN",
    rule_status: { source: "mock_fixture", label: "mock rule fixtures", rule_count: 1, rule_ids: ["MOCK-R-SQLI"], note: "RAW-TOKEN" },
    model: { model_version: "m-1", note: "RAW-TOKEN" } },
  privacy: { hidden_in_representation: 1, hidden_in_tokens: 2, note: "RAW-TOKEN" },
  masking: { masked_field_count: 2, masked_fields: ["headers.authorization", "payload.password"] },
  masked_representation: { method: "POST", path: "/login",
    headers: { authorization: "[MASKED_AUTH]" },
    payload: { password: "[MASKED_SECRET]", q: "' OR 1=1 --", username: "HUNTER2-RAW" } },
  tokens: { total: 4, returned: 4, truncated: false, items: [
    { token: "HUNTER2-RAW", role: "literal_api" }, { token: "[MASKED_AUTH]", role: "masked" },
    { token: "[SEQ_START]", role: "control" }, { token: "x", role: "evil_role" } ] },
  tensor: { max_length: 128, token_count_before_truncation: 4, tokens_in_tensor: 4, padded_positions: 124,
    payload_truncated: false, unknown_token_count: 0, vocabulary_size: 65, leaked: "RAW-TOKEN" },
  prediction: { score: 0.5, model_version: "m-1" },
  timings: { total_elapsed_ms: 1.5, stages: [
    { stage: "fusion", elapsed_ms: 0.1, success: true, summary: { rule_count: 1, leaked_text: "RAW-TOKEN", flag: true } },
    { stage: "RAW-TOKEN-STAGE", elapsed_ms: 9, success: true, summary: {} },
    { stage: "masking", elapsed_ms: 0.2, success: true, summary: { masked_field_count: 2 } } ] }
};
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_session_summary_contains_only_allowlisted_privacy_safe_fields():
    out = _run_session_js(HOSTILE_RESPONSE + r"""
const entry = summarize(data, { id: "suspicious", title: "Suspicious API request", request: { secret: "HUNTER2-RAW" } },
                        "2026-01-01T00:00:00.000Z", 7);
console.log(JSON.stringify(entry));
""")
    blob = json.dumps(out)
    for forbidden in ("HUNTER2-RAW", "RAW-TOKEN", "RAW-TOKEN-STAGE", "leaked", "raw_request", "' OR 1=1"):
        assert forbidden not in blob, forbidden
    assert sorted(out) == sorted([
        "seq", "observedAt", "scenarioId", "scenarioTitle", "eventId", "contract", "modelVersion", "score",
        "method", "path", "ruleSource", "ruleLabel", "ruleCount", "ruleIds", "maskedCount", "maskedFields",
        "tokens", "tensor", "hidden", "totalMs", "stages", "accuracyClaim",
    ])
    # Only counts of token roles, never token text; unknown roles are dropped.
    assert out["tokens"] == {
        "total": 4, "byRole": {"control": 1, "literal_api": 1, "masked": 1, "literal_rule": 0},
    }
    # Unknown stage names are dropped and stage summaries keep only counts/flags under safe keys.
    assert [s["stage"] for s in out["stages"]] == ["fusion", "masking"]
    assert out["stages"][0]["counts"] == {"rule_count": 1, "flag": True}
    assert out["maskedFields"] == [
        {"field": "headers.authorization", "placeholder": "[MASKED_AUTH]"},
        {"field": "payload.password", "placeholder": "[MASKED_SECRET]"},
    ]
    assert out["scenarioId"] == "suspicious" and out["accuracyClaim"] == "none"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_session_store_is_empty_bounded_ordered_and_clearable():
    out = _run_session_js(HOSTILE_RESPONSE + r"""
const scenario = { id: "benign", title: "Benign API request" };
const result = { initial: session.size() };
let notified = 0;
const off = session.subscribe(() => { notified += 1; });
session.add(data, scenario);
session.add(data, scenario);
result.afterTwo = session.list().map((e) => e.seq);
for (let i = 0; i < MAX_EVENTS + 5; i += 1) { session.add(data, scenario); }
result.bounded = session.size();
result.newestFirst = session.list()[0].seq > session.list()[1].seq;
result.notified = notified;
session.clear();
result.afterClear = session.size();
session.add(data, scenario);
result.seqAfterClear = session.list()[0].seq;
off();
console.log(JSON.stringify(result));
""")
    assert out["initial"] == 0  # nothing exists before the first analysis
    assert out["afterTwo"] == [2, 1]
    assert out["bounded"] == 50
    assert out["newestFirst"] is True
    assert out["notified"] == 2 + 55
    assert out["afterClear"] == 0
    assert out["seqAfterClear"] == 1


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_session_store_never_uses_browser_storage_or_network():
    js = _read("session.js")
    for forbidden in ("localStorage", "sessionStorage", "indexedDB", "document.", "window.", "fetch(", "XMLHttpRequest"):
        assert forbidden not in js, forbidden


# --- Logs & Traces ---------------------------------------------------------------


def test_logs_page_is_session_only_and_makes_no_requests():
    js = _read("page-logs.js")
    assert "fetch(" not in js
    assert 'from "./session.js"' in js
    assert "Member 4 Session Traces" in js
    assert "not persisted gateway logs" in js
    assert "not distributed tracing" in js
    assert "dashboard local time" in js
    assert "not when the request was made" in js
    assert "No analyses in this session yet" in js  # empty state before the first analysis


def test_logs_page_only_reads_from_the_session_store_and_never_invents_data():
    js = _read("page-logs.js")
    for forbidden in ("Math.random", "setInterval", '"ERROR"', '"WARN"', '"INFO"', "req_", "trc_", "2024-"):
        assert forbidden not in js, forbidden
    # Filters, selection and sort act on stored events only.
    for needle in ("function matches(", "view.selected", "view.newestFirst", "session.list()", "session.clear()"):
        assert needle in js


def test_logs_page_has_timeline_flow_and_detail_for_the_six_stages():
    js = _read("page-logs.js")
    for stage in ("fusion", "masking", "normalization", "tokenization", "tensorization", "model_inference"):
        assert f"{stage}:" in js
    for title in ("Trace timeline", "Member 4 component flow", "Event details & correlation", "Session events"):
        assert title in js
    assert "Not implemented; no agreed contract" in js  # the Member 3 hand-off is shown as not implemented


def test_anomaly_page_feeds_the_store_only_after_a_successful_response():
    js = _read("page-anomaly.js")
    block = js[js.index("const outcome = await api.analyze"): js.index("async function loadScenarios")]
    assert block.count("session.add(") == 1
    assert block.index("if (outcome.ok)") < block.index("session.add(")


# --- System Components -------------------------------------------------------------


def test_components_page_uses_real_health_and_no_other_module_status():
    js = _read("page-components.js")
    assert 'from "./shell.js"' in js and "shell.health" in js
    assert "fetch(" not in js
    # Only Member 4 may carry a live status.
    live_flags = [line for line in js.splitlines() if "live: true" in line]
    assert len(live_flags) == 1 and "tokenization_ai/" in live_flags[0]
    assert js.count('badge("Not connected", "pending")') == 1  # the single non-live branch


@pytest.mark.parametrize(
    "needle",
    [
        "Tested in-process",
        "Not integrated",
        "Pending contract",
        "Not measured",
        "to_member3_output raises NotImplementedError",
        "not part of docker-compose",
        "never calls it",
        "generic StubModel",
        "not Member 4's trained model",
        "port 8004",
    ],
)
def test_components_page_states_the_documented_integration_truth(needle):
    assert needle in _read("page-components.js")


def test_components_page_does_not_claim_operation_or_invent_metrics():
    js = _read("page-components.js")
    for forbidden in (
        "Healthy", "Operational", "Running", "Uptime", "uptime:", "req/s", "CPU Usage", "Throughput", "Instances",
        "99.", "Math.random", "setInterval", "fetch(",
    ):
        assert forbidden not in js, forbidden
    assert "KPI 5" in js and "No percentage is reported" in js


def test_components_edges_distinguish_tested_from_proposed():
    js = _read("page-components.js")
    assert js.count('"tested")') == 1  # exactly one tested connection (Member 1 -> Member 4)
    assert js.count('"pending")') >= 4
    assert "arr-tested" in js and "arr-pending" in js
    css = _read("dashboard.css")
    assert ".dg-pending" in css and "stroke-dasharray" in css


def test_member1_integration_evidence_named_on_the_page_exists():
    assert (ROOT.parent / "tests" / "tokenization_ai" / "test_member1_integration.py").exists()
    assert "test_member1_integration.py" in _read("page-components.js")
    assert (ROOT / "INTEGRATION_STATUS.md").exists()


# --- shared dashboard discipline over every file ----------------------------------


def test_every_dashboard_script_keeps_the_source_level_protections():
    for path in sorted(DASH.glob("*.js")):
        js = path.read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")
        for forbidden in (
            "localStorage", "sessionStorage", "indexedDB", "document.cookie", "console.", "innerHTML", "outerHTML",
            "insertAdjacentHTML", "document.write", "eval(", "new Function", "XMLHttpRequest", "sendBeacon",
            "WebSocket", "http://", "https://",
        ):
            assert forbidden not in js, (path.name, forbidden)


def test_only_the_api_module_uses_fetch():
    for path in sorted(DASH.glob("*.js")):
        if path.name != "api.js":
            assert "fetch(" not in path.read_text(encoding="utf-8"), path.name


def test_navigation_lists_functional_and_pending_pages_consistently():
    routes = _read("routes.js")
    assert routes.count("id: ") == 9
    for rid in ("dashboard", "traffic", "anomaly", "rules", "decisions", "logs", "components", "reports", "settings"):
        assert f'id: "{rid}"' in routes
    app = _read("app.js")
    assert "logs: logs" in app and "components: components" in app and "anomaly: anomaly" in app


def test_scenarios_still_return_the_documented_real_backend_values(client):
    scenarios = {
        s["id"]: s["request"]
        for s in json.loads((DEMO_STATIC / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    }
    benign = client.post("/v1/analyze", json=scenarios["benign"]).json()
    suspicious = client.post("/v1/analyze", json=scenarios["suspicious"]).json()
    assert (benign["tokens"]["total"], benign["tensor"]["padded_positions"], benign["masking"]["masked_field_count"]) == (26, 102, 0)
    assert (suspicious["tokens"]["total"], suspicious["tensor"]["padded_positions"], suspicious["masking"]["masked_field_count"]) == (44, 84, 2)
    assert benign["prediction"]["score"] < 0.01 < 0.9 < suspicious["prediction"]["score"]
