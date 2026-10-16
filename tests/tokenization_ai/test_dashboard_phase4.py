"""Phase 4 regression tests: the Reports page, documented KPI evidence and exports.

The export formatter, the session-report model and the evidence data are pure
modules, so they are executed for real in Node. Page-level behaviour (loading,
live updates, clear/reload resets, downloads, layout, console errors) is
verified by a separate real-Chrome run.
"""

import json
import re
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
README = (ROOT / "README.md").read_text(encoding="utf-8")
NODE = shutil.which("node")
EVIDENCE_DIR = Path(r"C:\Users\kavin\Downloads\LogiShield_PP1_Coding_Evidence\KPI_Evaluation")
SECRETS = ["HUNTER2-RAW", "RAW-TOKEN", "demo-not-a-real-token"]


def _read(name):
    return (DASH / name).read_text(encoding="utf-8")


def _node(source_files, driver):
    source = chr(10).join(_read(f) for f in source_files)
    # Explicit UTF-8 bytes: the sources contain non-ASCII characters and the Windows locale is not UTF-8.
    result = subprocess.run(
        [NODE, "--input-type=module", "-"], input=(source + chr(10) + driver).encode("utf-8"),
        capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return json.loads(result.stdout.decode("utf-8").strip().splitlines()[-1])


@pytest.fixture
def client():
    return TestClient(create_app(_build_demo_service()))


# --- hosting / routing -----------------------------------------------------------


@pytest.mark.parametrize("name", ["page-reports.js", "reports-model.js", "export-format.js", "evidence.js"])
def test_reports_assets_are_allowlisted_and_served(client, name):
    assert name in DASHBOARD_ASSETS
    response = client.get(f"/dashboard/assets/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")


def test_reports_route_is_functional_and_wired():
    routes = _read("routes.js")
    assert 'id: "reports", label: "Reports", icon: "reports", kind: "reports"' in routes
    assert "reports: reports" in _read("app.js")


def test_original_demo_untouched(client):
    assert sorted(p.name for p in DEMO_STATIC.iterdir()) == [
        "demo.css", "demo.js", "favicon.svg", "index.html", "scenarios.json",
    ]
    assert client.get("/demo").status_code == 200
    assert client.get("/demo/assets/page-reports.js").status_code == 404


# --- export: allowlist, CSV safety (Node) -------------------------------------------

HOSTILE_ENTRY = r"""
const hostile = {
  seq: 7, observedAt: "2026-10-15T09:00:50.225Z", scenarioId: "suspicious", scenarioTitle: "=HYPERLINK(\"http://x\",\"y\")",
  eventId: "+cmd|' /C calc'!A0", modelVersion: "-2+3", score: 0.5, method: "POST", path: "@SUM(1,1)", ruleSource: "mock_fixture",
  ruleLabel: "mock rule fixtures", ruleCount: 1, ruleIds: ["MOCK-R-SQLI"], maskedCount: 2,
  maskedFields: [{ field: "headers.authorization", placeholder: "[M]" }, { field: "payload.password", placeholder: "[M]" }],
  tokens: { total: 44, byRole: { control: 31, literal_api: 8, masked: 2, literal_rule: 3 }, items: [{ token: "HUNTER2-RAW" }] },
  tensor: { maxLength: 128, beforeTruncation: 44, tokensInTensor: 44, paddedPositions: 84, payloadTruncated: false,
            unknownTokens: 0, vocabularySize: 65 },
  hidden: { inRepresentation: 0, inTokens: 0 }, totalMs: 0.25,
  stages: [{ stage: "fusion", ms: 0.01, ok: true, counts: { rule_count: 1 } }, { stage: "masking", ms: 0.07, ok: true, counts: {} }],
  accuracyClaim: "none",
  // properties that must never be exported even if present on an object
  rawRequest: { password: "HUNTER2-RAW" }, authorization: "Bearer RAW-TOKEN", masked_representation: { password: "HUNTER2-RAW" },
  extra: "RAW-TOKEN"
};
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_json_export_contains_only_allowlisted_fields_and_no_secrets():
    out = _node(["export-format.js"], HOSTILE_ENTRY + r"""
console.log(JSON.stringify(buildJsonReport([hostile], "2026-10-15T09:01:00.000Z")));
""")
    assert out["report_type"] == "member4-session-summaries" and out["event_count"] == 1
    assert "uncalibrated" in out["scope"] and "no raw requests or secrets" in out["scope"]
    blob = json.dumps(out)
    for secret in SECRETS:
        assert secret not in blob, secret
    event = out["events"][0]
    assert sorted(event) == sorted([
        "sequence", "observed_at_utc", "scenario_id", "scenario_title", "event_id", "model_version",
        "anomaly_score_uncalibrated", "method", "path", "rule_source", "rule_count", "masked_field_count",
        "masked_fields", "tokens_total", "tokens_by_role", "tensor_max_length", "tokens_in_tensor",
        "padded_positions", "payload_truncated", "unknown_tokens", "pipeline_total_ms", "stage_ms",
    ])
    assert event["masked_fields"] == ["headers.authorization", "payload.password"]  # paths only, no placeholders or values
    assert event["stage_ms"]["fusion"] == 0.01 and event["stage_ms"]["tokenization"] is None


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_csv_export_is_allowlisted_quoted_and_formula_safe():
    out = _node(["export-format.js"], HOSTILE_ENTRY + r"""
const csv = buildCsv([hostile]);
console.log(JSON.stringify({ csv, columns: CSV_COLUMNS, cells: {
  eq: csvCell("=1+1"), plus: csvCell("+1"), minus: csvCell("-1"), at: csvCell("@a"), tab: csvCell("\tx"), cr: csvCell("\rx"),
  space: csvCell("  =1"), comma: csvCell("a,b"), quote: csvCell('say "hi"'), newline: csvCell("a\nb"),
  number: csvCell(-1), zero: csvCell(0), bool: csvCell(true), nul: csvCell(null), safe: csvCell("/login"),
  both: csvCell('=SUM("a",1)')
}, filename: exportFilename("csv", "2026-10-15T09:00:50.225Z"), jsonName: exportFilename("json", "2026-10-15T09:00:50.225Z") }));
""")
    csv_text = out["csv"]
    lines = csv_text.rstrip("\r\n").split("\r\n")
    assert lines[0] == ",".join(out["columns"]) and len(lines) == 2
    for secret in SECRETS:
        assert secret not in csv_text, secret
    cells = out["cells"]
    # Formula triggers are neutralised with a leading apostrophe; numbers and plain text are untouched.
    assert cells["eq"] == "'=1+1" and cells["plus"] == "'+1" and cells["minus"] == "'-1" and cells["at"] == "'@a"
    assert cells["tab"] == "'\tx" and cells["space"] == "'  =1"
    assert cells["cr"].startswith('"\'\r')  # apostrophe-prefixed and quoted because of the carriage return
    assert cells["number"] == "-1" and cells["zero"] == "0" and cells["bool"] == "true" and cells["nul"] == ""
    assert cells["safe"] == "/login"
    # RFC 4180 quoting.
    assert cells["comma"] == '"a,b"' and cells["quote"] == '"say ""hi"""' and cells["newline"] == '"a\nb"'
    assert cells["both"] == "\"'=SUM(\"\"a\"\",1)\""
    # Hostile text fields are neutralised: after real CSV parsing no cell starts with a formula trigger.
    import csv
    import io
    parsed = list(csv.reader(io.StringIO(csv_text)))
    assert len(parsed) == 2 and len(parsed[1]) == len(parsed[0])
    row = dict(zip(parsed[0], parsed[1]))
    assert row["scenario_title"].startswith("'=HYPERLINK") and row["event_id"].startswith("'+cmd")
    assert row["model_version"] == "'-2+3" and row["path"] == "'@SUM(1,1)"
    for name, cell in row.items():
        assert not cell.startswith(("=", "+", "-", "@", chr(9), chr(13))), (name, cell)
    assert row["masked_fields"] == "headers.authorization;payload.password"
    assert out["filename"] == "member4-session-summaries-20261015T090050Z.csv"
    assert out["jsonName"].endswith(".json") and "member4" in out["jsonName"] and "session" in out["jsonName"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_empty_export_is_header_only_and_has_no_events():
    out = _node(["export-format.js"], r"""
console.log(JSON.stringify({ csv: buildCsv([]), json: buildJsonReport([], "2026-01-01T00:00:00.000Z") }));
""")
    assert out["csv"].count("\r\n") == 1 and out["json"]["event_count"] == 0 and out["json"]["events"] == []


def test_export_module_is_pure_and_the_page_only_exports_on_click():
    fmt = _read("export-format.js")
    for forbidden in ("document.", "window.", "fetch(", "localStorage", "sessionStorage", "Blob", "URL."):
        assert forbidden not in fmt, forbidden
    page = _read("page-reports.js")
    assert "fetch(" not in page and "localStorage" not in page and "sessionStorage" not in page
    # The only callers of the download path are the two button handlers.
    assert page.count("doExport(") == 3  # definition + JSON + CSV button handlers
    assert "json.disabled = empty" in page and "csv.disabled = empty" in page
    assert "Nothing to export" in page
    assert page.count("download(") == 3  # definition + 2 call sites inside doExport
    # No automatic persistence or timers driving exports.
    for forbidden in ("setInterval", "requestIdleCallback", "beforeunload", "unload"):
        assert forbidden not in page


# --- session-report model (Node) -------------------------------------------------------

MODEL_ENTRY = r"""
function entry(seq, scenarioId, score, tokens, masked, roles, unknown, truncated) {
  return { seq, observedAt: "2026-01-01T00:00:00.000Z", scenarioId, scenarioTitle: scenarioId, eventId: "e" + seq, modelVersion: "m",
    score, method: "GET", path: "/", ruleSource: "none", ruleLabel: "none", ruleCount: 0, ruleIds: [], maskedCount: masked, maskedFields: [],
    tokens: { total: tokens, byRole: roles }, tensor: { maxLength: 128, beforeTruncation: tokens, tokensInTensor: tokens,
      paddedPositions: 128 - tokens, payloadTruncated: truncated, unknownTokens: unknown, vocabularySize: 65 },
    hidden: { inRepresentation: 0, inTokens: 0 }, totalMs: 0.2, stages: [], accuracyClaim: "none" };
}
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_report_model_is_empty_without_analyses_and_aggregates_exactly_with_them():
    out = _node(["overview-model.js", "reports-model.js"], MODEL_ENTRY + r"""
const empty = buildSessionReport([], buildOverviewModel([]));
const list = [
  entry(3, "suspicious", 0.9991, 44, 2, { control: 31, literal_api: 8, masked: 2, literal_rule: 3 }, 0, false),
  entry(2, "benign", 0.0016, 26, 0, { control: 20, literal_api: 6, masked: 0, literal_rule: 0 }, 1, true),
  entry(1, "suspicious", 0.9, 44, 2, { control: 31, literal_api: 8, masked: 2, literal_rule: 3 }, 0, false)
];
const full = buildSessionReport(list, buildOverviewModel(list));
console.log(JSON.stringify({ empty: { hasData: empty.hasData, count: empty.count, withMasking: empty.withMasking,
  maskedFieldsTotal: empty.maskedFieldsTotal, tokensTotal: empty.tokensTotal, roleTotals: empty.roleTotals,
  scoreMin: empty.scoreMin, scoreMax: empty.scoreMax, rows: empty.rows.length },
  full: { count: full.count, withMasking: full.withMasking, maskedFieldsTotal: full.maskedFieldsTotal, tokensTotal: full.tokensTotal,
          roleTotals: full.roleTotals, unknownTotal: full.unknownTotal, truncated: full.truncated, scoreMin: full.scoreMin,
          scoreMax: full.scoreMax, rows: full.rows.length, latestSeq: full.overview.latest.seq } }));
""")
    assert out["empty"] == {
        "hasData": False, "count": 0, "withMasking": None, "maskedFieldsTotal": None, "tokensTotal": None,
        "roleTotals": None, "scoreMin": None, "scoreMax": None, "rows": 0,
    }
    full = out["full"]
    assert full["count"] == 3 and full["withMasking"] == 2 and full["maskedFieldsTotal"] == 4
    assert full["tokensTotal"] == 44 + 26 + 44
    assert full["roleTotals"] == {"control": 82, "literal_api": 22, "masked": 4, "literal_rule": 6}
    assert full["unknownTotal"] == 1 and full["truncated"] == 1
    assert full["scoreMin"] == 0.0016 and full["scoreMax"] == 0.9991
    assert full["rows"] == 3 and full["latestSeq"] == 3


def test_report_model_is_pure():
    js = _read("reports-model.js")
    for forbidden in ("document.", "window.", "fetch(", "localStorage", "Math.random", "import "):
        assert forbidden not in js, forbidden


# --- documented evidence: accuracy and separation --------------------------------------


def _evidence_json():
    return _node(
        ["evidence.js"],
        "console.log(JSON.stringify({ KPIS, LATENCY_BOUNDARIES, LATENCY_NOTES, METHODOLOGY, EVIDENCE_INTEGRITY }));",
    )


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_evidence_covers_kpi1_to_kpi5_with_required_fields():
    ev = _evidence_json()
    assert [k["id"] for k in ev["KPIS"]] == ["kpi1", "kpi2", "kpi3", "kpi4", "kpi5"]
    for k in ev["KPIS"]:
        for field in ("title", "target", "result", "sample", "verdict", "context", "limits", "refs"):
            assert k[field], (k["id"], field)
    kpi5 = ev["KPIS"][4]
    assert kpi5["kind"] == "pending" and kpi5["result"] == "Not measured"
    assert "point" not in kpi5 and "mean" not in kpi5 and "interval" not in kpi5  # no number to misread
    assert "Member 3" in kpi5["verdict"][0]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_evidence_does_not_overclaim_statistics_or_scope():
    ev = _evidence_json()
    by_id = {k["id"]: k for k in ev["KPIS"]}
    for kid in ("kpi1", "kpi2", "kpi3"):
        k = by_id[kid]
        assert "not statistically demonstrated" in k["verdict"][0]
        assert k["interval"][0] < k["targetValue"]  # the interval's lower bound is below the target
        assert any("synthetic" in line.lower() or "Synthetic" in line for line in k["limits"] + [k["sample"]])
    kpi4 = by_id["kpi4"]
    text = " ".join(kpi4["limits"] + kpi4["context"])
    for needle in ("Not production API latency", "not an SLA", "isolated model-inference", "single run", "Single machine"):
        assert needle in text or needle in kpi4["verdict"][0], needle
    assert "browser" in " ".join(kpi4["limits"]).lower()  # explicitly not comparable with browser timings
    joined = json.dumps(ev["METHODOLOGY"])
    for needle in ("synthetic", "uncalibrated", "KPI 5", "Member 3", "production latency or an SLA"):
        assert needle in joined, needle


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_evidence_numbers_match_the_documented_numbers_in_the_repository_readme():
    ev = _evidence_json()
    blob = json.dumps(ev["KPIS"]) + json.dumps(ev["LATENCY_BOUNDARIES"]) + json.dumps(ev["LATENCY_NOTES"])
    for needle, readme_form in [
        ("31 / 31", "31 / 31"), ("30 / 30", "30 / 30"), ("3719 / 3719", "3719 / 3719"), ("0.3413", "0.3413"),
        ("0.3481", "0.3481"), ("0.3200", "0.3200"), ("0.3371", "0.3371"), ("40.41%", "40.41%"), ("2.7007", "2.7007"),
        ("0.3605", "0.3605"), ("0.3580", "0.3580"), ("5.1237", "5.1237"), ("ea50a49", "ea50a49"),
    ]:
        assert needle in blob or needle in json.dumps(ev["EVIDENCE_INTEGRITY"]), needle
        assert readme_form in README, readme_form
    assert "KPI 5" in README and "not measured" in README.lower()


@pytest.mark.skipif(NODE is None or not EVIDENCE_DIR.exists(), reason="archived evidence folder not present")
def test_evidence_numbers_match_the_archived_machine_readable_reports():
    ev = _evidence_json()
    by_id = {k["id"]: k for k in ev["KPIS"]}
    kpi = json.loads((EVIDENCE_DIR / "kpi_report.json").read_text(encoding="utf-8"))
    r = kpi["results"]
    k1 = r["kpi1_input_alignment"]["valid_event_alignment"]
    k2 = r["kpi2_token_mapping"]
    k3 = r["kpi3_aligned_tensor_correctness"]
    assert (k1["numerator"], k1["denominator"], k1["target_percent"]) == (31, 31, 90.0)
    assert by_id["kpi1"]["interval"] == k1["wilson95_percent"] and by_id["kpi1"]["result"].startswith("31 / 31")
    assert by_id["kpi2"]["interval"] == k2["sequence_exact_match"]["wilson95_percent"]
    assert "3719 / 3719" in by_id["kpi2"]["sample"] and k2["token_level_match"]["numerator"] == 3719
    t = k3["tensor_correctness"]
    assert (t["numerator"], t["denominator"], t["target_percent"]) == (30, 30, 95.0)
    assert by_id["kpi3"]["interval"] == t["wilson95_percent"]
    assert k3["unknown_token_rate"]["observed_unknown"] == 99 and k3["unknown_token_rate"]["literal_positions"] == 245
    assert k3["sequence_length_mix"] == {"exact_fit": 0, "padded": 21, "truncated": 9}

    post = json.loads((EVIDENCE_DIR / "Post_D3_Latency" / "latency_report.json").read_text(encoding="utf-8"))
    orig = json.loads((EVIDENCE_DIR / "latency_report.json").read_text(encoding="utf-8"))
    assert round(post["primary"]["pooled_mean_ms"], 4) == by_id["kpi4"]["mean"]
    assert round(post["primary"]["worst_batch_mean_ms"], 4) == by_id["kpi4"]["worst"]
    assert post["primary"]["target_ms"] == by_id["kpi4"]["targetValue"] and post["primary"]["target_met"] is True
    assert f"{orig['primary']['pooled_mean_ms']:.4f}" in by_id["kpi4"]["baseline"]
    assert f"{orig['primary']['worst_batch_mean_ms']:.4f}" in by_id["kpi4"]["baseline"]
    rows = {(x["boundary"], x["stratum"]): x for x in _flatten_latency(post)}
    shown = {b[0]: b[2] for b in ev["LATENCY_BOUNDARIES"]}
    for boundary in ("B1", "B1T", "B2", "B3"):
        assert f"{rows[(boundary, 'S6_mix_all_valid')]['mean']:.4f}" == shown[boundary], boundary
    assert post["config"]["batches"] == 5 and post["config"]["reps_per_stratum_per_batch"] == 1000
    # Informational B3 figures quoted on the page.
    b3 = rows[("B3", "S6_mix_all_valid")]
    assert f"p95 {b3['p95']:.4f}" in " ".join(x[3] for x in ev["LATENCY_BOUNDARIES"])
    assert f"p99 {b3['p99']:.4f}" in " ".join(x[3] for x in ev["LATENCY_BOUNDARIES"])
    s4 = rows[("B3", "S4_large_seeded_80_fields")]
    notes = Path(DASH / "evidence.js").read_text(encoding="utf-8")
    assert f"{s4['p95']:.4f} ms" in notes and f"{s4['max']:.1f} ms outlier" in notes


def _flatten_latency(report):
    return [
        {"boundary": boundary, "stratum": stratum, "mean": stats["pooled"]["mean_ms"], "p95": stats["pooled"]["p95_ms"],
         "p99": stats["pooled"]["p99_ms"], "max": stats["pooled"]["max_ms"]}
        for boundary, strata in report["results"].items() for stratum, stats in strata.items()
    ]


# --- page wording / separation -----------------------------------------------------------


def test_reports_page_keeps_session_and_documented_evidence_separate():
    js = _read("page-reports.js")
    assert "Current Browser Session " + chr(0x2014) + " Member 4 Prototype" in js
    assert "Member 4 Research Evaluation Evidence" in js
    assert js.index("Current Browser Session") < js.index("Member 4 Research Evaluation Evidence")
    # Evidence data never flows into, or out of, the session store; the page only reads the store.
    assert "session.add(" not in js and "session.clear(" not in js
    lowered = js.lower()
    for needle in ("not the per-request timings", "not production traffic", "not the kpi 4 benchmark"):
        assert needle in lowered, needle
    assert "from \"./evidence.js\"" in js and "session.list()" in js


def test_reports_page_makes_no_forbidden_claims():
    text = _read("page-reports.js") + _read("evidence.js")
    for forbidden in (
        "All systems", "Operational", "Healthy", "Uptime", "req/s", "Critical", "Incidents", "Allowed Requests",
        "Blocked Requests", "Math.random", "setInterval", "detection rate", "attack detection accuracy",
    ):
        assert forbidden not in text, forbidden
    for needle in ("uncalibrated", "Not measured", "Pending", "synthetic"):
        assert needle in text.lower() or needle in text, needle


def test_reports_discipline_across_all_new_scripts():
    for name in ("page-reports.js", "reports-model.js", "export-format.js", "evidence.js"):
        js = _read(name).replace("http://www.w3.org/2000/svg", "")
        for forbidden in (
            "localStorage", "sessionStorage", "indexedDB", "document.cookie", "console.", "innerHTML", "outerHTML",
            "insertAdjacentHTML", "document.write", "eval(", "new Function", "XMLHttpRequest", "sendBeacon",
            "WebSocket", "http://", "https://", "RegExp", "Math.random", "[MASKED",
        ):
            assert forbidden not in js, (name, forbidden)


def test_session_store_allowlist_is_unchanged_by_phase4():
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
