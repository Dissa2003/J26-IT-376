"""Tests for the non-model audit of the synthetic dataset (draft protocol heldout-eval-1.2.0-draft).

The audit trains nothing and computes no accuracy, PR-AUC or similar. These tests use only data fabricated
by the generator, plus small deliberate corruptions to show that each check can fail.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.audit as audit
from tokenization_ai.heldout_eval.audit import (
    actual_body_format,
    build_pilot_report,
    check_content_type_semantics,
    check_determinism,
    check_feature_separation,
    check_split,
    distributions,
    normalized_mutual_information,
    pool_sharing,
    scan_label_terms,
    shortcut_statistics,
    split_feature_overlap,
    structural_coverage,
)
from tokenization_ai.heldout_eval import synthetic_catalog as cat
from tokenization_ai.heldout_eval.generator import GeneratorConfig, SyntheticDataset, generate
from tokenization_ai.heldout_eval.splits import make_record

CFG = GeneratorConfig(seed=11, n_templates=120)


@pytest.fixture(scope="module")
def ds() -> SyntheticDataset:
    return generate(CFG)


def with_record_replaced(dataset: SyntheticDataset, index: int, **overrides) -> SyntheticDataset:
    rec = dataset.records[index]
    args = dict(method=rec.features.method, path=rec.features.path, label=rec.meta.label,
                body=rec.features.body, headers=dict(rec.features.headers), timestamp=rec.meta.timestamp,
                source_id=rec.meta.source_id)
    args.update(overrides)
    new = make_record(overrides.pop("record_id", rec.record_id), **args)
    records = list(dataset.records)
    records[index] = new
    return dataclasses.replace(dataset, records=tuple(records))


# -- label / family term scan -------------------------------------------------------------------------------------


def test_generated_data_has_no_label_terms(ds):
    assert scan_label_terms(ds) == []


@pytest.mark.parametrize(
    "overrides,category",
    [
        (dict(path="/orders/attack"), "path"),
        (dict(body='{"note":"this is a malicious request"}'), "body"),
        (dict(headers={"user-agent": "ExampleClient/1.0", "x-label": "1"}), "header_name"),
        (dict(headers={"user-agent": "benign client"}), "header_value"),
        (dict(source_id="normal-source"), "source_id"),
    ],
)
def test_label_terms_are_detected_in_each_field_category(ds, overrides, category):
    findings = scan_label_terms(with_record_replaced(ds, 0, **overrides))
    assert findings and all(f.severity == "error" for f in findings)
    assert any(f"'{category}'" in f.message for f in findings), [f.message for f in findings]


def test_family_names_and_template_ids_are_detected(ds):
    family = ds.metadata[ds.records[0].record_id].family
    assert scan_label_terms(with_record_replaced(ds, 0, path=f"/x/{family}"))
    tid = ds.metadata[ds.records[0].record_id].template_id
    assert scan_label_terms(with_record_replaced(ds, 0, body=json.dumps({"k": tid})))


def test_term_findings_never_echo_request_text(ds):
    secret_path = "/orders/ATTACK-SECRET-PATH-TEXT"
    findings = scan_label_terms(with_record_replaced(ds, 0, path=secret_path))
    assert findings
    assert all("secret-path-text" not in f.message.lower() for f in findings)


# -- feature separation ---------------------------------------------------------------------------------------------


def test_feature_separation_is_clean_for_generated_data(ds):
    assert check_feature_separation(ds) == []


def test_missing_metadata_for_a_record_is_an_error(ds):
    broken = dict(ds.metadata)
    del broken[ds.records[0].record_id]
    findings = check_feature_separation(dataclasses.replace(ds, metadata=broken))
    assert any("no generation metadata" in f.message for f in findings)


def test_non_feature_objects_in_the_features_slot_are_detected(ds):
    rec = ds.records[0]
    bad = dataclasses.replace(rec, features="not features")  # type: ignore[arg-type]
    records = (bad,) + tuple(ds.records[1:])
    findings = check_feature_separation(dataclasses.replace(ds, records=records))
    assert any("not RequestFeatures" in f.message for f in findings)


# -- pool sharing --------------------------------------------------------------------------------------------------------


def test_pool_sharing_reports_shared_and_exclusive_values(ds):
    out = pool_sharing(ds)
    kinds = out["by_kind"]
    assert {"method", "user_agent", "role", "header_name", "resource"} <= set(kinds)
    assert kinds["user_agent"]["only_in_class_0"] == [] and kinds["user_agent"]["only_in_class_1"] == []
    assert kinds["method"]["shared"] >= 4


def test_an_exclusive_method_is_flagged(ds):
    idx = next(i for i, r in enumerate(ds.records) if r.meta.label == 1 and r.features.method == "GET")
    out = pool_sharing(with_record_replaced(ds, idx, method="TRACE"))
    assert "TRACE" in out["by_kind"]["method"]["only_in_class_1"]
    assert any("method" in w["message"] for w in out["warnings"])


# -- determinism ---------------------------------------------------------------------------------------------------------


def test_determinism_check_passes_for_a_valid_config():
    assert check_determinism(GeneratorConfig(seed=4, n_templates=30)) == []


# -- distributions and split ---------------------------------------------------------------------------------------------


def test_distributions_are_consistent(ds):
    d = distributions(ds)
    assert d["records"] == len(ds.records) == d["records_suspicious"] + d["records_normal"]
    assert sum(d["template_composition"].values()) == d["groups"] == ds.stats.actual_templates
    assert d["template_composition"]["mixed"] == ds.stats.templates_mixed
    assert sum(d["group_size"]["histogram"].values()) == d["groups"]
    assert d["template_level_containing_suspicious_share"] > d["template_level_all_suspicious_share"]
    assert d["record_level_suspicious_share"] != d["template_level_containing_suspicious_share"]
    assert sum(d["records_by_family"].values()) == d["records"]
    assert d["exact_duplicate_records"] == 0


def test_split_is_feasible_and_isolated_for_the_synthetic_data(ds):
    info, findings = check_split(ds)
    assert info["feasible"] and info["meets_requirements"]
    assert [f for f in findings if f.severity == "error"] == []
    assert sum(s["records"] for s in info["per_split"].values()) == len(ds.records)
    for s in info["per_split"].values():
        assert s["suspicious"] > 0 and s["normal"] > 0
    assert len(info["manifest_sha256"]) == 64


def test_split_check_reports_a_single_class_dataset_without_hiding_it():
    normal_only = generate(GeneratorConfig(seed=5, n_templates=40, family_weights={"crud_read": 1, "crud_write": 1}))
    info, findings = check_split(normal_only)
    assert info["feasible"] and not info["meets_requirements"]
    assert any(f.check == "split" and f.severity == "warning" for f in findings)
    assert not any(f.severity == "error" for f in findings)  # isolation still holds


def test_check_split_is_reproducible(ds):
    assert check_split(ds)[0] == check_split(ds)[0]


# -- shortcut statistics (no model) ---------------------------------------------------------------------------------------------


def test_normalized_mutual_information_edge_cases():
    labels = [0, 1] * 20
    assert normalized_mutual_information(labels, labels) == pytest.approx(1.0)
    assert normalized_mutual_information(["a"] * 40, labels) == 0.0
    assert normalized_mutual_information([1, 2, 3, 4] * 10, [0] * 40) == 0.0
    assert normalized_mutual_information([], []) == 0.0
    independent = normalized_mutual_information([0, 0, 1, 1] * 10, [0, 1, 0, 1] * 10)
    assert independent == pytest.approx(0.0, abs=1e-9)


def test_shortcut_statistics_cover_content_and_metadata_features(ds):
    stats = shortcut_statistics(ds)
    feats = stats["features"]
    assert {"method", "body_shape", "header_name_set", "role", "request_template"} <= set(feats)
    meta = {k for k, v in feats.items() if v["kind"] == "metadata_only"}
    assert meta == set(audit.METADATA_ONLY_FEATURES)
    for entry in feats.values():
        assert 0.0 <= entry["normalized_mi"] <= 1.0 and 0.0 <= entry["class_exclusive_record_share"] <= 1.0
    assert "pass/fail" in stats["note"]


def test_shortcut_statistics_are_deterministic(ds):
    assert shortcut_statistics(ds) == shortcut_statistics(ds)


def test_an_exclusive_method_triggers_a_shortcut_warning():
    base = generate(GeneratorConfig(seed=6, n_templates=60))
    records = tuple(
        make_record(r.record_id, method="PATCH" if r.meta.label == 1 else "GET", path=r.features.path,
                    label=r.meta.label, body=r.features.body, headers=dict(r.features.headers),
                    timestamp=r.meta.timestamp, source_id=r.meta.source_id) for r in base.records)
    rigged = dataclasses.replace(base, records=records)
    stats = shortcut_statistics(rigged)
    assert stats["features"]["method"]["normalized_mi"] == pytest.approx(1.0)
    assert any("'method'" in w["message"] for w in stats["warnings"])


def test_a_label_correlated_timestamp_triggers_the_metadata_warning():
    base = generate(GeneratorConfig(seed=6, n_templates=60))
    records = tuple(
        make_record(r.record_id, method=r.features.method, path=r.features.path, label=r.meta.label,
                    body=r.features.body, headers=dict(r.features.headers),
                    timestamp=(1.0 if r.meta.label == 0 else 1e9) + i, source_id=r.meta.source_id)
        for i, r in enumerate(base.records))
    stats = shortcut_statistics(dataclasses.replace(base, records=records))
    assert any("timestamp_decile" in w["message"] for w in stats["warnings"])


def test_warning_parameters_are_explicit_and_recorded(ds):
    stats = shortcut_statistics(ds, min_support=3, warn_exclusive_share=0.5)
    assert stats["min_support"] == 3 and stats["warn_exclusive_share"] == 0.5
    assert len(stats["warnings"]) <= len(shortcut_statistics(ds)["warnings"])


def test_split_feature_overlap_is_descriptive_bounded_and_reproducible(ds):
    one = split_feature_overlap(ds)
    assert one == split_feature_overlap(ds) and one["available"]
    for entry in one["features"].values():
        for split in ("validation", "test"):
            assert 0.0 <= entry[split]["share_value_seen_in_train"] <= 1.0
            assert 0.0 <= entry[split]["share_of_shared_with_single_class_train_value"] <= 1.0


# -- pilot report -----------------------------------------------------------------------------------------------------------------


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_pilot_report_is_json_safe_complete_and_reproducible():
    one, two = build_pilot_report(CFG), build_pilot_report(CFG)
    assert one == two
    text = json.dumps(one, allow_nan=False)
    assert one["protocol"] == audit.PROTOCOL_LABEL == "heldout-eval-1.2.0-draft"
    for section in ("config", "generation", "distributions", "split", "pool_sharing", "shortcut_statistics",
                    "split_feature_overlap", "errors", "warnings", "notice"):
        assert section in one
    assert one["construction_checks_passed"] is True and one["errors"] == []
    assert "does not show" in one["notice"] and "No model was trained" in one["notice"]
    assert len(text) > 1000


def test_pilot_report_holds_no_raw_request_content_and_no_performance_metrics(ds):
    report = build_pilot_report(CFG)
    text = json.dumps(report)
    for rec in ds.records[:50]:
        assert rec.features.path not in text and "Bearer" not in text
        if rec.features.body:
            assert rec.features.body not in text
    forbidden = ("accuracy", "auc", "precision", "recall", "f1", "roc", "pr_auc")
    for key in _keys(report):
        assert not any(w in key.lower() for w in forbidden), key


def test_pilot_report_surfaces_construction_errors_when_the_data_is_corrupt(monkeypatch, ds):
    corrupt = with_record_replaced(ds, 0, path="/orders/attack")
    monkeypatch.setattr(audit, "generate", lambda cfg: corrupt if cfg.seed == CFG.seed else ds)
    report = build_pilot_report(CFG)
    assert report["construction_checks_passed"] is False
    assert any(e["check"] == "label_terms" for e in report["errors"])


def test_pilot_report_includes_warnings_from_every_source(ds):
    report = build_pilot_report(CFG)
    checks = {w["check"] for w in report["warnings"]}
    assert "shortcut" in checks  # request-content association is expected by construction and surfaced


def test_audit_module_uses_only_the_standard_library_and_local_modules():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or "").split(".")[0])
    assert imported <= {"__future__", "dataclasses", "json", "math", "random", "urllib", "collections", "typing",
                        ".", ".generator", ".policy_sim", ".splits"}, imported


# -- content-type semantics and template-supported exclusivity (D3a-R) -----------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [("", "none"), ('{"a":1}', "json"), ("[1,2]", "json"), ("<note>some words</note>", "xml"),
     ("a=1&b=two", "form"), ("just some plain words", "plain"), ('"quoted json scalar"', "plain")],
)
def test_actual_body_format_classifies_by_content(body, expected):
    assert actual_body_format(body) == expected


def test_generated_data_passes_the_content_type_semantics_check(ds):
    assert check_content_type_semantics(ds) == []


def test_a_normal_request_with_an_incompatible_declared_type_is_an_error(ds):
    idx = next(i for i, r in enumerate(ds.records)
               if ds.metadata[r.record_id].family == "crud_write" and r.features.body)
    headers = dict(ds.records[idx].features.headers)
    headers["content-type"] = "text/plain"  # a JSON body declared as plain text
    findings = check_content_type_semantics(with_record_replaced(ds, idx, headers=headers))
    assert findings and findings[0].severity == "error" and findings[0].check == "content_type_semantics"


def test_a_suspicious_mismatch_record_whose_type_actually_matches_is_an_error(ds):
    idx = next(i for i, r in enumerate(ds.records)
               if ds.metadata[r.record_id].family == "content_type_mismatch" and r.meta.label == 1)
    rec = ds.records[idx]
    matching = cat.CONTENT_TYPES_BY_FORMAT[actual_body_format(rec.features.body)][0]
    headers = dict(rec.features.headers)
    headers["content-type"] = matching
    assert check_content_type_semantics(with_record_replaced(ds, idx, headers=headers))


@pytest.fixture(scope="module")
def pilot_ds() -> SyntheticDataset:
    return generate(GeneratorConfig())  # pilot scale: every content type has enough requests to occur in both classes


def test_the_declared_value_alone_never_decides_the_label(pilot_ds):
    by_value: dict[str, set[int]] = {}
    for r in pilot_ds.records:
        if r.features.body:
            by_value.setdefault(dict(r.features.headers)["content-type"], set()).add(r.meta.label)
    assert all(labels == {0, 1} for labels in by_value.values())


def test_template_supported_exclusivity_ignores_values_that_belong_to_a_single_template(ds):
    tid = next(m.template_id for m in ds.metadata.values() if m.label == 1 and m.label_kind == "suspicious")
    idxs = [i for i, r in enumerate(ds.records) if ds.metadata[r.record_id].template_id == tid]
    rigged = ds
    for i in idxs:
        headers = dict(rigged.records[i].features.headers)
        headers["user-agent"] = "SingleTemplateAgent/1.0"
        rigged = with_record_replaced(rigged, i, headers=headers)
    stats = shortcut_statistics(rigged, min_support=3)
    entry = stats["features"]["user_agent"]
    # legacy measure sees it (shares are rounded to 4 decimals in the report, hence the 5e-5 tolerance)
    assert entry["class_exclusive_record_share"] >= len(idxs) / len(ds.records) - 5e-5
    assert entry["class_exclusive_template_share"] == 0.0  # but it is one template, not a cross-template shortcut
    assert stats["min_template_support"] == 3


def test_a_value_shared_by_many_templates_and_one_class_does_warn():
    base = generate(GeneratorConfig(seed=6, n_templates=60))
    records = tuple(
        make_record(r.record_id, method=r.features.method, path=r.features.path, label=r.meta.label,
                    body=r.features.body,
                    headers={**dict(r.features.headers), "user-agent": "RiggedAgent/1.0" if r.meta.label else "ExampleClient/1.0"},
                    timestamp=r.meta.timestamp, source_id=r.meta.source_id) for r in base.records)
    stats = shortcut_statistics(dataclasses.replace(base, records=records))
    assert stats["features"]["user_agent"]["class_exclusive_template_share"] > 0.9
    assert any("'user_agent'" in w["message"] for w in stats["warnings"])


def test_pilot_report_runs_the_content_type_semantics_check_and_lists_no_error(ds):
    report = build_pilot_report(CFG)
    assert report["construction_checks_passed"] is True
    assert not any(e["check"] == "content_type_semantics" for e in report["errors"])


# -- structural coverage (D4-0; descriptive) and the protocol label --------------------------------------------------


def _rec(rid, method, path, label, headers=None, body=""):
    return make_record(rid, method=method, path=path, label=label, body=body,
                       headers=headers or {"user-agent": "T/1"}, timestamp=1.0, source_id="s1")


def _hand_dataset(records):
    """A dataset built from hand-made records (metadata stays empty: the coverage report must not need it)."""
    return SyntheticDataset(config=CFG, records=tuple(records), metadata={}, stats=None, fingerprint="x")


def _cell(rows, **cell):
    return next(r for r in rows if r["cell"] == cell)


def test_protocol_label_matches_the_protocol_document_header():
    import re

    text = (Path(audit.__file__).parent / "PROTOCOL.md").read_text(encoding="utf-8")
    declared = re.search(r"\*\*Protocol version:\*\* `([^`]+)`", text).group(1)
    assert audit.PROTOCOL_LABEL == declared
    from tokenization_ai.heldout_eval import splits

    assert splits.PROTOCOL_VERSION == "heldout-eval-1.0.0-draft"  # split-manifest provenance is deliberately untouched


def test_path_category_and_query_follow_the_simpol_observation_logic():
    from tokenization_ai.heldout_eval.policy_sim import observe

    recs = [_rec("a", "GET", "/admin/orders/x", 0), _rec("b", "GET", "/files/orders?page=2", 1),
            _rec("c", "DELETE", "/orders/55", 0), _rec("d", "POST", "/auth/login", 0), _rec("e", "GET", "/health", 0)]
    cov = structural_coverage(_hand_dataset(recs))
    cats = {r["cell"]["path_category"] for r in cov["path_category"]}
    assert cats == {observe(r.features).path_category for r in recs} == {
        "admin_area", "files_area", "resource_area", "auth_area", "status_area"}
    assert _cell(cov["method_by_path_category_by_query"], method="GET", path_category="files_area", query="present")
    assert _cell(cov["method_by_path_category_by_query"], method="DELETE", path_category="resource_area", query="absent")


def test_counting_semantics_one_mixed_template_is_one_template():
    # Three variants of one template (same method and shape): two normal, one suspicious -> a single mixed template.
    mixed = [_rec(f"m{i}", "GET", f"/orders/{100 + i}", label) for i, label in enumerate((0, 0, 1))]
    normal_only = [_rec("n1", "GET", "/invoices/7", 0), _rec("n2", "GET", "/invoices/8", 0)]
    susp_only = [_rec("s1", "GET", "/tickets/3", 1)]
    cov = structural_coverage(_hand_dataset(mixed + normal_only + susp_only))
    row = _cell(cov["method_by_path_category"], method="GET", path_category="resource_area")
    assert row["records"] == {"normal": 4, "suspicious": 2}
    t = row["templates"]
    assert t["distinct"] == 3 and (t["normal_only"], t["suspicious_only"], t["mixed"]) == (1, 1, 1)
    assert t["with_normal"] == 2 and t["with_suspicious"] == 2  # overlap on the mixed template; not additive
    assert cov["distinct_templates"] == 3 and cov["template_composition"] == {"normal_only": 1, "suspicious_only": 1, "mixed": 1}


def test_class_exclusive_and_sparse_cells():
    recs = [_rec(f"x{i}", "DELETE", f"/{res}/{i + 1}", 0) for i, res in enumerate(("orders", "invoices", "tickets"))]
    recs.append(_rec("y1", "POST", "/orders", 1, body="{}",
                     headers={"user-agent": "T/1", "content-type": "application/json"}))
    cov = structural_coverage(_hand_dataset(recs))
    ex = cov["class_exclusive_cells"]["method_by_path_category"]
    assert [(e["cell"]["method"], e["exclusive_to"]) for e in ex] == [("DELETE", "normal")]
    sp = cov["sparse_cells"]["method_by_path_category"]
    assert [(s["cell"]["method"], s["templates"]["distinct"]) for s in sp] == [("POST", 1)]  # exclusive needs >=3 templates


def test_coverage_is_deterministic_ordered_and_json_safe(ds):
    one, two = structural_coverage(ds), structural_coverage(ds)
    assert one == two and json.dumps(one, allow_nan=False) == json.dumps(two, allow_nan=False)
    for level in ("path_category", "method_by_path_category", "method_by_path_category_by_query"):
        keys = [tuple(r["cell"].values()) for r in one[level]]
        assert keys == sorted(keys) and len(keys) == len(set(keys))
        assert sum(r["records"]["normal"] + r["records"]["suspicious"] for r in one[level]) == len(ds.records)
        for r in one[level]:
            t = r["templates"]
            assert t["distinct"] == t["normal_only"] + t["suspicious_only"] + t["mixed"]
    assert one["records"] == len(ds.records) and one["distinct_templates"] == ds.stats.actual_templates
    comp = one["template_composition"]
    assert (comp["normal_only"], comp["suspicious_only"], comp["mixed"]) == (
        ds.stats.templates_all_normal, ds.stats.templates_all_suspicious, ds.stats.templates_mixed)


def test_coverage_holds_no_raw_content_metadata_or_performance_metrics(ds):
    cov = structural_coverage(ds)
    text = json.dumps(cov)
    for rec in ds.records[:50]:
        assert rec.features.path not in text and "Bearer" not in text
    for fam in cat.family_names():
        assert fam not in text
    banned = ("accuracy", "auc", "precision", "recall", "f1", "roc", "prediction", "score")
    keys = {k.lower() for k in _keys(cov)}
    assert not any(b in k for k in keys for b in banned)


def test_coverage_is_an_additive_field_that_leaves_existing_report_content_unchanged():
    report = build_pilot_report(CFG)
    assert report["structural_coverage"] == structural_coverage(generate(CFG))
    without = [k for k in report if k != "structural_coverage"]
    assert without == ["protocol", "config", "generation", "distributions", "split", "pool_sharing",
                       "shortcut_statistics", "split_feature_overlap", "errors", "warnings",
                       "construction_checks_passed", "notice"]
    # Exclusive cells are never promoted to warnings or errors.
    assert not any("structural" in w["check"] for w in report["warnings"] + report["errors"])
    # The established thresholds are unchanged.
    assert (audit.DEFAULT_MIN_SUPPORT, audit.DEFAULT_MIN_TEMPLATE_SUPPORT, audit.DEFAULT_WARN_EXCLUSIVE_SHARE) == (5, 3, 0.05)


def test_coverage_on_the_development_pilot_reports_the_documented_structure():
    dev = generate(GeneratorConfig(seed=6101, n_templates=250))  # development seed only
    cov = structural_coverage(dev)
    excl = {tuple(e["cell"].values()) for e in cov["class_exclusive_cells"]["method_by_path_category"]}
    assert excl == {("GET", "status_area")}  # SIM-P06's documented structural association
    files = _cell(cov["method_by_path_category"], method="GET", path_category="files_area")
    assert files["templates"]["with_normal"] >= 3 and files["templates"]["with_suspicious"] >= 3  # SIM-P07: both classes
    delete = _cell(cov["method_by_path_category"], method="DELETE", path_category="resource_area")
    assert delete["templates"]["with_normal"] >= 1 and delete["templates"]["with_suspicious"] >= 1  # SIM-P04: both classes
    assert not any(r["cell"]["path_category"] == "status_area" and r["records"]["suspicious"] for r in cov["path_category"])


def test_coverage_code_does_not_import_a_model_or_the_prediction_pipeline():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not names & {"model", "service", "tensorizer", "tokenizer", "vocabulary"}
