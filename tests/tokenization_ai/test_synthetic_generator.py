"""Tests for the deterministic synthetic request generator (draft protocol heldout-eval-1.2.0-draft).

All data is fabricated by the generator under test. No model is trained and no detection performance
is measured.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import socket
import urllib.parse
from collections import Counter
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.generator as gen
from tokenization_ai.heldout_eval import synthetic_catalog as cat
from tokenization_ai.heldout_eval.generator import (
    GENERATOR_VERSION,
    GeneratorConfig,
    GeneratorConfigError,
    GeneratorError,
    generate,
)
from tokenization_ai.heldout_eval.splits import EvalRecord, RequestFeatures, exact_fingerprint, group_id_for

PILOT = GeneratorConfig()  # about 250 templates, 4-8 variants


@pytest.fixture(scope="module")
def pilot():
    return generate(PILOT)


@pytest.fixture(scope="module")
def small():
    return generate(GeneratorConfig(seed=7, n_templates=80))


# -- determinism and seeds -------------------------------------------------------------------------------


def test_same_config_gives_byte_identical_records_and_fingerprint(small):
    again = generate(GeneratorConfig(seed=7, n_templates=80))
    assert again.fingerprint == small.fingerprint
    assert again.records == small.records
    assert dict(again.metadata) == dict(small.metadata)


def test_different_seeds_change_the_data(small):
    other = generate(GeneratorConfig(seed=8, n_templates=80))
    assert other.fingerprint != small.fingerprint
    assert {r.record_id for r in other.records} != {r.record_id for r in small.records}


def test_changing_any_parameter_changes_the_fingerprint(small):
    for change in (dict(n_templates=81), dict(variants_max=9), dict(hard_fraction=0.31),
                   dict(mixed_suspicious_probability=0.41)):
        cfg = dataclasses.replace(GeneratorConfig(seed=7, n_templates=80), **change)
        assert generate(cfg).fingerprint != small.fingerprint, change


def test_generator_does_not_use_network_or_global_randomness(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    import random

    random.seed(1)
    one = generate(GeneratorConfig(seed=3, n_templates=30)).fingerprint
    random.seed(999)
    assert generate(GeneratorConfig(seed=3, n_templates=30)).fingerprint == one


def test_generator_modules_import_only_the_standard_library_and_local_modules():
    for module in (gen,):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(("." * node.level) + (node.module or "").split(".")[0])
        allowed = {"__future__", "hashlib", "json", "math", "random", "urllib", "collections", "dataclasses",
                   "typing", ".", ".splits"}
        assert imported <= allowed, imported


# -- configuration validation ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(seed=True), dict(seed="1"), dict(seed=1.5),
        dict(n_templates=9), dict(n_templates=20001), dict(n_templates=True), dict(n_templates="50"),
        dict(variants_min=1), dict(variants_min=5, variants_max=4), dict(variants_max=51),
        dict(mixed_suspicious_probability=0), dict(mixed_suspicious_probability=1),
        dict(mixed_suspicious_probability=float("nan")), dict(hard_fraction=-0.1), dict(hard_fraction=True),
        dict(max_attempt_factor=1),
        dict(family_weights={}), dict(family_weights={"nonexistent": 1}), dict(family_weights={"crud_read": -1}),
        dict(family_weights={"crud_read": 0}), dict(family_weights={"crud_read": True}),
        dict(family_weights="crud_read"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(GeneratorConfigError):
        GeneratorConfig(**kwargs)


def test_default_config_documents_itself():
    d = GeneratorConfig().to_dict()
    assert d["generator_version"] == GENERATOR_VERSION and d["catalog_version"] == cat.CATALOG_VERSION
    assert d["seed"] == gen.DEFAULT_DATASET_SEED and d["family_weights"] == cat.default_weights()
    json.dumps(d)


# -- valid records and metadata separation ---------------------------------------------------------------------------


def test_records_are_valid_unique_and_opaque(pilot):
    ids = [r.record_id for r in pilot.records]
    assert len(ids) == len(set(ids)) == len(pilot.records)
    for r in pilot.records:
        assert isinstance(r, EvalRecord) and isinstance(r.features, RequestFeatures)
        assert re.fullmatch(r"r[0-9a-f]{12}", r.record_id)
        assert re.fullmatch(r"s[0-9a-f]{8}", r.meta.source_id)
        assert r.meta.label in (0, 1) and r.meta.timestamp is not None
        assert r.features.method in cat.HTTP_METHODS
        assert r.features.path.startswith("/")
        assert bool(r.features.body) == (r.features.method in ("POST", "PUT", "PATCH"))


def test_metadata_is_separate_complete_and_consistent(pilot):
    assert set(pilot.metadata) == {r.record_id for r in pilot.records}
    for r in pilot.records:
        m = pilot.metadata[r.record_id]
        assert m.label == r.meta.label
        assert m.family in cat.FAMILY_BY_NAME and m.label_kind == cat.FAMILY_BY_NAME[m.family].label_kind
        assert m.hard_case in ("", "negative", "positive")
    names = {f.name for f in dataclasses.fields(RequestFeatures)}
    assert names == {"method", "path", "body", "headers"}


def test_family_and_label_names_never_appear_in_request_fields_or_ids(pilot):
    forbidden = set(cat.BANNED_LABEL_WORDS) | set(cat.family_names()) | {m.template_id for m in pilot.metadata.values()}
    for r in pilot.records:
        text = " ".join([r.record_id, r.meta.source_id, r.features.path.lower(), r.features.body.lower(),
                         *(f"{k} {v}".lower() for k, v in r.features.headers)])
        for word in forbidden:
            assert word.lower() not in text, word


def test_record_ids_source_ids_and_timestamps_are_independent_of_the_label(pilot):
    recs = pilot.records
    labels = [r.meta.label for r in recs]
    n = len(recs)
    changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    assert changes > n // 4  # shuffled, not grouped by class
    half = n // 2
    first, second = sum(labels[:half]) / half, sum(labels[half:]) / (n - half)
    assert abs(first - second) < 0.10
    ts = [r.meta.timestamp for r in recs]
    assert ts == sorted(ts) and len(set(ts)) == n
    by_source: dict[str, set[int]] = {}
    for r in recs:
        by_source.setdefault(r.meta.source_id, set()).add(r.meta.label)
    assert all(v == {0, 1} for v in by_source.values())
    ids_by_class = {k: Counter(r.record_id[1] for r in recs if r.meta.label == k) for k in (0, 1)}
    assert set(ids_by_class[0]) & set(ids_by_class[1])


def test_no_real_data_or_credentials(pilot):
    for r in pilot.records:
        h = dict(r.features.headers)
        if "authorization" in h:
            assert h["authorization"].startswith("Bearer syn-")
        assert h["user-agent"] in cat.USER_AGENTS
    text = json.dumps([[r.features.path, r.features.body] for r in pilot.records])
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)
    assert "https://" not in text


# -- templates, grouping and mixed-label scenarios -------------------------------------------------------------------------


def test_every_variant_normalizes_to_its_templates_group(pilot):
    for r in pilot.records:
        assert group_id_for(r.features) == pilot.metadata[r.record_id].template_id


def test_actual_template_count_and_variant_ranges(pilot):
    s = pilot.stats
    assert s.requested_templates == 250 and s.actual_templates + s.shortfall == 250
    sizes = Counter(m.template_id for m in pilot.metadata.values())
    assert len(sizes) == s.actual_templates
    assert all(PILOT.variants_min <= c <= PILOT.variants_max for c in sizes.values())
    assert s.n_records == len(pilot.records) == sum(sizes.values())


def test_template_composition_counts_are_consistent_with_the_records(pilot):
    by_template: dict[str, set[int]] = {}
    for r in pilot.records:
        by_template.setdefault(pilot.metadata[r.record_id].template_id, set()).add(r.meta.label)
    s = pilot.stats
    assert s.templates_all_normal == sum(1 for v in by_template.values() if v == {0})
    assert s.templates_all_suspicious == sum(1 for v in by_template.values() if v == {1})
    assert s.templates_mixed == sum(1 for v in by_template.values() if v == {0, 1})
    assert s.templates_all_normal + s.templates_all_suspicious + s.templates_mixed == s.actual_templates


def test_mixed_label_templates_come_only_from_mixed_families_and_each_has_both_labels(pilot):
    fam_of = {m.template_id: m.family for m in pilot.metadata.values()}
    labels: dict[str, set[int]] = {}
    for r in pilot.records:
        labels.setdefault(pilot.metadata[r.record_id].template_id, set()).add(r.meta.label)
    assert pilot.stats.templates_mixed > 0
    for tid, lab in labels.items():
        kind = cat.FAMILY_BY_NAME[fam_of[tid]].label_kind
        if kind == "mixed":
            assert lab == {0, 1}, fam_of[tid]
        elif kind == "normal":
            assert lab == {0}
        else:
            assert lab == {1}


def _actual_format(body: str) -> str:
    """Test-local, independent classification of a body by its content (not by any header)."""
    if not body:
        return "none"
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        return "json"
    if body.lstrip().startswith("<"):
        return "xml"
    if "=" in body and " " not in body.strip():
        return "form"
    return "plain"


def test_labels_follow_documented_construction(pilot):
    for r in pilot.records:
        m = pilot.metadata[r.record_id]
        headers = dict(r.features.headers)
        if m.family == "role_action_mismatch":
            assert (r.meta.label == 1) == (headers["x-role"] != "admin")
        elif m.family == "content_type_mismatch":
            matches = headers["content-type"] in cat.CONTENT_TYPES_BY_FORMAT[_actual_format(r.features.body)]
            assert (r.meta.label == 1) == (not matches)
        elif m.family == "admin_legit":
            assert headers["x-role"] == "admin" and r.meta.label == 0
        elif m.family == "mass_assignment":
            assert headers["x-role"] != "admin" and r.meta.label == 1
        elif m.label_kind == "normal":
            assert r.meta.label == 0
        elif m.label_kind == "suspicious":
            assert r.meta.label == 1


def test_content_type_mismatch_is_incompatibility_with_the_actual_body_not_the_declared_value(pilot):
    by_value: dict[str, set[int]] = {}
    for r in pilot.records:
        body_format = _actual_format(r.features.body)
        if body_format == "none":
            continue
        declared = dict(r.features.headers)["content-type"]
        matches = declared in cat.CONTENT_TYPES_BY_FORMAT[body_format]
        if pilot.metadata[r.record_id].family != "content_type_mismatch":
            assert matches, "a request outside the mismatch family declared an incompatible type"
        by_value.setdefault(declared, set()).add(r.meta.label)
    # Every declared type is legitimate for its own format, so no value identifies a class by itself.
    assert {"text/plain", "application/xml"} <= set(by_value)
    for value, labels in by_value.items():
        assert labels == {0, 1}, value
    families = {pilot.metadata[r.record_id].family for r in pilot.records if _actual_format(r.features.body) in ("xml", "plain")}
    assert "alt_content_type" in families  # legitimate plain-text and XML requests exist


def test_content_types_by_format_are_disjoint_and_cover_the_four_formats():
    flat = [t for types in cat.CONTENT_TYPES_BY_FORMAT.values() for t in types]
    assert set(cat.CONTENT_TYPES_BY_FORMAT) == {"json", "xml", "form", "plain"}
    assert len(flat) == len(set(flat))


def test_privileged_looking_field_names_occur_in_both_classes_and_role_context_decides(pilot):
    names = set(cat.PRIVILEGED_FIELDS_OBVIOUS + cat.PRIVILEGED_FIELDS_QUIET)
    seen = {0: set(), 1: set()}
    for r in pilot.records:
        if _actual_format(r.features.body) == "json":
            seen[r.meta.label] |= names & set(json.loads(r.features.body))
    assert seen[0] and seen[1] and (seen[0] & seen[1])
    for r in pilot.records:
        m = pilot.metadata[r.record_id]
        if m.family == "admin_privileged_update":
            assert dict(r.features.headers)["x-role"] == "admin" and r.meta.label == 0


def _json_depth(value, level=0):
    if isinstance(value, dict):
        return max([level + 1] + [_json_depth(v, level + 1) for v in value.values()])
    if isinstance(value, list):
        return max([level + 1] + [_json_depth(v, level + 1) for v in value])
    return level


def test_nested_bodies_exist_in_both_classes_and_only_excess_depth_is_the_documented_anomaly(pilot):
    normal_depths, deep_anomaly = [], []
    for r in pilot.records:
        if _actual_format(r.features.body) != "json":
            continue
        depth = _json_depth(json.loads(r.features.body))
        m = pilot.metadata[r.record_id]
        if r.meta.label == 0:
            normal_depths.append(depth)
        elif m.family == "structure_anomaly" and "payload_tree" in r.features.body:
            deep_anomaly.append(depth)
    assert 5 <= max(normal_depths) <= 6  # legitimately nested configs exist, and are bounded
    assert deep_anomaly and min(deep_anomaly) >= 7  # the intentional, documented structural anomaly


def test_comparable_structures_are_present_in_both_classes(pilot):
    stats = {k: {"depth": set(), "fields": set(), "query": set(), "body_bits": set()} for k in (0, 1)}
    for r in pilot.records:
        d = stats[r.meta.label]
        path, _, query = r.features.path.partition("?")
        d["depth"].add(len([s for s in path.split("/") if s]))
        d["query"].add(len(query.split("&")) if query else 0)
        d["body_bits"].add(len(r.features.body).bit_length())
        if _actual_format(r.features.body) == "json":
            d["fields"].add(len(json.loads(r.features.body)))
    assert max(stats[1]["body_bits"]) >= 8 and max(stats[0]["body_bits"]) >= 8  # large bodies in both classes
    assert {5, 6} <= stats[0]["depth"] & stats[1]["depth"]  # deep paths in both classes
    assert max(stats[1]["fields"]) >= 8 and max(stats[0]["fields"]) >= 8  # many-field bodies in both classes
    assert 3 in stats[0]["query"] and 3 in stats[1]["query"]  # several query parameters in both classes
    assert {5, 6, 7} & stats[0]["fields"] and {5, 6, 7} & stats[1]["fields"]


def test_every_sub_resource_and_resource_is_used_by_both_classes(pilot):
    used = {"sub": {0: set(), 1: set()}, "res": {0: set(), 1: set()}}
    for r in pilot.records:
        for seg in r.features.path.partition("?")[0].lower().split("/"):
            if seg in cat.SUB_RESOURCES:
                used["sub"][r.meta.label].add(seg)
            if seg in cat.RESOURCES:
                used["res"][r.meta.label].add(seg)
    assert used["sub"][0] == used["sub"][1] == set(cat.SUB_RESOURCES)
    assert used["res"][0] == used["res"][1] == set(cat.RESOURCES)


def test_unauthenticated_and_authenticated_endpoints_both_carry_normal_and_suspicious_requests(pilot):
    for has_auth in (False, True):
        labels = {r.meta.label for r in pilot.records if ("authorization" in dict(r.features.headers)) == has_auth}
        assert labels == {0, 1}


@pytest.mark.parametrize("bits", [5, 6, 7])
def test_sized_text_stays_inside_one_length_bucket_so_templates_cannot_drift(bits):
    import random

    for seed in range(60):
        for prefix, suffix in (("", ""), ("<note>", "</note>")):
            text = gen._sized_text(random.Random(seed), bits, prefix, suffix)
            assert len(text).bit_length() == bits
            assert "=" not in text


def test_round_robin_pickers_cover_every_item_before_repeating():
    import random

    cycler = gen._Cycler(cat.RESOURCES, random.Random(1))
    first_round = [cycler.next() for _ in range(len(cat.RESOURCES))]
    assert sorted(first_round) == sorted(cat.RESOURCES)
    second_round = [cycler.next() for _ in range(len(cat.RESOURCES))]
    assert sorted(second_round) == sorted(cat.RESOURCES) and first_round != second_round


def test_scenario_coverage_in_the_pilot(pilot):
    fams = {m.family for m in pilot.metadata.values()}
    assert fams >= {"crud_read", "crud_write", "search_filter", "auth_login", "json_create_nested", "admin_legit",
                    "sqli_like", "traversal_like", "cmdi_like", "markup_like", "mass_assignment",
                    "role_action_mismatch", "content_type_mismatch", "structure_anomaly",
                    "hn_keyword_text", "hn_dotted_paths", "hn_long_payload", "shared_endpoint_text",
                    "alt_content_type", "admin_privileged_update", "hn_nested_config"}
    assert pilot.stats.hard_negative_records > 0 and pilot.stats.hard_positive_records > 0
    for r in pilot.records:
        m = pilot.metadata[r.record_id]
        if m.hard_case == "negative":
            assert r.meta.label == 0
        if m.hard_case == "positive":
            assert r.meta.label == 1


def test_pilot_scale_and_prevalence_are_reported_not_forced(pilot):
    s = pilot.stats
    assert 1000 <= s.n_records <= 2000
    suspicious_templates = s.templates_all_suspicious + s.templates_mixed
    assert 0.25 <= suspicious_templates / s.actual_templates <= 0.45  # descriptive range, not a requirement
    records_share = sum(r.meta.label for r in pilot.records) / s.n_records
    assert records_share != suspicious_templates / s.actual_templates  # template and record prevalence differ


def test_shortfall_is_reported_when_normalization_collapses_templates():
    cfg = GeneratorConfig(seed=1, n_templates=500, family_weights={"health_status": 1}, max_attempt_factor=40)
    ds = generate(cfg)
    s = ds.stats
    assert 0 < s.actual_templates < 500 and s.shortfall == 500 - s.actual_templates > 0
    assert s.collapsed_specs > 0 and s.attempts == 500 * 40
    assert {m.family for m in ds.metadata.values()} == {"health_status"}


def test_family_weights_select_families_and_zero_weight_excludes():
    ds = generate(GeneratorConfig(seed=2, n_templates=40, family_weights={"crud_read": 1, "sqli_like": 1}))
    assert {m.family for m in ds.metadata.values()} <= {"crud_read", "sqli_like"}


# -- shared pools -------------------------------------------------------------------------------------------------------------


def test_methods_headers_user_agents_and_roles_come_from_shared_pools(pilot):
    per_class = {0: {"methods": set(), "ua": set(), "roles": set(), "hdr": set()},
                 1: {"methods": set(), "ua": set(), "roles": set(), "hdr": set()}}
    for r in pilot.records:
        d, h = per_class[r.meta.label], dict(r.features.headers)
        d["methods"].add(r.features.method)
        d["ua"].add(h["user-agent"])
        d["hdr"] |= set(h)
        if "x-role" in h:
            d["roles"].add(h["x-role"])
    for d in per_class.values():
        assert d["methods"] <= set(cat.HTTP_METHODS)
        assert d["ua"] <= set(cat.USER_AGENTS)
        assert d["roles"] <= set(cat.ROLES_LOW) | {cat.ROLE_ADMIN}
    assert per_class[0]["ua"] == per_class[1]["ua"] == set(cat.USER_AGENTS)
    assert {"GET", "POST"} <= per_class[0]["methods"] & per_class[1]["methods"]
    assert per_class[0]["roles"] == per_class[1]["roles"]
    assert per_class[0]["hdr"] - per_class[1]["hdr"] == set() and per_class[1]["hdr"] - per_class[0]["hdr"] == set()


def test_resources_are_shared_across_classes(pilot):
    seen = {0: set(), 1: set()}
    for r in pilot.records:
        for seg in r.features.path.partition("?")[0].lower().split("/"):
            if seg in cat.RESOURCES:
                seen[r.meta.label].add(seg)
    assert len(seen[0] & seen[1]) >= len(cat.RESOURCES) - 3


def test_unauthenticated_endpoints_carry_both_classes(pilot):
    labels = {r.meta.label for r in pilot.records if "authorization" not in dict(r.features.headers)}
    assert labels == {0, 1}


# -- duplicates and splitting compatibility ---------------------------------------------------------------------------------------


def test_no_exact_duplicate_requests_are_generated(pilot):
    prints = Counter(exact_fingerprint(r.features) for r in pilot.records)
    assert max(prints.values()) == 1
    assert pilot.stats.duplicate_variants_skipped >= 0


def test_pilot_statistics_are_reproducible_and_pinned(pilot):
    again = generate(PILOT)
    assert again.fingerprint == pilot.fingerprint
    s = pilot.stats
    # Regression pins for the default pilot config (a deliberate guard: any generator change must update them).
    assert (s.actual_templates, s.n_records) == (250, 1466)
    assert (s.templates_all_normal, s.templates_all_suspicious, s.templates_mixed) == (168, 49, 33)
    assert pilot.fingerprint == PINNED_PILOT_FINGERPRINT


# Pinned to the Python version used when written (random-module algorithms can differ between versions).
# synthgen-3 / synthetic-catalog-3, development seed 6101 (the default). Earlier, burned pilots (history only):
# synthgen-2 seed 5101 d24fbec1688beb5e..., 1501 records; synthgen-1 dcd042946fc3571b..., 1487 records.
PINNED_PILOT_FINGERPRINT = "908627770c1c07cabc3832a7e3720e80c0c60b6fa97607cd18b71b035388ec00"


def test_generator_error_type_exists_for_template_drift():
    assert issubclass(GeneratorError, RuntimeError)


# -- synthgen-3: versions, seed plan, G1 (normal files paths) and G2 (comparable DELETE) ----------------------------
# Only the development seed 6101 is generated here. The confirmatory seed 7101 and the reserve seeds 7102 / 7103
# are never generated by any test.

DEV_SEED = 6101
NOT_FOR_TESTS = {5101, 7101, 7102, 7103}  # burned pilot, confirmatory, reserves


@pytest.fixture(scope="module")
def dev_large():
    """Development-seed dataset at the planned full-study template scale (descriptive structure checks only)."""
    return generate(GeneratorConfig(seed=DEV_SEED, n_templates=2000))


def _first_segment(path: str) -> str:
    return next((s for s in path.partition("?")[0].split("/") if s), "")


def _category(record: EvalRecord) -> str:
    from tokenization_ai.heldout_eval.policy_sim import observe

    return observe(record.features).path_category


def _templates_by_class(ds, selector) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {0: set(), 1: set()}
    for r in ds.records:
        if selector(r):
            out[r.meta.label].add(ds.metadata[r.record_id].template_id)
    return out


def _is_resource_delete(r: EvalRecord) -> bool:
    return r.features.method == "DELETE" and _category(r) == "resource_area"


def test_generator_and_catalog_versions_and_default_seed():
    assert GENERATOR_VERSION == "synthgen-3" and cat.CATALOG_VERSION == "synthetic-catalog-3"
    assert gen.DEFAULT_DATASET_SEED == DEV_SEED and GeneratorConfig().seed == DEV_SEED
    assert gen.DEFAULT_DATASET_SEED not in NOT_FOR_TESTS
    d = GeneratorConfig().to_dict()
    assert d["generator_version"] == "synthgen-3" and d["catalog_version"] == "synthetic-catalog-3"


def test_family_weights_are_unchanged_by_the_corrections():
    assert sum(f.weight for f in cat.FAMILIES) == 106
    assert {f.name for f in cat.FAMILIES} == {
        "crud_read", "crud_write", "search_filter", "auth_login", "json_create_nested", "admin_legit", "health_status",
        "hn_keyword_text", "hn_dotted_paths", "hn_long_payload", "alt_content_type", "admin_privileged_update",
        "hn_nested_config", "sqli_like", "traversal_like", "cmdi_like", "markup_like", "mass_assignment",
        "structure_anomaly", "role_action_mismatch", "content_type_mismatch", "shared_endpoint_text"}


def test_g1_files_paths_occur_in_both_classes(pilot, dev_large):
    for ds, minimum in ((pilot, 3), (dev_large, 30)):
        by_class = _templates_by_class(ds, lambda r: _first_segment(r.features.path) == "files")
        assert len(by_class[0]) >= minimum and len(by_class[1]) >= minimum, {k: len(v) for k, v in by_class.items()}


def test_g1_normal_files_requests_are_ordinary_reads_and_suspicious_ones_keep_their_construction(dev_large):
    families = {0: set(), 1: set()}
    for r in dev_large.records:
        if _first_segment(r.features.path) != "files":
            continue
        m = dev_large.metadata[r.record_id]
        families[r.meta.label].add(m.family)
        path = r.features.path.partition("?")[0].lower()
        traversal = ".." in path.split("/") or "%2f" in path
        if r.meta.label == 0:
            assert m.family in ("crud_read", "hn_dotted_paths") and m.label_kind == "normal"
            assert r.features.method == "GET" and not r.features.body and not traversal
            assert "authorization" in dict(r.features.headers)
        else:
            assert m.family == "traversal_like" and traversal  # unchanged: suspicious only by the construction
    assert families[0] == {"crud_read", "hn_dotted_paths"} and families[1] == {"traversal_like"}


def test_g1_files_paths_no_longer_decide_the_class_for_policy_applicability(dev_large):
    from tokenization_ai.heldout_eval.policy_sim import simulate_rules_for_record

    classes = {r.meta.label for r in dev_large.records if "SIM-P07" in simulate_rules_for_record(r).policy_ids}
    assert classes == {0, 1}


def test_g2_delete_cells_hold_both_classes_in_every_structure(dev_large):
    for has_query in (False, True):
        by_class = _templates_by_class(
            dev_large, lambda r, q=has_query: _is_resource_delete(r) and ("?" in r.features.path) == q)
        assert len(by_class[0]) >= 3 and len(by_class[1]) >= 3, (has_query, {k: len(v) for k, v in by_class.items()})


def test_g2_delete_requests_share_method_body_content_type_and_authorization_structure(dev_large):
    seen = {0: [], 1: []}
    for r in dev_large.records:
        if _is_resource_delete(r):
            h = dict(r.features.headers)
            assert r.features.body == "" and "content-type" not in h  # no body shape or declared type to mark a class
            seen[r.meta.label].append("authorization" in h)
    assert seen[0] and seen[1]
    assert set(seen[0]) == set(seen[1]) == {True}  # authorization-header presence does not mark a class


def test_g2_labels_follow_construction_not_the_method(dev_large):
    for r in dev_large.records:
        if not _is_resource_delete(r):
            continue
        m = dev_large.metadata[r.record_id]
        path, _, query = r.features.path.partition("?")
        values = [v for _, v in urllib.parse.parse_qsl(query, keep_blank_values=True)]
        special = any(re.search(r"[^a-z0-9 ]", v) for v in values)  # payload helpers always contain a symbol
        traversal = ".." in path.split("/") or "%2f" in path.lower()
        if r.meta.label == 0:
            assert m.family == "crud_write" and not traversal and not special
        else:
            assert m.family in ("traversal_like", "sqli_like", "cmdi_like", "markup_like")
            assert traversal if m.family == "traversal_like" else special


def test_g2_both_classes_use_the_same_query_builder_for_delete(dev_large):
    names = {0: set(), 1: set()}
    for r in dev_large.records:
        if _is_resource_delete(r) and "?" in r.features.path:
            _, _, query = r.features.path.partition("?")
            names[r.meta.label] |= {kv.split("=")[0] for kv in query.split("&")}
    assert names[0] and names[1]
    allowed = set(cat.QUERY_TEXT) | set(cat.QUERY_NUM) | set(cat.QUERY_TOKEN)
    assert names[0] <= allowed and names[1] <= allowed
    assert names[0] & set(cat.QUERY_TEXT) and names[1] & set(cat.QUERY_TEXT)


def test_no_class_exclusive_method_and_path_category_cell_other_than_the_documented_status_calls(dev_large):
    cells: dict[tuple[str, str], dict[int, set[str]]] = {}
    for r in dev_large.records:
        cell = cells.setdefault((r.features.method, _category(r)), {0: set(), 1: set()})
        cell[r.meta.label].add(dev_large.metadata[r.record_id].template_id)
    exclusive = {k for k, v in cells.items() if (len(v[0]) >= 3 and not v[1]) or (len(v[1]) >= 3 and not v[0])}
    # GET on status paths are health calls by construction: SIM-P06's documented structural limitation (PROTOCOL 12.3).
    assert exclusive == {("GET", "status_area")}, exclusive


def test_simpol_1_is_unchanged_and_p12_stays_inert(pilot, dev_large):
    from tokenization_ai.heldout_eval import policy_sim as sim

    assert sim.CATALOG_VERSION == "simpol-1"
    assert sim.CATALOG_SHA256 == "79273ba6dac5cfa453e258024c2a9e4510e041b68c8db8b79781c52b4de1ba79"
    assert len(sim.POLICIES) == 13
    for ds in (pilot, dev_large):
        applied = Counter(pid for r in ds.records for pid in sim.simulate_rules_for_record(r).policy_ids)
        assert "SIM-P12" not in applied  # no anonymous request reaches a protected path: not fed by any scenario
        assert applied["SIM-P04"] and applied["SIM-P07"] and applied["SIM-P06"]


def test_construction_checks_pass_for_the_development_pilot_and_features_carry_no_metadata(pilot):
    from tokenization_ai.heldout_eval import audit

    findings = (audit.scan_label_terms(pilot) + audit.check_feature_separation(pilot)
                + audit.check_content_type_semantics(pilot) + audit.check_determinism(pilot.config))
    split_info, split_findings = audit.check_split(pilot)
    errors = [f for f in findings + split_findings if f.severity == "error"]
    assert not errors, [f.message for f in errors]
    # This is a construction check only; it does not show the generator is free of leakage or shortcuts.
    assert {f.name for f in dataclasses.fields(RequestFeatures)} == {"method", "path", "body", "headers"}
    for r in pilot.records:
        text = " ".join([r.features.path, r.features.body, *(f"{k} {v}" for k, v in r.features.headers)]).lower()
        for fam in cat.family_names():
            assert fam not in text


def test_changed_scenarios_keep_requests_normalizing_to_their_template(dev_large):
    for r in dev_large.records:
        assert group_id_for(r.features) == dev_large.metadata[r.record_id].template_id
