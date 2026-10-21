"""Tests for the synthetic scenario catalog (draft protocol heldout-eval-1.1.0-draft).

Everything is fabricated and harmless; nothing here trains a model or measures detection quality.
"""

from __future__ import annotations

import random
import re

import pytest

from tokenization_ai.heldout_eval import synthetic_catalog as cat

POOLS = {
    "RESOURCES": cat.RESOURCES, "SUB_RESOURCES": cat.SUB_RESOURCES, "USER_AGENTS": cat.USER_AGENTS,
    "ACCEPT_VALUES": cat.ACCEPT_VALUES, "ROLES_LOW": cat.ROLES_LOW, "WORDS": cat.WORDS,
    "FIRST_NAMES": cat.FIRST_NAMES, "FIELD_NAMES_TEXT": cat.FIELD_NAMES_TEXT, "FIELD_NAMES_NUM": cat.FIELD_NAMES_NUM,
    "FIELD_NAMES_BOOL": cat.FIELD_NAMES_BOOL, "FIELD_NAMES_LIST": cat.FIELD_NAMES_LIST,
    "FIELD_NAMES_OBJECT": cat.FIELD_NAMES_OBJECT, "QUERY_NUM": cat.QUERY_NUM, "QUERY_TOKEN": cat.QUERY_TOKEN,
    "QUERY_TEXT": cat.QUERY_TEXT, "PRIVILEGED_OBVIOUS": cat.PRIVILEGED_FIELDS_OBVIOUS,
    "PRIVILEGED_QUIET": cat.PRIVILEGED_FIELDS_QUIET, "HARD_NEGATIVE_PHRASES": cat.HARD_NEGATIVE_PHRASES,
}


def test_family_names_are_unique_and_kinds_valid():
    names = [f.name for f in cat.FAMILIES]
    assert len(names) == len(set(names)) == len(cat.FAMILY_BY_NAME)
    assert {f.label_kind for f in cat.FAMILIES} == {"normal", "suspicious", "mixed"}
    assert cat.family_names() == tuple(names)


def test_every_family_is_documented_with_a_label_rule_and_a_positive_weight():
    for f in cat.FAMILIES:
        assert f.description.strip() and f.label_rule.strip(), f.name
        assert isinstance(f.weight, int) and f.weight > 0, f.name
        assert f.hard_case in ("", "negative", "positive")
    assert cat.default_weights() == {f.name: f.weight for f in cat.FAMILIES}


def test_required_scenario_coverage_is_present():
    by_kind = {k: {f.name for f in cat.FAMILIES if f.label_kind == k} for k in ("normal", "suspicious", "mixed")}
    assert {"crud_read", "crud_write", "search_filter", "auth_login", "json_create_nested", "admin_legit",
            "health_status", "alt_content_type", "admin_privileged_update", "hn_nested_config"} <= by_kind["normal"]
    assert {"sqli_like", "traversal_like", "cmdi_like", "markup_like", "mass_assignment",
            "structure_anomaly"} <= by_kind["suspicious"]
    assert {"role_action_mismatch", "content_type_mismatch", "shared_endpoint_text"} <= by_kind["mixed"]
    assert any(f.hard_case == "negative" for f in cat.FAMILIES)
    assert any(f.hard_case == "positive" for f in cat.FAMILIES)


def test_hard_negative_families_are_normal_and_hard_positive_families_are_not():
    for f in cat.FAMILIES:
        if f.hard_case == "negative":
            assert f.label_kind == "normal", f.name
        if f.hard_case == "positive":
            assert f.label_kind in ("suspicious", "mixed"), f.name


def test_no_stateful_or_multi_request_scenarios_are_defined():
    for f in cat.FAMILIES:
        text = f"{f.name} {f.description} {f.label_rule}".lower()
        for word in ("stateful", "multi-request", "sequence of", "brute", "stuffing", "rate limit", "replay"):
            assert word not in text, (f.name, word)


def test_weight_shares_sum_to_one_and_target_roughly_a_third_containing_suspicious():
    shares = cat.weight_shares()
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert 0.25 <= shares["suspicious"] + shares["mixed"] <= 0.40


@pytest.mark.parametrize("pool_name", sorted(POOLS))
def test_shared_pools_contain_no_banned_label_words_or_family_names(pool_name):
    for item in POOLS[pool_name]:
        low = item.lower()
        for word in cat.BANNED_LABEL_WORDS:
            assert word not in low, (pool_name, item, word)
        for family in cat.family_names():
            assert family not in low


@pytest.mark.parametrize("pool_name", ["RESOURCES", "SUB_RESOURCES", "WORDS", "FIRST_NAMES", "QUERY_NUM",
                                       "QUERY_TOKEN", "QUERY_TEXT", "FIELD_NAMES_TEXT", "FIELD_NAMES_NUM"])
def test_name_and_word_pools_are_letters_only_so_normalization_cannot_merge_templates(pool_name):
    for item in POOLS[pool_name]:
        assert re.fullmatch(r"[a-z_]+", item), (pool_name, item)


def test_pools_have_no_duplicates():
    for name, pool in POOLS.items():
        assert len(pool) == len(set(pool)), name


def test_ordinary_words_avoid_sql_markup_and_shell_vocabulary():
    risky = {"select", "union", "drop", "insert", "delete", "update", "script", "alert", "whoami", "passwd", "echo"}
    assert not (set(cat.WORDS) & risky)
    assert not (set(cat.FIRST_NAMES) & risky)


def test_ordinary_text_helpers_are_deterministic_and_never_numeric_only():
    for seed in range(5):
        a = cat.plain_phrase(random.Random(seed), 1, 4)
        assert a == cat.plain_phrase(random.Random(seed), 1, 4)
        assert re.search(r"[a-z]", a) and not a.isdigit()
    assert cat.fake_username(random.Random(1)) == cat.fake_username(random.Random(1))
    assert cat.fake_secret(random.Random(1)).startswith("Fx")


def test_hard_negative_phrases_are_ordinary_text_with_risky_words():
    joined = " ".join(cat.HARD_NEGATIVE_PHRASES).lower()
    for word in ("select", "union", "drop", "insert", "delete", "update"):
        assert word in joined
    for phrase in cat.HARD_NEGATIVE_PHRASES:
        assert re.search(r"[a-z]", phrase)


@pytest.mark.parametrize("helper", [cat.sql_like, cat.cmd_like, cat.markup_like])
def test_payload_helpers_are_deterministic_nonempty_harmless_strings(helper):
    for hard in (False, True):
        for seed in range(8):
            one, two = helper(random.Random(seed), hard), helper(random.Random(seed), hard)
            assert one == two and one.strip()
            assert re.search(r"[A-Za-z]", one) and not one.isdigit()
            assert len(one) < 120
            low = one.lower()
            assert not any(w in low for w in cat.BANNED_LABEL_WORDS)


@pytest.mark.parametrize("helper", [cat.sql_like, cat.cmd_like, cat.markup_like])
def test_hard_variants_differ_from_easy_ones(helper):
    easy = {helper(random.Random(s), False) for s in range(30)}
    hard = {helper(random.Random(s), True) for s in range(30)}
    assert easy != hard


def test_traversal_segments_easy_and_hard_forms():
    easy = cat.traversal_segments(random.Random(3), False)
    hard = cat.traversal_segments(random.Random(3), True)
    assert ".." in easy and all(isinstance(s, str) and s for s in easy)
    assert len(hard) == 1 and "%2f" in hard[0].lower()
    assert cat.traversal_segments(random.Random(3), False) == easy


def test_privileged_field_pools_are_disjoint_and_nonempty():
    assert cat.PRIVILEGED_FIELDS_OBVIOUS and cat.PRIVILEGED_FIELDS_QUIET
    assert not set(cat.PRIVILEGED_FIELDS_OBVIOUS) & set(cat.PRIVILEGED_FIELDS_QUIET)
    assert not set(cat.PRIVILEGED_FIELDS_OBVIOUS) & set(cat.FIELD_NAMES_TEXT + cat.FIELD_NAMES_NUM + cat.FIELD_NAMES_BOOL)


def test_catalog_has_no_io_or_pipeline_imports():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(cat.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or "").split(".")[0])
    assert imported <= {"__future__", "random", "dataclasses", "typing"}, imported


def test_content_type_semantics_are_documented_as_actual_body_incompatibility():
    family = cat.FAMILY_BY_NAME["content_type_mismatch"]
    assert "ACTUAL" in family.label_rule and "never decides" in family.label_rule
    assert family.label_kind == "mixed"
    assert set(cat.CONTENT_TYPES_BY_FORMAT) == {"json", "xml", "form", "plain"}
    flat = [t for types in cat.CONTENT_TYPES_BY_FORMAT.values() for t in types]
    assert len(flat) == len(set(flat)) and "text/plain" in flat and "application/xml" in flat


def test_new_legitimate_families_are_hard_negatives_with_label_zero_rules():
    for name in ("alt_content_type", "admin_privileged_update", "hn_nested_config"):
        f = cat.FAMILY_BY_NAME[name]
        assert f.label_kind == "normal" and f.hard_case == "negative" and f.label_rule.startswith("always 0")


def test_mass_assignment_is_defined_by_non_admin_role_context():
    assert "NON-admin" in cat.FAMILY_BY_NAME["mass_assignment"].description


def test_traversal_tails_include_single_segment_tails_so_depths_overlap_with_normal_paths():
    assert any("/" not in t for t in cat.TRAVERSAL_TAILS) and any("/" in t for t in cat.TRAVERSAL_TAILS)
