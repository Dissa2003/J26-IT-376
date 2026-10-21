"""Tests for the held-out study's leakage-safe splitting (draft protocol heldout-eval-1.0.0-draft).

Every request here is FABRICATED. Nothing in this file is, or resembles a measurement on, a
real dataset, and no model is trained or scored.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.splits as splits
from tokenization_ai.heldout_eval.splits import (
    FORBIDDEN_FEATURE_FIELD_NAMES,
    PRIMARY_SPLIT_SEED,
    ROBUSTNESS_SPLIT_SEEDS,
    SPLIT_NAMES,
    EvalRecord,
    InvalidRecordError,
    IsolationViolationError,
    LabelConflictError,
    RequestFeatures,
    SplitError,
    SplitInfeasibleError,
    SplitMetadata,
    SplitResult,
    exact_fingerprint,
    group_id_for,
    make_record,
    request_template,
    split_records,
    time_forward_split,
    verify_split_isolation,
)

SENTINEL = "FABRICATED-SENTINEL-TEXT"


def nm(i: int) -> str:
    """Letters-only name for template i; digits would be normalized away and merge templates."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(97 + r) + out
    return out


def feats(method="GET", path="/x", body="", headers=None) -> RequestFeatures:
    return make_record("r", method=method, path=path, label=0, body=body, headers=headers).features


def synthetic(n_benign_templates=24, n_attack_templates=12, per_template=5, start_ts=1000.0) -> list[EvalRecord]:
    """Fabricated records: each template has several near-duplicate variants (same shape, new values)."""
    out: list[EvalRecord] = []
    k = 0
    for t in range(n_benign_templates):
        for j in range(per_template):
            out.append(make_record(
                f"b{k}", method="GET", path=f"/benign{nm(t)}/item/{1000 + j * 7}", label=0,
                timestamp=start_ts + k, source_id=f"src{k % 5}"))
            k += 1
    for t in range(n_attack_templates):
        for j in range(per_template):
            out.append(make_record(
                f"a{k}", method="POST", path=f"/attack{nm(t)}/login?user=u{j}&q=x{'y' * j}", label=1,
                body=f"name=u{j}&pw=p{j}", timestamp=start_ts + k, source_id=f"src{k % 5}"))
            k += 1
    return out


# -- request template normalization -------------------------------------------------


def test_numbers_hex_and_uuids_are_normalized():
    a = request_template(feats(path="/orders/123/items/9"))
    b = request_template(feats(path="/orders/456/items/77"))
    assert a == b
    assert request_template(feats(path="/f/0a1b2c3d4e5f")) == request_template(feats(path="/f/ffeeddccbbaa"))
    u1, u2 = "123e4567-e89b-12d3-a456-426614174000", "00000000-0000-0000-0000-000000000000"
    assert request_template(feats(path=f"/u/{u1}")) == request_template(feats(path=f"/u/{u2}"))


def test_case_slashes_trailing_slash_and_percent_encoding_are_normalized():
    base = request_template(feats(path="/api/users"))
    for variant in ("/API/Users", "/api//users", "/api/users/", "/api/%75sers", "/api/users#frag"):
        assert request_template(feats(path=variant)) == base, variant
    assert request_template(feats(method="get", path="/api/users")) == base


def test_query_values_become_classes_and_parameter_order_is_ignored():
    a = request_template(feats(path="/s?q=hello&page=2"))
    b = request_template(feats(path="/s?page=9&q=world"))
    assert a == b
    assert a != request_template(feats(path="/s?q=hello"))  # different parameters
    assert a != request_template(feats(path="/s?q=&page=2"))  # empty value is its own class
    assert request_template(feats(path="/s?q=1")) != request_template(feats(path="/s?q=abc"))


def test_near_duplicate_attack_variants_share_a_group():
    variants = ["/p?id=1' OR '1'='1", "/p?id=2' UNION SELECT a", "/p?id=zz--"]
    ids = {group_id_for(feats(path=v)) for v in variants}
    assert len(ids) == 1


def test_body_shape_is_normalized_for_json_and_forms():
    j1 = request_template(feats(method="POST", path="/a", body='{"u": "x", "n": 1}'))
    j2 = request_template(feats(method="POST", path="/a", body='{"n": 99, "u": "totally different"}'))
    assert j1 == j2
    assert j1 != request_template(feats(method="POST", path="/a", body='{"u": "x"}'))
    f1 = request_template(feats(method="POST", path="/a", body="u=abc&p=1"))
    f2 = request_template(feats(method="POST", path="/a", body="p=22&u=zzz"))
    assert f1 == f2
    assert request_template(feats(method="POST", path="/a", body="")) != f1
    raw = request_template(feats(method="POST", path="/a", body="free text here"))
    assert raw.endswith("raw:" + str(len("free text here").bit_length()))


def test_structurally_different_requests_get_different_groups():
    assert group_id_for(feats(path="/a")) != group_id_for(feats(path="/b"))
    assert group_id_for(feats(method="GET", path="/a")) != group_id_for(feats(method="POST", path="/a"))


def test_grouping_is_deterministic_and_label_independent():
    one = make_record("r1", method="GET", path="/x/1", label=0).features
    two = make_record("r2", method="GET", path="/x/2", label=1).features
    assert group_id_for(one) == group_id_for(one) == group_id_for(two)
    assert re.fullmatch(r"[0-9a-f]{16}", group_id_for(one))


def test_deeply_nested_json_is_handled():
    deep = "[" * 40 + "1" + "]" * 40
    assert request_template(feats(method="POST", path="/a", body=deep)).startswith("POST|")


# -- reproducibility and isolation -----------------------------------------------------


def test_same_seed_gives_identical_assignments_and_manifest_hash():
    recs = synthetic()
    one = split_records(recs, seed=PRIMARY_SPLIT_SEED)
    two = split_records(list(reversed(recs)), seed=PRIMARY_SPLIT_SEED)
    assert one.assignments == two.assignments
    assert one.manifest_sha256 == two.manifest_sha256 and len(one.manifest_sha256) == 64
    assert one.to_manifest() == two.to_manifest()


def test_different_seeds_change_the_split_but_every_split_stays_valid():
    recs = synthetic()
    primary = split_records(recs, seed=PRIMARY_SPLIT_SEED)
    others = [split_records(recs, seed=s) for s in ROBUSTNESS_SPLIT_SEEDS]
    assert any(o.assignments != primary.assignments for o in others)
    for result in [primary, *others]:
        assert verify_split_isolation(result, recs) == []
        assert result.report.meets_requirements


def test_no_group_or_template_overlap_between_splits():
    recs = synthetic()
    result = split_records(recs)
    by_group: dict[str, set[str]] = {}
    for r in recs:
        by_group.setdefault(group_id_for(r.features), set()).add(result.assignments[r.record_id])
    assert all(len(s) == 1 for s in by_group.values())
    assert set(result.assignments.values()) == set(SPLIT_NAMES)
    assert sum(len(result.members(s)) for s in SPLIT_NAMES) == len(recs)


def test_isolation_verifier_detects_tampered_assignments():
    recs = synthetic()
    result = split_records(recs)
    victim = next(r for r in recs if r.record_id.startswith("b"))
    twin = next(r for r in recs if r.record_id != victim.record_id and group_id_for(r.features) == group_id_for(victim.features))
    tampered = dict(result.assignments)
    tampered[twin.record_id] = next(s for s in SPLIT_NAMES if s != result.assignments[victim.record_id])
    bad = SplitResult(assignments=tampered, group_of=result.group_of, excluded=result.excluded, report=result.report)
    problems = verify_split_isolation(bad, recs)
    assert problems and any("spans splits" in p for p in problems)


def test_exact_duplicates_never_cross_splits():
    recs = synthetic()
    dup_source = recs[0]
    dups = [make_record(f"dup{i}", method="GET", path=dup_source.features.path, label=0, timestamp=5000.0 + i) for i in range(6)]
    result = split_records(recs + dups)
    splits_seen = {result.assignments[r.record_id] for r in [dup_source, *dups]}
    assert len(splits_seen) == 1
    assert result.report.exact_duplicate_records >= 6
    assert verify_split_isolation(result, recs + dups) == []


def test_drop_exact_duplicates_keeps_one_and_records_the_rest():
    recs = synthetic()
    dups = [make_record(f"dup{i}", method="GET", path=recs[0].features.path, label=0) for i in range(3)]
    result = split_records(recs + dups, drop_exact_duplicates=True)
    group = [r for r in recs + dups if exact_fingerprint(r.features) == exact_fingerprint(recs[0].features)]
    kept = [r for r in group if r.record_id in result.assignments]
    dropped = [r for r in group if r.record_id in result.excluded]
    assert len(kept) == 1 and len(dropped) == len(group) - 1
    assert all(result.excluded[r.record_id] == "exact_duplicate_dropped" for r in dropped)
    assert result.report.n_excluded == len(dropped)


def test_headers_distinguish_exact_duplicates_but_not_templates():
    a = make_record("a", method="GET", path="/h", label=0, headers={"User-Agent": "one"}).features
    b = make_record("b", method="GET", path="/h", label=0, headers={"User-Agent": "two"}).features
    assert exact_fingerprint(a) != exact_fingerprint(b)
    assert group_id_for(a) == group_id_for(b)


# -- label conflicts ---------------------------------------------------------------------


def _with_conflict():
    recs = synthetic()
    clash = [
        make_record("c1", method="GET", path="/conflict/1", label=0),
        make_record("c2", method="GET", path="/conflict/1", label=1),
    ]
    return recs, clash


def test_exact_duplicate_label_conflict_is_an_error_by_default_without_content():
    recs, clash = _with_conflict()
    with pytest.raises(LabelConflictError) as caught:
        split_records(recs + clash)
    assert "conflict" in str(caught.value)
    assert "/conflict/1" not in str(caught.value)


def test_exact_conflict_can_be_excluded_and_is_reported():
    recs, clash = _with_conflict()
    result = split_records(recs + clash, exact_conflict_policy="exclude")
    assert {r.record_id: result.excluded.get(r.record_id) for r in clash} == {
        "c1": "exact_duplicate_label_conflict", "c2": "exact_duplicate_label_conflict"}
    assert result.report.exact_duplicate_label_conflicts == 1
    assert all(r.record_id not in result.assignments for r in clash)


def test_template_with_mixed_labels_is_reported_but_kept_together():
    recs = synthetic()
    benign = make_record("m1", method="GET", path="/mix/search?q=hello", label=0)
    attack = make_record("m2", method="GET", path="/mix/search?q=' OR 1=1 --", label=1)
    assert group_id_for(benign.features) == group_id_for(attack.features)
    assert exact_fingerprint(benign.features) != exact_fingerprint(attack.features)
    result = split_records(recs + [benign, attack])
    assert result.report.mixed_label_templates == 1
    assert result.assignments["m1"] == result.assignments["m2"]
    assert result.report.exact_duplicate_label_conflicts == 0


# -- class distribution ---------------------------------------------------------------------


def test_every_split_holds_both_classes_in_a_feasible_dataset():
    result = split_records(synthetic())
    for name in SPLIT_NAMES:
        counts = result.report.counts[name]
        assert counts["positives"] >= 1 and counts["negatives"] >= 1
        assert counts["records"] == counts["positives"] + counts["negatives"]
    assert sum(result.report.counts[n]["records"] for n in SPLIT_NAMES) == result.report.n_assigned


def test_single_class_data_is_infeasible_strict_and_reported_when_not_strict():
    benign_only = [r for r in synthetic() if r.meta.label == 0]
    with pytest.raises(SplitInfeasibleError, match="positive"):
        split_records(benign_only)
    result = split_records(benign_only, strict=False)
    assert not result.report.meets_requirements
    assert any("positive" in p for p in result.report.problems)
    assert verify_split_isolation(result, benign_only) == []  # isolation is never relaxed


def test_rare_class_concentrated_in_one_group_cannot_fill_every_split():
    recs = [r for r in synthetic() if r.meta.label == 0]
    recs += [make_record(f"p{i}", method="POST", path=f"/only-attack/login?x={i}", label=1) for i in range(4)]
    with pytest.raises(SplitInfeasibleError, match="positive"):
        split_records(recs)


def test_min_per_class_is_enforced():
    recs = synthetic(n_attack_templates=3, per_template=2)
    with pytest.raises(SplitInfeasibleError):
        split_records(recs, min_per_class=50)


# -- small and impossible data ------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2])
def test_too_few_groups_fail_explicitly(count):
    recs = [make_record(f"r{i}", method="GET", path=f"/only{nm(i)}", label=i % 2) for i in range(count)]
    with pytest.raises(SplitInfeasibleError, match="group"):
        split_records(recs)


def test_many_records_in_two_groups_still_fail():
    recs = [make_record(f"r{i}", method="GET", path=f"/g{nm(i % 2)}/{i}", label=i % 2) for i in range(40)]
    with pytest.raises(SplitInfeasibleError, match="group"):
        split_records(recs)


def test_empty_input_is_rejected():
    with pytest.raises(InvalidRecordError, match="no records"):
        split_records([])


def test_a_dominant_group_prevents_a_close_fit_and_is_not_hidden():
    recs = [make_record(f"big{i}", method="GET", path=f"/dominant/{i}", label=i % 2) for i in range(60)]
    recs += [make_record(f"s{i}", method="GET", path=f"/small{nm(i)}/x", label=i % 2) for i in range(12)]
    with pytest.raises(SplitInfeasibleError, match="deviate"):
        split_records(recs)
    loose = split_records(recs, strict=False)
    assert loose.report.max_ratio_deviation > 0.10
    assert any("deviate" in p for p in loose.report.problems)
    assert verify_split_isolation(loose, recs) == []


# -- invalid records and parameters -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="", path="/x", label=0),
        dict(method="GET 1", path="/x", label=0),
        dict(method=5, path="/x", label=0),
        dict(method="GET", path="", label=0),
        dict(method="GET", path=None, label=0),
        dict(method="GET", path="/x", label=2),
        dict(method="GET", path="/x", label=True),
        dict(method="GET", path="/x", label="1"),
        dict(method="GET", path="/x", label=None),
        dict(method="GET", path="/x", label=0, body=5),
        dict(method="GET", path="/x", label=0, timestamp=float("nan")),
        dict(method="GET", path="/x", label=0, timestamp=float("inf")),
        dict(method="GET", path="/x", label=0, timestamp=True),
        dict(method="GET", path="/x", label=0, timestamp="now"),
        dict(method="GET", path="/x", label=0, source_id=""),
        dict(method="GET", path="/x", label=0, headers={"a": 1}),
        dict(method="GET", path="/x", label=0, headers=["a"]),
    ],
)
def test_invalid_record_fields_are_rejected(kwargs):
    with pytest.raises(InvalidRecordError):
        make_record("rid", **kwargs)


@pytest.mark.parametrize("rid", ["", " ", " lead", "trail ", None, 5])
def test_invalid_record_ids_are_rejected(rid):
    with pytest.raises(InvalidRecordError):
        make_record(rid, method="GET", path="/x", label=0)


def test_invalid_record_errors_never_echo_values():
    with pytest.raises(InvalidRecordError) as caught:
        make_record("rid", method="GET", path=SENTINEL, label=7)
    assert SENTINEL not in str(caught.value)


def test_duplicate_record_ids_and_foreign_items_are_rejected():
    recs = synthetic()
    with pytest.raises(InvalidRecordError, match="duplicate record_id"):
        split_records(recs + [recs[0]])
    with pytest.raises(InvalidRecordError, match="not an EvalRecord"):
        split_records(recs + [{"record_id": "x"}])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ratios=(0.5, 0.5)),
        dict(ratios=(0.5, 0.3, 0.3)),
        dict(ratios=(1.0, 0.0, 0.0)),
        dict(ratios=(0.7, 0.2, "x")),
        dict(ratios=(0.7, float("nan"), 0.3)),
        dict(min_per_class=0),
        dict(min_per_class=True),
        dict(exact_conflict_policy="ignore"),
        dict(ratio_tolerance=0),
        dict(seed=True),
        dict(seed="1"),
        dict(seed=1.5),
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(SplitError):
        split_records(synthetic(), **kwargs)


def test_custom_ratios_are_respected_approximately():
    result = split_records(synthetic(), ratios=(0.6, 0.2, 0.2), ratio_tolerance=0.15)
    assert abs(result.report.achieved_ratios[0] - 0.6) < 0.15
    assert result.report.ratios == (0.6, 0.2, 0.2)


# -- separation of features, labels and response metadata ----------------------------------------------


def test_feature_type_has_no_label_response_or_detector_field():
    names = {f.name for f in dataclasses.fields(RequestFeatures)}
    assert names == {"method", "path", "body", "headers"}
    assert not (names & FORBIDDEN_FEATURE_FIELD_NAMES)
    meta_names = {f.name for f in dataclasses.fields(SplitMetadata)}
    assert meta_names == {"label", "timestamp", "source_id"}
    assert not (names & meta_names)


def test_template_fingerprint_and_group_accept_only_request_features():
    rec = make_record("r", method="GET", path="/x", label=1)
    for bad in (rec, rec.meta, {"method": "GET", "path": "/x"}, "GET /x"):
        with pytest.raises(TypeError):
            request_template(bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            exact_fingerprint(bad)  # type: ignore[arg-type]


def test_labels_timestamps_and_sources_cannot_change_a_request_group_or_fingerprint():
    a = make_record("a", method="GET", path="/x/1", label=0, timestamp=1.0, source_id="s1")
    b = make_record("b", method="GET", path="/x/1", label=1, timestamp=99.0, source_id="s2")
    assert a.features == b.features
    assert group_id_for(a.features) == group_id_for(b.features)
    assert exact_fingerprint(a.features) == exact_fingerprint(b.features)


def test_response_and_detector_fields_are_not_accepted_by_make_record():
    for extra in ("response_status", "modsecurity_decision", "capec", "attack_type"):
        with pytest.raises(TypeError):
            make_record("r", method="GET", path="/x", label=0, **{extra: "x"})  # type: ignore[arg-type]


def test_records_are_immutable():
    rec = make_record("r", method="GET", path="/x", label=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.meta.label = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.features.path = "/y"  # type: ignore[misc]


# -- provenance without raw content -----------------------------------------------------------------------


def test_manifest_is_json_safe_and_contains_no_raw_request_content():
    recs = [
        make_record(r.record_id, method=r.features.method, path=f"{r.features.path}&{SENTINEL}" if "?" in r.features.path else f"{r.features.path}/{SENTINEL}",
                    label=r.meta.label, body=f"{SENTINEL}={r.record_id}", headers={"x-secret": SENTINEL}, timestamp=r.meta.timestamp)
        for r in synthetic()
    ]
    result = split_records(recs)
    text = json.dumps(result.to_manifest(), allow_nan=False)
    assert SENTINEL not in text and "x-secret" not in text
    manifest = result.to_manifest()
    assert {row["record_id"] for row in manifest["records"]} == {r.record_id for r in recs}
    assert all(re.fullmatch(r"[0-9a-f]{16}", row["group_id"]) for row in manifest["records"])
    assert manifest["report"]["protocol_version"] == splits.PROTOCOL_VERSION
    assert manifest["report"]["template_version"] == splits.TEMPLATE_VERSION
    assert len(manifest["report"]["dataset_fingerprint"]) == 64


def test_manifest_hash_changes_when_the_data_or_seed_changes():
    recs = synthetic()
    base = split_records(recs, seed=1).manifest_sha256
    assert split_records(recs, seed=1).manifest_sha256 == base
    assert split_records(recs, seed=2).manifest_sha256 != base
    changed = recs[:-1] + [make_record("zz", method="GET", path="/different/shape/here", label=0)]
    assert split_records(changed, seed=1).report.dataset_fingerprint != split_records(recs, seed=1).report.dataset_fingerprint


# -- source grouping is optional and never the only key -------------------------------------------------------


def test_same_template_different_sources_are_still_grouped():
    recs = synthetic()
    a = make_record("s1", method="GET", path="/shared/1", label=0, source_id="one")
    b = make_record("s2", method="GET", path="/shared/2", label=0, source_id="two")
    result = split_records(recs + [a, b])
    assert result.assignments["s1"] == result.assignments["s2"]


def test_merge_by_source_keeps_a_sources_requests_together_across_templates():
    recs = []
    for t in range(30):
        for j in range(3):
            recs.append(make_record(f"t{t}_{j}", method="GET" if t % 3 else "POST", path=f"/svc{nm(t)}/r{nm(j)}x/y{nm(t)}", label=t % 2,
                                    source_id=f"src{t % 6}"))
    merged = split_records(recs, merge_by_source=True, strict=False)
    for s in range(6):
        in_source = {merged.assignments[r.record_id] for r in recs if r.meta.source_id == f"src{s}"}
        assert len(in_source) == 1
    assert verify_split_isolation(merged, recs) == []
    unmerged = split_records(recs, strict=False)
    assert any(len({unmerged.assignments[r.record_id] for r in recs if r.meta.source_id == f"src{s}"}) > 1 for s in range(6))


# -- time-forward split ---------------------------------------------------------------------------------------------


def interleaved_in_time() -> list[EvalRecord]:
    """The synthetic set re-timestamped so both classes occur across the whole timeline."""
    out = []
    for r in synthetic():
        index = int(r.record_id[1:])
        is_benign = r.record_id.startswith("b")
        base = ((index if is_benign else index - 120) // 5) * (10 if is_benign else 21)
        out.append(make_record(r.record_id, method=r.features.method, path=r.features.path, label=r.meta.label,
                               body=r.features.body, timestamp=float(base + index % 5), source_id=r.meta.source_id))
    return out


def test_time_forward_orders_splits_in_time_and_is_deterministic():
    recs = interleaved_in_time()
    one = time_forward_split(recs)
    two = time_forward_split(list(reversed(recs)))
    assert one.assignments == two.assignments and one.manifest_sha256 == two.manifest_sha256
    by_id = {r.record_id: r for r in recs}
    latest = {name: max(by_id[rid].meta.timestamp for rid in one.members(name)) for name in SPLIT_NAMES}
    assert latest["train"] <= latest["validation"] <= latest["test"]
    assert verify_split_isolation(one, recs) == []
    assert one.report.method == "time_forward" and one.report.seed is None


def test_time_forward_requires_timestamps_on_every_record():
    recs = synthetic()
    recs[3] = make_record("nots", method="GET", path="/no/timestamp", label=0)
    with pytest.raises(InvalidRecordError, match="timestamp"):
        time_forward_split(recs)


def test_time_forward_puts_a_recurring_template_wholly_in_the_later_split():
    recs = []
    k = 0
    for t in range(30):
        for j in range(4):
            recs.append(make_record(f"x{k}", method="GET", path=f"/tf{nm(t)}/a/{j}", label=t % 2,
                                    timestamp=float(k)))
            k += 1
    # a template that recurs at the very start and the very end of the timeline
    recs.append(make_record("early", method="GET", path="/recurring/1", label=0, timestamp=-1.0))
    recs.append(make_record("late", method="GET", path="/recurring/2", label=1, timestamp=10_000.0))
    result = time_forward_split(recs, strict=False)
    assert result.assignments["early"] == result.assignments["late"] == "test"
    assert result.report.straddling_groups >= 1
    assert verify_split_isolation(result, recs) == []


def test_time_forward_reports_missing_class_in_a_split():
    recs = [make_record(f"r{i}", method="GET", path=f"/tfx{nm(i)}/p", label=0 if i < 80 else 1, timestamp=float(i)) for i in range(100)]
    with pytest.raises(SplitInfeasibleError, match="positive|negative"):
        time_forward_split(recs)


# -- scope: isolation of this module and the request-only (Arm A) path --------------------------------------


def test_splitting_module_imports_only_the_standard_library():
    tree = ast.parse(Path(splits.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or "").split(".")[0])
    assert imported <= {"__future__", "hashlib", "json", "math", "random", "re", "urllib", "collections",
                        "dataclasses", "typing"}, imported


def test_existing_pipeline_accepts_an_empty_rule_list_for_the_request_only_arm():
    """Documents (does not change) the supported request-only path used by comparison A."""
    from tokenization_ai.demo import _build_demo_service
    from tokenization_ai.models import UnifiedApiContext, VerifiedRuleInfo

    service = _build_demo_service()
    context = UnifiedApiContext(event_id="ra-1", method="GET", path="/fabricated/path")
    analysis = service.analyze(context, VerifiedRuleInfo(event_id="ra-1", rules=[]))
    assert analysis.sequence.tokens[-3:] == ["[RULES_START]", "[RULES_END]", "[SEQ_END]"]
    assert 0.0 <= analysis.prediction.score <= 1.0
