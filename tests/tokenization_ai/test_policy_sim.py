"""Tests for the simulated policy rule catalog (draft protocol heldout-eval-1.1.0-draft, catalog simpol-1).

The rules are SIMULATED: deterministic functions of observable request properties. They are not Member 2
output, carry no verification status, and no Member 2 VerificationReport is fabricated here. No model is
trained and no detection performance is computed.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import socket
import sys
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.policy_diagnostics as diag
import tokenization_ai.heldout_eval.policy_sim as sim
from tokenization_ai.adapters import InputContractError
from tokenization_ai.fusion import InputFusion
from tokenization_ai.heldout_eval import synthetic_catalog as cat
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.policy_sim import (
    ALLOWED_ATTRIBUTES,
    CATALOG_SHA256,
    CATALOG_VERSION,
    NOT_VERIFIED_STATUS,
    POLICIES,
    PolicyDef,
    PolicySimError,
    SimulatedRuleSet,
    applicable_policies,
    catalog_fingerprint,
    observe,
    simulate_rules,
    simulate_rules_for_record,
    validate_catalog,
)
from tokenization_ai.heldout_eval.splits import EvalRecord, RequestFeatures, SplitMetadata, make_record
from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.models import UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.normalization import Normalizer
from tokenization_ai.tokenizer import HybridTokenizer

PINNED_CATALOG_SHA256 = "79273ba6dac5cfa453e258024c2a9e4510e041b68c8db8b79781c52b4de1ba79"


def feats(method="GET", path="/orders", body="", headers=None) -> RequestFeatures:
    return make_record("r", method=method, path=path, label=0, body=body, headers=headers).features


def ids(features) -> list[str]:
    return [p.rule_id for p in applicable_policies(features)]


@pytest.fixture(scope="module")
def pilot():
    return generate(GeneratorConfig())


@pytest.fixture(scope="module")
def small():
    return generate(GeneratorConfig(seed=3, n_templates=60))


# -- catalog -------------------------------------------------------------------------------------------------


def test_catalog_has_neutral_sequential_unique_ids_and_documented_version():
    assert CATALOG_VERSION == "simpol-1"
    rule_ids = [p.rule_id for p in POLICIES]
    assert rule_ids == [f"SIM-P{i:02d}" for i in range(1, len(POLICIES) + 1)]
    assert len(set(rule_ids)) == len(rule_ids)


def test_catalog_text_never_names_a_label_a_family_or_a_scenario():
    banned = [w.lower() for w in cat.BANNED_LABEL_WORDS] + [n.lower() for n in cat.family_names()]
    banned += ["sql", "traversal", "injection", "markup", "mass", "mismatch", "suspicious"]
    for p in POLICIES:
        text = " ".join([p.rule_id, p.action, p.resource, p.effect]).lower()
        assert not any(b in text for b in banned), text


def test_attributes_are_allowlisted_and_the_catalog_validates():
    assert ALLOWED_ATTRIBUTES == ("action", "resource", "effect")
    validate_catalog(POLICIES)
    for p in POLICIES:
        assert p.effect in ("Allow", "Deny")
        assert p.applies_when


def test_catalog_fingerprint_is_stable_and_pinned():
    assert catalog_fingerprint() == catalog_fingerprint() == CATALOG_SHA256
    assert len(CATALOG_SHA256) == 64
    assert CATALOG_SHA256 == PINNED_CATALOG_SHA256  # deliberate guard: a catalog change must update this and the protocol


def test_fingerprint_changes_when_any_part_of_the_catalog_changes():
    base = catalog_fingerprint()
    changed_when = list(POLICIES)
    changed_when[0] = dataclasses.replace(POLICIES[0], applies_when={**POLICIES[0].applies_when, "methods": ("GET",)})
    changed_attr = list(POLICIES)
    changed_attr[1] = dataclasses.replace(POLICIES[1], resource="other-things")
    assert catalog_fingerprint(changed_when) != base
    assert catalog_fingerprint(changed_attr) != base
    assert catalog_fingerprint(POLICIES[:-1]) != base


def _with(index: int, **kw) -> list[PolicyDef]:
    out = list(POLICIES)
    out[index] = dataclasses.replace(POLICIES[index], **kw)
    return out


@pytest.mark.parametrize(
    "catalog",
    [
        [],
        list(POLICIES) + [POLICIES[0]],  # duplicate id
        _with(0, rule_id="P01"),
        _with(0, rule_id="SIM-P1"),
        _with(0, rule_id="SIM-PATTACK"),
        _with(0, action="Attack"),
        _with(0, resource="a b"),
        _with(0, resource=""),
        _with(0, resource="malicious-area"),
        _with(0, resource="sqli_like"),
        _with(0, effect="Permit"),
        _with(0, applies_when={}),
        _with(0, applies_when={"label": True}),
        _with(0, applies_when={"methods": "GET"}),
        _with(0, applies_when={"methods": ("TRACE",)}),
        _with(0, applies_when={"path_category": ("moon",)}),
        _with(0, applies_when={"authorization": "yes"}),
        _with(0, applies_when={"body": 1}),
    ],
)
def test_invalid_catalogs_are_rejected(catalog):
    with pytest.raises(PolicySimError):
        validate_catalog(catalog)


# -- observable properties ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,category",
    [("/admin/orders/status", "admin_area"), ("/auth/login", "auth_area"), ("/account/reset", "auth_area"),
     ("/portal/session", "auth_area"), ("/files/orders/etc/passwd", "files_area"),
     ("/documents/orders/1/report.final.pdf", "documents_area"), ("/health", "status_area"), ("/ping", "status_area"),
     ("/status/orders", "status_area"), ("/orders", "resource_area"), ("/Orders/12", "resource_area"),
     ("/ADMIN/x", "admin_area"), ("/files?x=1", "files_area")],
)
def test_path_category_comes_from_the_first_segment_only(path, category):
    assert observe(feats(path=path)).path_category == category


@pytest.mark.parametrize(
    "declared,cls",
    [("application/json", "json"), ("application/json; charset=utf-8", "json"), ("APPLICATION/JSON", "json"),
     ("application/xml", "xml"), ("text/xml", "xml"), ("application/x-www-form-urlencoded", "form"),
     ("text/plain", "plain"), ("image/png", "other")],
)
def test_declared_type_class_uses_the_header_value(declared, cls):
    f = feats(method="POST", body='{"a":1}', headers={"Content-Type": declared})
    assert observe(f).declared_type == cls


def test_declared_type_never_uses_the_body_content_and_is_none_without_a_header():
    json_body_declared_plain = feats(method="POST", body='{"a":1}', headers={"content-type": "text/plain"})
    assert observe(json_body_declared_plain).declared_type == "plain"
    assert observe(feats(method="POST", body="a=b")).declared_type == "none"


def test_authorization_is_presence_only_and_role_values_are_ignored():
    with_auth = feats(headers={"authorization": "Bearer anything", "x-role": "admin"})
    other_values = feats(headers={"authorization": "different", "x-role": "viewer"})
    assert observe(with_auth) == observe(other_values)
    assert observe(with_auth).has_authorization and not observe(feats()).has_authorization


def test_body_and_query_are_presence_only():
    a = observe(feats(method="POST", path="/x?q=one", body="first body"))
    b = observe(feats(method="POST", path="/x?q=a-very-different-value", body="{}"))
    assert a == dataclasses.replace(b, declared_type=a.declared_type)
    assert a.has_body and a.has_query and not observe(feats()).has_body and not observe(feats()).has_query


def test_method_is_upper_cased():
    assert observe(feats(method="get")).method == "GET"


# -- applicability ---------------------------------------------------------------------------------------------------


AUTH = {"authorization": "Bearer x"}


@pytest.mark.parametrize(
    "request_kwargs,expected",
    [
        (dict(method="GET", path="/orders", headers=AUTH), ["SIM-P02"]),
        (dict(method="GET", path="/orders?page=1", headers=AUTH), ["SIM-P02", "SIM-P13"]),
        (dict(method="GET", path="/orders"), ["SIM-P12"]),
        (dict(method="POST", path="/orders", body='{"a":1}', headers={**AUTH, "content-type": "application/json"}),
         ["SIM-P03", "SIM-P09"]),
        (dict(method="DELETE", path="/orders/5", headers=AUTH), ["SIM-P04"]),
        (dict(method="GET", path="/admin/orders/status", headers=AUTH), ["SIM-P01"]),
        (dict(method="POST", path="/auth/login", body="u=a&p=b",
              headers={"content-type": "application/x-www-form-urlencoded"}), ["SIM-P05", "SIM-P11"]),
        (dict(method="GET", path="/health"), ["SIM-P06"]),
        (dict(method="GET", path="/files/orders/x", headers=AUTH), ["SIM-P07"]),
        (dict(method="GET", path="/documents/orders/1/a.final.pdf"), ["SIM-P08", "SIM-P12"]),  # no authorization header
        (dict(method="GET", path="/documents/orders/1/a.final.pdf", headers=AUTH), ["SIM-P08"]),
        (dict(method="PUT", path="/orders", body="plain words here", headers={**AUTH, "content-type": "text/plain"}),
         ["SIM-P03", "SIM-P10"]),
        (dict(method="PATCH", path="/admin/orders/status", body='{"a":1}',
              headers={"content-type": "application/json"}), ["SIM-P01", "SIM-P09", "SIM-P12"]),
        (dict(method="OPTIONS", path="/orders", headers=AUTH), []),
    ],
)
def test_applicability_for_crafted_requests(request_kwargs, expected):
    assert ids(feats(**request_kwargs)) == expected


def test_applicability_is_in_catalog_order_with_no_duplicates(pilot):
    order = {p.rule_id: i for i, p in enumerate(POLICIES)}
    for r in pilot.records:
        got = simulate_rules_for_record(r).policy_ids
        assert list(got) == sorted(got, key=order.get)
        assert len(set(got)) == len(got)


def test_zero_coverage_policy_exists_when_the_data_has_no_matching_request(pilot):
    assigned = diag.rule_assignments(pilot.records)
    used = {pid for rules in assigned.values() for pid in rules}
    assert "SIM-P12" not in used  # no anonymous request reaches a protected path in the pilot; reported, not tuned


# -- determinism and independence from labels / metadata ----------------------------------------------------------------


def test_assignments_are_deterministic_and_independent_of_header_order(small):
    for r in small.records:
        one, two = simulate_rules_for_record(r), simulate_rules_for_record(r)
        assert one == two
    shuffled_headers = tuple(reversed(small.records[0].features.headers))
    flipped = dataclasses.replace(small.records[0].features, headers=shuffled_headers)
    assert ids(flipped) == ids(small.records[0].features)


def test_assignments_do_not_depend_on_labels_ids_sources_timestamps_or_family(small):
    base = diag.rule_assignments(small.records)
    changed = []
    for i, r in enumerate(small.records):
        changed.append(make_record(f"x{i:06d}", method=r.features.method, path=r.features.path, label=1 - r.meta.label,
                                   body=r.features.body, headers=dict(r.features.headers),
                                   timestamp=float(len(small.records) - i), source_id=f"zzz{i % 3}"))
    after = diag.rule_assignments(changed)
    assert [base[r.record_id] for r in small.records] == [after[r.record_id] for r in changed]


def test_simulator_never_reads_record_metadata():
    class Booby:
        def __getattribute__(self, name):
            raise AssertionError("record.meta must not be read")

    record = EvalRecord(record_id="rbooby", features=feats(path="/orders", headers=AUTH), meta=Booby())  # type: ignore[arg-type]
    assert simulate_rules_for_record(record).policy_ids == ("SIM-P02",)


def test_simulator_source_never_accesses_label_or_metadata_attributes():
    tree = ast.parse(Path(sim.__file__).read_text(encoding="utf-8"))
    forbidden = {"meta", "label", "source_id", "timestamp", "family", "template_id", "metadata", "variant_index",
                 "hard_case", "label_kind"}
    accessed = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (accessed & forbidden), accessed & forbidden


def test_simulate_rules_signature_accepts_only_an_event_id_and_request_features():
    import inspect

    assert list(inspect.signature(simulate_rules).parameters) == ["event_id", "features"]
    assert list(inspect.signature(applicable_policies).parameters)[0] == "features"
    for bad in (make_record("r", method="GET", path="/x", label=1), SplitMetadata(label=1), {"method": "GET"}, "GET /x", None):
        with pytest.raises(TypeError):
            simulate_rules("e1", bad)  # type: ignore[arg-type]


def test_family_metadata_cannot_influence_the_result(small):
    # Same features, different generation metadata: simulate_rules has no way to receive it.
    r = small.records[0]
    assert simulate_rules(r.record_id, r.features) == simulate_rules(r.record_id, r.features)
    assert small.metadata[r.record_id].family in cat.FAMILY_BY_NAME


# -- malformed input ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        RequestFeatures(method="", path="/x"),
        RequestFeatures(method="   ", path="/x"),
        RequestFeatures(method=None, path="/x"),  # type: ignore[arg-type]
        RequestFeatures(method="GET", path="no-slash"),
        RequestFeatures(method="GET", path=""),
        RequestFeatures(method="GET", path=None),  # type: ignore[arg-type]
        RequestFeatures(method="GET", path="/x", body=5),  # type: ignore[arg-type]
        RequestFeatures(method="GET", path="/x", headers=5),  # type: ignore[arg-type]
        RequestFeatures(method="GET", path="/x", headers=(1, 2, 3)),  # type: ignore[arg-type]
    ],
)
def test_malformed_requests_are_rejected_without_echoing_content(bad):
    with pytest.raises(PolicySimError) as caught:
        simulate_rules("e1", bad)
    assert "no-slash" not in str(caught.value)


@pytest.mark.parametrize("event_id", ["", " ", " lead", "trail ", None, 5])
def test_invalid_event_ids_are_rejected(event_id):
    with pytest.raises(PolicySimError):
        simulate_rules(event_id, feats())  # type: ignore[arg-type]


def test_simulate_rules_for_record_requires_an_eval_record():
    with pytest.raises(TypeError):
        simulate_rules_for_record(feats())  # type: ignore[arg-type]


# -- output format and provenance -------------------------------------------------------------------------------------------


def test_output_format_and_explicit_simulated_provenance():
    result = simulate_rules("evt-1", feats(path="/orders?page=1", headers=AUTH))
    assert isinstance(result, SimulatedRuleSet) and isinstance(result.rule_info, VerifiedRuleInfo)
    assert result.rule_info.event_id == "evt-1"
    assert [r.rule_id for r in result.rule_info.rules] == list(result.policy_ids) == ["SIM-P02", "SIM-P13"]
    for rule in result.rule_info.rules:
        assert isinstance(rule, VerifiedRule) and set(rule.attributes) == set(ALLOWED_ATTRIBUTES)
    assert result.provenance == "simulated" and result.verification_status == NOT_VERIFIED_STATUS
    assert result.catalog_version == CATALOG_VERSION and result.catalog_sha256 == CATALOG_SHA256


def test_no_rule_or_attribute_claims_verification():
    for r in generate(GeneratorConfig(seed=2, n_templates=30)).records:
        result = simulate_rules_for_record(r)
        text = json.dumps(result.rule_info.model_dump(mode="json")).lower()
        assert "verified" not in text and "formal" not in text and "z3" not in text and "member 2" not in text
        assert result.verification_status == "not_verified_simulated"


def test_rule_ids_are_unique_within_every_request(small):
    for r in small.records:
        rules = simulate_rules_for_record(r).rule_info.rules
        assert len({x.rule_id for x in rules}) == len(rules)


def test_an_empty_rule_list_is_valid_and_is_not_a_verification_claim():
    result = simulate_rules("evt-2", feats(method="OPTIONS", path="/orders", headers=AUTH))
    assert result.rule_info.rules == [] and result.verification_status == NOT_VERIFIED_STATUS


def test_the_api_rule_source_cannot_represent_simulated_rules_and_the_simulator_stays_out_of_the_api():
    from tokenization_ai.app import RuleSource

    assert {s.value for s in RuleSource} == {"none", "mock_fixture", "verified"}  # no 'simulated' value exists
    for name in ("app.py", "service.py", "demo.py", "adapters.py", "policy_report_adapter.py", "anomaly_signal.py"):
        text = (Path(sim.__file__).parent.parent / name).read_text(encoding="utf-8")
        assert "policy_sim" not in text and "heldout_eval" not in text, name


# -- isolation ------------------------------------------------------------------------------------------------------------------


def test_no_member2_code_or_network_is_used(monkeypatch, small):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for r in small.records:
        simulate_rules_for_record(r)


def test_importing_and_running_the_simulator_loads_no_member_service_modules():
    """Checked in a fresh interpreter: other tests legitimately import other members' modules in this process."""
    import os
    import subprocess

    root = Path(sim.__file__).resolve().parents[2]
    code = "\n".join([
        "import json, sys",
        f"sys.path.insert(0, {str(root)!r})",
        "import tokenization_ai.heldout_eval.policy_sim as s, tokenization_ai.heldout_eval.policy_diagnostics",
        "from tokenization_ai.heldout_eval.splits import make_record",
        "r = make_record('r1', method='GET', path='/orders?page=1', label=0, headers={'authorization': 'x'})",
        "s.simulate_rules_for_record(r)",
        "banned = ('src', 'policy_verifier', 'api', 'api_gateway', 'engine', 'ingestion', 'processing', 'httpx', 'requests', 'redis')",
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in banned)))",
    ])
    env = {"PYTHONDONTWRITEBYTECODE": "1", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=root, capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_simulator_imports_only_the_standard_library_and_local_modules():
    tree = ast.parse(Path(sim.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert imported <= {"__future__", "hashlib", "json", "re", "collections.abc", "dataclasses", "typing",
                        "..models", ".", ".splits"}, imported


# -- compatibility with the existing Member 4 pipeline --------------------------------------------------------------------------------


def _context(record: EvalRecord) -> UnifiedApiContext:
    return diag.to_context(record)


def test_simulated_rules_fuse_mask_normalize_and_tokenize_with_literal_rule_roles(small):
    record = next(r for r in small.records if len(simulate_rules_for_record(r).policy_ids) >= 2)
    result = simulate_rules_for_record(record)
    fused = InputFusion().fuse(_context(record), result.rule_info)
    masked = PrivacyMasker().mask_with_provenance(fused)
    normalized = Normalizer().normalize_with_provenance(masked.fused, masked.masked_paths)
    seq = HybridTokenizer().tokenize(normalized.fused, normalized.masked_paths)
    rule_literals = [t for t, role in zip(seq.tokens, seq.roles) if role == "literal_rule"]
    for pid in result.policy_ids:
        assert pid in rule_literals
    assert seq.tokens.index("[RULES_START]") < seq.tokens.index(result.policy_ids[0]) < seq.tokens.index("[RULES_END]")


def test_rule_tokens_cost_twenty_tokens_per_rule_with_the_existing_tokenizer(small):
    def length(rule_info, record):
        fused = InputFusion().fuse(_context(record), rule_info)
        masked = PrivacyMasker().mask_with_provenance(fused)
        norm = Normalizer().normalize_with_provenance(masked.fused, masked.masked_paths)
        return len(HybridTokenizer().tokenize(norm.fused, norm.masked_paths).tokens)

    for record in small.records[:40]:
        result = simulate_rules_for_record(record)
        bare = VerifiedRuleInfo(event_id=record.record_id, rules=[])
        assert length(result.rule_info, record) - length(bare, record) == 20 * len(result.policy_ids)


def test_fusion_still_rejects_rules_for_a_different_event(small):
    record = small.records[0]
    other = simulate_rules("a-different-event", record.features)
    with pytest.raises(InputContractError):
        InputFusion().fuse(_context(record), other.rule_info)


def test_existing_tensorizer_rejects_instead_of_truncating_rules_that_cannot_fit(small):
    from tokenization_ai.tensorizer import Tensorizer
    from tokenization_ai.vocabulary import Vocabulary

    record = next(r for r in small.records if len(simulate_rules_for_record(r).policy_ids) == 2)
    fused = InputFusion().fuse(_context(record), simulate_rules_for_record(record).rule_info)
    masked = PrivacyMasker().mask_with_provenance(fused)
    norm = Normalizer().normalize_with_provenance(masked.fused, masked.masked_paths)
    seq = HybridTokenizer().tokenize(norm.fused, norm.masked_paths)
    vocabulary = Vocabulary()
    vocabulary.fit([seq])
    with pytest.raises(ValueError, match="refusing to drop rule"):
        Tensorizer(vocabulary, max_length=45).tensorize(seq)


# -- diagnostics (descriptive; not a model) ------------------------------------------------------------------------------------------------


def test_diagnostics_are_json_safe_deterministic_and_hold_no_raw_content_or_performance_metrics(small):
    one = diag.rule_diagnostics(small, include_token_budget=False)
    assert one == diag.rule_diagnostics(small, include_token_budget=False)
    text = json.dumps(one, allow_nan=False)
    for r in small.records[:30]:
        assert r.features.path not in text and "Bearer" not in text
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(str(k).lower())
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(one)
    assert not any(w in k for k in keys for w in ("accuracy", "auc", "precision", "recall", "f1", "roc"))
    assert one["catalog"]["provenance"] == "simulated" and one["catalog"]["verification_status"] == NOT_VERIFIED_STATUS
    assert "not proof" in one["note"] and "Descriptive" in one["note"]


def test_pilot_diagnostics_regression_pins(pilot):
    report = diag.rule_diagnostics(pilot)
    assert report["records"] == 1501 and report["catalog"]["sha256"] == PINNED_CATALOG_SHA256
    assert report["rules_per_request"]["histogram"] == {"1": 177, "2": 1324} or report["rules_per_request"]["histogram"] == {1: 177, 2: 1324}
    assert report["empty_rule_share"] == 0.0 and report["records_with_duplicate_policy_ids"] == 0
    assert report["policies_with_zero_coverage"] == ["SIM-P12"]
    flags = " ".join(report["attention_flags"])
    assert "SIM-P07" in flags and "SIM-P12" in flags  # surfaced, not tuned away
    budget = report["token_budget"]
    assert budget["rule_tokens_per_rule"] == 20
    assert budget["tokens_arm_b"]["mean"] > budget["tokens_arm_a"]["mean"]
    assert budget["by_max_length"]["128"]["B_simulated_rules"]["longer_than_max_length"] == 18
    assert budget["by_max_length"]["128"]["A_request_only"]["longer_than_max_length"] == 0
    assert budget["by_max_length"]["256"]["B_simulated_rules"]["longer_than_max_length"] == 0
    assert budget["by_max_length"]["128"]["records_rejected_in_either_arm"]["count"] == 0
    mean_rules = report["rules_per_request"]["mean"]
    assert budget["rule_tokens_per_request"]["mean"] == pytest.approx(20 * mean_rules, abs=0.02)


def test_diagnostics_flag_a_policy_that_applies_to_one_class_only(small):
    rigged_records = []
    for r in small.records:
        label = 1 if "SIM-P03" in simulate_rules_for_record(r).policy_ids else r.meta.label
        rigged_records.append(make_record(r.record_id, method=r.features.method, path=r.features.path, label=label,
                                          body=r.features.body, headers=dict(r.features.headers),
                                          timestamp=r.meta.timestamp, source_id=r.meta.source_id))
    rigged = dataclasses.replace(small, records=tuple(rigged_records))
    report = diag.rule_diagnostics(rigged, include_token_budget=False)
    assert report["per_policy"]["SIM-P03"]["class_0_records"] == 0
    assert any("SIM-P03" in f and "only one class" in f for f in report["attention_flags"])


def test_diagnostics_module_is_the_only_place_that_reads_labels():
    sim_text = Path(sim.__file__).read_text(encoding="utf-8")
    assert ".label" not in sim_text  # attribute access to a label; the AST test above checks this structurally too
    assert ".meta.label" in Path(diag.__file__).read_text(encoding="utf-8")
