"""Tests for the paired A/B/C arm builders (arms-1; draft protocol heldout-eval-1.2.0-draft).

Small hand-made records and development-seed (6101) requests only. No model is fitted or scored, no detection metric is computed, and the
confirmatory seeds 7101-7103 are never generated.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.arms as arms
from tokenization_ai.fusion import InputFusion
from tokenization_ai.heldout_eval import arm_c as arm_c_module
from tokenization_ai.heldout_eval.arm_c import encode_request, serialize_masked_request
from tokenization_ai.heldout_eval.arms import (
    ARM_A,
    ARM_B,
    ARM_C,
    ArmBuildError,
    ArmInputs,
    PairedExclusionRequired,
    PairingError,
    assert_disjoint,
    build_arm_inputs,
    build_provenance,
    fit_shared_vocabulary,
    new_untrained_model,
    prepare_split,
    propose_exclusions,
    training_arm_b_lengths,
    unknown_token_report,
)
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.policy_sim import CATALOG_SHA256, POLICIES, simulate_rules_for_record
from tokenization_ai.heldout_eval.request_context import request_to_context
from tokenization_ai.heldout_eval.splits import EvalRecord, SplitMetadata, make_record, split_records
from tokenization_ai.heldout_eval.token_budget import select_shared_max_length
from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import VerifiedRuleInfo
from tokenization_ai.normalization import NormalizationResult, Normalizer
from tokenization_ai.tokenizer import HybridTokenizer


def _rec(rid, path="/orders/5", method="GET", body="", headers=None, label=0):
    return make_record(rid, method=method, path=path, label=label, body=body,
                       headers=headers if headers is not None else {"user-agent": "T/1"}, timestamp=1.0, source_id="s1")


@pytest.fixture(scope="module")
def dev():
    return generate(GeneratorConfig(seed=6101, n_templates=120))  # development seed only


@pytest.fixture(scope="module")
def dev_splits(dev):
    res = split_records(dev.records, seed=4401, strict=False)
    by = {r.record_id: r for r in dev.records}
    return {s: [by[i] for i in res.members(s)] for s in ("train", "validation", "test")}


@pytest.fixture(scope="module")
def prepared(dev_splits):
    return {s: prepare_split(recs, s) for s, recs in dev_splits.items()}


@pytest.fixture(scope="module")
def frozen(prepared):
    return fit_shared_vocabulary(prepared["train"])


@pytest.fixture(scope="module")
def decision(prepared):
    return select_shared_max_length(training_arm_b_lengths(prepared["train"]))


def _independent_front(record, rules):
    ctx = request_to_context(record)
    fused = InputFusion().fuse(ctx, rules if rules is not None else VerifiedRuleInfo(event_id=ctx.event_id, rules=[]))
    masked = PrivacyMasker().mask_with_provenance(fused)
    norm = Normalizer().normalize_with_provenance(masked.fused, masked.masked_paths)
    return norm, HybridTokenizer().tokenize(norm.fused, norm.masked_paths)


# -- identifiers --------------------------------------------------------------------------------------------------------


def test_versioned_identifiers():
    assert arms.ARMS_VERSION == "arms-1" and (ARM_A, ARM_B, ARM_C) == ("A_request_only", "B_simulated_rules", "C_char3")
    assert arms.MIN_FREQUENCY_PROPOSED == 2
    assert CATALOG_SHA256 == "79273ba6dac5cfa453e258024c2a9e4510e041b68c8db8b79781c52b4de1ba79" and len(POLICIES) == 13


# -- the three arms --------------------------------------------------------------------------------------------------------


def test_arm_a_has_no_rules_and_matches_an_independent_front_end(prepared, dev_splits):
    for p, rec in zip(prepared["train"].requests, sorted(dev_splits["train"], key=lambda r: r.record_id)):
        seq = p.sequence_a
        assert seq.tokens[-3:] == ["[RULES_START]", "[RULES_END]", "[SEQ_END]"] and "[RULE_START]" not in seq.tokens
        assert seq == _independent_front(rec, None)[1]


def test_arm_b_receives_only_simpol_1_rules_and_keeps_simulated_provenance(prepared, dev_splits):
    catalog_ids = {p.rule_id for p in POLICIES}
    n_with_rules = 0
    for p, rec in zip(prepared["train"].requests, sorted(dev_splits["train"], key=lambda r: r.record_id)):
        rs = simulate_rules_for_record(rec)
        assert p.rule_set == rs
        assert (rs.provenance, rs.verification_status, rs.catalog_version, rs.catalog_sha256) == (
            "simulated", "not_verified_simulated", "simpol-1", CATALOG_SHA256)
        assert set(p.rule_set.policy_ids) <= catalog_ids
        assert p.sequence_b.tokens.count("[RULE_START]") == len(rs.policy_ids)
        assert p.sequence_b == _independent_front(rec, rs.rule_info)[1]
        n_with_rules += bool(rs.policy_ids)
        # Arm B differs from Arm A only by the rule tokens
        a, b = p.sequence_a.tokens, p.sequence_b.tokens
        assert b[:len(a) - 3] == a[:-3] and len(b) - len(a) == 20 * len(rs.policy_ids)
    assert n_with_rules > 0


def test_arm_c_gets_the_rule_free_real_normalization_result_and_no_rules(prepared, dev_splits, monkeypatch):
    seen = []
    real = arms.encode_request

    def spy(result):
        seen.append(result)
        return real(result)

    monkeypatch.setattr(arms, "encode_request", spy)
    recs = dev_splits["validation"][:15]
    out = prepare_split(recs, "validation")
    assert len(seen) == len(recs)
    for result in seen:
        assert isinstance(result, NormalizationResult) and result.fused.rule_info.rules == []
    for p, rec in zip(out.requests, sorted(recs, key=lambda r: r.record_id)):
        norm, _ = _independent_front(rec, None)
        assert p.model_input_c == encode_request(norm)
        text = serialize_masked_request(norm)
        assert "[rule" not in text and "sim-p" not in text


def test_real_stages_are_used_in_order_for_every_arm(dev_splits, monkeypatch):
    calls = []
    for cls, name in ((InputFusion, "fuse"), (PrivacyMasker, "mask_with_provenance"),
                      (Normalizer, "normalize_with_provenance"), (HybridTokenizer, "tokenize")):
        original = getattr(cls, name)

        def make(orig, label):
            def wrapper(self, *a, **k):
                calls.append(label)
                return orig(self, *a, **k)
            return wrapper

        monkeypatch.setattr(cls, name, make(original, name))
    prepare_split(dev_splits["test"][:1], "test")
    assert calls.count("fuse") == 2 and calls.count("tokenize") == 2 and calls.count("normalize_with_provenance") == 2
    assert calls.count("mask_with_provenance") >= 2  # two front ends, plus the Arm C privacy-boundary re-check
    assert calls[:4] == ["fuse", "mask_with_provenance", "normalize_with_provenance", "tokenize"]  # the production stage order


# -- pairing, determinism, split identity ------------------------------------------------------------------------------------


def test_one_representation_per_arm_for_exactly_the_same_records(prepared, frozen, decision, dev_splits):
    for name, recs in dev_splits.items():
        inputs = build_arm_inputs(prepared[name], frozen, decision)
        ids = tuple(sorted(r.record_id for r in recs))
        assert inputs.record_ids == ids == prepared[name].record_ids
        for arm in (inputs.arm_a, inputs.arm_b, inputs.arm_c):
            assert tuple(m.event_id for m in arm) == ids
        assert len(inputs.arm_a) == len(inputs.arm_b) == len(inputs.arm_c) == len(recs)


def test_paired_mismatches_are_detected(prepared, frozen, decision):
    ok = build_arm_inputs(prepared["test"], frozen, decision)
    base = dict(split="test", record_ids=ok.record_ids, arm_a=ok.arm_a, arm_b=ok.arm_b, arm_c=ok.arm_c, max_length=ok.max_length)
    ArmInputs(**base)
    with pytest.raises(PairingError):
        ArmInputs(**{**base, "arm_c": ok.arm_c[:-1]})                      # a missing record in one arm
    with pytest.raises(PairingError):
        ArmInputs(**{**base, "arm_b": ok.arm_b[::-1]})                      # same records, different order
    with pytest.raises(PairingError):
        ArmInputs(**{**base, "record_ids": ok.record_ids[:-1] + (ok.record_ids[0],)})  # duplicate id
    with pytest.raises(PairingError):
        ArmInputs(**{**base, "max_length": ok.max_length + 1})              # A/B not at the shared budget
    with pytest.raises(PairingError):
        ArmInputs(**{**base, "outcomes_a": ("fits",)})


def test_deterministic_and_independent_of_input_order(dev_splits):
    recs = dev_splits["validation"][:40]
    one, two = prepare_split(recs, "validation"), prepare_split(list(reversed(recs)), "validation")
    assert one == two and one == prepare_split(recs, "validation")
    assert one.record_ids == tuple(sorted(r.record_id for r in recs))


def test_split_identity_is_explicit_and_input_is_validated():
    r = _rec("a")
    for bad in ("training", "", "Train", None):
        with pytest.raises(ArmBuildError):
            prepare_split([r], bad)  # type: ignore[arg-type]
    with pytest.raises(ArmBuildError):
        prepare_split([], "train")
    with pytest.raises(ArmBuildError):
        prepare_split([r, _rec("a", "/other")], "train")  # duplicate ids
    with pytest.raises(ArmBuildError):
        prepare_split([r, "not a record"], "train")  # type: ignore[list-item]


def test_splits_stay_separate(prepared):
    assert_disjoint(prepared["train"], prepared["validation"], prepared["test"])
    with pytest.raises(PairingError):
        assert_disjoint(prepared["train"], prepared["train"])
    twin = prepare_split([_rec(prepared["train"].record_ids[0])], "test")
    with pytest.raises(PairingError):
        assert_disjoint(prepared["train"], twin)


def test_labels_and_metadata_cannot_influence_any_arm(dev_splits):
    recs = dev_splits["test"][:30]
    flipped = [EvalRecord(r.record_id, r.features, SplitMetadata(label=1 - r.meta.label, timestamp=9e9, source_id="zz")) for r in recs]
    assert prepare_split(flipped, "test") == prepare_split(recs, "test")
    tree = ast.parse(Path(arms.__file__).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"meta", "label", "family", "timestamp", "source_id", "template_id", "features", "predict"}


# -- privacy ----------------------------------------------------------------------------------------------------------------------


def _sensitive():
    return _rec("sens1", "/reset?token=SECRETQUERY&page=2", "POST",
                '{"password":"hunter2SECRET","email":"a.person@example.com","note":"ordinary"}',
                {"authorization": "Bearer SECRETTOKEN-abc", "cookie": "sid=SECRETCOOKIE", "content-type": "application/json",
                 "x-api-key": "SECRETKEY-9"})


SECRETS = ("SECRETQUERY", "hunter2SECRET", "a.person@example.com", "SECRETTOKEN", "SECRETCOOKIE", "SECRETKEY")


def test_sensitive_values_are_masked_in_every_arm_before_any_feature_is_built():
    p = prepare_split([_sensitive()], "train").requests[0]
    for s in SECRETS:
        assert s.lower() not in " ".join(p.sequence_a.tokens).lower() and s.lower() not in " ".join(p.sequence_b.tokens).lower()
    assert "[API:HEADER_VALUE_MASKED]" in p.sequence_a.tokens
    text = serialize_masked_request(_independent_front(_sensitive(), None)[0])
    for s in SECRETS:
        assert s.lower() not in text
    assert p.model_input_c == encode_request(_independent_front(_sensitive(), None)[0])


def test_errors_and_diagnostics_never_echo_request_content(monkeypatch):
    def boom(record):
        raise ValueError("leaked: hunter2SECRET SECRETTOKEN")

    monkeypatch.setattr(arms, "request_to_context", boom)
    with pytest.raises(ArmBuildError) as info:
        prepare_split([_sensitive()], "train")
    assert "hunter2SECRET" not in str(info.value) and "SECRETTOKEN" not in str(info.value) and "sens1" in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__ is True


def test_public_summaries_and_provenance_hold_no_request_content(frozen):
    sp = prepare_split([_sensitive(), _rec("n1", "/orders/1", headers={"user-agent": "T/1"})], "train")
    fz = fit_shared_vocabulary(sp)
    dec = select_shared_max_length(training_arm_b_lengths(sp))
    blob = json.dumps([sp.summary(), build_provenance([sp], fz, dec), unknown_token_report(sp, fz),
                       build_arm_inputs(sp, fz, dec).summary()], allow_nan=False)
    for s in SECRETS + ("Bearer", "/reset", "ordinary"):
        assert s.lower() not in blob.lower(), s


# -- shared vocabulary (train only) ---------------------------------------------------------------------------------------------


def _vocab_tokens(frozen_vocab):
    return set(frozen_vocab.vocabulary.decode(list(range(len(frozen_vocab.vocabulary)))))


def test_vocabulary_can_only_be_fitted_on_the_training_split(prepared):
    for name in ("validation", "test"):
        with pytest.raises(ArmBuildError):
            fit_shared_vocabulary(prepared[name])
    with pytest.raises(ArmBuildError):
        fit_shared_vocabulary("train")  # type: ignore[arg-type]


def test_no_validation_or_test_literal_enters_the_vocabulary():
    train = prepare_split([_rec("t1", "/orders/alpha"), _rec("t2", "/orders/alpha"), _rec("t3", "/orders/alpha")], "train")
    val = prepare_split([_rec("v1", "/orders/zzvalonly"), _rec("v2", "/orders/zzvalonly")], "validation")
    test = prepare_split([_rec("x1", "/orders/zztestonly"), _rec("x2", "/orders/zztestonly")], "test")
    fz = fit_shared_vocabulary(train, min_frequency=2)
    tokens = _vocab_tokens(fz)
    assert "alpha" in tokens and "zzvalonly" not in tokens and "zztestonly" not in tokens
    before = (len(fz.vocabulary), fz.record["sha256"])
    dec = select_shared_max_length(training_arm_b_lengths(train))
    for sp in (val, test):
        build_arm_inputs(sp, fz, dec)
        rep = unknown_token_report(sp, fz)
        assert rep[ARM_A]["unknown_tokens"] >= 2 and rep["vocabulary_sha256"] == fz.record["sha256"]
    assert (len(fz.vocabulary), fz.record["sha256"]) == before  # nothing was refitted by encoding other splits
    unk = fz.vocabulary.unk_id
    ids = fz.vocabulary.encode(val.requests[0].sequence_a)
    assert ids[val.requests[0].sequence_a.tokens.index("zzvalonly")] == unk


def test_min_frequency_counts_each_request_once_not_once_per_arm():
    train = prepare_split([_rec("t1", "/orders/common"), _rec("t2", "/orders/common"), _rec("t3", "/orders/single")], "train")
    fz = fit_shared_vocabulary(train, min_frequency=2)
    tokens = _vocab_tokens(fz)
    # "single" occurs in exactly one request (although that request appears in both Arm A and Arm B): it must be unknown.
    assert "common" in tokens and "single" not in tokens
    assert fz.record["min_frequency"] == 2 and fz.record["min_frequency_status"] == "proposed_not_frozen"
    assert fit_shared_vocabulary(train, min_frequency=1).record["min_frequency_status"] == "non_default_setting"
    assert "single" in _vocab_tokens(fit_shared_vocabulary(train, min_frequency=1))


def test_vocabulary_is_deterministic_and_independent_of_record_order(dev_splits):
    recs = dev_splits["train"][:80]
    a = fit_shared_vocabulary(prepare_split(recs, "train"))
    b = fit_shared_vocabulary(prepare_split(list(reversed(recs)), "train"))
    assert a.record == b.record and a.vocabulary.pad_id == b.vocabulary.pad_id and a.vocabulary.unk_id == b.vocabulary.unk_id
    assert re.fullmatch(r"[0-9a-f]{64}", a.record["sha256"]) and a.record["fitted_on_split"] == "train"
    assert a.vocabulary.decode([a.vocabulary.pad_id, a.vocabulary.unk_id]) == ["[PAD]", "[UNK]"]


def test_unknown_tokens_are_exposed_in_training_and_counted_equally_for_a_and_b(prepared, frozen):
    rep = unknown_token_report(prepared["train"], frozen)
    assert rep[ARM_A]["unknown_tokens"] > 0  # min_frequency=2 exposes the unknown token during training
    assert rep[ARM_A]["unknown_tokens"] == rep[ARM_B]["unknown_tokens"]  # rules add only known tokens: one shared vocabulary
    assert rep[ARM_B]["literal_tokens"] > rep[ARM_A]["literal_tokens"]


def test_arm_c_has_no_vocabulary_stage():
    tree = ast.parse(Path(arm_c_module.__file__).read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "vocabulary" not in modules and "tensorizer" not in modules


# -- shared budget -----------------------------------------------------------------------------------------------------------


def test_one_selected_budget_applies_to_both_arms_and_is_not_hard_coded(prepared, frozen, decision):
    inputs = build_arm_inputs(prepared["validation"], frozen, decision)
    for m in inputs.arm_a + inputs.arm_b:
        assert len(m.input_ids) == len(m.attention_mask) == decision.selected_max_length == inputs.max_length
    consts = {n.value for n in ast.walk(ast.parse(Path(arms.__file__).read_text(encoding="utf-8")))
              if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool)}
    assert not consts & {128, 192, 256, 384, 512}
    failed = select_shared_max_length({"r": 999})
    with pytest.raises(ArmBuildError):
        build_arm_inputs(prepared["validation"], frozen, failed)
    with pytest.raises(ArmBuildError):
        build_arm_inputs(prepared["validation"], frozen, "192")  # type: ignore[arg-type]


def _wide(rid, n=60):
    return _rec(rid, "/orders", "POST", json.dumps({f"field{i:02d}": f"value{i}" for i in range(n)}),
                {"user-agent": "T/1", "content-type": "application/json"})


def test_payload_truncation_is_allowed_and_counted_never_hidden():
    train = prepare_split([_rec(f"t{i}", f"/orders/{i}") for i in range(5)], "train")
    dec = select_shared_max_length(training_arm_b_lengths(train))
    assert dec.selected_max_length == 128
    test = prepare_split([_wide("w1"), _rec("w2", "/orders/9")], "test")
    fz = fit_shared_vocabulary(train, min_frequency=1)
    inputs = build_arm_inputs(test, fz, dec)
    assert inputs.outcomes_b == ("fits", "truncated") or inputs.outcomes_b == ("truncated", "fits")
    assert inputs.summary()["arm_b"]["truncated"] == 1 and inputs.max_length == 128
    assert len(inputs.record_ids) == 2  # nothing dropped
    assert dec.selected_max_length == 128  # the frozen decision is untouched


def test_framing_rejections_fail_closed_with_an_id_only_proposal():
    train = prepare_split([_rec(f"t{i}", f"/orders/{i}") for i in range(5)], "train")
    dec = select_shared_max_length(training_arm_b_lengths(train))
    huge = _rec("h1", "/orders", "GET", headers={f"x-h{i:03d}": "v" for i in range(80)})
    test = prepare_split([huge, _rec("ok1", "/orders/1")], "test")
    prop = propose_exclusions(test, dec)
    assert [r.record_id for r in prop.records] == ["h1"] and set(prop.records[0].arms) == {ARM_A, ARM_B}
    assert prop.to_dict()["action"] == "proposal_only_nothing_excluded"
    with pytest.raises(PairedExclusionRequired) as info:
        build_arm_inputs(test, fit_shared_vocabulary(train, min_frequency=1), dec)
    assert info.value.proposal == prop and "h1" not in str(info.value) and "1 record(s)" in str(info.value)
    assert propose_exclusions(prepare_split([_rec("ok1", "/orders/1")], "test"), dec).records == ()


def test_training_lengths_for_selection_come_from_training_only(prepared):
    lengths = training_arm_b_lengths(prepared["train"])
    assert set(lengths) == set(prepared["train"].record_ids)
    with pytest.raises(ArmBuildError):
        training_arm_b_lengths(prepared["validation"])


# -- model preparation (no training) ---------------------------------------------------------------------------------------------


def test_model_factory_returns_fresh_untrained_instances_and_validates():
    a, b = new_untrained_model(256), new_untrained_model(256)
    assert isinstance(a, LightweightAnomalyModel) and a is not b
    assert a.num_buckets == 256 and a.is_trained is False and not a.weights.any() and a.training_loss_history == []
    for bad in (0, -1, True, 2.5, "8"):
        with pytest.raises(ArmBuildError):
            new_untrained_model(bad)  # type: ignore[arg-type]


def test_no_model_is_trained_or_scored_anywhere_in_the_module():
    tree = ast.parse(Path(arms.__file__).read_text(encoding="utf-8"))
    called = [(n.func.attr, ast.unparse(n.func.value)) for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not [c for c in called if c[0] in ("predict", "evaluate", "score")]
    fits = [c for c in called if c[0] == "fit"]
    assert fits == [("fit", "vocabulary")]  # the only fit is Vocabulary.fit on the training split


# -- provenance and boundaries --------------------------------------------------------------------------------------------------------


def test_provenance_records_versions_hashes_counts_and_simulated_status(prepared, frozen, decision):
    prov = build_provenance(list(prepared.values()), frozen, decision)
    json.dumps(prov, allow_nan=False)
    assert prov["arms"] == {"A": ARM_A, "B": ARM_B, "C": ARM_C}
    assert prov["request_to_context"] == "request_to_context-v1" and prov["arm_c_encoder"]["encoder_id"] == "armc-char3-v1"
    assert prov["generator_version"] == "synthgen-3" and prov["synthetic_catalog_version"] == "synthetic-catalog-3"
    rules = prov["arm_b_rules"]
    assert (rules["catalog_version"], rules["catalog_sha256"], rules["provenance"], rules["verification_status"]) == (
        "simpol-1", CATALOG_SHA256, "simulated", "not_verified_simulated")
    assert "not Member 2 verified" in rules["note"]
    assert prov["budget"]["rule_id"] == "budget-rule-1" and prov["vocabulary"]["sha256"] == frozen.record["sha256"]
    assert prov["paired_records"] == {s: len(p.requests) for s, p in prepared.items()} and prov["no_model_trained"] is True
    assert {"masking.py", "normalization.py", "tokenizer.py", "vocabulary.py", "tensorizer.py", "arms.py", "arm_c.py"} <= set(prov["source_sha256"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in prov["source_sha256"].values())
    assert '"verified"' not in json.dumps(prov)
    with pytest.raises(PairingError):
        build_provenance([prepared["train"], prepared["train"]])


def test_module_has_no_cross_member_or_network_dependencies():
    tree = ast.parse(Path(arms.__file__).read_text(encoding="utf-8"))
    imported = {n.module.split(".")[0] if n.module else "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    imported |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {"socket", "requests", "httpx", "urllib", "shared_contracts", "ingestion", "api_gateway", "processing", "engine"}
    relative = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level >= 1 and n.module}
    assert not relative & {"policy_report_adapter", "app", "service", "dashboard", "adapters"}


def test_development_workflow_end_to_end_without_training(prepared, frozen, decision):
    assert_disjoint(*prepared.values())
    assert decision.status == "selected" and frozen.record["fitted_on_split"] == "train"
    for name, sp in prepared.items():
        inputs = build_arm_inputs(sp, frozen, decision)
        assert inputs.summary()["paired_records"] == len(sp.requests)
        assert all(set(m.attention_mask) == {1} for m in inputs.arm_c)
