"""Tests for the Arm C encoder armc-char3-v1 (draft protocol heldout-eval-1.2.0-draft, section 12.10).

Hand-made contexts and development-seed (6101) requests only. No model is fitted or scored, no detection metric is computed, and the
confirmatory seeds 7101-7103 are never generated.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import socket
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.arm_c as arm_c
from tokenization_ai.fusion import InputFusion
from tokenization_ai.heldout_eval.arm_c import (
    ENCODER_ID,
    ArmCEncodingError,
    char_ngram_ids,
    encode_request,
    encoder_parameters,
    serialize_masked_request,
)
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.request_context import request_to_context
from tokenization_ai.heldout_eval.splits import EvalRecord, SplitMetadata, make_record
from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import ModelInput, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.normalization import NormalizationResult, Normalizer


def _prepare(ctx, rules=None):
    fused = InputFusion().fuse(ctx, VerifiedRuleInfo(event_id=ctx.event_id, rules=rules or []))
    masked = PrivacyMasker().mask_with_provenance(fused)
    return Normalizer().normalize_with_provenance(masked.fused, masked.masked_paths)


def _ctx(**kw):
    base = dict(event_id="evt1", method="post", path="/Orders/5?Q=Abc", headers={"User-Agent": "T/1"}, payload={"b": 1, "a": "X"})
    base.update(kw)
    return UnifiedApiContext(**base)


def _record_result(record):
    return _prepare(request_to_context(record))


@pytest.fixture(scope="module")
def dev():
    return generate(GeneratorConfig(seed=6101, n_templates=120))  # development seed only


def _blake(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big") & 0x7FFFFFFF


# -- identifier and configuration --------------------------------------------------------------------------------------


def test_encoder_identifier_and_parameters_are_stable():
    assert ENCODER_ID == "armc-char3-v1"
    p = encoder_parameters()
    assert (p["ngram"], p["hash"], p["digest_size"], p["id_mask"]) == (3, "blake2b", 8, 0x7FFFFFFF)
    assert p["percent_decoding"] is False and p["padding"] is False and p["truncation"] is False and p["rules"] is False
    json.dumps(dict(p))


# -- serialization -----------------------------------------------------------------------------------------------------------


def test_exact_serialization_format():
    text = serialize_masked_request(_prepare(_ctx()))
    assert text == 'post /orders/5?q=abc\nuser-agent: t/1\n\n{"a":"x","b":1}'


def test_headers_are_sorted_canonically_regardless_of_insertion_order_or_case():
    a = _prepare(_ctx(headers={"X-Role": "admin", "accept": "*/*", "User-Agent": "T"}))
    b = _prepare(_ctx(headers={"user-agent": "T", "ACCEPT": "*/*", "x-role": "admin"}))
    assert serialize_masked_request(a) == serialize_masked_request(b)
    lines = serialize_masked_request(a).split("\n\n")[0].split("\n")[1:]
    assert [line.split(":")[0] for line in lines] == ["accept", "user-agent", "x-role"]


def test_payload_key_order_does_not_matter_at_any_depth():
    a = _prepare(_ctx(payload={"z": {"b": 1, "a": [1, {"d": 1, "c": 2}]}, "a": True}))
    b = _prepare(_ctx(payload={"a": True, "z": {"a": [1, {"c": 2, "d": 1}], "b": 1}}))
    assert serialize_masked_request(a) == serialize_masked_request(b) and encode_request(a) == encode_request(b)


def test_ascii_only_lowercasing_leaves_other_characters_unchanged():
    text = serialize_masked_request(_prepare(_ctx(path="/ÉCOLE/AbC", payload={"k": "ÀÉ ABC"})))
    head, body = text.split("\n\n")
    assert head.split("\n")[0] == "post /École/abc"  # A-Z folded; É kept (str.lower() would have changed it)
    assert body == r'{"k":"\u00c0\u00c9 abc"}'  # payload: non-ASCII escaped by JSON, ASCII letters folded
    assert body.isascii()


def test_percent_encoding_is_not_decoded_only_lowercased():
    text = serialize_masked_request(_prepare(_ctx(path="/o/Search?Term=%27+OR+1%3D1%2F..%2F")))
    assert "term=%27+or+1%3d1%2f..%2f" in text and "'" not in text.split("\n")[0]


def test_query_string_is_kept_inside_the_path_line_unsplit():
    first = serialize_masked_request(_prepare(_ctx(path="/a/b?x=1&y=2"))).split("\n")[0]
    assert first == "post /a/b?x=1&y=2"


def test_newlines_in_content_cannot_forge_lines_or_the_separator():
    ctx = _ctx(headers={"x-note": "a\r\nevil: 1\n\nbody"}, path="/p\nq", payload={"t": "l1\nl2\n\nl3"})
    text = serialize_masked_request(_prepare(ctx))
    assert text.count("\n\n") == 1 and "\r" not in text
    assert "x-note: a\\r\\nevil: 1\\n\\nbody" in text and "post /p\\nq" in text


def test_json_and_unicode_edge_cases():
    payload = {"emoji": "😀", "nul": None, "f": 1.5, "t": True, "big": 10**20, "nested": {"é": ["ü", {}]}, "empty": {}}
    text = serialize_masked_request(_prepare(_ctx(payload=payload)))
    body = text.split("\n\n", 1)[1]
    assert body.isascii()  # ensure_ascii: reproducible without Unicode tables
    assert json.loads(body)["big"] == 10**20 and json.loads(body)["f"] == 1.5
    assert serialize_masked_request(_prepare(_ctx(payload={}))).endswith("\n\n{}")


def test_payload_that_cannot_be_serialized_is_a_meaningful_failure():
    with pytest.raises(ArmCEncodingError):
        serialize_masked_request(_prepare(_ctx(payload={"k": b"bytes"})))


def test_serialization_is_independently_reproducible(dev):
    # An independent re-implementation of the documented format reproduces the encoder output for every request.
    for r in dev.records:
        res = _record_result(r)
        c = res.fused.context
        head = [f"{c.method} {c.path}"] + [f"{k}: {c.headers[k]}" for k in sorted(c.headers)]
        body = json.dumps(c.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        expected = "\n".join(head) + "\n\n" + body
        expected = "".join(chr(ord(ch) + 32) if "A" <= ch <= "Z" else ch for ch in expected)
        assert serialize_masked_request(res) == expected


# -- character 3-grams and hashing --------------------------------------------------------------------------------------------


def test_overlapping_trigrams_in_order_with_known_answers():
    assert char_ngram_ids("abcde") == [869635417, 769125405, 1199159168] == [_blake("abc"), _blake("bcd"), _blake("cde")]


def test_short_and_empty_input_yield_no_trigrams():
    assert char_ngram_ids("") == [] and char_ngram_ids("ab") == [] and len(char_ngram_ids("abc")) == 1
    assert len(char_ngram_ids("a" * 50)) == 48


def test_trigrams_are_code_point_windows_hashed_as_utf8():
    ids = char_ngram_ids("aé😀b")
    assert ids == [_blake("aé😀"), _blake("é😀b")]


def test_ids_are_stable_non_negative_ints_below_two_to_the_31():
    ids = char_ngram_ids("The quick brown fox jumps over the lazy dog " * 5)
    assert all(type(i) is int and 0 <= i <= 0x7FFFFFFF for i in ids)
    assert ids == char_ngram_ids("The quick brown fox jumps over the lazy dog " * 5)
    assert len(set(ids)) > 20


def test_ngram_function_validates_arguments():
    for bad in (None, 5, b"abc"):
        with pytest.raises(ArmCEncodingError):
            char_ngram_ids(bad)  # type: ignore[arg-type]
    for bad_n in (0, -1, True, 2.5):
        with pytest.raises(ArmCEncodingError):
            char_ngram_ids("abcd", bad_n)  # type: ignore[arg-type]
    assert len(char_ngram_ids("abcd", 2)) == 3


def test_builtin_hash_is_never_used():
    src = Path(arm_c.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "hash" not in calls and "blake2b" in src


# -- ModelInput and the unchanged model ------------------------------------------------------------------------------------


def test_model_input_shape_types_and_mask():
    mi = encode_request(_prepare(_ctx()))
    assert isinstance(mi, ModelInput) and mi.event_id == "evt1"
    assert len(mi.input_ids) == len(mi.attention_mask) > 0
    assert set(mi.attention_mask) == {1} and all(type(i) is int and i >= 0 for i in mi.input_ids)
    text = serialize_masked_request(_prepare(_ctx()))
    assert len(mi.input_ids) == len(text) - 2  # no hidden padding or truncation


def test_a_very_long_request_is_neither_truncated_nor_padded():
    big = _prepare(_ctx(payload={f"field{i:03d}": "value " * 10 for i in range(200)}))
    mi = encode_request(big)
    assert len(mi.input_ids) == len(serialize_masked_request(big)) - 2 > 10000


@pytest.mark.parametrize("buckets", [64, 1024, 4096])
def test_unchanged_model_feature_path_counts_unigrams_and_adjacent_pairs(buckets):
    mi = encode_request(_prepare(_ctx()))
    model = LightweightAnomalyModel(num_buckets=buckets)
    model._validate_model_input(mi)  # the existing validation accepts it
    feats = model._features(mi.input_ids, mi.attention_mask)
    ids = mi.input_ids
    expected = [0.0] * buckets
    for i in ids:
        expected[model._unigram_bucket(i)] += 1  # one character 3-gram
    for a, b in zip(ids, ids[1:]):
        expected[model._bigram_bucket(a, b)] += 1  # two adjacent, overlapping 3-grams = a 4-character span
    assert list(feats) == expected and feats.sum() == 2 * len(ids) - 1


def test_adjacent_trigram_pairs_span_four_characters():
    text = "abcd"
    a, b = char_ngram_ids(text)
    assert (a, b) == (_blake("abc"), _blake("bcd"))  # overlapping windows share "bc": the pair determines "abcd"


def test_encoding_neither_creates_nor_fits_model_weights():
    assert "model" not in {n.module for n in ast.walk(ast.parse(Path(arm_c.__file__).read_text(encoding="utf-8")))
                           if isinstance(n, ast.ImportFrom)}
    fresh = LightweightAnomalyModel(num_buckets=128)
    encode_request(_prepare(_ctx()))
    assert fresh.is_trained is False and not fresh.weights.any() and fresh.training_loss_history == []
    # fit() warm-starts from existing weights, so each future configuration must use a fresh instance (documented).
    other = LightweightAnomalyModel(num_buckets=128)
    assert other is not fresh and not other.weights.any()


def test_same_model_class_and_bucket_setting_serve_all_arms():
    mi = encode_request(_prepare(_ctx()))
    for buckets in (1024, 4096):
        LightweightAnomalyModel(num_buckets=buckets)._validate_model_input(mi)


# -- privacy boundary --------------------------------------------------------------------------------------------------


SENSITIVE = dict(
    path="/reset?token=tok-SECRET-123&page=2&email=a.person@example.com",
    headers={"Authorization": "Bearer abc.def.ghi-SECRET", "Cookie": "sid=SECRET-COOKIE", "X-Api-Key": "key-SECRET-9",
             "User-Agent": "T/1"},
    payload={"password": "hunter2-SECRET", "email": "a.person@example.com", "card": "4111 1111 1111 1111",
             "profile": {"ssn": "123-45-6789", "note": "ordinary note"}, "client_ip": "203.0.113.9"},
)
SECRETS = ("tok-secret-123", "ghi-secret", "secret-cookie", "key-secret-9", "hunter2", "a.person@example.com",
           "a.person%40example.com", "4111", "123-45-6789", "203.0.113.9", "abc.def.ghi")


def test_existing_masking_protects_sensitive_values_before_arm_c_encoding():
    ctx = _ctx(**SENSITIVE)
    assert any(s.lower() in json.dumps([ctx.path, ctx.headers, ctx.payload]).lower() for s in SECRETS)  # raw context is sensitive
    text = serialize_masked_request(_prepare(ctx))
    for s in SECRETS:
        assert s.lower() not in text, s
    assert "[masked_auth]" in text and "ordinary note" in text  # placeholders appear; ordinary content is kept


def test_an_unmasked_raw_context_is_refused_by_a_content_based_check():
    ctx = _ctx(**SENSITIVE)
    fused = InputFusion().fuse(ctx, VerifiedRuleInfo(event_id="evt1", rules=[]))
    raw_but_normalized = Normalizer().normalize_with_provenance(fused, frozenset())  # normalization only, masking skipped
    with pytest.raises(ArmCEncodingError) as info:
        encode_request(raw_but_normalized)
    assert "SECRET" not in str(info.value) and "hunter2" not in str(info.value)


def test_a_forged_provenance_set_does_not_make_an_unmasked_record_acceptable():
    ctx = _ctx(**SENSITIVE)
    fused = InputFusion().fuse(ctx, VerifiedRuleInfo(event_id="evt1", rules=[]))
    forged = NormalizationResult(fused=Normalizer().normalize(fused), masked_paths=frozenset({("headers", "authorization")}))
    with pytest.raises(ArmCEncodingError):
        encode_request(forged)


def test_a_masked_but_not_normalized_record_is_refused():
    fused = InputFusion().fuse(_ctx(headers={"User-Agent": "T"}), VerifiedRuleInfo(event_id="evt1", rules=[]))
    masked_only = PrivacyMasker().mask_with_provenance(fused)
    with pytest.raises(ArmCEncodingError):
        encode_request(NormalizationResult(fused=masked_only.fused, masked_paths=masked_only.masked_paths))


@pytest.mark.parametrize("bad", [None, {}, "text", 5])
def test_wrong_input_types_are_refused(bad):
    with pytest.raises(ArmCEncodingError):
        encode_request(bad)  # type: ignore[arg-type]


def test_records_contexts_and_fused_inputs_are_not_accepted_as_input():
    record = make_record("rid", method="GET", path="/x", label=1, headers={"user-agent": "T"})
    ctx = request_to_context(record)
    fused = InputFusion().fuse(ctx, VerifiedRuleInfo(event_id="rid", rules=[]))
    for bad in (record, ctx, fused):
        with pytest.raises(ArmCEncodingError):
            encode_request(bad)  # type: ignore[arg-type]


def test_arm_c_is_request_only_and_refuses_records_that_carry_rules():
    rule = VerifiedRule(rule_id="SIM-P01", attributes={"action": "manage", "resource": "admin-area", "effect": "Allow"})
    with pytest.raises(ArmCEncodingError):
        encode_request(_prepare(_ctx(), rules=[rule]))


def test_serialized_text_has_no_rule_tokens_or_structural_tags(dev):
    for r in dev.records[::9]:
        text = serialize_masked_request(_record_result(r))
        assert not re.search(r"\[(rules?|api|seq)[_:\]]|sim-p\d\d|\[rule", text)


# -- feature isolation ---------------------------------------------------------------------------------------------------------


def test_labels_family_and_metadata_never_influence_or_enter_the_encoding(dev):
    for r in dev.records[::5]:
        flipped = EvalRecord(r.record_id, r.features, SplitMetadata(label=1 - r.meta.label, timestamp=9e9, source_id="zz"))
        assert encode_request(_record_result(flipped)) == encode_request(_record_result(r))
        text = serialize_masked_request(_record_result(r))
        assert r.record_id not in text and (r.meta.source_id or "\0") not in text
        for banned in ("synthgen", "synthetic-catalog", "template", "scenario"):
            assert banned not in text


def test_event_id_is_only_the_model_input_identifier():
    a = encode_request(_prepare(_ctx(event_id="alpha")))
    b = encode_request(_prepare(_ctx(event_id="beta")))
    assert a.event_id == "alpha" and b.event_id == "beta" and a.input_ids == b.input_ids


def test_encoder_code_reads_no_label_or_metadata_and_has_no_network_or_member2_dependency():
    tree = ast.parse(Path(arm_c.__file__).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"label", "meta", "timestamp", "source_id", "record_id", "features", "predict", "fit"}
    imported = {n.module.split(".")[0] if n.module else "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported <= {"__future__", "hashlib", "json", "collections", "", "masking", "models", "normalization"} | {""}
    assert not imported & {"socket", "requests", "httpx", "urllib", "shared_contracts", "policy_sim", "splits", "generator"}


def test_encoding_does_not_use_the_network(monkeypatch, dev):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    encode_request(_record_result(dev.records[0]))


def test_input_is_not_mutated_and_encoding_is_deterministic(dev):
    for r in dev.records[:100]:
        res = _record_result(r)
        before = copy.deepcopy(res)
        one, two = encode_request(res), encode_request(res)
        assert one == two and res == before
        one.input_ids.clear()  # mutating an output cannot affect later encodings
        assert encode_request(res) == two


# -- development-seed preprocessing checks (label-free) ---------------------------------------------------------------


def test_every_development_request_encodes_without_leaking_credentials(dev):
    n_auth = 0
    for r in dev.records:
        res = _record_result(r)
        text = serialize_masked_request(res)
        mi = encode_request(res)
        assert len(mi.input_ids) == len(text) - 2 and max(mi.input_ids) <= 0x7FFFFFFF
        assert "syn-" not in text and not re.search(r"fx[a-z]{8}zq", text)
        if "\nauthorization:" in text:
            n_auth += 1
            assert "authorization: [masked_auth]" in text
    assert n_auth > 0
