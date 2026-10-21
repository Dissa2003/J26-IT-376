"""Tests for request_to_context-v1 (draft protocol heldout-eval-1.2.0-draft, section 12.9).

Uses only fabricated requests from the generator (development seed 6101) and small hand-made records. No model is
trained and no detection performance is measured.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.request_context as rc
from tokenization_ai.adapters import to_unified_api_context
from tokenization_ai.fusion import InputFusion
from tokenization_ai.heldout_eval.generator import GeneratorConfig, generate
from tokenization_ai.heldout_eval.policy_diagnostics import to_context
from tokenization_ai.heldout_eval.request_context import (
    MAPPING_ID,
    ContextMappingError,
    request_to_context,
    source_sha256,
)
from tokenization_ai.heldout_eval.splits import EvalRecord, RequestFeatures, SplitMetadata, make_record
from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.models import UnifiedApiContext, VerifiedRuleInfo
from tokenization_ai.normalization import Normalizer
from tokenization_ai.tokenizer import HybridTokenizer


@pytest.fixture(scope="module")
def dev():
    return generate(GeneratorConfig(seed=6101, n_templates=250))  # development seed only


def _rec(rid="rid1", method="GET", path="/orders/5", body="", headers=None, label=0):
    return make_record(rid, method=method, path=path, label=label, body=body,
                       headers=headers if headers is not None else {"user-agent": "T/1"}, timestamp=1.0, source_id="s1")


def _sequence(record):
    ctx = request_to_context(record)
    fused = InputFusion().fuse(ctx, VerifiedRuleInfo(event_id=ctx.event_id, rules=[]))
    masked = PrivacyMasker().mask_with_provenance(fused)
    norm = Normalizer().normalize_with_provenance(masked.fused, masked.masked_paths)
    return norm, HybridTokenizer().tokenize(norm.fused, norm.masked_paths)


def test_mapping_id_is_versioned():
    assert MAPPING_ID == "request_to_context-v1"


# -- field mapping ---------------------------------------------------------------------------------------------------


def test_method_path_query_and_event_id_are_preserved_unsplit_and_undecoded():
    r = _rec("opaque123", "DELETE", "/orders/55?term=a%27+b&page=2")
    ctx = request_to_context(r)
    assert isinstance(ctx, UnifiedApiContext)
    assert (ctx.event_id, ctx.method) == ("opaque123", "DELETE")
    assert ctx.path == "/orders/55?term=a%27+b&page=2"  # whole string, query included, not decoded
    assert ctx.client_ip is None and ctx.threat_score is None and ctx.timestamp is None


def test_headers_are_copied_faithfully_and_content_type_is_derived_from_the_header():
    h = {"user-agent": "T/1", "content-type": "application/xml", "authorization": "Bearer syn-0123456789abcdef"}
    ctx = request_to_context(_rec(headers=h, body="<note>x y</note>", method="POST"))
    assert ctx.headers == h  # raw, including the credential: masking is a later stage
    assert ctx.content_type == "application/xml"
    assert request_to_context(_rec(headers={"user-agent": "T/1"})).content_type is None
    assert request_to_context(_rec(headers={"Content-Type": "text/plain"})).content_type == "text/plain"


def test_duplicate_header_names_are_rejected_not_merged():
    bad = EvalRecord("r1", RequestFeatures("GET", "/x", "", (("A", "1"), ("a", "2"))), SplitMetadata(0))
    with pytest.raises(ContextMappingError):
        request_to_context(bad)


# -- body conventions ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("body, expected", [
    ("", {}),
    ('{"b":2,"a":{"c":[1,2]}}', {"b": 2, "a": {"c": [1, 2]}}),
    ("name=amber&note=a+b&empty=", {"name": "amber", "note": "a b", "empty": ""}),
    ("amber river maple", {"text": "amber river maple"}),
    ("<note>amber river maple cedar</note>", {"text": "<note>amber river maple cedar</note>"}),
    ('[1,2,3]', {"text": "[1,2,3]"}),      # JSON but not an object
    ('"just a string"', {"text": '"just a string"'}),
    ("a=1 b=2", {"text": "a=1 b=2"}),      # contains a space: not form-like
])
def test_body_conventions(body, expected):
    assert request_to_context(_rec(method="POST", body=body)).payload == expected


def test_body_is_decided_by_content_not_by_the_declared_type():
    json_body = '{"a":1}'
    for declared in ("application/json", "application/xml", "text/plain", "application/x-www-form-urlencoded"):
        ctx = request_to_context(_rec(method="POST", body=json_body, headers={"content-type": declared}))
        assert ctx.payload == {"a": 1} and ctx.content_type == declared


# -- purity, determinism, immutability -------------------------------------------------------------------------------


def test_deterministic_and_does_not_mutate_the_input(dev):
    for r in dev.records[:200]:
        before = copy.deepcopy(r)
        one, two = request_to_context(r), request_to_context(r)
        assert one == two and r == before
        one.payload.clear()  # mutating a result must not touch the record
        assert r == before
    assert [request_to_context(r) for r in dev.records] == [request_to_context(r) for r in dev.records]


def test_result_depends_only_on_request_content_not_label_or_metadata(dev):
    for r in dev.records[:300]:
        other = EvalRecord(r.record_id, r.features, SplitMetadata(label=1 - r.meta.label, timestamp=9e9, source_id="zz"))
        assert request_to_context(other) == request_to_context(r)


def test_only_the_record_id_and_features_are_read():
    src = Path(rc.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "meta" not in attrs and "label" not in attrs and "timestamp" not in attrs and "source_id" not in attrs
    assert "record_id" in attrs and "features" in attrs
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"service", "model", "tensorizer", "vocabulary", "policy_sim", "generator", "audit"}


def test_no_simulated_rules_are_attached_by_the_adapter():
    ctx = request_to_context(_rec())
    assert not hasattr(ctx, "rules") and "rule" not in " ".join(UnifiedApiContext.model_fields)


# -- malformed requests ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, "GET /x", {"method": "GET"}, RequestFeatures("GET", "/x")])
def test_non_records_are_rejected(bad):
    with pytest.raises(ContextMappingError):
        request_to_context(bad)  # type: ignore[arg-type]


def test_wrong_field_types_and_empty_values_are_rejected_without_echoing_content():
    secret = "S3CR3T-PATH"
    cases = [
        EvalRecord("r", RequestFeatures(123, "/x"), SplitMetadata(0)),                       # type: ignore[arg-type]
        EvalRecord("r", RequestFeatures("GET", None), SplitMetadata(0)),                      # type: ignore[arg-type]
        EvalRecord("r", RequestFeatures("GET", "/x", 5), SplitMetadata(0)),                   # type: ignore[arg-type]
        EvalRecord("r", RequestFeatures("GET", "/x", "", (("a", 1),)), SplitMetadata(0)),     # type: ignore[arg-type]
        EvalRecord("r", RequestFeatures("GET", "/x", "", 7), SplitMetadata(0)),               # type: ignore[arg-type]
        EvalRecord("r", RequestFeatures("", "/" + secret), SplitMetadata(0)),                 # empty method
        EvalRecord("", RequestFeatures("GET", "/" + secret), SplitMetadata(0)),               # empty event id
    ]
    for bad in cases:
        with pytest.raises(ContextMappingError) as info:
            request_to_context(bad)
        assert secret not in str(info.value)


# -- compatibility with the existing stages and conventions ----------------------------------------------------------


def test_every_generated_request_flows_through_fusion_masking_normalization_and_tokenization(dev):
    for r in dev.records[::7]:
        norm, seq = _sequence(r)
        assert seq.tokens[0] == "[SEQ_START]" and seq.tokens[-1] == "[SEQ_END]" and len(seq.tokens) == len(seq.roles)


def test_equivalent_to_the_existing_diagnostic_mapping_for_every_generated_request(dev):
    assert len(dev.records) > 1000
    for r in dev.records:
        assert request_to_context(r) == to_context(r), r.record_id
    # The diagnostic mapping rebuilds the headers dict (content-type last); the dict content is identical.


def test_matches_the_production_adapter_convention_for_path_headers_and_event_id():
    r = _rec("evt-1", "POST", "/search?q=a&page=2", body='{"k":"v"}',
             headers={"content-type": "application/json", "user-agent": "T/1"})
    ours = request_to_context(r)
    prod = to_unified_api_context({
        "event_id": "evt-1",
        "metadata": {"method": "POST", "path": "/search?q=a&page=2", "content_type": "application/json",
                     "headers": {"content-type": "application/json", "user-agent": "T/1"}},
        "payload": {"k": "v"},
    })
    assert ours == prod  # same single-string path with its query, same header dict, same payload dict


def test_query_stays_inside_one_atomic_path_token_the_documented_limitation():
    _, seq = _sequence(_rec(path="/orders/search?q=amber+river&page=3"))
    segs = [seq.tokens[i + 1] for i, t in enumerate(seq.tokens) if t == "[API:PATH_SEG]"]
    assert segs == ["orders", "search?q=amber+river&page=3"]  # not split, not decoded
    _, enc = _sequence(_rec(path="/orders/search?term=%27+OR+1%3D1"))
    assert any(t == "search?term=%27+OR+1%3D1" for t in enc.tokens)  # not percent-decoded


def test_sensitive_values_are_handled_by_the_existing_masking_stage_not_the_adapter():
    r = _rec(method="POST", path="/reset?token=abc123&page=2", body='{"passphrase":"Fxsecretvalue","note":"hi"}',
             headers={"authorization": "Bearer syn-0123456789abcdef", "content-type": "application/json"})
    ctx = request_to_context(r)
    assert ctx.headers["authorization"] == "Bearer syn-0123456789abcdef" and ctx.payload["passphrase"] == "Fxsecretvalue"
    norm, seq = _sequence(r)
    text = " ".join(seq.tokens)
    assert "syn-0123456789abcdef" not in text and "Fxsecretvalue" not in text and "abc123" not in text
    assert "[API:HEADER_VALUE_MASKED]" in seq.tokens and ("headers", "authorization") in norm.masked_paths


def test_event_id_label_and_metadata_never_enter_feature_tokens(dev):
    for r in dev.records[::5]:
        _, seq = _sequence(r)
        joined = " ".join(seq.tokens)
        assert r.record_id not in joined and (r.meta.source_id or "\0") not in joined
        for banned in ("synthetic-catalog", "synthgen", "label", "template", "scenario"):
            assert banned not in joined.lower()


def test_same_mapped_request_serves_arm_a_and_arm_b_context_identically(dev):
    r = dev.records[0]
    a = request_to_context(r)
    b = request_to_context(r)
    assert a == b and a.event_id == r.record_id  # rules are attached later, per arm, by the arm builders


def test_source_hash_is_stable_across_line_endings():
    import hashlib

    raw = Path(rc.__file__).read_bytes()
    assert source_sha256() == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() and len(source_sha256()) == 64
