"""Tests for the Member 4 KPI 1-3 evaluation framework.

These verify the *framework*: fixture integrity, oracle independence,
scoring arithmetic, and that the runner actually detects defects (via
deliberately broken components). They do NOT assert that any KPI target is
met; achieved rates are produced by the runner, not hard-coded here.
"""

import ast
import inspect
import json
from pathlib import Path

import pytest

from tokenization_ai.evaluation import generate, runner, spec
from tokenization_ai.evaluation.__main__ import main as cli_main
from tokenization_ai.evaluation.runner import Components
from tokenization_ai.fusion import InputFusion
from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.models import FusedInput, ModelInput, TokenSequence
from tokenization_ai.tensorizer import Tensorizer
from tokenization_ai.tokenizer import HybridTokenizer

FIXTURES = runner.FIXTURE_DIR


@pytest.fixture(scope="module")
def docs():
    return runner.load_fixtures()


@pytest.fixture(scope="module")
def baseline_report():
    return runner.run_evaluation()


# --- fixture and oracle integrity -------------------------------------------


def test_every_expected_sequence_is_accepted_by_the_independent_grammar(docs):
    checked = 0
    for name in (*runner.EVAL_FILES, *runner.DEV_FILES):
        for case in docs[name]["cases"]:
            tokens, roles = spec.parse_compact(case["expect"]["tokens"])
            assert spec.well_formed(tokens, roles) == [], case["id"]
            checked += 1
    assert checked == 49


def test_reference_encoder_agrees_with_hand_written_fixtures(docs):
    compatible = [c for c in docs["curated_eval.json"]["cases"] if c.get("reference_compatible")]
    assert len(compatible) >= 6
    for case in compatible:
        ctx = case["input"]["context"]
        meta = ctx["metadata"]
        rules = (case["input"]["rule_info"] or {"rules": []})["rules"]
        tokens, roles, masked = spec.reference_encode(
            meta["method"], meta["path"], meta.get("headers", {}), ctx["payload"], rules
        )
        expected_tokens, expected_roles = spec.parse_compact(case["expect"]["tokens"])
        assert (tokens, roles) == (expected_tokens, expected_roles), case["id"]
        assert masked == case["expect"]["masked_paths"], case["id"]


def test_frozen_generated_fixtures_regenerate_identically_from_the_seeds():
    for name, document in generate.build().items():
        frozen = (FIXTURES / name).read_text(encoding="utf-8")
        assert generate.serialize(document) == frozen, name
    assert (generate.DEV_SEED, generate.EVAL_SEED) == (1101, 2202)


def test_dev_and_eval_groups_are_disjoint(docs):
    dev_ids = {c["id"] for n in runner.DEV_FILES for c in docs[n]["cases"]}
    eval_ids = {c["id"] for n in runner.EVAL_FILES for c in docs[n]["cases"]}
    assert dev_ids and eval_ids and not dev_ids & eval_ids
    dev_events = {c["input"]["context"]["event_id"] for n in runner.DEV_FILES for c in docs[n]["cases"]}
    eval_events = {c["input"]["context"]["event_id"] for n in runner.EVAL_FILES for c in docs[n]["cases"]}
    assert not dev_events & eval_events


def test_grammar_acceptor_rejects_malformed_sequences():
    tokens, roles = spec.parse_compact(
        ["C:[SEQ_START]", "C:[API:METHOD]", "A:GET", "C:[API:PATH_START]", "C:[API:PATH_END]",
         "C:[API:HEADERS_START]", "C:[API:HEADERS_END]", "C:[API:PAYLOAD_START]", "C:[API:OBJ_START]",
         "C:[API:OBJ_END]", "C:[API:PAYLOAD_END]", "C:[RULES_START]", "C:[RULES_END]", "C:[SEQ_END]"]
    )
    assert spec.well_formed(tokens, roles) == []
    assert spec.well_formed(tokens[:-1], roles[:-1])  # missing SEQ_END
    bad_roles = list(roles)
    bad_roles[2] = "control"  # method literal given a control role
    assert spec.well_formed(tokens, bad_roles)
    unbalanced_t = tokens[:9] + tokens[10:]
    unbalanced_r = roles[:9] + roles[10:]
    assert spec.well_formed(unbalanced_t, unbalanced_r)
    masked_t = list(tokens)
    masked_t[2] = "[MASKED_PII]"
    masked_r = list(roles)
    masked_r[2] = "masked"  # masked token in a slot that takes an ordinary literal
    assert spec.well_formed(masked_t, masked_r)


def test_runner_does_not_depend_on_the_anomaly_model():
    imported = set()
    for node in ast.walk(ast.parse(inspect.getsource(runner))):
        if isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "..model" not in imported
    assert "LightweightAnomalyModel" not in imported


# --- scoring arithmetic ------------------------------------------------------


def test_wilson_interval_and_rate_arithmetic():
    assert runner.rate(9, 10)["percent"] == 90.0
    assert runner.rate(0, 0) == {
        "numerator": 0, "denominator": 0, "percent": None, "wilson95_percent": None,
    }
    lo, hi = runner.wilson_interval(10, 10)
    assert hi == 100.0 and 70.0 < lo < 75.0
    lo, hi = runner.wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 30.0


# --- report structure and denominators ---------------------------------------


def test_report_structure_and_denominators(baseline_report):
    report = baseline_report
    assert report["protocol_version"] == "1.0.0"
    assert "Not real-world" in report["scope_statement"] or "Not real-world" in report["scope_statement"].replace("not", "Not")
    assert report["data"] == {**report["data"], "valid_eval_cases": 31, "invalid_cases": 5, "dev_cases": 18}
    k1 = report["results"]["kpi1_input_alignment"]
    assert k1["valid_event_alignment"]["denominator"] == 31
    assert k1["invalid_event_rejection"]["denominator"] == 5
    k2 = report["results"]["kpi2_token_mapping"]
    assert k2["sequence_exact_match"]["denominator"] == 31
    k3 = report["results"]["kpi3_aligned_tensor_correctness"]
    assert k3["tensor_correctness"]["denominator"] == 30  # 31 valid minus 1 capacity-rejection case
    assert k3["capacity_rejection"]["denominator"] == 1
    assert set(k3["checks"]) == set(runner._CHECKS)
    assert {c["type"] for c in k3["checks"].values()} == {"invariant", "independent"}
    assert set(k1["valid_event_alignment"]["by_group"]) == {"curated_eval", "generated_eval", "designed"}
    assert any("accuracy" in text for text in report["limitations"])


def test_invalid_events_are_not_in_the_alignment_denominator(baseline_report):
    k1 = baseline_report["results"]["kpi1_input_alignment"]
    assert k1["valid_event_alignment"]["denominator"] == 31
    assert "Reported separately" in k1["invalid_event_rejection"]["note"]


def test_unknown_token_rate_is_reported_separately_and_not_zero_by_construction(baseline_report):
    unk = baseline_report["results"]["kpi3_aligned_tensor_correctness"]["unknown_token_rate"]
    assert unk["literal_positions"] > 0
    assert unk["expected_unknown"] > 0  # eval literals unseen in the dev corpus exist
    assert unk["agrees_with_independent_expectation"] is True
    assert "not a failure" in unk["note"]


def test_every_length_regime_is_exercised(baseline_report):
    mix = baseline_report["results"]["kpi3_aligned_tensor_correctness"]["sequence_length_mix"]
    assert mix["padded"] > 0 and mix["truncated"] > 0


def test_evaluation_is_deterministic(baseline_report):
    again = runner.run_evaluation()
    assert again["results_sha256"] == baseline_report["results_sha256"]


# --- the runner must detect real defects -------------------------------------


def _numbers(report):
    r = report["results"]
    return (
        r["kpi1_input_alignment"]["valid_event_alignment"]["numerator"],
        r["kpi1_input_alignment"]["invalid_event_rejection"]["numerator"],
        r["kpi2_token_mapping"]["sequence_exact_match"]["numerator"],
        r["kpi2_token_mapping"]["masking_provenance"]["event_exact_match"]["numerator"],
        r["kpi3_aligned_tensor_correctness"]["tensor_correctness"]["numerator"],
    )


def test_detects_a_tokenizer_that_drops_a_token(baseline_report):
    class DropsLast(HybridTokenizer):
        def tokenize(self, fused, masked_paths=frozenset()):
            seq = super().tokenize(fused, masked_paths)
            return TokenSequence(event_id=seq.event_id, tokens=seq.tokens[:-1], roles=seq.roles[:-1])

    report = runner.run_evaluation(Components(tokenizer=DropsLast()))
    k2 = report["results"]["kpi2_token_mapping"]
    assert k2["sequence_exact_match"]["numerator"] == 0
    assert k2["sequence_exact_match"]["mapping_errors"]
    assert k2["grammar_conformance_of_actual_output"]["numerator"] == 0
    assert k2["token_level_match"]["percent"] < 100


def test_detects_a_tokenizer_that_mislabels_a_role():
    class MislabelsMasked(HybridTokenizer):
        def tokenize(self, fused, masked_paths=frozenset()):
            seq = super().tokenize(fused, masked_paths)
            roles = ["literal_api" if r == "masked" else r for r in seq.roles]
            return TokenSequence(event_id=seq.event_id, tokens=seq.tokens, roles=roles)

    report = runner.run_evaluation(Components(tokenizer=MislabelsMasked()))
    k2 = report["results"]["kpi2_token_mapping"]
    assert k2["sequence_exact_match"]["numerator"] < k2["sequence_exact_match"]["denominator"]
    assert k2["role_correctness"]["percent"] < 100
    assert k2["token_level_match"]["percent"] < 100


def test_detects_fusion_that_alters_fields():
    class UppercasesPath(InputFusion):
        def fuse(self, context, rule_info):
            return super().fuse(context.model_copy(update={"path": context.path.upper()}), rule_info)

    report = runner.run_evaluation(Components(fusion=UppercasesPath()))
    valid = report["results"]["kpi1_input_alignment"]["valid_event_alignment"]
    assert valid["numerator"] < valid["denominator"]
    assert any(m["field"] == "path" for m in valid["mismatches"])


def test_detects_fusion_that_accepts_mismatched_events():
    class Lenient(InputFusion):
        def fuse(self, context, rule_info):
            return FusedInput.model_construct(context=context, rule_info=rule_info)

    report = runner.run_evaluation(Components(fusion=Lenient()))
    invalid = report["results"]["kpi1_input_alignment"]["invalid_event_rejection"]
    assert invalid["numerator"] == invalid["denominator"] - 1
    assert any(d["outcome"] == "accepted" for d in invalid["details"])
    # Valid-event alignment is unaffected, and kept apart from rejection.
    assert report["results"]["kpi1_input_alignment"]["valid_event_alignment"]["numerator"] == 31


def test_detects_a_masker_that_masks_nothing():
    class NoMasking(PrivacyMasker):
        def mask_with_provenance(self, fused):
            from tokenization_ai.masking import MaskingResult

            return MaskingResult(fused=fused, masked_paths=frozenset())

    report = runner.run_evaluation(Components(masker=NoMasking()))
    prov = report["results"]["kpi2_token_mapping"]["masking_provenance"]
    assert prov["path_false_negative"] > 0
    assert prov["path_recall"] == 0.0
    assert report["results"]["kpi2_token_mapping"]["sequence_exact_match"]["numerator"] < 31


def test_detects_a_tensorizer_that_does_not_pad():
    class NoPad(Tensorizer):
        def tensorize(self, sequence):
            full = super().tensorize(sequence)
            n = sum(full.attention_mask)
            return ModelInput(
                event_id=full.event_id, input_ids=full.input_ids[:n], attention_mask=full.attention_mask[:n]
            )

    report = runner.run_evaluation(Components(tensorizer_factory=lambda v, length: NoPad(v, length)))
    k3 = report["results"]["kpi3_aligned_tensor_correctness"]
    correct = k3["tensor_correctness"]
    assert correct["numerator"] < correct["denominator"]
    # Only sequences that already fill max_length exactly (truncated ones) can pass.
    assert correct["numerator"] <= k3["sequence_length_mix"]["truncated"] + k3["sequence_length_mix"]["exact_fit"]
    assert k3["checks"]["length"]["passed"] < k3["checks"]["length"]["applicable"]


def test_detects_naive_truncation_that_drops_the_rule_segment():
    class NaiveTruncate(Tensorizer):
        def tensorize(self, sequence):
            if len(sequence.tokens) > self.max_length:
                sequence = TokenSequence(
                    event_id=sequence.event_id,
                    tokens=sequence.tokens[: self.max_length],
                    roles=sequence.roles[: self.max_length],
                )
            return super().tensorize(sequence)

    report = runner.run_evaluation(Components(tensorizer_factory=lambda v, length: NaiveTruncate(v, length)))
    checks = report["results"]["kpi3_aligned_tensor_correctness"]["checks"]
    assert checks["rule_segment_preserved"]["passed"] < checks["rule_segment_preserved"]["applicable"]
    assert checks["framing_preserved"]["passed"] < checks["framing_preserved"]["applicable"]
    # The capacity-rejection case is no longer rejected, and the runner says so.
    cap = report["results"]["kpi3_aligned_tensor_correctness"]["capacity_rejection"]
    assert cap["numerator"] == 0 and cap["details"][0]["outcome"] != "rejected_as_specified"


# --- command line -----------------------------------------------------------


def test_cli_writes_machine_and_human_readable_reports(tmp_path: Path):
    assert cli_main(["--out-dir", str(tmp_path), "--write-manifest"]) == 0
    assert cli_main(["--out-dir", str(tmp_path), "--write-manifest"]) == 2  # never overwrite the pre-run record
    assert cli_main(["--out-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "kpi_report.json").read_text(encoding="utf-8"))
    assert report["data"]["matches_pre_run_manifest"] is True
    text = (tmp_path / "kpi_report.md").read_text(encoding="utf-8")
    assert "Targets are proposal targets, not achieved real-world results" in text
    assert "Limitations" in text and "not detection accuracy" in text
