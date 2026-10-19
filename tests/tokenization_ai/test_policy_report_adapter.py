"""Tests for the isolated Member 2-style policy report compatibility adapter.

All report fixtures are SYNTHETIC mappings hand-written to resemble the fields Member 2's
``VerificationReport`` exposes (JSON form). They are not Member 2 output, can drift from
Member 2's real models, and passing these tests is NOT evidence of Member 2 integration.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

import tokenization_ai.policy_report_adapter as module
from tokenization_ai import adapters
from tokenization_ai.adapters import InputContractError
from tokenization_ai.fusion import InputFusion
from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import UnifiedApiContext, VerifiedRuleInfo
from tokenization_ai.normalization import Normalizer
from tokenization_ai.policy_report_adapter import (
    ALLOWED_ATTRIBUTES,
    MAX_POLICY_REPORTS,
    PolicyReportError,
    PolicyReportNotEligibleError,
    policy_reports_to_verified_rule_info,
)
from tokenization_ai.service import TokenizationAIService
from tokenization_ai.tensorizer import Tensorizer
from tokenization_ai.tokenizer import HybridTokenizer
from tokenization_ai.vocabulary import Vocabulary

CONTEXT = UnifiedApiContext(event_id="ctx-event-1", method="POST", path="/accounts")


def make_report(policy_id: str = "synthetic-P1", **overrides) -> dict:
    report = {
        "is_valid": True,
        "status": "UNSAT",
        "fol_expressions": ["synthetic formula text"],
        "conflicting_policies": None,
        "execution_time_ms": 1.0,
        "policy_id": policy_id,
        "original_policy": "SYNTHETIC-RAW-POLICY-TEXT",
        "attributes": {
            "subject": "Administrators",
            "action": "Delete",
            "resource": "UserAccounts",
            "effect": "Allow",
            "conditions": [],
            "interpretation": "role_permission",
        },
        "structured_policy": {"raw_text": "SYNTHETIC-RAW-POLICY-TEXT"},
        "inference_trace": [{"step_number": 1, "conclusion": "SYNTHETIC-TRACE"}],
        "constraints": [{"expression": "SYNTHETIC-CONSTRAINT"}],
        "solver_result": {
            "status": "UNSAT",
            "query_type": "entailment",
            "is_entailment": True,
            "explanation": "SYNTHETIC-EXPLANATION",
        },
        "conflicts": [],
        "rule_status": "eligible",
        "limitations": ["demonstration individual"],
    }
    report.update(overrides)
    return report


_DEFAULT = object()


def convert(reports=_DEFAULT, ids=_DEFAULT, context=CONTEXT):
    reports = [make_report()] if reports is _DEFAULT else reports
    ids = ["synthetic-P1"] if ids is _DEFAULT else ids
    return policy_reports_to_verified_rule_info(context, reports, applicable_policy_ids=ids)


# -- correct conversion ------------------------------------------------------


def test_eligible_report_converts_to_one_rule_with_allowlisted_attributes():
    info = convert()
    assert isinstance(info, VerifiedRuleInfo)
    assert len(info.rules) == 1
    rule = info.rules[0]
    assert rule.rule_id == "synthetic-P1"
    assert rule.attributes == {
        "subject": "Administrators",
        "action": "Delete",
        "resource": "UserAccounts",
        "effect": "Allow",
        "interpretation": "role_permission",
    }
    assert tuple(rule.attributes) == ALLOWED_ATTRIBUTES


def test_report_order_is_preserved_and_ids_may_be_any_collection():
    reports = [make_report("synthetic-C"), make_report("synthetic-A"), make_report("synthetic-B")]
    info = convert(reports, ids={"synthetic-A", "synthetic-B", "synthetic-C"})
    assert [r.rule_id for r in info.rules] == ["synthetic-C", "synthetic-A", "synthetic-B"]
    assert convert(reports, ids=("synthetic-B", "synthetic-C", "synthetic-A")).rules == info.rules


def test_deny_effect_and_other_interpretations_are_accepted():
    report = make_report(
        "synthetic-D",
        attributes={"subject": "Users", "action": "Delete", "resource": "Accounts",
                    "effect": "Deny", "conditions": [], "interpretation": "role_prohibition"},
    )
    assert convert([report], ["synthetic-D"]).rules[0].attributes["effect"] == "Deny"


def test_missing_conditions_key_is_treated_as_no_conditions():
    report = make_report()
    del report["attributes"]["conditions"]
    assert len(convert([report]).rules) == 1


def test_inputs_are_not_mutated():
    reports = [make_report()]
    before = copy.deepcopy(reports)
    convert(reports)
    assert reports == before


# -- event identity ----------------------------------------------------------


def test_event_id_comes_only_from_the_context():
    assert convert(context=CONTEXT).event_id == "ctx-event-1"
    other = UnifiedApiContext(event_id="another-event", method="GET", path="/x")
    assert convert(context=other).event_id == "another-event"


@pytest.mark.parametrize("value", ["ctx-event-1", "someone-elses-event", None, ""])
def test_a_report_carrying_its_own_event_id_is_rejected(value):
    with pytest.raises(PolicyReportError, match="event_id"):
        convert([make_report(event_id=value)])


def test_result_fuses_with_the_same_context_and_is_rejected_for_another():
    info = convert()
    assert InputFusion().fuse(CONTEXT, info).event_id == "ctx-event-1"
    other = UnifiedApiContext(event_id="different", method="GET", path="/x")
    with pytest.raises(InputContractError):
        InputFusion().fuse(other, info)


@pytest.mark.parametrize("bad", [None, {"event_id": "x", "method": "GET", "path": "/"}, "ctx-event-1"])
def test_context_must_be_a_unified_api_context(bad):
    with pytest.raises(PolicyReportError, match="context"):
        convert(context=bad)


# -- applicability -----------------------------------------------------------


@pytest.mark.parametrize(
    "ids",
    [
        ["synthetic-P1", "synthetic-P2"],  # asserted but no report
        ["synthetic-OTHER"],  # different id
        [],  # empty
        set(),
    ],
)
def test_applicability_must_match_reports_exactly(ids):
    with pytest.raises(PolicyReportError):
        convert(ids=ids)


def test_report_not_asserted_as_applicable_is_rejected():
    with pytest.raises(PolicyReportError, match="do not match"):
        convert([make_report("synthetic-P1"), make_report("synthetic-P2")], ["synthetic-P1"])


@pytest.mark.parametrize("ids", ["synthetic-P1", b"synthetic-P1", {"synthetic-P1": 1}, None, 5])
def test_applicability_must_be_a_real_collection_not_a_string(ids):
    with pytest.raises(PolicyReportError, match="applicable_policy_ids"):
        convert(ids=ids)


@pytest.mark.parametrize("bad_id", ["", " ", " lead", "trail ", "line\nbreak", 5, None, "x" * 65])
def test_applicable_ids_must_be_valid_ids(bad_id):
    with pytest.raises(PolicyReportError, match="applicable_policy_ids"):
        convert(ids=["synthetic-P1", bad_id])


def test_duplicate_asserted_ids_are_rejected():
    with pytest.raises(PolicyReportError, match="duplicates"):
        convert(ids=["synthetic-P1", "synthetic-P1"])


# -- policy ids --------------------------------------------------------------


@pytest.mark.parametrize("bad_id", [None, "", "   ", 5, ["x"], "x" * 65, "has\ttab", " padded "])
def test_report_policy_id_must_be_present_and_valid(bad_id):
    report = make_report()
    report["policy_id"] = bad_id
    with pytest.raises(PolicyReportError, match=r"reports\[0\]: policy_id"):
        convert([report])


def test_missing_policy_id_is_rejected():
    report = make_report()
    del report["policy_id"]
    with pytest.raises(PolicyReportError, match="policy_id"):
        convert([report])


def test_duplicate_report_policy_ids_are_rejected():
    with pytest.raises(PolicyReportError, match="duplicate"):
        convert([make_report("synthetic-P1"), make_report("synthetic-P1")], ["synthetic-P1"])


# -- eligibility -------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_valid": False},
        {"is_valid": 1},
        {"is_valid": "true"},
        {"is_valid": None},
        {"rule_status": "rejected"},
        {"rule_status": "unsupported"},
        {"rule_status": "conflicted"},
        {"rule_status": "ELIGIBLE"},
        {"status": "SAT"},
        {"status": "UNKNOWN"},
        {"status": None},
        {"conflicting_policies": ["synthetic-P2"]},
    ],
)
def test_ineligible_reports_are_rejected_with_a_specific_error(overrides):
    with pytest.raises(PolicyReportNotEligibleError):
        convert([make_report(**overrides)])


@pytest.mark.parametrize(
    "solver_change",
    [
        {"status": "SAT"},
        {"status": "UNKNOWN"},
        {"is_entailment": False},
        {"is_entailment": 1},
        {"is_entailment": None},
        {"query_type": "consistency"},
        {"query_type": None},
    ],
)
def test_solver_result_must_show_an_unsat_negated_entailment_query(solver_change):
    report = make_report()
    report["solver_result"] = {**report["solver_result"], **solver_change}
    with pytest.raises(PolicyReportNotEligibleError):
        convert([report])


def test_unsat_alone_is_not_enough():
    report = make_report(is_valid=False, rule_status="rejected")
    assert report["status"] == "UNSAT"
    with pytest.raises(PolicyReportNotEligibleError):
        convert([report])


def test_sat_status_is_not_treated_as_a_verified_rule():
    report = make_report(status="SAT")
    report["solver_result"]["status"] = "SAT"
    with pytest.raises(PolicyReportNotEligibleError):
        convert([report])


def test_reports_with_conflicts_are_rejected():
    with pytest.raises(PolicyReportNotEligibleError, match="conflicts"):
        convert([make_report(conflicts=[{"conflict_id": "c1"}])])


def test_reports_with_conditions_are_rejected_rather_than_silently_dropped():
    report = make_report()
    report["attributes"]["conditions"] = [{"attribute": "SYNTHETIC-SECRET", "operator": "equals", "value": "v"}]
    with pytest.raises(PolicyReportNotEligibleError, match="conditions") as caught:
        convert([report])
    assert "SYNTHETIC-SECRET" not in str(caught.value)


def test_one_ineligible_report_rejects_the_whole_input():
    good, bad = make_report("synthetic-P1"), make_report("synthetic-P2", rule_status="rejected")
    with pytest.raises(PolicyReportNotEligibleError, match=r"reports\[1\]"):
        convert([good, bad], ["synthetic-P1", "synthetic-P2"])


# -- malformed input ---------------------------------------------------------


@pytest.mark.parametrize("reports", [[], (), None, "report", b"report", {"policy_id": "x"}, 5, {1, 2}])
def test_reports_container_must_be_a_non_empty_sequence(reports):
    with pytest.raises(PolicyReportError):
        convert(reports)


def test_empty_reports_never_yield_an_empty_rule_list():
    with pytest.raises(PolicyReportError, match="empty"):
        convert([])


@pytest.mark.parametrize("item", [None, "x", 5, ["a"], make_report])
def test_each_report_must_be_a_mapping(item):
    with pytest.raises(PolicyReportError, match=r"reports\[0\] must be a mapping"):
        convert([item])


@pytest.mark.parametrize("field", ["solver_result", "conflicts", "attributes"])
@pytest.mark.parametrize("bad", [None, "x", 5])
def test_required_structures_must_have_the_right_type(field, bad):
    report = make_report()
    report[field] = bad
    with pytest.raises(PolicyReportError):
        convert([report])


@pytest.mark.parametrize("field", ["solver_result", "conflicts", "attributes", "is_valid", "rule_status", "status"])
def test_missing_required_fields_are_rejected(field):
    report = make_report()
    del report[field]
    with pytest.raises(PolicyReportError):
        convert([report])


@pytest.mark.parametrize("name", list(ALLOWED_ATTRIBUTES))
@pytest.mark.parametrize("bad", [None, "", "  ", 5, ["x"], "x" * 65, "new\nline", " padded"])
def test_every_allowlisted_attribute_must_be_present_and_valid(name, bad):
    report = make_report()
    report["attributes"][name] = bad
    with pytest.raises(PolicyReportError, match=rf"attributes\.{name}"):
        convert([report])


def test_missing_allowlisted_attribute_is_rejected():
    report = make_report()
    del report["attributes"]["resource"]
    with pytest.raises(PolicyReportError, match="attributes.resource"):
        convert([report])


@pytest.mark.parametrize("effect", ["allow", "Permit", "ALLOW", "Challenge"])
def test_unknown_effect_is_rejected(effect):
    report = make_report()
    report["attributes"]["effect"] = effect
    with pytest.raises(PolicyReportError, match="effect"):
        convert([report])


@pytest.mark.parametrize("interpretation", ["Role_Permission", "custom", "only role"])
def test_unknown_interpretation_is_rejected(interpretation):
    report = make_report()
    report["attributes"]["interpretation"] = interpretation
    with pytest.raises(PolicyReportError, match="interpretation"):
        convert([report])


def test_report_count_is_capped():
    reports = [make_report(f"synthetic-{i}") for i in range(MAX_POLICY_REPORTS + 1)]
    with pytest.raises(PolicyReportError, match="too many"):
        convert(reports, [r["policy_id"] for r in reports])


def test_maximum_report_count_is_accepted():
    reports = [make_report(f"synthetic-{i}") for i in range(MAX_POLICY_REPORTS)]
    assert len(convert(reports, [r["policy_id"] for r in reports]).rules) == MAX_POLICY_REPORTS


def test_errors_are_input_contract_errors():
    assert issubclass(PolicyReportError, InputContractError)
    assert issubclass(PolicyReportNotEligibleError, PolicyReportError)


# -- privacy -----------------------------------------------------------------


def test_nothing_outside_the_allowlist_reaches_the_rule():
    info = convert()
    dumped = json.dumps(info.model_dump(mode="json"))
    for forbidden in ("SYNTHETIC-RAW-POLICY-TEXT", "SYNTHETIC-TRACE", "SYNTHETIC-CONSTRAINT",
                      "SYNTHETIC-EXPLANATION", "synthetic formula text", "demonstration individual",
                      "conditions", "rule_status", "UNSAT"):
        assert forbidden not in dumped.replace("ctx-event-1", "")
    assert set(info.rules[0].attributes) == set(ALLOWED_ATTRIBUTES)


def test_unknown_extra_report_fields_are_ignored_not_copied():
    report = make_report(extra_secret="SYNTHETIC-EXTRA-SECRET")
    report["attributes"]["extra_attr"] = "SYNTHETIC-EXTRA-ATTR"
    dumped = json.dumps(convert([report]).model_dump(mode="json"))
    assert "SYNTHETIC-EXTRA" not in dumped


@pytest.mark.parametrize("which", ["policy_id", "attribute", "applicable", "condition", "effect"])
def test_error_messages_never_echo_supplied_values(which):
    secret = "SYNTHETIC-SENSITIVE-VALUE\n"
    report = make_report()
    ids = ["synthetic-P1"]
    if which == "policy_id":
        report["policy_id"] = secret
    elif which == "attribute":
        report["attributes"]["subject"] = secret
    elif which == "applicable":
        ids = [secret]
    elif which == "condition":
        report["attributes"]["conditions"] = [{"value": secret}]
    else:
        report["attributes"]["effect"] = secret.strip()
    with pytest.raises(PolicyReportError) as caught:
        convert([report], ids)
    assert "SYNTHETIC-SENSITIVE-VALUE" not in str(caught.value)


def test_duplicate_and_mismatch_errors_do_not_echo_ids():
    with pytest.raises(PolicyReportError) as caught:
        convert([make_report("secret-policy-a")], ["secret-policy-b"])
    assert "secret-policy" not in str(caught.value)


# -- compatibility with the existing Member 4 pipeline ------------------------


def _service(max_length: int) -> TokenizationAIService:
    fusion, masker, normalizer, tokenizer = InputFusion(), PrivacyMasker(), Normalizer(), HybridTokenizer()
    info = convert()
    context = CONTEXT
    fused = fusion.fuse(context, info)
    masked = masker.mask_with_provenance(fused)
    normalized = normalizer.normalize_with_provenance(masked.fused, masked.masked_paths)
    sequence = tokenizer.tokenize(normalized.fused, normalized.masked_paths)
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    tensorizer = Tensorizer(vocabulary, max_length=max_length)
    model = LightweightAnomalyModel(num_buckets=64)
    model.fit([tensorizer.tensorize(sequence)] * 2, [0, 1], epochs=2, learning_rate=0.1)
    return TokenizationAIService(fusion, masker, normalizer, tokenizer, tensorizer, model)


def test_converted_rules_run_through_the_real_service_with_literal_rule_roles():
    service = _service(max_length=128)
    info = convert()
    analysis = service.analyze(CONTEXT, info)
    assert analysis.prediction.event_id == "ctx-event-1"
    assert 0.0 <= analysis.prediction.score <= 1.0
    roles = analysis.sequence.roles
    assert roles.count("literal_rule") >= 1 + len(ALLOWED_ATTRIBUTES) * 2
    rule_literals = [t for t, r in zip(analysis.sequence.tokens, roles) if r == "literal_rule"]
    assert "synthetic-P1" in rule_literals and "Administrators" in rule_literals
    assert "SYNTHETIC-RAW-POLICY-TEXT" not in analysis.sequence.tokens


def test_rules_that_do_not_fit_raise_instead_of_being_truncated():
    service = _service(max_length=128)
    many = [make_report(f"synthetic-{i}") for i in range(8)]
    info = convert(many, [r["policy_id"] for r in many])
    with pytest.raises(ValueError, match="refusing to drop rule"):
        service.predict(CONTEXT, info)


def test_two_converted_runs_are_deterministic():
    assert convert() == convert()


# -- isolation and scope ------------------------------------------------------


def test_module_imports_nothing_from_other_members_and_does_no_io():
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert imported <= {"__future__", "re", "collections.abc", "typing", ".adapters", ".models"}, imported
    for banned in ("src", "api", "engine", "api_gateway", "ingestion", "shared_contracts",
                   "httpx", "requests", "urllib", "socket", "redis", "policy_verifier"):
        assert banned not in imported


def test_existing_adapters_are_unchanged_in_behaviour():
    with pytest.raises(NotImplementedError):
        adapters.to_member3_output(None)  # type: ignore[arg-type]
    assert not hasattr(adapters, "policy_reports_to_verified_rule_info")


def test_existing_rule_input_adapter_still_works_for_mock_style_rules():
    info = adapters.to_verified_rule_info({"event_id": "e", "rules": [{"rule_id": "MOCK-R-1", "severity": "high"}]})
    assert info.rules[0].rule_id == "MOCK-R-1"
