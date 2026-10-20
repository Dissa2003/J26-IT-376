"""Tests for confirmatory-scoring-1 (draft protocol heldout-eval-1.2.0-draft, section 12.12.10).

DEVELOPMENT-ONLY and NON-CONFIRMATORY. A small development-seed (6101) chain on a throw-away FROZEN fixture repository with a monkeypatched seed tuple runs
the whole path (authorization, training, validation selection, one-time gate, test scoring) on disposable files. The numbers produced here are properties of
a disposable fixture: they are never reported, and the tests assert only structure, settings and internal consistency. No confirmatory seed, dataset,
manifest, record, marker or result is created; the real repository's DRAFT protocol cannot open the gate.
"""

from __future__ import annotations

import ast
import json
import os
import stat
from pathlib import Path

import pytest

import _heldout_chain as hc
import tokenization_ai.heldout_eval.attempt_gate as gate
import tokenization_ai.heldout_eval.confirmatory_scoring as cs
import tokenization_ai.heldout_eval.detection_metrics as dm
import tokenization_ai.heldout_eval.manifest as mf
import tokenization_ai.heldout_eval.model_artifacts as ma
import tokenization_ai.heldout_eval.numerics as numerics
import tokenization_ai.heldout_eval.runner as rn
import tokenization_ai.heldout_eval.selection_record as sr
import tokenization_ai.heldout_eval.training as tr
from tokenization_ai.heldout_eval.arms import ARM_A, ARM_B, ARM_C, ARMS, build_arm_inputs, prepare_split


@pytest.fixture(scope="module", autouse=True)
def development_seed_stands_in():
    patch = pytest.MonkeyPatch()
    patch.setattr(mf, "CONFIRMATORY_SEEDS", (hc.DEV_SEED,))
    yield
    patch.undo()


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    c = hc.build_chain(tmp_path_factory.mktemp("scoring"))
    c.plan = tr.build_training_plan(c.auth, **hc.plan_args(c))
    c.artifact_dir = c.root / "artifacts"
    c.artifact_dir.mkdir()
    c.trained = tr.train_candidates(c.plan, artifact_dir=c.artifact_dir, repo_root=c.repo)
    (c.root / "records").mkdir()
    c.selection = tr.select_on_validation(c.plan, c.trained, validation=c.part.validation, manifest_bytes=c.manifest_bytes,
                                          expected_manifest_sha256=c.manifest_sha, repo_root=c.repo, expected_run=c.expected,
                                          record_path=c.root / "records" / "selection.json", created_utc=hc.CREATED)
    c.record_bytes = Path(c.selection.record_path).read_bytes()
    c.record = mf.parse_manifest_bytes(c.record_bytes)
    c.val_inputs = rn.encode_validation(c.cfg, c.part.validation)
    c.evidence = sr.ValidationEvidence(arm_inputs=c.val_inputs, labels_by_id=c.part.validation.labels_by_id(), frozen=c.cfg.frozen,
                                       split_result=c.split, artifact_dir=c.trained.artifact_dir, artifact_digests=dict(c.trained.digests))
    test_ids = set(c.part.sealed_test.record_ids)
    c.source = [r for r in c.dataset.records if r.record_id in test_ids]
    c.counter = 0
    return c


class Provider:
    def __init__(self, records, root=None, experiment_id=None):
        self.records, self.calls, self.root, self.experiment_id, self.marker_state = records, 0, root, experiment_id, None

    def __call__(self):
        self.calls += 1
        if self.root is not None:
            self.marker_state = gate.inspect_attempt(self.root, self.experiment_id).state
        return list(self.records)


def dirs(c, tmp_path):
    attempts, results = tmp_path / "attempts", tmp_path / "results"
    attempts.mkdir()
    results.mkdir()
    return attempts, results


def score(c, attempts, results, provider=None, **over):
    kwargs = dict(attempt_root=attempts, results_dir=results, manifest_bytes=c.manifest_bytes, expected_manifest_sha256=c.manifest_sha,
                  record_bytes=c.record_bytes, expected_record_sha256=c.selection.record_sha256, repo_root=c.repo, expected_run=c.expected,
                  evidence=c.evidence, sealed_test=c.part.sealed_test, label_provider=provider or Provider(c.source), train_config=c.cfg,
                  created_utc=hc.CREATED)
    kwargs.update(over)
    return cs.score_test_once(**kwargs)


def experiment(c) -> str:
    return gate.experiment_identity(c.manifest)


@pytest.fixture(scope="module")
def scored(chain, tmp_path_factory):
    root = tmp_path_factory.mktemp("scored")
    attempts, results = root / "attempts", root / "results"
    attempts.mkdir()
    results.mkdir()
    provider = Provider(chain.source, attempts, experiment(chain))
    outcome = score(chain, attempts, results, provider)
    return outcome, provider, attempts, results


# -- the happy path on a disposable fixture -----------------------------------------------------------------------------------------------------


def test_scoring_runs_once_through_the_gate_and_writes_an_aggregate_result(chain, scored):
    outcome, provider, attempts, results = scored
    assert provider.calls == 1 and provider.marker_state == "consumed"  # labels were requested only after the marker existed
    assert gate.inspect_attempt(attempts, experiment(chain)).state == "consumed"
    path = Path(outcome.result_path)
    data = path.read_bytes()
    assert path.parent == results and mf.sha256_hex(data) == outcome.result_sha256 and not (path.stat().st_mode & stat.S_IWRITE)
    result = mf.parse_manifest_bytes(data)
    assert result["result_version"] == "confirmatory-test-result-1" and result["manifest_sha256"] == chain.manifest_sha
    assert result["selection_record_sha256"] == chain.selection.record_sha256 and result["experiment_id"] == experiment(chain)
    assert result["selected_candidates"] == dict(chain.selection.selected_candidates)
    assert result["test_records"] == len(chain.part.sealed_test) and result["post_test_changes"].startswith("none")
    assert set(result["per_arm"]) == set(ARMS) and len(result["secondary"]) == 2


def test_the_primary_contrast_uses_the_fixed_bootstrap_and_decision_rule(chain, scored):
    result = mf.parse_manifest_bytes(Path(scored[0].result_path).read_bytes())
    primary = result["primary"]
    assert primary["statistic"] == f"AP({ARM_B}) - AP({ARM_A})" and primary["replicates_requested"] == 2000 and primary["seed"] == 8201
    assert primary["confidence"] == 0.95 and primary["min_valid_replicates"] == 1900 and primary["resampling_unit"] == "normalized_template_group"
    assert primary["undefined_replicates_replaced"] is False and "replicate_differences" not in primary
    assert primary["decision"] == cs.decision_category(primary) == scored[0].primary_decision
    assert primary["decision"] in ("evidence_of_synthetic_improvement", "possible_worse_synthetic_performance", "no_improvement_claimed", "inconclusive")
    assert {s["statistic"] for s in result["secondary"]} == {f"AP({ARM_C}) - AP({ARM_A})", f"AP({ARM_C}) - AP({ARM_B})"}
    assert all(s["role"] == "secondary_descriptive" and s["replicates_requested"] == 2000 for s in result["secondary"])


def test_the_reported_values_are_reproducible_from_the_selected_artifacts_and_frozen_thresholds(chain, scored):
    """Independent recomputation inside the test (the test may read the fixture's labels; the scored values are never reported)."""
    result = mf.parse_manifest_bytes(Path(scored[0].result_path).read_bytes())
    prepared = prepare_split(chain.part.sealed_test.blinded_records(), "test")
    inputs = build_arm_inputs(prepared, chain.cfg.frozen, chain.cfg.decision)
    truth = {r.record_id: r.meta.label for r in chain.source}
    labels = [truth[i] for i in inputs.record_ids]
    rates = ma.learning_rates_from_manifest(chain.manifest)
    logits = {}
    for arm in ARMS:
        cid = chain.selection.selected_candidates[arm]
        model = ma.load_model_artifact(chain.trained.artifact_dir, expected_sha256=chain.trained.digests[cid], expected_arm=arm, expected_candidate_id=cid,
                                       expected_manifest_sha256=chain.manifest_sha, expected_learning_rate=rates[cid]).model
        logits[arm] = numerics.raw_logits(model, list(getattr(inputs, {ARM_A: "arm_a", ARM_B: "arm_b", ARM_C: "arm_c"}[arm])))
        threshold = chain.record["arms"][arm]["threshold"]["value"]
        per = result["per_arm"][arm]
        assert per["threshold"] == threshold and per["threshold_scale"] == "raw_logit"
        assert per["average_precision"] == pytest.approx(dm.average_precision(logits[arm], labels), abs=1e-12)
        assert per["roc_auc"] == pytest.approx(dm.roc_auc(logits[arm], labels), abs=1e-12)
        counts = dm.confusion_counts(logits[arm], labels, threshold)
        assert (per["confusion"]["tp"], per["confusion"]["fp"], per["confusion"]["tn"], per["confusion"]["fn"]) == (counts.tp, counts.fp, counts.tn, counts.fn)
    ids = list(inputs.record_ids)
    boot = dm.paired_group_bootstrap(dict(zip(ids, logits[ARM_A])), dict(zip(ids, logits[ARM_B])), dict(zip(ids, labels)),
                                     dict(chain.part.sealed_test.group_by_id))
    assert result["primary"]["difference"] == pytest.approx(boot.difference, abs=1e-12) and result["primary"]["replicates_valid"] == boot.replicates_valid
    if boot.interval is not None:
        assert result["primary"]["interval"] == pytest.approx(list(boot.interval), abs=1e-12)


def test_the_result_contains_no_per_record_labels_scores_or_request_content(chain, scored):
    text = Path(scored[0].result_path).read_text(encoding="ascii")
    for rec in chain.source:
        assert rec.record_id not in text and rec.features.path not in text
    result = json.loads(text)
    forbidden_keys = {"labels", "label", "scores", "score", "logits", "logit", "predictions", "prediction", "record_ids", "body", "bodies"}

    def walk(value):
        if isinstance(value, dict):
            assert not set(value) & forbidden_keys
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            assert len(value) < 10  # intervals, limitations and the two secondary contrasts only: no per-record arrays
            for item in value:
                walk(item)

    walk(result)


def test_only_the_three_validation_selected_models_are_evaluated(chain, tmp_path, monkeypatch):
    attempts, results = dirs(chain, tmp_path)
    loads = []
    opened = {"done": False}
    real_open, real_load = gate.open_test_attempt, ma.load_model_artifact

    def open_spy(**kwargs):
        out = real_open(**kwargs)
        opened["done"] = True
        return out

    def load_spy(*args, **kwargs):
        if opened["done"]:
            loads.append(kwargs["expected_candidate_id"])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(gate, "open_test_attempt", open_spy)
    monkeypatch.setattr(ma, "load_model_artifact", load_spy)
    score(chain, attempts, results)
    assert sorted(loads) == sorted(chain.selection.selected_candidates.values())


# -- one attempt only ---------------------------------------------------------------------------------------------------------------------------


def test_a_second_attempt_is_refused_and_the_result_is_untouched(chain, scored):
    outcome, _, attempts, results = scored
    before = Path(outcome.result_path).read_bytes()
    provider = Provider(chain.source)
    with pytest.raises(gate.AttemptAlreadyConsumed):
        score(chain, attempts, results, provider)
    assert provider.calls == 0 and Path(outcome.result_path).read_bytes() == before


def test_a_failure_after_the_marker_is_reported_and_never_retried(chain, tmp_path, monkeypatch):
    attempts, results = dirs(chain, tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("SECRET-TEST-DATA-1010")

    monkeypatch.setattr(cs, "build_arm_inputs", boom)
    with pytest.raises(gate.PostMarkerFailure) as caught:
        score(chain, attempts, results)
    assert caught.value.stage == "test_scoring" and caught.value.category == "scoring_failed" and "SECRET" not in str(caught.value)
    assert gate.inspect_attempt(attempts, experiment(chain)).state == "consumed" and list(results.iterdir()) == []
    report = json.loads(Path(caught.value.report_path).read_bytes())
    assert report["stage"] == "test_scoring" and report["retry_authorized"] is False and b"SECRET" not in Path(caught.value.report_path).read_bytes()
    monkeypatch.undo()
    provider = Provider(chain.source)
    with pytest.raises(gate.AttemptAlreadyConsumed):
        score(chain, attempts, results, provider)
    assert provider.calls == 0


def test_an_existing_result_file_is_never_overwritten(chain, tmp_path):
    attempts, results = dirs(chain, tmp_path)
    existing = results / cs.result_filename(experiment(chain))
    existing.write_bytes(b"someone else's result")
    with pytest.raises(gate.PostMarkerFailure):
        score(chain, attempts, results)
    assert existing.read_bytes() == b"someone else's result"


def test_failures_before_the_gate_create_nothing(chain, tmp_path):
    attempts, results = dirs(chain, tmp_path)
    import dataclasses
    bad_cfg = dataclasses.replace(chain.cfg, frozen=dataclasses.replace(chain.cfg.frozen, record={**chain.cfg.frozen.record, "sha256": "0" * 64}))
    for over in (dict(train_config=bad_cfg), dict(train_config="cfg"), dict(results_dir=tmp_path / "missing"), dict(results_dir=chain.repo),
                 dict(manifest_bytes=b"not json")):
        provider = Provider(chain.source)
        with pytest.raises(cs.ScoringError):
            score(chain, attempts, results, provider, **over)
        assert provider.calls == 0 and list(attempts.iterdir()) == [] and list(results.iterdir()) == []


def test_the_gate_still_refuses_wrong_records_draft_protocols_and_the_real_repository(chain, tmp_path):
    attempts, results = dirs(chain, tmp_path)
    for over in (dict(expected_record_sha256="0" * 64), dict(expected_manifest_sha256="0" * 64), dict(repo_root=hc.REPO_ROOT),
                 dict(expected_run=__import__("dataclasses").replace(chain.expected, approved_protocol_sha256=None))):
        provider = Provider(chain.source)
        with pytest.raises(gate.GateVerificationError):
            score(chain, attempts, results, provider, **over)
        assert provider.calls == 0 and list(attempts.iterdir()) == [] and list(results.iterdir()) == []
    assert mf.collect_protocol(hc.REPO_ROOT)["status"] == "DRAFT"


# -- rule and hygiene -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("boot,expected", [
    ({"status": "ok", "interval": [0.01, 0.2]}, "evidence_of_synthetic_improvement"),
    ({"status": "ok", "interval": [0.0, 0.2]}, "no_improvement_claimed"),
    ({"status": "ok", "interval": [-0.2, 0.1]}, "no_improvement_claimed"),
    ({"status": "ok", "interval": [-0.2, -0.01]}, "possible_worse_synthetic_performance"),
    ({"status": "insufficient_valid_replicates", "interval": None}, "inconclusive"),
    ({"status": "undefined_point_estimate", "interval": None}, "inconclusive"),
    ({"status": "ok", "interval": None}, "inconclusive"),
])
def test_the_pre_specified_decision_rule(boot, expected):
    assert cs.decision_category(boot) == expected


def test_the_scoring_module_cannot_tune_retrain_or_bypass_the_gate():
    text = Path(cs.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"fit", "select_threshold", "select_candidate", "train_candidates", "select_on_validation", "recover_test_labels",
                         "create_attempt_marker", "authorize_training", "unlink", "remove", "rename", "replace"}
    assert "open_test_attempt" in called and not any(seed in text for seed in ("7101", "7102", "7103"))
    assert "tokenization_ai/heldout_eval/confirmatory_scoring.py" in mf.REQUIRED_SOURCE_FILES


def test_the_runner_test_access_boundary_stays_disabled():
    with pytest.raises(rn.AuthorizationUnavailable):
        rn.request_test_access()
    assert rn.TEST_SCORING_AVAILABLE is False
