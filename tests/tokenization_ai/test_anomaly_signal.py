"""Tests for the standalone, Member 4-owned AnomalySignal output model."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

import tokenization_ai.anomaly_signal as signal_module
from tokenization_ai import adapters
from tokenization_ai.anomaly_signal import (
    AnomalySignal,
    to_anomaly_signal,
    to_anomaly_signal_dict,
)
from tokenization_ai.models import AnomalyPrediction

VALID = dict(event_id="synthetic-1", score=0.25, model_version="synthetic-model-v0")


def _prediction(**overrides) -> AnomalyPrediction:
    return AnomalyPrediction(**{**VALID, **overrides})


def test_conversion_preserves_event_id_score_and_model_version():
    signal = to_anomaly_signal(_prediction(event_id="evt-9", score=0.7312, model_version="m-3"))
    assert isinstance(signal, AnomalySignal)
    assert (signal.event_id, signal.score, signal.model_version) == ("evt-9", 0.7312, "m-3")


@pytest.mark.parametrize("score", [0.0, 1.0, 0.5])
def test_boundary_and_mid_scores_are_unchanged(score):
    assert to_anomaly_signal(_prediction(score=score)).score == score


def test_fixed_metadata_values():
    signal = to_anomaly_signal(_prediction())
    assert signal.schema_version == "anomaly-signal-1"
    assert signal.score_semantics == "uncalibrated_model_output"
    assert signal.is_probability is False
    assert signal.calibrated is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "anomaly-signal-2"),
        ("score_semantics", "probability"),
        ("is_probability", True),
        ("calibrated", True),
        ("is_probability", 0),
        ("calibrated", "false"),
    ],
)
def test_metadata_cannot_be_overridden(field, value):
    with pytest.raises(ValidationError):
        AnomalySignal(**VALID, **{field: value})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.0001, 1.0001, 2, -1])
def test_invalid_scores_are_rejected(bad):
    with pytest.raises(ValidationError):
        AnomalySignal(**{**VALID, "score": bad})


@pytest.mark.parametrize("bad", ["0.5", None, True, [0.5]])
def test_non_numeric_scores_are_rejected(bad):
    with pytest.raises(ValidationError):
        AnomalySignal(**{**VALID, "score": bad})


@pytest.mark.parametrize("field", ["event_id", "model_version"])
def test_empty_or_non_string_identifiers_are_rejected(field):
    for bad in ("", 5, None):
        with pytest.raises(ValidationError):
            AnomalySignal(**{**VALID, field: bad})


@pytest.mark.parametrize("extra", ["label", "threshold", "decision", "threat_score", "rule_aligned_signal", "anything"])
def test_unknown_fields_are_rejected(extra):
    with pytest.raises(ValidationError):
        AnomalySignal(**VALID, **{extra: "x"})


def test_signal_is_immutable():
    signal = to_anomaly_signal(_prediction())
    with pytest.raises(ValidationError):
        signal.score = 0.9


def test_no_decision_like_fields_exist():
    fields = set(AnomalySignal.model_fields)
    assert fields == {
        "schema_version", "event_id", "score", "model_version",
        "score_semantics", "is_probability", "calibrated",
    }
    forbidden = {"label", "threshold", "decision", "threat_score", "rule_aligned_signal", "allow", "block"}
    assert not (fields & forbidden)
    assert not (set(to_anomaly_signal_dict(_prediction())) & forbidden)


def test_dict_is_json_safe_and_round_trips():
    data = to_anomaly_signal_dict(_prediction(score=0.125))
    text = json.dumps(data, allow_nan=False)
    restored = AnomalySignal.model_validate_json(text)
    assert restored == to_anomaly_signal(_prediction(score=0.125))
    assert AnomalySignal.model_validate(json.loads(text)) == restored
    assert data == {
        "schema_version": "anomaly-signal-1", "event_id": "synthetic-1", "score": 0.125,
        "model_version": "synthetic-model-v0", "score_semantics": "uncalibrated_model_output",
        "is_probability": False, "calibrated": False,
    }


def test_non_prediction_input_is_rejected():
    with pytest.raises(TypeError):
        to_anomaly_signal({"event_id": "x", "score": 0.1, "model_version": "m"})  # type: ignore[arg-type]


def test_conversion_does_not_mutate_the_prediction():
    prediction = _prediction()
    before = prediction.model_dump()
    to_anomaly_signal(prediction)
    to_anomaly_signal_dict(prediction)
    assert prediction.model_dump() == before


def test_existing_member3_adapter_is_still_unimplemented():
    with pytest.raises(NotImplementedError):
        adapters.to_member3_output(_prediction())


def test_module_imports_no_other_member_services_and_does_no_io():
    tree = ast.parse(Path(signal_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    allowed = {"__future__", "typing", "pydantic", ".models"}
    assert imported <= allowed, imported
    for banned in ("engine", "api_gateway", "ingestion", "api", "src", "shared_contracts",
                   "httpx", "requests", "urllib", "socket", "redis"):
        assert banned not in imported
