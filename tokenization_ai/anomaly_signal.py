"""Member 4-owned anomaly signal: a proposed output format, not an agreed contract.

``AnomalySignal`` is a strict, immutable, self-describing wrapper around an
existing :class:`~tokenization_ai.models.AnomalyPrediction`. It exists so the
meaning of the score travels with the number.

What this module is NOT:

- It is not a completed or agreed Member 3 interface. No Member 3 input
  contract exists; score interpretation, threshold ownership, transport and
  failure handling are open questions for Member 3.
- It performs no I/O: no HTTP, no network calls, no imports of other members'
  modules, and no mapping of ``score`` onto any other component's field.
- It carries no label, threshold, decision, ``threat_score`` or rule-aligned
  signal. Those are deliberately absent, and ``extra="forbid"`` keeps them out.

The score is an uncalibrated model output. It is not a probability of attack.
Any future evaluation that uses this module needs updated source provenance
under a new protocol version; the historical KPI evidence does not cover it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import AnomalyPrediction

ANOMALY_SIGNAL_SCHEMA_VERSION = "anomaly-signal-1"


class AnomalySignal(BaseModel):
    """One event's anomaly score, with fixed metadata stating how to read it."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["anomaly-signal-1"] = ANOMALY_SIGNAL_SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    model_version: str = Field(min_length=1)
    score_semantics: Literal["uncalibrated_model_output"] = "uncalibrated_model_output"
    is_probability: Literal[False] = False
    calibrated: Literal[False] = False

    @field_validator("is_probability", "calibrated", mode="before")
    @classmethod
    def _must_be_the_bool_false(cls, value: Any) -> Any:
        # ``Literal[False]`` alone also accepts 0 (0 == False); require the real bool.
        if value is not False:
            raise ValueError("must be the boolean False")
        return value


def to_anomaly_signal(prediction: AnomalyPrediction) -> AnomalySignal:
    """Wrap an existing prediction; ``event_id``, ``score`` and ``model_version`` are unchanged."""
    if not isinstance(prediction, AnomalyPrediction):
        raise TypeError("prediction must be an AnomalyPrediction")
    return AnomalySignal(
        event_id=prediction.event_id,
        score=prediction.score,
        model_version=prediction.model_version,
    )


def to_anomaly_signal_dict(prediction: AnomalyPrediction) -> dict[str, Any]:
    """Return the signal as a JSON-safe dict (``json.dumps`` works on it directly)."""
    return to_anomaly_signal(prediction).model_dump(mode="json")
