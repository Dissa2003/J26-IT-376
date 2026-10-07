"""Internal data models passed between the tokenization and prediction stages."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UnifiedContext(BaseModel):
    """Unified API context received from the upstream pipeline."""

    event_id: str
    context: dict[str, Any] = Field(default_factory=dict)


class RuleInfo(BaseModel):
    """Verified rule information accompanying a unified context."""

    event_id: str
    rules: list[dict[str, Any]] = Field(default_factory=list)


class FusedInput(BaseModel):
    """Single structured record produced by input fusion."""

    event_id: str
    fields: dict[str, Any] = Field(default_factory=dict)


class TokenSequence(BaseModel):
    """Ordered tokens produced by the hybrid tokenizer."""

    event_id: str
    tokens: list[str] = Field(default_factory=list)


class ModelInput(BaseModel):
    """Fixed-length token ids and attention mask ready for the model."""

    event_id: str
    input_ids: list[int] = Field(default_factory=list)
    attention_mask: list[int] = Field(default_factory=list)


class AnomalyPrediction(BaseModel):
    """Anomaly prediction emitted for Member 3."""

    event_id: str
    score: float = Field(ge=0.0, le=1.0)
    model_version: str
