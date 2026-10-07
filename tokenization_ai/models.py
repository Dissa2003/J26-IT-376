"""Internal data models passed between the tokenization and prediction stages."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UnifiedApiContext(BaseModel):
    """One intercepted API request as supplied by Member 1."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    path: str = Field(min_length=1)
    content_type: str | None = None
    client_ip: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    threat_score: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp: datetime | None = None


class VerifiedRule(BaseModel):
    """A single verified rule; fields beyond the id are kept as received."""

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class VerifiedRuleInfo(BaseModel):
    """Verified rule information for one event as supplied by Member 2."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(min_length=1)
    rules: list[VerifiedRule] = Field(default_factory=list)


class FusedInput(BaseModel):
    """API context paired with the verified rule information for the same event."""

    model_config = ConfigDict(frozen=True)

    context: UnifiedApiContext
    rule_info: VerifiedRuleInfo

    @model_validator(mode="after")
    def _same_event(self) -> FusedInput:
        if self.context.event_id != self.rule_info.event_id:
            raise ValueError(
                "context and rule_info refer to different events: "
                f"{self.context.event_id!r} != {self.rule_info.event_id!r}"
            )
        return self

    @property
    def event_id(self) -> str:
        return self.context.event_id


# Names used by the stage skeletons until they are implemented.
UnifiedContext = UnifiedApiContext
RuleInfo = VerifiedRuleInfo


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
