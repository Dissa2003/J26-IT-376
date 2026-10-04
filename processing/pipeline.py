from __future__ import annotations

from shared_contracts.interfaces import FeatureTransformer, SchemaValidator
from shared_contracts.models import FeatureVector, RawEvent


class BasicValidator(SchemaValidator):
    def validate(self, event: RawEvent) -> None:
        if not event.source:
            raise ValueError("source is required")
        bad = [k for k, v in event.payload.items() if isinstance(v, bool) or not isinstance(v, (int, float, str))]
        if bad:
            raise ValueError(f"unsupported payload field types: {bad}")


class NumericFeatureTransformer(FeatureTransformer):
    """Keeps numeric payload fields as features; stub for real engineering."""

    def transform(self, event: RawEvent) -> FeatureVector:
        feats = {k: float(v) for k, v in event.payload.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return FeatureVector(event_id=event.event_id, features=feats)