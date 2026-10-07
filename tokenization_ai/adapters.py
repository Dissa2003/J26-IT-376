"""Adapters between shared wire contracts and Member 4 internal models."""

from __future__ import annotations

from typing import Any

from shared_contracts.models import RawEvent

from .models import AnomalyPrediction, UnifiedContext


def to_unified_context(event: RawEvent) -> UnifiedContext:
    """Convert an upstream event into the internal unified context."""
    raise NotImplementedError


def to_member3_output(prediction: AnomalyPrediction) -> Any:
    """Convert an anomaly prediction into the contract expected by Member 3."""
    raise NotImplementedError
