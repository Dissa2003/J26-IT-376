"""Adapters between upstream inputs and Member 4 internal models."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from shared_contracts.models import RawEvent

from .models import AnomalyPrediction, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo

# Keys Member 1 adds to ``RawEvent.payload`` for intercepted HTTP requests.
HTTP_METADATA_KEY = "_http_metadata"
THREAT_SCORE_KEY = "_threat_score"

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class InputContractError(ValueError):
    """Raised when an upstream input does not satisfy a Member 4 contract."""


def to_unified_api_context(source: RawEvent | Mapping[str, Any]) -> UnifiedApiContext:
    """Convert a Member 1 event into a unified API context.

    Accepts a ``RawEvent`` (or its dict form) whose payload carries the
    interception metadata, or an interception envelope with top-level
    ``event_id``, ``metadata``, ``payload`` and optional ``threat_score``.
    """
    if isinstance(source, RawEvent):
        return _context_from_event(source.event_id, source.payload, source.timestamp)
    if not isinstance(source, Mapping):
        raise InputContractError("unified API context must be a RawEvent or a mapping")
    if "metadata" in source:
        _require(source, ("event_id", "metadata", "payload"), "interception envelope")
        return _build_context(
            source["event_id"],
            source["metadata"],
            source["payload"],
            source.get("threat_score"),
            source.get("timestamp"),
        )
    _require(source, ("event_id", "payload"), "event")
    return _context_from_event(source["event_id"], source["payload"], source.get("timestamp"))


def to_verified_rule_info(source: Mapping[str, Any] | str | bytes) -> VerifiedRuleInfo:
    """Convert verified rule information from a mapping or JSON document.

    The input must provide ``event_id`` and a ``rules`` list. Every rule needs
    a ``rule_id``; its remaining keys are preserved as ``attributes``.
    """
    what = "verified rule information"
    if isinstance(source, (str, bytes)):
        try:
            source = json.loads(source)
        except ValueError as exc:
            raise InputContractError(f"{what} is not valid JSON") from exc
    if not isinstance(source, Mapping):
        raise InputContractError(f"{what} must be a mapping or a JSON object")
    _require(source, ("event_id", "rules"), what)
    rules = source["rules"]
    if not isinstance(rules, list):
        raise InputContractError(f"{what} field 'rules' must be a list")
    return _validated(
        VerifiedRuleInfo,
        what,
        event_id=source["event_id"],
        rules=[_to_verified_rule(rule, index) for index, rule in enumerate(rules)],
    )


def to_member3_output(prediction: AnomalyPrediction) -> Any:
    """Convert an anomaly prediction into the contract expected by Member 3."""
    raise NotImplementedError


def _context_from_event(event_id: Any, payload: Any, timestamp: Any = None) -> UnifiedApiContext:
    """Split interception metadata out of an event payload."""
    if not isinstance(payload, Mapping):
        raise InputContractError("event field 'payload' must be a mapping")
    _require(payload, (HTTP_METADATA_KEY,), "event payload")
    body = {
        key: value
        for key, value in payload.items()
        if key not in (HTTP_METADATA_KEY, THREAT_SCORE_KEY)
    }
    return _build_context(
        event_id,
        payload[HTTP_METADATA_KEY],
        body,
        payload.get(THREAT_SCORE_KEY),
        timestamp,
    )


def _build_context(
    event_id: Any, metadata: Any, payload: Any, threat_score: Any, timestamp: Any = None
) -> UnifiedApiContext:
    if not isinstance(metadata, Mapping):
        raise InputContractError("HTTP metadata must be a mapping")
    _require(metadata, ("method", "path"), "HTTP metadata")
    return _validated(
        UnifiedApiContext,
        "unified API context",
        event_id=event_id,
        method=metadata["method"],
        path=metadata["path"],
        content_type=metadata.get("content_type"),
        client_ip=metadata.get("client_ip"),
        headers=metadata.get("headers") or {},
        payload=payload,
        threat_score=threat_score,
        timestamp=timestamp,
    )


def _to_verified_rule(rule: Any, index: int) -> VerifiedRule:
    what = f"rules[{index}]"
    if not isinstance(rule, Mapping):
        raise InputContractError(f"{what} must be a mapping")
    _require(rule, ("rule_id",), what)
    attributes = {key: value for key, value in rule.items() if key != "rule_id"}
    return _validated(VerifiedRule, what, rule_id=rule["rule_id"], attributes=attributes)


def _require(data: Mapping[str, Any], fields: Iterable[str], what: str) -> None:
    """Raise if any required field is absent or null."""
    missing = [field for field in fields if data.get(field) is None]
    if missing:
        raise InputContractError(f"{what} is missing required fields: {missing}")


def _validated(model: type[_ModelT], what: str, **fields: Any) -> _ModelT:
    """Build a model, reporting field errors as a contract violation."""
    try:
        return model(**fields)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise InputContractError(f"invalid {what}: {details}") from exc
