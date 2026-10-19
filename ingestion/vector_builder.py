"""Unified Context Vector construction and protobuf serialization.

Includes Rich Mathematical Vector Mapping (WBS 4.4 / FR4 / FR5) combining probabilistic
threat scores, structural/syntactic payload dimensions, and numeric leaves into a high-dimensional
vector representation for downstream SMT/Z3 solvers and neuro-symbolic tokenizers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message
from google.protobuf.message_factory import GetMessageClass
from google.protobuf.struct_pb2 import Struct

HIGH_THREAT_THRESHOLD = 0.70
LOW_THREAT_THRESHOLD = 0.30


def extract_threat_score(payload: Mapping[str, Any]) -> float:
    """Return the normalized threat score carried by an event payload."""
    value = payload.get("_threat_score", payload.get("threat_score", 0.0))
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, score))


def _context_message_class() -> type[message.Message]:
    """Build the wire schema once through protobuf's descriptor registry."""
    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="ingestion_context.proto",
        package="dissanayaka.ingestion",
        syntax="proto3",
        dependency=["google/protobuf/struct.proto"],
    )
    context = file_descriptor.message_type.add(name="ContextEnvelope")
    field = context.field.add(name="event_id", number=1, label=1, type=9)
    field = context.field.add(name="threat_score", number=2, label=1, type=1)
    field = context.field.add(name="unified_context_vector", number=3, label=3, type=1)
    field = context.field.add(
        name="masked_payload",
        number=4,
        label=1,
        type=11,
        type_name=".google.protobuf.Struct",
    )
    pool = descriptor_pool.Default()
    try:
        descriptor = pool.FindMessageTypeByName(
            "dissanayaka.ingestion.ContextEnvelope"
        )
    except KeyError:
        descriptor = pool.Add(file_descriptor).message_types_by_name["ContextEnvelope"]
    return GetMessageClass(descriptor)


ContextEnvelope = _context_message_class()


class UnifiedContextVectorBuilder:
    """Build deterministic mathematical context vectors and compact protobuf envelopes."""

    def build(
        self,
        payload: Mapping[str, Any],
        threat_score: float,
        include_structural_features: bool | None = None,
    ) -> list[float]:
        """Return a mathematical context vector.
        
        When include_structural_features is True or when payload is string-heavy (has no numeric leaves),
        maps [threat_score, payload_byte_length, structural_depth, masked_pii_count, syntactic_anomaly_token_density, ...numeric_leaves].
        When numeric leaves exist and include_structural_features is False/None, preserves compact [threat_score, ...numeric_leaves].
        """
        score = min(1.0, max(0.0, float(threat_score)))
        numeric_leaves: list[float] = []
        self._collect_numbers(payload, numeric_leaves)

        should_include_structural = (
            include_structural_features
            if include_structural_features is not None
            else (len(numeric_leaves) == 0)
        )

        if should_include_structural:
            byte_length = float(len(json.dumps(dict(payload), default=str)))
            depth = float(self._compute_structural_depth(payload))
            pii_count = float(self._count_masked_pii(payload))
            token_density = float(self._compute_syntactic_anomaly_token_density(payload))
            return [score, byte_length, depth, pii_count, token_density, *numeric_leaves]

        return [score, *numeric_leaves]

    def serialize(
        self,
        event_id: str,
        masked_payload: Mapping[str, Any],
        threat_score: float,
        vector: list[float] | None = None,
    ) -> bytes:
        """Serialize the masked payload, score, and vector as protobuf bytes."""
        context_vector = vector or self.build(masked_payload, threat_score)
        payload_struct = Struct()
        json_format.ParseDict(dict(masked_payload), payload_struct)
        envelope = ContextEnvelope(
            event_id=event_id,
            threat_score=threat_score,
            unified_context_vector=context_vector,
            masked_payload=payload_struct,
        )
        return envelope.SerializeToString()

    @classmethod
    def _compute_structural_depth(cls, value: Any, current_depth: int = 1) -> int:
        """Compute maximum nesting depth of dictionary or sequence."""
        if isinstance(value, Mapping) and value:
            return max(cls._compute_structural_depth(v, current_depth + 1) for v in value.values())
        if isinstance(value, (list, tuple)) and value:
            return max(cls._compute_structural_depth(v, current_depth + 1) for v in value)
        return current_depth

    @classmethod
    def _count_masked_pii(cls, value: Any) -> int:
        """Count occurrences of redacted PII markers."""
        count = 0
        if isinstance(value, str):
            if "[REDACTED" in value:
                count += 1
        elif isinstance(value, Mapping):
            for v in value.values():
                count += cls._count_masked_pii(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                count += cls._count_masked_pii(v)
        return count

    @classmethod
    def _compute_syntactic_anomaly_token_density(cls, value: Any) -> float:
        """Compute density of syntactic metacharacters (SQLi, XSS, scripts) across payload strings."""
        total_chars = 0
        syntax_chars = 0
        syntax_set = {"'", '"', ';', '<', '>', '=', '(', ')', '-', '/', '*', '`', '$', '%'}

        def collect_str_stats(val: Any) -> None:
            nonlocal total_chars, syntax_chars
            if isinstance(val, str):
                total_chars += len(val)
                syntax_chars += sum(1 for ch in val if ch in syntax_set)
            elif isinstance(val, Mapping):
                for v in val.values():
                    collect_str_stats(v)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    collect_str_stats(v)

        collect_str_stats(value)
        if total_chars == 0:
            return 0.0
        return round(syntax_chars / total_chars, 4)

    @classmethod
    def _collect_numbers(cls, value: Any, output: list[float]) -> None:
        """Collect finite numeric leaves from nested mappings and sequences."""
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            output.append(float(value))
        elif isinstance(value, Mapping):
            for child in value.values():
                cls._collect_numbers(child, output)
        elif isinstance(value, (list, tuple)):
            for child in value:
                cls._collect_numbers(child, output)
