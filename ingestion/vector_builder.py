"""Unified Context Vector construction and protobuf serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message
from google.protobuf.message_factory import GetMessageClass
from google.protobuf.struct_pb2 import Struct


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
    """Build deterministic numeric vectors and compact protobuf envelopes."""

    def build(self, payload: Mapping[str, Any], threat_score: float) -> list[float]:
        """Return ``[threat_score, ...numeric payload values]`` in traversal order."""
        values: list[float] = [float(threat_score)]
        self._collect_numbers(payload, values)
        return values

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
