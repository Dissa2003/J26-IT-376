"""Protobuf message classes for ``szcfm_context.proto``.

The descriptor is registered at import time from a ``FileDescriptorProto`` that
mirrors the ``.proto`` file, so the classes are genuine protobuf messages and
``SerializeToString()`` produces the standard binary wire format.
"""

from __future__ import annotations

from google.protobuf import descriptor_pb2, descriptor_pool
from google.protobuf.message_factory import GetMessageClass

PROTO_PACKAGE = "logishield.szcfm.v1"
PROTO_FILE = "szcfm_context.proto"
ENVELOPE_SCHEMA_VERSION = "1"

_F = descriptor_pb2.FieldDescriptorProto
_OPTIONAL, _REPEATED = _F.LABEL_OPTIONAL, _F.LABEL_REPEATED


def _add(message, name: str, number: int, type_: int, label: int = _OPTIONAL, type_name: str = "") -> None:
    field = message.field.add(name=name, number=number, type=type_, label=label)
    if type_name:
        field.type_name = type_name


def _file_descriptor() -> descriptor_pb2.FileDescriptorProto:
    file = descriptor_pb2.FileDescriptorProto(
        name=PROTO_FILE,
        package=PROTO_PACKAGE,
        syntax="proto3",
        dependency=["google/protobuf/struct.proto"],
    )

    metadata = file.message_type.add(name="GatewayMetadata")
    _add(metadata, "method", 1, _F.TYPE_STRING)
    _add(metadata, "path", 2, _F.TYPE_STRING)
    _add(metadata, "content_type", 3, _F.TYPE_STRING)
    _add(metadata, "client_role", 4, _F.TYPE_STRING)
    _add(metadata, "authorization_present", 5, _F.TYPE_BOOL)
    _add(metadata, "authorization_scheme", 6, _F.TYPE_STRING)
    headers_entry = metadata.nested_type.add(name="HeadersEntry")
    headers_entry.options.map_entry = True
    _add(headers_entry, "key", 1, _F.TYPE_STRING)
    _add(headers_entry, "value", 2, _F.TYPE_STRING)
    _add(
        metadata, "headers", 7, _F.TYPE_MESSAGE, _REPEATED,
        f".{PROTO_PACKAGE}.GatewayMetadata.HeadersEntry",
    )

    envelope = file.message_type.add(name="SZCFMEnvelope")
    _add(envelope, "schema_version", 1, _F.TYPE_STRING)
    _add(envelope, "event_id", 2, _F.TYPE_STRING)
    _add(envelope, "request_id", 3, _F.TYPE_STRING)
    _add(envelope, "correlation_id", 4, _F.TYPE_STRING)
    _add(envelope, "source", 5, _F.TYPE_STRING)
    _add(envelope, "ingested_at_unix_ns", 6, _F.TYPE_INT64)
    _add(envelope, "metadata", 7, _F.TYPE_MESSAGE, type_name=f".{PROTO_PACKAGE}.GatewayMetadata")
    _add(envelope, "masked_payload", 8, _F.TYPE_MESSAGE, type_name=".google.protobuf.Struct")
    _add(envelope, "feature_schema_version", 9, _F.TYPE_STRING)
    _add(envelope, "feature_vector", 10, _F.TYPE_DOUBLE, _REPEATED)
    _add(envelope, "has_prior_threat_score", 11, _F.TYPE_BOOL)
    _add(envelope, "prior_threat_score", 12, _F.TYPE_DOUBLE)
    return file


def _register():
    pool = descriptor_pool.Default()
    try:
        file_descriptor = pool.FindFileByName(PROTO_FILE)
    except KeyError:
        # Importing struct_pb2 registers google/protobuf/struct.proto in the pool.
        from google.protobuf import struct_pb2  # noqa: F401

        file_descriptor = pool.Add(_file_descriptor())
        file_descriptor = pool.FindFileByName(PROTO_FILE)
    return (
        GetMessageClass(file_descriptor.message_types_by_name["GatewayMetadata"]),
        GetMessageClass(file_descriptor.message_types_by_name["SZCFMEnvelope"]),
    )


GatewayMetadata, SZCFMEnvelope = _register()
ENVELOPE_MESSAGE_TYPE = SZCFMEnvelope.DESCRIPTOR.full_name
