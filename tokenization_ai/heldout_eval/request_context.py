"""``request_to_context-v1``: maps one synthetic evaluation request to the pipeline's input context.

Evaluation infrastructure only (draft protocol heldout-eval-1.2.0-draft, section 12.9). It changes no production
module. It is a pure, deterministic function of the request content: the record's label, scenario family, template
id, split, source id, timestamp and any prediction are never read (only ``record.record_id`` and
``record.features`` are touched), and nothing here trains or scores a model.

Mapping (everything not listed below is passed through unchanged)
-----------------------------------------------------------------
* ``event_id``     the opaque synthetic ``record_id``. It is the context identifier only; the tokenizer never emits it.
* ``method``       as given. ``Normalizer`` upper-cases it later.
* ``path``         the complete string **including** any ``?query``, unsplit and undecoded. This matches the production
                   adapter (``adapters._build_context`` passes ``metadata["path"]`` as one string; ``UnifiedApiContext`` has
                   no query field). ``PrivacyMasker._mask_path`` already masks sensitive query values.
* ``headers``      every request header, copied (names as given, values untouched), including ``content-type`` and
                   ``authorization``. Masking of credentials is left to the existing ``PrivacyMasker``; this function
                   never inspects or removes a value. Duplicate header names (after lower-casing) are rejected rather
                   than silently merged.
* ``content_type`` the value of the ``content-type`` header (case-insensitive name), or ``None``. The tokenizer does not
                   read this field; the header itself is what the model sees.
* ``payload``      from the body, decided by the **content** of the body and never by the declared content type or by any
                   label (so a content-type mismatch stays visible to the model only as the combination of a header
                   literal and a payload shape):

  1. empty body                                   -> ``{}``
  2. body that parses as a JSON object           -> that dict
  3. form-like body (contains ``=`` and, once stripped, no space) -> ``urllib.parse.parse_qsl(keep_blank_values=True)``
     as a ``{name: str}`` dict (a repeated name keeps its last value)
  4. anything else (plain text, XML, a JSON non-object) -> ``{"text": body}``

Production-faithful versus an evaluation convention
---------------------------------------------------
Faithful to production: the field names and types of ``UnifiedApiContext``, the single-string path with its query, the
header dict, and the stage order that follows (fusion, masking, normalization, tokenization; see ``service._front``).
Evaluation convention (production only guarantees ``payload: dict``): rules 3 and 4 above. They are the same rules as the
diagnostic mapping ``policy_diagnostics.to_context``; they are frozen here as a dedicated, hashable function.

Known limitation (kept, not fixed): the production tokenizer splits a path on ``/``, so a query string stays inside the
last path segment and becomes one atomic literal. Query text is therefore (almost) invisible to Arms A and B, while a
character-n-gram arm sees it. This module does not change that.

Simulated rules are not attached here; the arm builders attach them (empty for Arm A, ``simpol-1`` for Arm B).
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

from pydantic import ValidationError

from ..models import UnifiedApiContext
from .splits import EvalRecord

MAPPING_ID = "request_to_context-v1"
TEXT_PAYLOAD_KEY = "text"


class ContextMappingError(ValueError):
    """The request cannot be mapped to a context. Messages never echo request content."""


def request_to_context(record: EvalRecord) -> UnifiedApiContext:
    """Map one evaluation record to a ``UnifiedApiContext`` (``request_to_context-v1``). Pure and deterministic."""
    if not isinstance(record, EvalRecord):
        raise ContextMappingError("request_to_context-v1 accepts an EvalRecord")
    features = record.features
    if not isinstance(features.method, str) or not isinstance(features.path, str) or not isinstance(features.body, str):
        raise ContextMappingError("method, path and body must be strings")
    headers = _headers(features.headers)
    content_type = next((v for k, v in headers.items() if k.lower() == "content-type"), None)
    try:
        return UnifiedApiContext(
            event_id=record.record_id, method=features.method, path=features.path, content_type=content_type,
            headers=headers, payload=_payload(features.body),
        )
    except ValidationError as exc:
        raise ContextMappingError("the request does not satisfy the unified context contract") from exc


def _headers(pairs: object) -> dict[str, str]:
    try:
        items = [(k, v) for k, v in pairs]  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ContextMappingError("headers must be (name, value) pairs") from exc
    out: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in items:
        if not isinstance(name, str) or not isinstance(value, str):
            raise ContextMappingError("header names and values must be strings")
        folded = name.strip().lower()
        if folded in seen:
            raise ContextMappingError("duplicate header names are not supported")
        seen.add(folded)
        out[name] = value
    return out


def _payload(body: str) -> dict[str, object]:
    if body == "":
        return {}
    try:
        parsed = json.loads(body)
    except (ValueError, RecursionError):  # not JSON (or absurdly nested): fall through to the other rules
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if "=" in body and " " not in body.strip():
        return dict(urllib.parse.parse_qsl(body, keep_blank_values=True))
    return {TEXT_PAYLOAD_KEY: body}


def source_sha256() -> str:
    """SHA-256 of this module's source with line endings normalized to LF (stable across checkouts), for the manifest."""
    text = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()
