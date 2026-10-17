"""Adapter from SR-BH 2020 raw CSV rows to LogiShield's UnifiedApiContext / VerifiedRuleInfo.

Dataset provenance
-------------------
SR-BH 2020 (Harvard Dataverse, DOI ``10.7910/DVN/OGOIXX``). License confirmed
directly from the Dataverse API's own ``license`` field as CC0 1.0 (Phase 12) --
not assumed from a mirror or secondary description. 907,814 real HTTP requests
captured from a single WordPress honeypot over 12 days in July 2020, logged
(never blocked) by ModSecurity 2.9.2 + OWASP CRS 3.3.0 in detection-only mode,
then manually reviewed and CAPEC-labeled.

The raw 38-column schema was confirmed byte-for-byte against the live file's
own header row via small, explicit ``HTTP Range`` requests (never a full
download) in Phase 13A, and spot-checked further via several additional
scattered Range samples in Phase 13B:

- 5 network/timing columns: ``timestamp``, ``src_ip``, ``src_port``, ``dst_ip``,
  ``dst_port``
- 15 request-side columns: ``request_http_method``, ``request_http_request``,
  ``request_http_protocol``, ``request_user_agent``, ``request_referer``,
  ``request_host``, ``request_origin``, ``request_cookie``,
  ``request_content_type``, ``request_accept``, ``request_accept_language``,
  ``request_accept_encoding``, ``request_do_not_track``, ``request_connection``,
  ``request_body``
- 4 response-side columns: ``response_http_protocol``,
  ``response_http_status_code``, ``response_http_status_message``,
  ``response_content_length``
- 14 label columns: ``000 - Normal`` plus 13 CAPEC category flags

Phase 13B's additional sampling (176 structurally-complete rows recovered from
5 scattered 16 KB windows, each validated against the known 38-column shape
before being trusted -- a 9-row boundary fragment was correctly discarded by
that check) found: ``request_http_request`` does carry a query string when
present (~12% of the sample); every sampled row's body was either empty (all
GET rows) or a short, consistently-shaped string starting with the same
character class across every non-empty case observed, consistent with
URL-encoded form data and with no JSON-shaped body observed; and every
sampled row's ``000 - Normal``/CAPEC flags were mutually consistent (never
both "normal" and "any CAPEC category" at once). All of this is a small-sample
finding, not a dataset-wide guarantee -- see :func:`parse_request_body` and
:func:`extract_binary_label` for how this module still verifies rather than
assumes consistency on every row it is given.

Scope and safety contract
--------------------------
This module converts exactly one already-parsed raw CSV row (a mapping keyed
by the raw column names above) into Member-4-facing objects. It never reads
the whole dataset and never downloads anything itself.

- **Allowlisted** into the model-facing :class:`UnifiedApiContext`: HTTP
  method, path (including any query string -- carried through verbatim,
  never split or re-derived, since this module does not invent structure the
  dataset itself doesn't provide), request-side headers, and a best-effort
  parse of the request body (see :func:`parse_request_body`).
- **Denylisted** -- see :data:`DENYLISTED_COLUMNS` -- and never read by any
  function in this module at all, let alone forwarded: every ``response_*``
  column, every label column (``000 - Normal`` and all 13 CAPEC columns, read
  only by the separate :func:`extract_binary_label`, never by the
  context-building functions), ``timestamp``, and every network-identifier
  column (``src_ip``, ``src_port``, ``dst_ip``, ``dst_port``). A future
  dataset-grouping/splitting pass may need ``src_ip``/``timestamp`` from the
  raw row directly -- that is this module's caller's business, not this
  module's; nothing here reads, returns, or depends on them.
- **Unknown columns tolerated, never forwarded**: every value this module
  reads is read by an explicit, fixed column name (see
  :data:`RAW_TO_HEADER_NAME`, :data:`REQUIRED_COLUMNS`, :data:`LABEL_COLUMNS`).
  Nothing here ever iterates over ``row``'s own keys to decide what to
  forward, so an extra column the dataset doesn't currently have cannot
  silently become a model feature just by existing in some future export.
- **Masking is never bypassed, never duplicated**: this module's output is an
  ordinary, unmasked :class:`UnifiedApiContext` -- exactly like any other
  LogiShield input -- that must still go through
  ``PrivacyMasker.mask_with_provenance`` like every other request. This
  module does not mask anything itself and must not be mistaken for a
  substitute for that stage.
- **No body structure is invented**: :func:`parse_request_body` only ever
  returns a successfully-parsed JSON object or URL-encoded form as a dict;
  anything else is reported as :attr:`BodyParseOutcome.UNSUPPORTED` with an
  empty payload, never a guessed structure.
- **Labels are kept out of the model-facing objects entirely**:
  :func:`extract_binary_label` is a fully separate function operating only on
  the label columns; nothing in :func:`to_unified_api_context` reads a label
  column, and nothing in :func:`extract_binary_label` reads a non-label
  column. There is no shared object type that carries both, so a future
  caller cannot accidentally wire a label into a feature by holding onto the
  wrong field of some combined result.
- **Rules are never real rules**: :func:`to_empty_verified_rule_info` returns
  an empty ``rules`` list for every SR-BH 2020 row, because this dataset
  carries no Formal Verification formal-verification output at all. That empty list is
  not, and must never be described as, evidence that formal verification was
  performed, attempted, or skipped for the event -- see ``service.py``'s own
  ``predict`` docstring for the same point about an empty rules list.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import ValidationError

from .models import UnifiedApiContext, VerifiedRuleInfo

# -- Raw schema, confirmed against the live file's own header row (Phase 13A/13B) --

NORMAL_COLUMN = "000 - Normal"

CAPEC_COLUMNS = (
    "272 - Protocol Manipulation",
    "242 - Code Injection",
    "88 - OS Command Injection",
    "126 - Path Traversal",
    "66 - SQL Injection",
    "16 - Dictionary-based Password Attack",
    "310 - Scanning for Vulnerable Software",
    "153 - Input Data Manipulation",
    "248 - Command Injection",
    "274 - HTTP Verb Tampering",
    "194 - Fake the Source of Data",
    "34 - HTTP Response Splitting",
    "33 - HTTP Request Smuggling",
)

LABEL_COLUMNS = (NORMAL_COLUMN,) + CAPEC_COLUMNS

# Columns this module must never read into a model-facing object, under any
# circumstance -- response-side data, every label, every network/timing
# identifier. See the module docstring's "Denylisted" section.
DENYLISTED_COLUMNS = frozenset(
    {
        "timestamp",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "response_http_protocol",
        "response_http_status_code",
        "response_http_status_message",
        "response_content_length",
    }
) | frozenset(LABEL_COLUMNS)

# Required for every row: a request with no method or no path is not a
# parseable HTTP request at all, so this is treated as a structural error,
# not an optional/missing field.
REQUIRED_COLUMNS = ("request_http_method", "request_http_request")

# Request-side header columns this module allowlists into UnifiedApiContext.headers.
# request_http_protocol is deliberately absent here: it is not response-side
# data, but it is also outside the explicit "method / path / headers / body"
# allowlist this phase was scoped to, so it is conservatively left unread.
RAW_TO_HEADER_NAME: dict[str, str] = {
    "request_user_agent": "User-Agent",
    "request_referer": "Referer",
    "request_host": "Host",
    "request_origin": "Origin",
    "request_cookie": "Cookie",
    "request_content_type": "Content-Type",
    "request_accept": "Accept",
    "request_accept_language": "Accept-Language",
    "request_accept_encoding": "Accept-Encoding",
    "request_do_not_track": "DNT",
    "request_connection": "Connection",
}


class SrBh2020AdapterError(ValueError):
    """Raised when a raw SR-BH 2020 row cannot be safely converted."""


class LabelExtractionError(SrBh2020AdapterError):
    """Raised when a row's label columns are missing or contradictory.

    Never silently repaired -- see :func:`extract_binary_label`.
    """


class BodyParseOutcome(str, Enum):
    """How :func:`parse_request_body` interpreted one row's ``request_body``."""

    EMPTY = "empty"
    JSON = "json"
    FORM_URLENCODED = "form_urlencoded"
    UNSUPPORTED = "unsupported"


def parse_request_body(raw_body: str) -> tuple[dict[str, Any], BodyParseOutcome]:
    """Best-effort, non-fabricating parse of one row's raw request body.

    Tries, in order, the two structured formats this dataset's bodies are
    expected to take (see the module docstring's Phase 13B sampling note):
    a JSON *object* (a JSON array/string/number is not attempted -- there is
    no dict-shaped representation to offer without inventing one), then
    standard URL-encoded form data. Neither attempt raises on failure; if
    both fail, the body is reported as :attr:`BodyParseOutcome.UNSUPPORTED`
    with an empty payload -- this function never guesses at a structure it
    cannot actually parse.
    """
    if raw_body == "":
        return {}, BodyParseOutcome.EMPTY

    try:
        parsed = json.loads(raw_body)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed, BodyParseOutcome.JSON

    try:
        form = urllib.parse.parse_qs(raw_body, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        form = None
    if form:
        payload: dict[str, Any] = {
            key: (values[0] if len(values) == 1 else values) for key, values in form.items()
        }
        return payload, BodyParseOutcome.FORM_URLENCODED

    return {}, BodyParseOutcome.UNSUPPORTED


def to_unified_api_context(
    row: Mapping[str, Any], event_id: str
) -> tuple[UnifiedApiContext, BodyParseOutcome]:
    """Convert one raw SR-BH 2020 row into a :class:`UnifiedApiContext`.

    ``event_id`` is supplied by the caller (e.g. a stable row index), never
    derived from a raw column -- SR-BH 2020 has no natural request-id column,
    and deriving one from ``timestamp`` would mean reading a denylisted
    column for a purpose that isn't even labeling or splitting.

    Reads only the columns named in :data:`REQUIRED_COLUMNS` and
    :data:`RAW_TO_HEADER_NAME`, plus ``request_body`` -- never any column in
    :data:`DENYLISTED_COLUMNS`, and never any column not already named by
    this module, regardless of what else ``row`` contains.

    Raises :class:`SrBh2020AdapterError` if a required column is missing or
    empty, or if the resulting fields fail :class:`UnifiedApiContext`'s own
    validation (e.g. empty method/path). Never includes a raw field's value
    in its own error message -- see the module docstring's masking note;
    even an error about a malformed row must not become a channel for
    leaking whatever was actually in it.
    """
    for column in REQUIRED_COLUMNS:
        if not row.get(column):
            raise SrBh2020AdapterError(f"missing or empty required column: {column!r}")

    method = str(row["request_http_method"])
    path = str(row["request_http_request"])

    headers: dict[str, str] = {}
    for raw_column, header_name in RAW_TO_HEADER_NAME.items():
        value = row.get(raw_column)
        if value:
            headers[header_name] = str(value)

    raw_body = row.get("request_body") or ""
    payload, body_outcome = parse_request_body(str(raw_body))

    try:
        context = UnifiedApiContext(
            event_id=event_id, method=method, path=path, headers=headers, payload=payload
        )
    except ValidationError as exc:
        raise SrBh2020AdapterError(
            f"row for event {event_id!r} failed UnifiedApiContext validation: "
            f"{'; '.join(error['msg'] for error in exc.errors())}"
        ) from exc

    return context, body_outcome


def to_empty_verified_rule_info(event_id: str) -> VerifiedRuleInfo:
    """Return the (always-empty) rule info for one SR-BH 2020 event.

    SR-BH 2020 carries no Formal Verification formal-verification output, so every
    event gets an empty ``rules`` list. This is valid input to
    ``TokenizationAIService`` (see ``service.py``'s own docstring) and is
    not, and must never be described as, a claim that formal verification
    was performed, attempted, or skipped for this event.
    """
    return VerifiedRuleInfo(event_id=event_id, rules=[])


def _read_label_flag(row: Mapping[str, Any], column: str) -> int:
    """Read one label column as a strict 0/1 flag.

    Never includes the offending value in its error message (it could, in a
    misaligned/corrupted row, actually hold unrelated raw content such as a
    cookie or body fragment) -- only the column name and the value's type.
    """
    if column not in row:
        raise LabelExtractionError(f"missing label column: {column!r}")
    value = row[column]
    if value in (0, 1) or value in ("0", "1"):
        return int(value)
    raise LabelExtractionError(
        f"label column {column!r} has a non-binary value (type={type(value).__name__})"
    )


def extract_binary_label(row: Mapping[str, Any]) -> int:
    """Return the binary anomaly label (0=benign, 1=anomalous) for one row.

    Mapping, confirmed non-inverted in Phase 13: ``000 - Normal == 1`` means
    benign (``0``); ``000 - Normal == 0`` means anomalous (``1``), which is
    cross-checked against the 13 CAPEC columns rather than trusted alone.

    Raises :class:`LabelExtractionError` -- never silently repairs -- if any
    label column is missing, non-binary, or contradictory: ``000 - Normal``
    and at least one CAPEC column both ``1`` at once, or ``000 - Normal``
    ``0`` with every CAPEC column ``0`` (an anomalous row with no attack
    category at all). Phase 13B's own 176-row sample had zero contradictions,
    but that is a small-sample finding, not a dataset-wide guarantee -- this
    function is what actually enforces consistency row by row.
    """
    normal_flag = _read_label_flag(row, NORMAL_COLUMN)
    capec_flags = [_read_label_flag(row, column) for column in CAPEC_COLUMNS]
    any_capec = any(flag == 1 for flag in capec_flags)

    if normal_flag == 1 and any_capec:
        raise LabelExtractionError(
            f"contradictory labels: {NORMAL_COLUMN!r}=1 but at least one CAPEC column=1"
        )
    if normal_flag == 0 and not any_capec:
        raise LabelExtractionError(
            f"contradictory labels: {NORMAL_COLUMN!r}=0 but no CAPEC column=1"
        )

    return 0 if normal_flag == 1 else 1
