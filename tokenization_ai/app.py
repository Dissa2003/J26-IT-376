"""Local FastAPI wrapper that exposes the LogiShield pipeline for inspection.

``POST /v1/analyze`` runs the *existing* :class:`TokenizationAIService`
(fusion -> masking -> normalization -> tokenization -> tensorization ->
anomaly score) on one request and returns a privacy-safe *projection* of
what each stage produced. Nothing here re-implements a pipeline stage and
nothing here changes what the model is given.

Two layers, deliberately separate
----------------------------------
1. **Internal pipeline** -- the complete post-masking record is tokenized,
   tensorized and scored exactly as in any other caller of
   ``TokenizationAIService``. The projection below never feeds back into it.
2. **Privacy-safe demonstration output** -- a fail-closed projection built
   only for display. Masking is heuristic (key names, email/card/IP-shaped
   values, sensitive query parameters), so "not masked" does NOT mean
   "safe". The projection therefore inverts the default: a caller-supplied
   literal is shown only when its safety is *explicitly established*;
   everything else is replaced by a typed placeholder.

What counts as explicitly safe
-------------------------------
- Reserved control tags and masking placeholders produced by the pipeline.
- The finite set of standard HTTP method names.
- A literal that the fitted vocabulary already knows. The vocabulary is
  frozen at fit time from the trusted training corpus, so membership says
  "this string came from that corpus", not "this request contained it"
  (API literals are checked in the API namespace, rule literals in the rule
  namespace). This is only as safe as the corpus: a vocabulary fitted on
  sensitive real traffic would need a different policy.
- A value located at a path the trusted masker reports as masked (its
  displayed content is always the placeholder constant).
- Booleans and nulls (no content beyond one bit).
Everything else -- unknown keys, values, header names, path segments, query
names/values, numbers, rule ids and rule attributes -- is hidden. Structure,
token roles, masking indicators, counts and timings are preserved.

Other contracts
----------------
- ``rule_source`` is mandatory so a response can never be mistaken for one
  that used genuinely verified rules: ``"none"``, ``"mock_fixture"`` or
  ``"verified"`` (asserted by the caller). This service never verifies
  rules; ``verification_performed_by_this_service`` is always ``false``.
- The anomaly score comes from a small model fitted on hand-written
  synthetic fixtures. It is not evidence of real-world detection accuracy.
- ``event_id`` is a caller-chosen correlation id and is echoed; it is
  restricted to a short identifier alphabet and must not carry secrets.
- Error bodies carry stable codes and generic text built only from fixed
  field names and error kinds; pydantic's ``input`` echo, exception
  messages and caller-supplied key names never reach a response or a log.
- Local development service only: bound to 127.0.0.1, no authentication.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shared_contracts.models import ErrorResponse
from api.routes import router as policy_router

from .adapters import InputContractError, to_unified_api_context, to_verified_rule_info
from .dashboard import install_dashboard
from .demo import _build_demo_service
from .masking import MASKED_AUTH, MASKED_IP, MASKED_PII, MASKED_SECRET
from .metrics import runtime_metrics_from_trace
from .models import TokenSequence
from .service import TokenizationAIService
from .tokenizer import TOKEN_ROLE_LITERAL_API, TOKEN_ROLE_LITERAL_RULE

log = logging.getLogger("tokenization_ai")

# Static demo client (presentation only; see static/demo.js). Served from the
# same origin as the API so no CORS or third-party origin is involved.
_STATIC_DIR = Path(__file__).parent / "static"
_DEMO_ASSETS = {
    "demo.css": "text/css; charset=utf-8",
    "demo.js": "text/javascript; charset=utf-8",
    "scenarios.json": "application/json",
    "favicon.svg": "image/svg+xml",
}
_DEMO_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

CONTRACT_VERSION = "logishield-analysis-v2"

MAX_BODY_BYTES = 64 * 1024
MAX_COLLECTION_ITEMS = 50
MAX_NESTING_DEPTH = 6
MAX_TOKENS_RETURNED = 256
MAX_TOKEN_CHARS = 64
MAX_MASKED_FIELDS_RETURNED = 50
MAX_RULE_IDS_RETURNED = 20

UNRECOGNIZED_VALUE = "[UNRECOGNIZED_VALUE]"
UNRECOGNIZED_KEY = "[UNRECOGNIZED_KEY]"
UNRECOGNIZED_SEGMENT = "[UNRECOGNIZED_SEGMENT]"
UNRECOGNIZED_PARAM = "[UNRECOGNIZED_PARAM]"
UNRECOGNIZED_METHOD = "[UNRECOGNIZED_METHOD]"
UNRECOGNIZED_LITERAL = "[UNRECOGNIZED_LITERAL]"
UNRECOGNIZED_RULE_LITERAL = "[UNRECOGNIZED_RULE_LITERAL]"
UNRECOGNIZED_RULE_ID = "[UNRECOGNIZED_RULE_ID]"
UNRECOGNIZED_NUMBER = "[UNRECOGNIZED_NUMBER]"
TRUNCATED_DEPTH = "[TRUNCATED_DEPTH]"

_HTTP_METHODS = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"}
)
_MASK_PLACEHOLDERS = frozenset({MASKED_AUTH, MASKED_SECRET, MASKED_PII, MASKED_IP})
_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_PATH_ROOTS = frozenset({"payload", "headers", "path", "client_ip"})

# Field names fixed by this API and the adapter contracts. Only these may
# appear in an error's field location; anything caller-chosen is replaced.
_REQUEST_FIELDS = frozenset({"body", "context", "rule_source", "rule_info"})
_CONTRACT_FIELDS = _REQUEST_FIELDS | frozenset(
    {
        "event_id",
        "metadata",
        "method",
        "path",
        "content_type",
        "client_ip",
        "headers",
        "payload",
        "threat_score",
        "timestamp",
        "rules",
        "rule_id",
        "attributes",
    }
)

DEFAULT_MODEL_NOTE = (
    "Demo model fitted on two hand-written synthetic fixtures. The score shows the "
    "pipeline is wired end to end; it is not evidence of real-world detection accuracy."
)
INJECTED_MODEL_NOTE = (
    "Caller-supplied service; training data and accuracy are not described by this API."
)
PROJECTION_NOTE = (
    "The model scored the complete post-masking input. This projection shows only masking "
    "placeholders, structure, and literals already known to the fitted vocabulary; every "
    "other caller-supplied value is hidden and does not alter what the model received."
)


class RuleSource(str, Enum):
    NONE = "none"
    MOCK_FIXTURE = "mock_fixture"
    VERIFIED = "verified"


class AnalyzeRequest(BaseModel):
    """Request body. ``context`` and ``rule_info`` use the existing adapter contracts."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "context": {
                        "event_id": "evt-demo-1",
                        "metadata": {
                            "method": "POST",
                            "path": "/login",
                            "headers": {"Authorization": "Bearer synthetic-not-a-token"},
                        },
                        "payload": {"username": "admin", "password": "synthetic-pass"},
                        "threat_score": 0.5,
                    },
                    "rule_source": "mock_fixture",
                    "rule_info": {
                        "event_id": "evt-demo-1",
                        "rules": [{"rule_id": "MOCK-R-SQLI", "severity": "high"}],
                    },
                }
            ]
        },
    )

    context: dict[str, Any] = Field(
        description=(
            "Ingestion interception envelope (event_id, metadata{method,path,...}, payload, "
            "optional threat_score/timestamp) or an event whose payload carries _http_metadata."
        )
    )
    rule_source: RuleSource = Field(
        description="Origin of the rules: 'none', 'mock_fixture', or 'verified' (caller-asserted)."
    )
    rule_info: dict[str, Any] | None = Field(
        default=None,
        description="Verified-rule information {event_id, rules:[{rule_id,...}]}. Omit for 'none'.",
    )


class RuleStatus(BaseModel):
    source: RuleSource
    label: str
    rule_count: int
    rule_ids: list[str]
    rule_ids_truncated: bool
    verification_performed_by_this_service: Literal[False] = False
    note: str


class ModelStatus(BaseModel):
    model_version: str
    note: str


class Provenance(BaseModel):
    input_origin: Literal["caller_supplied_request"] = "caller_supplied_request"
    rule_status: RuleStatus
    model: ModelStatus
    detection_accuracy_claim: Literal["none"] = "none"


class PrivacyView(BaseModel):
    output_kind: Literal["privacy_safe_projection"] = "privacy_safe_projection"
    policy: Literal["fail_closed_known_literals_only"] = "fail_closed_known_literals_only"
    hidden_in_representation: int
    hidden_in_tokens: int
    note: str = PROJECTION_NOTE


class MaskingView(BaseModel):
    masked_field_count: int
    masked_fields: list[str]
    masked_fields_truncated: bool


class MaskedRepresentation(BaseModel):
    method: str
    path: str
    headers: dict[str, Any]
    payload: dict[str, Any]
    truncated: bool


class TokenItem(BaseModel):
    token: str
    role: str


class TokenView(BaseModel):
    total: int
    returned: int
    truncated: bool
    items: list[TokenItem]


class TensorView(BaseModel):
    max_length: int
    token_count_before_truncation: int
    tokens_in_tensor: int
    padded_positions: int
    payload_truncated: bool
    unknown_token_count: int
    vocabulary_size: int


class PredictionView(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    model_version: str


class StageTimingView(BaseModel):
    stage: str
    elapsed_ms: float
    success: bool
    summary: dict[str, int | bool]


class TimingsView(BaseModel):
    total_elapsed_ms: float
    stages: list[StageTimingView]


class AnalyzeResponse(BaseModel):
    contract_version: Literal["logishield-analysis-v2"] = CONTRACT_VERSION
    event_id: str
    provenance: Provenance
    privacy: PrivacyView
    masking: MaskingView
    masked_representation: MaskedRepresentation
    tokens: TokenView
    tensor: TensorView
    prediction: PredictionView
    timings: TimingsView


_RULE_LABELS = {
    RuleSource.NONE: "no verified-rule input",
    RuleSource.MOCK_FIXTURE: "mock rule fixtures",
    RuleSource.VERIFIED: "caller-asserted verified rules (not verified by this service)",
}
_RULE_NOTES = {
    RuleSource.NONE: "No verified-rule input was supplied; this is not a verification result.",
    RuleSource.MOCK_FIXTURE: "Rules are illustrative mock fixtures, not genuine Formal Verification output.",
    RuleSource.VERIFIED: (
        "Rules were asserted by the caller to be verified; this service did not verify them."
    ),
}


def create_app(
    service: TokenizationAIService | None = None, model_note: str | None = None
) -> FastAPI:
    """Build the app. ``service`` is injectable; the default is the PP1 demo service."""
    if service is None:
        service = _build_demo_service()
        note = model_note or DEFAULT_MODEL_NOTE
    else:
        note = model_note or INJECTED_MODEL_NOTE

    app = FastAPI(title="tokenization_ai")

    def _ready() -> bool:
        return bool(service.tensorizer.vocabulary.is_fitted and service.model.is_trained)

    def _error(status: int, code: str, detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(error=code, detail=detail).model_dump(),
        )

    @app.middleware("http")
    async def _bound_body(request: Request, call_next):
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None:
                return _error(411, "length_required", "Content-Length header is required.")
            if not length.isdigit() or int(length) > MAX_BODY_BYTES:
                return _error(
                    413, "payload_too_large", f"Request body exceeds {MAX_BODY_BYTES} bytes."
                )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(_request: Request, exc: RequestValidationError):
        # Deliberately omit pydantic's "input"/"msg" and any caller-chosen key
        # name in "loc": they can quote request content.
        return _error(422, "invalid_request", _describe_errors(exc.errors()))

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception):
        log.error("unexpected analysis failure: %s", type(exc).__name__)
        return _error(500, "internal_error", "Unexpected server error.")

    @app.get("/health")
    def health():
        ready = _ready()
        body = {
            "status": "ok" if ready else "not_ready",
            "service": "tokenization_ai",
            "ready": ready,
        }
        return JSONResponse(status_code=200 if ready else 503, content=body)

    @app.get("/demo", include_in_schema=False)
    def demo_page():
        return HTMLResponse(
            (_STATIC_DIR / "index.html").read_text(encoding="utf-8"), headers=_DEMO_HEADERS
        )

    @app.get("/demo/assets/{name}", include_in_schema=False)
    def demo_asset(name: str):
        media_type = _DEMO_ASSETS.get(name)  # fixed allowlist; no path joining on input
        if media_type is None:
            return _error(404, "not_found", "Unknown demo asset.")
        return Response(
            (_STATIC_DIR / name).read_bytes(), media_type=media_type, headers=_DEMO_HEADERS
        )

    @app.post(
        "/v1/analyze",
        response_model=AnalyzeResponse,
        responses={
            411: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def analyze(request: AnalyzeRequest):
        if not _ready():
            return _error(503, "model_not_ready", "Vocabulary or model is not fitted/trained.")

        try:
            context = to_unified_api_context(request.context)
            rule_payload = request.rule_info
            if request.rule_source is RuleSource.NONE:
                if rule_payload is not None and rule_payload.get("rules"):
                    return _error(
                        422,
                        "rule_source_conflict",
                        "rule_source 'none' cannot be combined with a non-empty rule list.",
                    )
                rule_payload = {"event_id": context.event_id, "rules": []}
            elif rule_payload is None:
                return _error(
                    422,
                    "rule_info_required",
                    f"rule_source '{request.rule_source.value}' requires rule_info.",
                )
            rule_info = to_verified_rule_info(rule_payload)
        except InputContractError as exc:
            return _error(422, "input_contract_violation", _contract_detail(exc))

        if request.rule_source is not RuleSource.NONE and not rule_info.rules:
            return _error(
                422,
                "rule_source_conflict",
                f"rule_source '{request.rule_source.value}' requires at least one rule.",
            )
        if not _EVENT_ID.fullmatch(context.event_id):
            return _error(
                422,
                "invalid_event_id",
                "event_id must be 1-64 characters of letters, digits, '.', '_', ':' or '-'.",
            )
        if context.event_id != rule_info.event_id:
            return _error(
                422,
                "event_id_mismatch",
                "context.event_id and rule_info.event_id must refer to the same event.",
            )

        trace = service.tracer.start(context.event_id)
        try:
            result = service.analyze(context, rule_info, trace=trace)
        except InputContractError:
            return _error(422, "input_contract_violation", "Input violates the fusion contract.")
        except ValueError:
            return _error(
                422,
                "pipeline_input_rejected",
                "Input could not be represented within the model's maximum length.",
            )
        except RuntimeError:
            log.error("analysis failed: model or vocabulary not ready")
            return _error(503, "model_not_ready", "Vocabulary or model is not fitted/trained.")

        return _build_response(request.rule_source, rule_info, result, trace, note, service)

    install_dashboard(app)  # additive: GET /dashboard only; /demo and /v1/analyze unchanged
    app.include_router(policy_router)
    return app


# --- error-text sanitizing -------------------------------------------------


def _safe_loc(loc) -> str:
    parts = []
    for part in loc:
        if isinstance(part, int):
            parts.append(str(part))
        elif isinstance(part, str) and part in _CONTRACT_FIELDS:
            parts.append(part)
        else:
            parts.append("[key]")
    return ".".join(parts) or "[body]"


def _describe_errors(errors) -> str:
    return "; ".join(f"{_safe_loc(e['loc'])}: {e['type']}" for e in errors[:10])


def _contract_detail(exc: InputContractError) -> str:
    """Describe an adapter failure without relaying any caller-chosen text.

    Adapter messages built from a pydantic failure embed the field location,
    which can include caller-chosen header or payload key names; those are
    rebuilt here from the structured cause with unknown names replaced. The
    adapters' other messages are fixed strings naming only contract fields.
    """
    cause = exc.__cause__
    if isinstance(cause, ValidationError):
        return "invalid input: " + _describe_errors(cause.errors())
    return str(exc)


# --- fail-closed projection --------------------------------------------------


class _Projector:
    """Decides, per string, whether it is explicitly safe to display."""

    def __init__(self, service: TokenizationAIService, masked_paths) -> None:
        self._vocabulary = service.tensorizer.vocabulary
        self._masked_paths = masked_paths
        self._cache: dict[tuple[str, str], bool] = {}
        self.hidden = 0
        self.truncated = False

    def known(self, text: str, role: str = TOKEN_ROLE_LITERAL_API) -> bool:
        key = (text, role)
        cached = self._cache.get(key)
        if cached is None:
            probe = TokenSequence(event_id="probe", tokens=[text], roles=[role])
            cached = self._vocabulary.encode(probe)[0] != self._vocabulary.unk_id
            self._cache[key] = cached
        return cached

    def hide(self, placeholder: str) -> str:
        self.hidden += 1
        return placeholder

    def text(self, value: str, placeholder: str) -> str:
        return value if self.known(value) else self.hide(placeholder)

    def method(self, value: str) -> str:
        return value if value in _HTTP_METHODS else self.hide(UNRECOGNIZED_METHOD)

    def key(self, value: str, position: int) -> str:
        if self.known(value):
            return value
        self.hidden += 1
        return f"{UNRECOGNIZED_KEY}_{position}"

    def path(self, value: str) -> str:
        base, sep, query = value.partition("?")
        segments = [
            segment if not segment else self.text(segment, UNRECOGNIZED_SEGMENT)
            for segment in base.split("/")
        ]
        out = "/".join(segments)
        if sep:
            pairs = []
            for name, val in urllib.parse.parse_qsl(query, keep_blank_values=True):
                shown_name = self.text(name, UNRECOGNIZED_PARAM)
                shown_val = (
                    val
                    if val in _MASK_PLACEHOLDERS and ("path",) in self._masked_paths
                    else self.text(val, UNRECOGNIZED_VALUE)
                )
                pairs.append(f"{shown_name}={shown_val}")
            out += "?" + "&".join(pairs)
        return out

    def value(self, value: Any, path: tuple, depth: int) -> Any:
        if path in self._masked_paths:
            return value if value in _MASK_PLACEHOLDERS else self.hide(UNRECOGNIZED_VALUE)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return self.text(value, UNRECOGNIZED_VALUE)
        if isinstance(value, (int, float)):
            return value if self.known(str(value)) else self.hide(UNRECOGNIZED_NUMBER)
        if isinstance(value, dict):
            if depth >= MAX_NESTING_DEPTH:
                self.truncated = True
                return TRUNCATED_DEPTH
            if len(value) > MAX_COLLECTION_ITEMS:
                self.truncated = True
            return {
                self.key(str(k), i): self.value(v, path + (str(k),), depth + 1)
                for i, (k, v) in enumerate(list(value.items())[:MAX_COLLECTION_ITEMS])
            }
        if isinstance(value, (list, tuple)):
            if depth >= MAX_NESTING_DEPTH:
                self.truncated = True
                return TRUNCATED_DEPTH
            if len(value) > MAX_COLLECTION_ITEMS:
                self.truncated = True
            return [
                self.value(v, path + (i,), depth + 1)
                for i, v in enumerate(list(value)[:MAX_COLLECTION_ITEMS])
            ]
        return self.hide(UNRECOGNIZED_VALUE)

    def field_path(self, path: tuple) -> str:
        parts = []
        for index, part in enumerate(path):
            if isinstance(part, int):
                parts.append(str(part))
            elif index == 0 and part in _PATH_ROOTS:
                parts.append(part)
            else:
                parts.append(part if self.known(part) else UNRECOGNIZED_KEY)
        return ".".join(parts)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _build_response(rule_source, rule_info, result, trace, note, service) -> AnalyzeResponse:
    projector = _Projector(service, result.masked_paths)
    ctx = result.normalized.context

    representation = MaskedRepresentation(
        method=projector.method(ctx.method),
        path=_clip(projector.path(ctx.path), MAX_TOKEN_CHARS * 4),
        headers={
            projector.key(str(name), i): projector.value(val, ("headers", name), 1)
            for i, (name, val) in enumerate(list(ctx.headers.items())[:MAX_COLLECTION_ITEMS])
        },
        payload=projector.value(ctx.payload, ("payload",), 0),
        truncated=len(ctx.headers) > MAX_COLLECTION_ITEMS,
    )
    representation.truncated = representation.truncated or projector.truncated
    hidden_in_representation = projector.hidden

    masked_fields = sorted(projector.field_path(p) for p in result.masked_paths)
    masked_view = MaskingView(
        masked_field_count=len(masked_fields),
        masked_fields=[_clip(f, MAX_TOKEN_CHARS) for f in masked_fields[:MAX_MASKED_FIELDS_RETURNED]],
        masked_fields_truncated=len(masked_fields) > MAX_MASKED_FIELDS_RETURNED,
    )

    sequence = result.sequence
    projector.hidden = 0
    items: list[TokenItem] = []
    for token, role in list(zip(sequence.tokens, sequence.roles))[:MAX_TOKENS_RETURNED]:
        if role == TOKEN_ROLE_LITERAL_API:
            shown = (
                token
                if token in _HTTP_METHODS or projector.known(token)
                else projector.hide(UNRECOGNIZED_LITERAL)
            )
        elif role == TOKEN_ROLE_LITERAL_RULE:
            shown = (
                token
                if projector.known(token, TOKEN_ROLE_LITERAL_RULE)
                else projector.hide(UNRECOGNIZED_RULE_LITERAL)
            )
        else:
            # Control tags and masking placeholders: role comes from the
            # tokenizer, but the text must still be a reserved vocabulary token.
            shown = token if projector.known(token, role) else projector.hide(UNRECOGNIZED_LITERAL)
        items.append(TokenItem(token=_clip(shown, MAX_TOKEN_CHARS), role=role))
    token_view = TokenView(
        total=len(sequence.tokens),
        returned=len(items),
        truncated=len(sequence.tokens) > len(items),
        items=items,
    )
    hidden_in_tokens = projector.hidden

    tensorizer = service.tensorizer
    mask = result.model_input.attention_mask
    ids = result.model_input.input_ids
    unk_id = tensorizer.vocabulary.unk_id
    tensor_view = TensorView(
        max_length=tensorizer.max_length,
        token_count_before_truncation=len(sequence.tokens),
        tokens_in_tensor=sum(mask),
        padded_positions=len(mask) - sum(mask),
        payload_truncated=len(sequence.tokens) > tensorizer.max_length,
        unknown_token_count=sum(1 for i, m in zip(ids, mask) if m and i == unk_id),
        vocabulary_size=len(tensorizer.vocabulary),
    )

    rule_ids = [
        rule.rule_id
        if projector.known(rule.rule_id, TOKEN_ROLE_LITERAL_RULE)
        else UNRECOGNIZED_RULE_ID
        for rule in rule_info.rules[:MAX_RULE_IDS_RETURNED]
    ]
    rule_status = RuleStatus(
        source=rule_source,
        label=_RULE_LABELS[rule_source],
        rule_count=len(rule_info.rules),
        rule_ids=[_clip(r, MAX_TOKEN_CHARS) for r in rule_ids],
        rule_ids_truncated=len(rule_info.rules) > MAX_RULE_IDS_RETURNED,
        note=_RULE_NOTES[rule_source],
    )

    metrics = runtime_metrics_from_trace(trace)
    stages = [
        StageTimingView(
            stage=s.stage, elapsed_ms=s.elapsed_ms, success=s.success, summary=dict(s.summary)
        )
        for s in trace.stages
    ]

    prediction = result.prediction
    return AnalyzeResponse(
        event_id=prediction.event_id,
        provenance=Provenance(
            rule_status=rule_status,
            model=ModelStatus(model_version=prediction.model_version, note=note),
        ),
        privacy=PrivacyView(
            hidden_in_representation=hidden_in_representation,
            hidden_in_tokens=hidden_in_tokens,
        ),
        masking=masked_view,
        masked_representation=representation,
        tokens=token_view,
        tensor=tensor_view,
        prediction=PredictionView(score=prediction.score, model_version=prediction.model_version),
        timings=TimingsView(total_elapsed_ms=metrics.total_elapsed_ms, stages=stages),
    )


app = create_app()
