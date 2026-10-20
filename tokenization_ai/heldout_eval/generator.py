"""Deterministic synthetic API request generator (draft protocol heldout-eval-1.2.0-draft).

Generates FABRICATED requests from the scenario families in ``synthetic_catalog.py``. No network
traffic, no real users, credentials or private data, and no external datasets are involved. Nothing
here trains, tunes or scores a model.

Design points
-------------
* **Templates first.** The splitter groups requests by their normalized template (``tmpl-1``), so a
  generated *template* is a structural skeleton (method, path shape, query parameter names and value
  classes, body shape). Variants of one template change only *values*, so they normalize to the same
  template. Each variant is checked (``GeneratorError`` on drift). If the requested number of
  templates cannot be reached because normalization collapses candidates, the actual count is
  reported in ``GenerationStats`` and never silently forced.
* **Labels come from construction, not from features.** ``label`` and the scenario family live in
  ``GenerationMetadata`` and in ``EvalRecord.meta`` (split metadata); neither is a request field.
  Request ids, source ids and timestamps are opaque and independent of the label (records are
  shuffled before ids and timestamps are assigned).
* **Shared pools and comparable structures.** Resources, sub-resources, methods, headers, user agents,
  roles and word lists are the same for every class, and normal and suspicious families deliberately
  produce comparable path depths, body sizes, field counts, nested shapes, content types and
  privileged-looking field names, so these are not class markers on their own. Where a structural
  property *is* the documented anomaly (for example nesting deeper than the legitimate depth) the
  association is intentional and is listed in ``PROTOCOL.md``.
* **Mixed-label templates are deliberate.** For ``mixed`` families one template holds both labels
  (the first two variants are one of each), decided by the family's stated rule.
* **Content-type mismatch means incompatibility** between the declared type and the actual body
  format (json, xml, form or plain text), never the declared value alone.
* **Reproducible.** The same config yields byte-identical records and ``fingerprint``.

Out of scope in this phase: stateful or multi-request scenarios, simulated rules, Member 2 reports.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import urllib.parse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import synthetic_catalog as cat
from .splits import EvalRecord, exact_fingerprint, group_id_for, make_record

GENERATOR_VERSION = "synthgen-3"
# Development / debugging seed (PROTOCOL.md section 12.5). The earlier default 5101 belongs to the burned synthgen-2
# pilot; the confirmatory seed 7101 and reserve seeds 7102 / 7103 are never a default and are never generated here.
DEFAULT_DATASET_SEED = 6101  # arbitrary, fixed in advance

# synthgen-3 corrections (PROTOCOL.md section 12.2.1). Design constants, fixed before any synthgen-3 data existed and
# not tuned on labels, audits or models.
_FILES_ROOT_READ_P = 0.12       # G1: crud_read paths that sit under /files
_DELETE_QUERY_P = 0.5           # G2: normal DELETE calls (crud_write) that carry an ordinary text query parameter
_TRAVERSAL_DELETE_P = 0.3       # G2: traversal_like templates that are a DELETE on an ordinary resource path
_TEXT_DELETE_P = 0.2            # G2: query-slot templates of the text families that are a DELETE call
_TIMESTAMP_BASE = 1_700_000_000.0
_SOURCE_POOL = 40
_MAX_DEEP = 9
_PRIVILEGED_BOOL = ("is_admin", "verified")


class GeneratorConfigError(ValueError):
    """The generator configuration is invalid."""


class GeneratorError(RuntimeError):
    """The generator violated one of its own invariants (for example template drift)."""


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = DEFAULT_DATASET_SEED
    n_templates: int = 250
    variants_min: int = 4
    variants_max: int = 8
    mixed_suspicious_probability: float = 0.4
    hard_fraction: float = 0.3
    family_weights: Mapping[str, int] | None = None
    max_attempt_factor: int = 60

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise GeneratorConfigError("seed must be an integer")
        for name in ("n_templates", "variants_min", "variants_max", "max_attempt_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise GeneratorConfigError(f"{name} must be an integer")
        if not 10 <= self.n_templates <= 20000:
            raise GeneratorConfigError("n_templates must be between 10 and 20000")
        if self.variants_min < 2:
            raise GeneratorConfigError("variants_min must be at least 2 (mixed templates need both labels)")
        if not self.variants_min <= self.variants_max <= 50:
            raise GeneratorConfigError("variants_max must be between variants_min and 50")
        if self.max_attempt_factor < 2:
            raise GeneratorConfigError("max_attempt_factor must be at least 2")
        for name in ("mixed_suspicious_probability", "hard_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise GeneratorConfigError(f"{name} must be a number strictly between 0 and 1")
        if self.family_weights is not None:
            if not isinstance(self.family_weights, Mapping) or not self.family_weights:
                raise GeneratorConfigError("family_weights must be a non-empty mapping")
            for fam, weight in self.family_weights.items():
                if fam not in cat.FAMILY_BY_NAME:
                    raise GeneratorConfigError("family_weights names an unknown family")
                if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
                    raise GeneratorConfigError("family weights must be non-negative integers")
            if sum(self.family_weights.values()) <= 0:
                raise GeneratorConfigError("family_weights must have a positive total")

    def weights(self) -> dict[str, int]:
        base = cat.default_weights()
        if self.family_weights is None:
            return base
        return {name: int(self.family_weights.get(name, 0)) for name in base}

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_version": GENERATOR_VERSION, "catalog_version": cat.CATALOG_VERSION, "seed": self.seed,
            "n_templates": self.n_templates, "variants_min": self.variants_min, "variants_max": self.variants_max,
            "mixed_suspicious_probability": self.mixed_suspicious_probability, "hard_fraction": self.hard_fraction,
            "family_weights": self.weights(), "max_attempt_factor": self.max_attempt_factor,
        }


@dataclass(frozen=True)
class GenerationMetadata:
    """Label and scenario information. Kept apart from request features; never a model input."""

    label: int
    family: str
    label_kind: str
    template_id: str
    variant_index: int
    hard_case: str  # "", "negative" or "positive"


@dataclass(frozen=True)
class GenerationStats:
    requested_templates: int
    actual_templates: int
    collapsed_specs: int
    attempts: int
    n_records: int
    duplicate_variants_skipped: int
    templates_all_normal: int
    templates_all_suspicious: int
    templates_mixed: int
    records_by_family: Mapping[str, int]
    templates_by_family: Mapping[str, int]
    hard_negative_records: int
    hard_positive_records: int

    @property
    def shortfall(self) -> int:
        return self.requested_templates - self.actual_templates


@dataclass(frozen=True)
class SyntheticDataset:
    config: GeneratorConfig
    records: tuple[EvalRecord, ...]
    metadata: Mapping[str, GenerationMetadata]
    stats: GenerationStats
    fingerprint: str
    generator_version: str = GENERATOR_VERSION


# -- template skeletons -------------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateSpec:
    family: str
    method: str
    segments: tuple[str | None, ...]  # None marks a numeric id slot
    query: tuple[tuple[str, str], ...] = ()
    body: tuple[tuple[str, Any], ...] = ()
    body_format: str = "none"  # "json", "form", "xml", "plain" or "none"
    text_slot: tuple[str, str] | None = None  # ("query"|"body"|"header", name)
    auth: bool = True
    admin_path: bool = False
    spec_hard: bool = False
    raw_bits: int = 0  # fixed length bucket (bit length) for xml/plain bodies, so the template cannot drift


class _Cycler:
    """Round-robin over a pool in shuffled rounds, so every item is used before any repeats."""

    def __init__(self, pool: tuple[str, ...], rng: random.Random) -> None:
        self._pool, self._rng, self._queue = pool, rng, []

    def next(self) -> str:
        if not self._queue:
            self._queue = list(self._pool)
            self._rng.shuffle(self._queue)
        return self._queue.pop()


class _Pickers:
    """Balanced resource and sub-resource choice, separately for normal and suspicious/mixed families.

    Without this, random draws leave some resources or sub-resources unused by one class, which would be a
    marker of that class. Balance is per group of families, never per label of an individual request.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self._res = {g: _Cycler(cat.RESOURCES, rng) for g in ("normal", "other")}
        self._sub = {g: _Cycler(cat.SUB_RESOURCES, rng) for g in ("normal", "other")}

    def for_family(self, family: str) -> "_Bound":
        group = "normal" if cat.FAMILY_BY_NAME[family].label_kind == "normal" else "other"
        return _Bound(self._res[group], self._sub[group])


@dataclass
class _Bound:
    _res: _Cycler
    _sub: _Cycler

    def res(self) -> str:
        return self._res.next()

    def sub(self) -> str:
        return self._sub.next()


def _flat_body(rng: random.Random, n: int, text_slot: bool = False, lists: bool = True) -> tuple[tuple[str, Any], ...]:
    """``n`` distinct flat fields; about 15% of bodies also carry one list-valued field."""
    pools = [(cat.FIELD_NAMES_TEXT, "text"), (cat.FIELD_NAMES_NUM, "num"), (cat.FIELD_NAMES_BOOL, "bool")]
    fields: list[tuple[str, Any]] = []
    used: set[str] = set()
    if text_slot:
        name = rng.choice(cat.FIELD_NAMES_TEXT)
        fields.append((name, "text"))
        used.add(name)
    while len(fields) < n:
        names, kind = rng.choice(pools)
        name = rng.choice(names)
        if name not in used:
            used.add(name)
            fields.append((name, kind))
    if lists and rng.random() < 0.15:
        name = rng.choice(cat.FIELD_NAMES_LIST)
        if name not in used:
            fields.append((name, rng.choice(("text_list", "num_list"))))
    return tuple(fields)


def _long_body(rng: random.Random, text_slot: bool = False) -> tuple[tuple[str, Any], ...]:
    return _flat_body(rng, rng.randint(7, 12), text_slot=text_slot) + (("long_text", "long_text"),)


def _resource_path(rng: random.Random, pk: _Bound, item: bool, sub: bool) -> tuple[str | None, ...]:
    parts: list[str | None] = [pk.res()]
    if item:
        parts.append(None)
    if sub:
        parts.append(pk.sub())
    return tuple(parts)


def _nested_read_path(rng: random.Random, pk: _Bound) -> tuple[str | None, ...]:
    """A resource followed by 0-5 alternating id / sub-resource segments (depth 1-6)."""
    parts: list[str | None] = [pk.res()]
    for index in range(rng.randint(0, 5)):
        parts.append(None if index % 2 == 0 else pk.sub())
    return tuple(parts)


def _search_sub(rng: random.Random, pk: _Bound) -> str:
    """Sub-resource for text endpoints; any sub-resource may appear in either class."""
    if rng.random() < 0.5:
        return rng.choice(("search", "lookup"))
    return pk.sub()


def _query_params(rng: random.Random, paging: bool, token: bool) -> tuple[tuple[str, str], ...]:
    params: list[tuple[str, str]] = []
    if paging:
        for name in rng.sample(cat.QUERY_NUM, rng.randint(1, 2)):
            params.append((name, "num"))
    if token:
        params.append((rng.choice(cat.QUERY_TOKEN), "token"))
    return tuple(params)


def _text_query(rng: random.Random, slot: str) -> tuple[tuple[str, str], ...]:
    """A text slot plus the optional paging / filter parameters, identical for every class."""
    return ((slot, "text"),) + _query_params(rng, rng.random() < 0.5, rng.random() < 0.6)


def _login_path(rng: random.Random) -> tuple[str, ...]:
    return (rng.choice(("auth", "account", "portal")),
            rng.choice(("login", "refresh", "session", "verify", "reset", "register")))


def _login_spec(family: str, rng: random.Random, slot: bool) -> TemplateSpec:
    """Unauthenticated login-style endpoint; with ``slot`` the username is the text slot."""
    return TemplateSpec(
        family, "POST", _login_path(rng),
        body=(("username", "text" if slot else "user"), ("passphrase", "secret")),
        body_format=rng.choice(("json", "form")), text_slot=("body", "username") if slot else None, auth=False)


def _privileged_fields(rng: random.Random, hard: bool, used: set[str]) -> list[tuple[str, Any]]:
    pool = cat.PRIVILEGED_FIELDS_QUIET if hard else cat.PRIVILEGED_FIELDS_OBVIOUS
    out: list[tuple[str, Any]] = []
    for name in rng.sample(pool, rng.randint(1, 2)):
        if name not in used:
            used.add(name)
            out.append((name, "bool" if name in _PRIVILEGED_BOOL else "text"))
    return out


def _build_spec(family: str, rng: random.Random, hard_fraction: float, pickers: _Pickers) -> TemplateSpec:
    ch = rng.choice
    pk = pickers.for_family(family)
    if family == "crud_read":
        segs = _nested_read_path(rng, pk)
        if rng.random() < _FILES_ROOT_READ_P:  # G1: a routine read under a /files path
            segs = ("files",) + segs
        return TemplateSpec(family, "GET", segs,
                            query=_query_params(rng, rng.random() < 0.7, rng.random() < 0.5))
    if family == "crud_write":
        method = ch(("POST", "PUT", "PATCH", "DELETE"))
        has_body = method != "DELETE"
        if method == "DELETE" and rng.random() < _DELETE_QUERY_P:
            # G2 (N2): a delete with an ordinary text query parameter, built exactly like the text families' queries
            slot = ch(cat.QUERY_TEXT)
            return TemplateSpec(family, method, _resource_path(rng, pk, True, False),
                                query=_text_query(rng, slot), text_slot=("query", slot))
        return TemplateSpec(family, method, _resource_path(rng, pk, method != "POST", False),
                            body=_flat_body(rng, rng.randint(2, 7)) if has_body else (),
                            body_format="json" if has_body else "none")
    if family == "search_filter":
        slot = ch(cat.QUERY_TEXT)
        return TemplateSpec(family, "GET", (pk.res(), _search_sub(rng, pk)), query=_text_query(rng, slot),
                            text_slot=("query", slot))
    if family == "auth_login":
        return _login_spec(family, rng, slot=False)
    if family == "json_create_nested":
        body = list(_flat_body(rng, rng.randint(1, 2)))
        used = {n for n, _ in body}
        for pool, kind in ((cat.FIELD_NAMES_OBJECT, ("object", (("city", "text"), ("zone", "num")))),
                           (cat.FIELD_NAMES_LIST, "text_list")):
            name = ch(pool)
            if name not in used:
                used.add(name)
                body.append((name, kind))
        return TemplateSpec(family, ch(("POST", "PUT")), _resource_path(rng, pk, False, False), body=tuple(body),
                            body_format="json")
    if family == "admin_legit":
        method = ch(("GET", "POST", "PUT", "PATCH", "DELETE"))
        has_body = method in ("POST", "PUT", "PATCH")
        return TemplateSpec(family, method, ("admin", pk.res(), pk.sub()),
                            body=_flat_body(rng, rng.randint(1, 3)) if has_body else (),
                            body_format="json" if has_body else "none", admin_path=True)
    if family == "health_status":
        segs = (("health",), ("ping",), ("status", pk.res()),
                ("status", pk.res(), pk.sub()))[rng.randint(0, 3)]
        return TemplateSpec(family, "GET", segs, auth=False)
    if family == "hn_keyword_text":
        if rng.random() < 0.5:
            slot = ch(cat.QUERY_TEXT)
            return TemplateSpec(family, "GET", (pk.res(), _search_sub(rng, pk)), query=_text_query(rng, slot),
                                text_slot=("query", slot), spec_hard=True)
        body = _flat_body(rng, rng.randint(2, 3), text_slot=True)
        return TemplateSpec(family, "POST", _resource_path(rng, pk, False, False), body=body, body_format="json",
                            text_slot=("body", body[0][0]), spec_hard=True)
    if family == "hn_dotted_paths":
        name = f"{rng.choice(cat.WORDS)}.final.{ch(('pdf', 'txt', 'csv', 'png'))}"
        folders = tuple(rng.choice(cat.WORDS) for _ in range(rng.randint(0, 3)))
        root = ch(("documents", "files"))  # G1: ordinary dotted files also live under /files
        return TemplateSpec(family, "GET", (root, pk.res(), None, *folders, name),
                            query=_query_params(rng, rng.random() < 0.4, False), spec_hard=True)
    if family == "hn_long_payload":
        return TemplateSpec(family, "POST", _resource_path(rng, pk, False, False), body=_long_body(rng),
                            body_format="json", spec_hard=True)
    if family == "alt_content_type":
        fmt = ch(("plain", "xml"))
        return TemplateSpec(family, ch(("POST", "PUT")), _resource_path(rng, pk, False, rng.random() < 0.3),
                            body_format=fmt, raw_bits=ch((5, 6, 7)), spec_hard=True)
    if family == "admin_privileged_update":
        body = list(_flat_body(rng, rng.randint(1, 3), lists=False))
        used = {n for n, _ in body}
        body += _privileged_fields(rng, rng.random() < hard_fraction, used)
        return TemplateSpec(family, ch(("POST", "PUT", "PATCH")),
                            ("admin", pk.res(), pk.sub()), body=tuple(body),
                            body_format="json", admin_path=True, spec_hard=True)
    if family == "hn_nested_config":
        body = _flat_body(rng, rng.randint(1, 2), lists=False) + (("config", ("deep", rng.randint(3, 5))),)
        return TemplateSpec(family, ch(("POST", "PUT")), _resource_path(rng, pk, False, True), body=body,
                            body_format="json", spec_hard=True)
    if family in ("sqli_like", "cmdi_like", "markup_like"):
        roll = rng.random()
        if roll >= 0.8:
            return _login_spec(family, rng, slot=True)
        if roll < 0.4:
            slot = ch(cat.QUERY_TEXT)
            if rng.random() < _TEXT_DELETE_P:
                # G2 (S2): the same query slot on a DELETE of an item, with the path shape of the normal delete (N2)
                return TemplateSpec(family, "DELETE", _resource_path(rng, pk, True, False),
                                    query=_text_query(rng, slot), text_slot=("query", slot))
            return TemplateSpec(family, "GET", (pk.res(), _search_sub(rng, pk)),
                                query=_text_query(rng, slot), text_slot=("query", slot))
        large = rng.random() < 0.3
        body = _long_body(rng, text_slot=True) if large else _flat_body(rng, rng.randint(2, 3), text_slot=True)
        return TemplateSpec(family, ch(("POST", "PUT", "PATCH")), _resource_path(rng, pk, False, False), body=body,
                            body_format="json", text_slot=("body", body[0][0]))
    if family == "traversal_like":
        hard = rng.random() < hard_fraction
        if rng.random() < _TRAVERSAL_DELETE_P:
            # G2 (S1): a DELETE on an ordinary resource path whose tail is a traversal-like construction. No query, so
            # the delete cells (no query / text query) hold both classes.
            return TemplateSpec(family, "DELETE", (pk.res(),) + cat.traversal_segments(rng, hard), spec_hard=hard)
        segs = ("files", pk.res()) + cat.traversal_segments(rng, hard)
        return TemplateSpec(family, "GET", segs, query=_query_params(rng, rng.random() < 0.3, False), spec_hard=hard)
    if family == "mass_assignment":
        hard = rng.random() < hard_fraction
        body = list(_flat_body(rng, rng.randint(2, 7), lists=False))
        used = {n for n, _ in body}
        body += _privileged_fields(rng, hard, used)
        return TemplateSpec(family, ch(("POST", "PUT", "PATCH")), _resource_path(rng, pk, rng.random() < 0.5, False),
                            body=tuple(body), body_format="json", spec_hard=hard)
    if family == "structure_anomaly":
        body = list(_flat_body(rng, rng.randint(2, 3), lists=False))
        if rng.random() < 0.5:
            body.append(("payload_tree", ("deep", rng.randint(6, _MAX_DEEP))))
        else:
            used = {n for n, _ in body}
            name = ch([n for n in cat.FIELD_NAMES_NUM if n not in used])
            body.append((name, ch(("num_list", ("object", (("value", "num"),))))))
        return TemplateSpec(family, "POST", _resource_path(rng, pk, False, False), body=tuple(body), body_format="json")
    if family == "role_action_mismatch":
        method = ch(("GET", "POST", "PUT", "PATCH", "DELETE"))
        has_body = method in ("POST", "PUT", "PATCH")
        return TemplateSpec(family, method, ("admin", pk.res(), pk.sub()),
                            body=_flat_body(rng, rng.randint(1, 3)) if has_body else (),
                            body_format="json" if has_body else "none", admin_path=True)
    if family == "content_type_mismatch":
        fmt = ch(("json", "xml", "form", "plain"))
        body = _flat_body(rng, rng.randint(2, 3), lists=False) if fmt in ("json", "form") else ()
        return TemplateSpec(family, ch(("POST", "PUT")), _resource_path(rng, pk, False, False), body=body,
                            body_format=fmt, raw_bits=ch((5, 6, 7)) if fmt in ("xml", "plain") else 0)
    if family == "shared_endpoint_text":
        kind = rng.choice(("query", "body", "header", "login"))
        if kind == "login":
            return _login_spec(family, rng, slot=True)
        if kind == "query":
            slot = ch(cat.QUERY_TEXT)
            return TemplateSpec(family, "GET", (pk.res(), _search_sub(rng, pk)),
                                query=_text_query(rng, slot), text_slot=("query", slot))
        if kind == "body":
            body = _flat_body(rng, rng.randint(2, 3), text_slot=True)
            return TemplateSpec(family, ch(("POST", "PUT")), _resource_path(rng, pk, False, False), body=body,
                                body_format="json", text_slot=("body", body[0][0]))
        return TemplateSpec(family, "GET", _resource_path(rng, pk, False, True), text_slot=("header", cat.CLIENT_NOTE_HEADER))
    raise GeneratorError("unknown family")


# -- rendering ------------------------------------------------------------------------------------------------


def _sized_text(rng: random.Random, bits: int, prefix: str = "", suffix: str = "") -> str:
    """Words (wrapped in ``prefix``/``suffix``) whose total length has exactly ``bits`` bits.

    The splitter's body shape for non-JSON text is a length bucket, so xml and plain bodies of one template
    must stay inside one bucket or the template would drift. Words are at most 8 letters, so overshooting
    the lower bound 2**(bits-1) by one word never reaches 2**bits for bits >= 5.
    """
    lower = 2 ** (bits - 1)
    words: list[str] = []
    while len(prefix) + len(" ".join(words)) + len(suffix) < lower:
        words.append(rng.choice(cat.WORDS))
    return prefix + " ".join(words) + suffix


def _value(kind: Any, rng: random.Random, text: str | None = None) -> Any:
    if kind == "text":
        return text if text is not None else cat.plain_phrase(rng)
    if kind == "long_text":
        return cat.plain_phrase(rng, 12, 24)
    if kind == "num":
        return rng.randint(1, 9999)
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "text_list":
        return [cat.plain_phrase(rng, 1, 2) for _ in range(rng.randint(1, 3))]
    if kind == "num_list":
        return [rng.randint(1, 99) for _ in range(rng.randint(1, 3))]
    if kind == "null":
        return None
    if kind == "user":
        return cat.fake_username(rng)
    if kind == "secret":
        return cat.fake_secret(rng)
    if isinstance(kind, tuple) and kind[0] == "object":
        return {name: _value(sub, rng) for name, sub in kind[1]}
    if isinstance(kind, tuple) and kind[0] == "deep":
        node: Any = cat.plain_phrase(rng, 1, 2)
        for _ in range(kind[1]):
            node = {"inner": node}
        return node
    raise GeneratorError("unknown value kind")


def _slot_text(spec: TemplateSpec, rng: random.Random, suspicious: bool, hard: bool) -> tuple[str, str]:
    """Return (text, hard_case) for the spec's text slot."""
    fam = spec.family
    if fam == "hn_keyword_text":
        return cat.hard_negative_phrase(rng), "negative"
    if suspicious:
        style = {"sqli_like": "sql", "cmdi_like": "cmd", "markup_like": "markup"}.get(fam) or rng.choice(cat.PAYLOAD_STYLES)
        text = {"sql": cat.sql_like, "cmd": cat.cmd_like, "markup": cat.markup_like}[style](rng, hard)
        return text, "positive" if hard else ""
    if fam == "shared_endpoint_text" and rng.random() < 0.3:
        return cat.hard_negative_phrase(rng), "negative"
    return cat.plain_phrase(rng), ""


def _declared_type(spec: TemplateSpec, rng: random.Random, suspicious: bool) -> tuple[str, bool]:
    """Declared content type for a body, and whether a mismatch is a hard positive."""
    matching = cat.CONTENT_TYPES_BY_FORMAT[spec.body_format]
    if spec.family == "content_type_mismatch" and suspicious:
        others = [t for f, types in cat.CONTENT_TYPES_BY_FORMAT.items() if f != spec.body_format for t in types]
        declared = rng.choice(others)
        return declared, declared == cat.CONTENT_TYPE_JSON
    return rng.choice(matching), False


def _render(spec: TemplateSpec, rng: random.Random, suspicious: bool, hard: bool) -> tuple[dict[str, Any], str]:
    """Render one variant: returns (request fields, hard_case)."""
    family = cat.FAMILY_BY_NAME[spec.family]
    hard_case = "negative" if family.hard_case == "negative" else ""
    text, slot_hard = (None, "")
    if spec.text_slot is not None:
        text, slot_hard = _slot_text(spec, rng, suspicious, hard)
        hard_case = slot_hard or hard_case

    segs = [str(rng.randint(1, 99999)) if s is None else s for s in spec.segments]
    pairs: list[tuple[str, str]] = []
    for name, kind in spec.query:
        if kind == "num":
            pairs.append((name, str(rng.randint(1, 9999))))
        elif kind == "token":
            pairs.append((name, rng.choice(("asc", "desc", "name", "date", "active", "all", "summary"))))
        else:
            pairs.append((name, text if spec.text_slot == ("query", name) else cat.plain_phrase(rng)))
    path = "/" + "/".join(segs) + ("?" + urllib.parse.urlencode(pairs) if pairs else "")

    privileged = set(cat.PRIVILEGED_FIELDS_OBVIOUS + cat.PRIVILEGED_FIELDS_QUIET)
    body_obj: dict[str, Any] = {}
    for name, kind in spec.body:
        if spec.text_slot == ("body", name):
            body_obj[name] = text
        elif spec.family in ("mass_assignment", "admin_privileged_update") and name in privileged:
            body_obj[name] = True if kind == "bool" else rng.choice(("root", "superuser", "internal"))
        else:
            body_obj[name] = _value(kind, rng)
    body = ""
    if spec.body_format == "json":
        body = json.dumps(body_obj, sort_keys=True, separators=(",", ":"))
    elif spec.body_format == "form":
        body = urllib.parse.urlencode({k: str(v) for k, v in body_obj.items()})
    elif spec.body_format == "plain":
        body = _sized_text(rng, spec.raw_bits)
    elif spec.body_format == "xml":
        body = _sized_text(rng, spec.raw_bits, "<note>", "</note>")
    if spec.family in ("mass_assignment", "traversal_like") and spec.spec_hard:
        hard_case = "positive"

    headers: dict[str, str] = {
        "user-agent": rng.choice(cat.USER_AGENTS),
        "accept": rng.choice(cat.ACCEPT_VALUES),
    }
    if spec.body_format != "none":
        headers["content-type"], mismatch_hard = _declared_type(spec, rng, suspicious)
        if mismatch_hard:
            hard_case = "positive"
    if spec.auth:
        headers["authorization"] = "Bearer syn-" + "".join(rng.choice("0123456789abcdef") for _ in range(16))
        if spec.admin_path:
            admin = (not suspicious) if spec.family == "role_action_mismatch" else True
            headers[cat.ROLE_HEADER] = cat.ROLE_ADMIN if admin else rng.choice(cat.ROLES_LOW)
        elif spec.family == "mass_assignment":
            headers[cat.ROLE_HEADER] = rng.choice(cat.ROLES_LOW)  # non-admin by definition of the family
        else:
            headers[cat.ROLE_HEADER] = cat.ROLE_ADMIN if rng.random() < 0.15 else rng.choice(cat.ROLES_LOW)
    if spec.text_slot == ("header", cat.CLIENT_NOTE_HEADER):
        headers[cat.CLIENT_NOTE_HEADER] = text or cat.plain_phrase(rng)
    elif rng.random() < 0.25:
        headers[cat.CLIENT_NOTE_HEADER] = cat.plain_phrase(rng)
    if spec.family == "role_action_mismatch" and suspicious:
        hard_case = "positive" if headers[cat.ROLE_HEADER] == "editor" else hard_case
    return {"method": spec.method, "path": path, "body": body, "headers": headers}, hard_case


def _variant_labels(kind: str, count: int, rng: random.Random, p: float) -> list[bool]:
    if kind == "normal":
        return [False] * count
    if kind == "suspicious":
        return [True] * count
    return [False, True] + [rng.random() < p for _ in range(count - 2)]


def _opaque(seed: int, tag: str, position: int, length: int = 12) -> str:
    return hashlib.sha256(f"{GENERATOR_VERSION}|{seed}|{tag}|{position}".encode()).hexdigest()[:length]


# -- public entry point ---------------------------------------------------------------------------------------------


def generate(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Generate the synthetic dataset. Deterministic for a given config."""
    cfg = config or GeneratorConfig()
    rng = random.Random(f"{GENERATOR_VERSION}|{cfg.seed}")
    weights = cfg.weights()
    names = [n for n in cat.family_names() if weights[n] > 0]
    probs = [weights[n] for n in names]

    pickers = _Pickers(rng)
    specs: list[tuple[TemplateSpec, str]] = []
    seen: set[str] = set()
    attempts = collapsed = 0
    max_attempts = cfg.n_templates * cfg.max_attempt_factor
    while len(specs) < cfg.n_templates and attempts < max_attempts:
        attempts += 1
        spec = _build_spec(rng.choices(names, probs)[0], rng, cfg.hard_fraction, pickers)
        sample, _ = _render(spec, random.Random(0), False, False)
        gid = group_id_for(make_record("probe", method=sample["method"], path=sample["path"], label=0,
                                       body=sample["body"], headers=sample["headers"]).features)
        if gid in seen:
            collapsed += 1
            continue
        seen.add(gid)
        specs.append((spec, gid))
    if not specs:
        raise GeneratorError("no template could be generated")

    pending: list[tuple[str, int, dict[str, Any], int, str, TemplateSpec]] = []
    duplicates = 0
    for spec, gid in specs:
        kind = cat.FAMILY_BY_NAME[spec.family].label_kind
        count = rng.randint(cfg.variants_min, cfg.variants_max)
        labels = _variant_labels(kind, count, rng, cfg.mixed_suspicious_probability)
        used_prints: set[str] = set()
        for index, suspicious in enumerate(labels):
            for _try in range(30):
                hard = rng.random() < cfg.hard_fraction
                fields, hard_case = _render(spec, rng, suspicious, hard)
                features = make_record("probe", method=fields["method"], path=fields["path"], label=0,
                                       body=fields["body"], headers=fields["headers"]).features
                if group_id_for(features) != gid:
                    raise GeneratorError("template drift: a variant normalized to a different template")
                print_ = exact_fingerprint(features)
                if print_ not in used_prints:
                    used_prints.add(print_)
                    pending.append((gid, index, fields, int(suspicious), hard_case, spec))
                    break
            else:
                duplicates += 1

    rng.shuffle(pending)  # ids, timestamps and sources are assigned after this: independent of label
    records: list[EvalRecord] = []
    metadata: dict[str, GenerationMetadata] = {}
    for position, (gid, index, fields, label, hard_case, spec) in enumerate(pending):
        record_id = "r" + _opaque(cfg.seed, "id", position)
        rec = make_record(
            record_id, method=fields["method"], path=fields["path"], label=label, body=fields["body"],
            headers=fields["headers"], timestamp=_TIMESTAMP_BASE + position * 37.0 + rng.random() * 20.0,
            source_id="s" + _opaque(cfg.seed, "src", rng.randrange(_SOURCE_POOL), 8),
        )
        records.append(rec)
        metadata[record_id] = GenerationMetadata(
            label=label, family=spec.family, label_kind=cat.FAMILY_BY_NAME[spec.family].label_kind,
            template_id=gid, variant_index=index, hard_case=hard_case,
        )
    if len(metadata) != len(records):
        raise GeneratorError("record id collision")

    by_template: dict[str, set[int]] = {}
    for rec in records:
        by_template.setdefault(metadata[rec.record_id].template_id, set()).add(rec.meta.label)
    stats = GenerationStats(
        requested_templates=cfg.n_templates, actual_templates=len(specs), collapsed_specs=collapsed,
        attempts=attempts, n_records=len(records), duplicate_variants_skipped=duplicates,
        templates_all_normal=sum(1 for v in by_template.values() if v == {0}),
        templates_all_suspicious=sum(1 for v in by_template.values() if v == {1}),
        templates_mixed=sum(1 for v in by_template.values() if v == {0, 1}),
        records_by_family=dict(Counter(m.family for m in metadata.values())),
        templates_by_family=dict(Counter(spec.family for spec, _ in specs)),
        hard_negative_records=sum(1 for m in metadata.values() if m.hard_case == "negative"),
        hard_positive_records=sum(1 for m in metadata.values() if m.hard_case == "positive"),
    )
    fingerprint = hashlib.sha256(json.dumps(
        [cfg.to_dict(), [[r.record_id, r.features.method, r.features.path, r.features.body,
                          list(r.features.headers), r.meta.label, r.meta.timestamp, r.meta.source_id,
                          metadata[r.record_id].family] for r in records]],
        sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return SyntheticDataset(config=cfg, records=tuple(records), metadata=metadata, stats=stats, fingerprint=fingerprint)
