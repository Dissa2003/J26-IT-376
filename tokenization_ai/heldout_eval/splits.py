"""Leakage-safe, reproducible train/validation/test splitting for the held-out study.

DRAFT infrastructure for ``PROTOCOL.md`` (protocol ``heldout-eval-1.0.0-draft``). It is
deliberately self-contained: it imports nothing from the production pipeline, the model, the
existing KPI ``evaluation`` package or any other member's code, and it trains, scores and
measures nothing. It only decides which request belongs to which split, and proves that the
decision does not leak.

Safeguards (each is tested with fabricated requests only):

* Requests are grouped by a documented **normalized request template** (``TEMPLATE_VERSION``).
  A whole group lands in exactly one split, so near-duplicates never straddle splits.
* **Exact duplicates** (identical method, path, body and headers) share a template, so they
  are always together; the isolation check also verifies this independently. Exact
  duplicates carrying different labels are label noise and are reported (default: error).
* **Labels are used only for balancing the split and for class checks.** They never enter a
  template, a group id, a fingerprint or ``RequestFeatures``. Model-facing features
  (``RequestFeatures``) and split metadata (``SplitMetadata``) are separate types, and the
  feature type has no field for a label, response status, ModSecurity decision or attack
  annotation.
* **Source identity is optional and never the only key.** ``merge_by_source=True`` merges
  groups that share a source id on top of the template grouping.
* **Impossible or degenerate requests are explicit failures** (``SplitInfeasibleError``),
  never a silent compromise of isolation.
* **Reproducible:** the same records, parameters and seed give identical assignments and an
  identical ``manifest_sha256``.
* **Provenance without raw content:** the report and manifest hold only caller-supplied
  record ids, hashed group ids, counts and hashes. Use opaque record ids (for example row
  numbers), never request text.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "heldout-eval-1.0.0-draft"
TEMPLATE_VERSION = "tmpl-1"

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

# Arbitrary values fixed in advance, before any data exists; they are not tuned.
PRIMARY_SPLIT_SEED = 4401
ROBUSTNESS_SPLIT_SEEDS: tuple[int, ...] = (4402, 4403, 4404, 4405)

_MAX_SHAPE_DEPTH = 6

# Names that must never appear among model-facing feature fields.
FORBIDDEN_FEATURE_FIELD_NAMES = frozenset({
    "label", "labels", "target", "y", "is_attack", "attack", "capec", "category", "normal",
    "status", "status_code", "response", "response_status", "response_code",
    "modsecurity", "modsec", "rule_hit", "anomaly_score", "threat_score", "decision",
})


class SplitError(ValueError):
    """Base class for splitting errors. Messages never contain request content."""


class InvalidRecordError(SplitError):
    """A record is malformed (reported by position and field, never by value)."""


class LabelConflictError(SplitError):
    """Exact duplicate requests carry different labels."""


class SplitInfeasibleError(SplitError):
    """The data cannot be split under the stated requirements without compromising isolation."""


class IsolationViolationError(SplitError):
    """An internal isolation check failed (indicates a bug, never an expected condition)."""


# -- records ------------------------------------------------------------------


@dataclass(frozen=True)
class RequestFeatures:
    """Model-facing request content only. No label, response or detector field exists here."""

    method: str
    path: str
    body: str = ""
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SplitMetadata:
    """Split-only metadata. Never converted into model features."""

    label: int
    timestamp: float | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class EvalRecord:
    record_id: str
    features: RequestFeatures
    meta: SplitMetadata


def make_record(
    record_id: str,
    *,
    method: str,
    path: str,
    label: int,
    body: str | None = None,
    headers: Mapping[str, str] | None = None,
    timestamp: float | None = None,
    source_id: str | None = None,
) -> EvalRecord:
    """Validate and build one record. Raises :class:`InvalidRecordError` without echoing values."""
    if not isinstance(record_id, str) or not record_id or record_id != record_id.strip():
        raise InvalidRecordError("record_id must be a non-empty string without surrounding whitespace")
    where = "record"
    if not isinstance(method, str) or not re.fullmatch(r"[A-Za-z]{1,20}", method):
        raise InvalidRecordError(f"{where}: method must be 1-20 letters")
    if not isinstance(path, str) or not path:
        raise InvalidRecordError(f"{where}: path must be a non-empty string")
    if body is not None and not isinstance(body, str):
        raise InvalidRecordError(f"{where}: body must be a string or None")
    if type(label) is not int or label not in (0, 1):
        raise InvalidRecordError(f"{where}: label must be the integer 0 or 1")
    if timestamp is not None and (
        isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp)
    ):
        raise InvalidRecordError(f"{where}: timestamp must be a finite number or None")
    if source_id is not None and (not isinstance(source_id, str) or not source_id):
        raise InvalidRecordError(f"{where}: source_id must be a non-empty string or None")
    header_pairs: tuple[tuple[str, str], ...] = ()
    if headers is not None:
        if not isinstance(headers, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise InvalidRecordError(f"{where}: headers must map strings to strings")
        header_pairs = tuple(sorted((k.lower(), v) for k, v in headers.items()))
    return EvalRecord(
        record_id=record_id,
        features=RequestFeatures(method=method, path=path, body=body or "", headers=header_pairs),
        meta=SplitMetadata(label=label, timestamp=None if timestamp is None else float(timestamp), source_id=source_id),
    )


# -- normalized request template -----------------------------------------------

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HEX = re.compile(r"(?<![0-9a-z])[0-9a-f]{8,}(?![0-9a-z])")
_DIGITS = re.compile(r"\d+")


def _norm_token(text: str) -> str:
    text = _UUID.sub("<uuid>", text)
    text = _HEX.sub("<hex>", text)
    return _DIGITS.sub("<n>", text)


def _normalize_path(path: str) -> str:
    path = path.split("#", 1)[0]
    raw_path = path.split("?", 1)[0]
    decoded = urllib.parse.unquote_plus(raw_path).lower()
    decoded = re.sub(r"/{2,}", "/", decoded)
    if len(decoded) > 1:
        decoded = decoded.rstrip("/") or "/"
    return _norm_token(decoded)


def _value_class(value: str) -> str:
    if value == "":
        return "<empty>"
    if value.isdigit():
        return "<n>"
    return "<v>"


def _normalize_pairs(query: str) -> str:
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    shaped = sorted((_norm_token(urllib.parse.unquote_plus(k).lower()), _value_class(v)) for k, v in pairs)
    return "&".join(f"{k}={v}" for k, v in shaped)


def _shape(value: Any, depth: int = 0) -> str:
    if depth >= _MAX_SHAPE_DEPTH:
        return "~"
    if isinstance(value, dict):
        return "{" + ",".join(f"{_norm_token(str(k).lower())}:{_shape(v, depth + 1)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, list):
        return "[" + ",".join(sorted({_shape(v, depth + 1) for v in value})) + "]"
    if isinstance(value, bool):
        return "b"
    if isinstance(value, (int, float)):
        return "n"
    if value is None:
        return "z"
    return "s"


def _normalize_body(body: str) -> str:
    if body.strip() == "":
        return "<empty>"
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    else:
        if isinstance(parsed, (dict, list)):
            return "json:" + _shape(parsed)
    if "=" in body and " " not in body.strip() and "\n" not in body:
        return "form:" + _normalize_pairs(body)
    return f"raw:{max(1, len(body)).bit_length()}"


def request_template(features: RequestFeatures) -> str:
    """The documented normalized template string for a request (method, path, query, body shape).

    Digits, hex strings and UUIDs become placeholders; query values become classes
    (``<empty>``, ``<n>``, ``<v>``); the body is reduced to its structural shape. Request text
    that differs only in values therefore shares a template, which is the intent: near-duplicate
    attack variants must stay together. Only :class:`RequestFeatures` is accepted, so a label or
    response field cannot be passed in.
    """
    if not isinstance(features, RequestFeatures):
        raise TypeError("request_template accepts RequestFeatures only")
    path, _, query = features.path.split("#", 1)[0].partition("?")
    return "|".join([
        features.method.upper(),
        _normalize_path(path),
        _normalize_pairs(query),
        _normalize_body(features.body),
    ])


def group_id_for(features: RequestFeatures) -> str:
    """Stable, label-independent group id: a truncated SHA-256 of the template."""
    digest = hashlib.sha256(f"{TEMPLATE_VERSION}\n{request_template(features)}".encode("utf-8")).hexdigest()
    return digest[:16]


def exact_fingerprint(features: RequestFeatures) -> str:
    """SHA-256 of the exact request content (method, path, body, headers)."""
    if not isinstance(features, RequestFeatures):
        raise TypeError("exact_fingerprint accepts RequestFeatures only")
    canonical = json.dumps(
        [features.method, features.path, features.body, list(features.headers)], ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- result types ----------------------------------------------------------------


@dataclass(frozen=True)
class SplitReport:
    protocol_version: str
    template_version: str
    method: str
    seed: int | None
    ratios: tuple[float, float, float]
    parameters: Mapping[str, Any]
    n_input_records: int
    n_assigned: int
    n_excluded: int
    n_groups: int
    group_size_max: int
    group_size_median: float
    counts: Mapping[str, Mapping[str, int]]  # split -> {"records", "positives", "negatives", "groups"}
    achieved_ratios: tuple[float, float, float]
    max_ratio_deviation: float
    exact_duplicate_records: int
    exact_duplicate_label_conflicts: int
    mixed_label_templates: int
    straddling_groups: int
    problems: tuple[str, ...]
    dataset_fingerprint: str

    @property
    def meets_requirements(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "template_version": self.template_version,
            "method": self.method,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "parameters": dict(self.parameters),
            "n_input_records": self.n_input_records,
            "n_assigned": self.n_assigned,
            "n_excluded": self.n_excluded,
            "n_groups": self.n_groups,
            "group_size_max": self.group_size_max,
            "group_size_median": self.group_size_median,
            "counts": {k: dict(v) for k, v in self.counts.items()},
            "achieved_ratios": list(self.achieved_ratios),
            "max_ratio_deviation": self.max_ratio_deviation,
            "exact_duplicate_records": self.exact_duplicate_records,
            "exact_duplicate_label_conflicts": self.exact_duplicate_label_conflicts,
            "mixed_label_templates": self.mixed_label_templates,
            "straddling_groups": self.straddling_groups,
            "problems": list(self.problems),
            "meets_requirements": self.meets_requirements,
            "dataset_fingerprint": self.dataset_fingerprint,
        }


@dataclass(frozen=True)
class SplitResult:
    assignments: Mapping[str, str]  # record_id -> split name
    group_of: Mapping[str, str]  # record_id -> group id
    excluded: Mapping[str, str]  # record_id -> reason
    report: SplitReport
    manifest_sha256: str = field(default="")

    def members(self, split: str) -> list[str]:
        if split not in SPLIT_NAMES:
            raise KeyError(split)
        return sorted(rid for rid, name in self.assignments.items() if name == split)

    def to_manifest(self) -> dict[str, Any]:
        """JSON-safe provenance: parameters, counts, hashes and per-record split. No raw content."""
        return {
            "report": self.report.to_dict(),
            "records": [
                {"record_id": rid, "group_id": self.group_of[rid], "split": self.assignments[rid]}
                for rid in sorted(self.assignments)
            ],
            "excluded": [{"record_id": rid, "reason": why} for rid, why in sorted(self.excluded.items())],
            "manifest_sha256": self.manifest_sha256,
        }


# -- public API --------------------------------------------------------------------


def split_records(
    records: Iterable[EvalRecord],
    *,
    seed: int = PRIMARY_SPLIT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    min_per_class: int = 1,
    drop_exact_duplicates: bool = False,
    exact_conflict_policy: str = "error",
    merge_by_source: bool = False,
    ratio_tolerance: float = 0.10,
    strict: bool = True,
) -> SplitResult:
    """Group-aware, seeded, class-balanced train/validation/test split.

    ``min_per_class``: every split must hold at least this many records of each class.
    ``exact_conflict_policy``: ``"error"`` (default) or ``"exclude"`` for exact duplicates with
    differing labels. ``strict=True`` raises :class:`SplitInfeasibleError` when a requirement is
    not met; ``strict=False`` returns the result with ``report.problems`` filled in.
    Isolation is never relaxed in either mode.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError("seed must be an integer")
    return _build(
        records, method="group_random", seed=seed, ratios=ratios, min_per_class=min_per_class,
        drop_exact_duplicates=drop_exact_duplicates, exact_conflict_policy=exact_conflict_policy,
        merge_by_source=merge_by_source, ratio_tolerance=ratio_tolerance, strict=strict,
    )


def time_forward_split(
    records: Iterable[EvalRecord],
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    min_per_class: int = 1,
    drop_exact_duplicates: bool = False,
    exact_conflict_policy: str = "error",
    merge_by_source: bool = False,
    ratio_tolerance: float = 0.10,
    strict: bool = True,
) -> SplitResult:
    """Secondary robustness split: train precedes validation precedes test in time.

    Groups are ordered by their *latest* timestamp and cut by cumulative record count. A group
    that spans a boundary goes wholly to the later split, so no future member of a template is
    ever in an earlier split (later splits may contain some older records of a recurring
    template). Every record needs a timestamp. There is no random element.
    """
    return _build(
        records, method="time_forward", seed=None, ratios=ratios, min_per_class=min_per_class,
        drop_exact_duplicates=drop_exact_duplicates, exact_conflict_policy=exact_conflict_policy,
        merge_by_source=merge_by_source, ratio_tolerance=ratio_tolerance, strict=strict,
    )


def verify_split_isolation(result: SplitResult, records: Iterable[EvalRecord]) -> list[str]:
    """Return a list of isolation violations (empty means isolated). Content-safe messages."""
    by_id = {r.record_id: r for r in records}
    problems: list[str] = []
    group_splits: dict[str, set[str]] = {}
    template_splits: dict[str, set[str]] = {}
    print_splits: dict[str, set[str]] = {}
    for rid, split in result.assignments.items():
        rec = by_id.get(rid)
        if rec is None:
            problems.append("an assigned record id is missing from the supplied records")
            continue
        group_splits.setdefault(result.group_of.get(rid, group_id_for(rec.features)), set()).add(split)
        template_splits.setdefault(group_id_for(rec.features), set()).add(split)
        print_splits.setdefault(exact_fingerprint(rec.features), set()).add(split)
    problems += [f"group {gid} spans splits {sorted(s)}" for gid, s in sorted(group_splits.items()) if len(s) > 1]
    problems += [f"template {gid} spans splits {sorted(s)}" for gid, s in sorted(template_splits.items()) if len(s) > 1]
    problems += [f"exact duplicate {fp[:12]} spans splits {sorted(s)}" for fp, s in sorted(print_splits.items()) if len(s) > 1]
    if set(result.assignments) & set(result.excluded):
        problems.append("a record is both assigned and excluded")
    return problems


# -- internals ------------------------------------------------------------------------


def _build(
    records: Iterable[EvalRecord], *, method: str, seed: int | None, ratios: Sequence[float],
    min_per_class: int, drop_exact_duplicates: bool, exact_conflict_policy: str,
    merge_by_source: bool, ratio_tolerance: float, strict: bool,
) -> SplitResult:
    ratios_t = _check_ratios(ratios)
    if isinstance(min_per_class, bool) or not isinstance(min_per_class, int) or min_per_class < 1:
        raise SplitError("min_per_class must be an integer of at least 1")
    if exact_conflict_policy not in ("error", "exclude"):
        raise SplitError("exact_conflict_policy must be 'error' or 'exclude'")
    if not (0 < ratio_tolerance <= 1):
        raise SplitError("ratio_tolerance must be in (0, 1]")

    recs = _validate_records(records, require_timestamp=(method == "time_forward"))
    by_id = {r.record_id: r for r in recs}
    fingerprints = {r.record_id: exact_fingerprint(r.features) for r in recs}
    dataset_fp = hashlib.sha256("\n".join(sorted(fingerprints.values())).encode("ascii")).hexdigest()

    # Exact duplicates: counts, label conflicts, optional dropping.
    by_print: dict[str, list[EvalRecord]] = {}
    for r in recs:
        by_print.setdefault(fingerprints[r.record_id], []).append(r)
    duplicate_records = sum(len(v) - 1 for v in by_print.values() if len(v) > 1)
    conflicting = {fp for fp, v in by_print.items() if len({x.meta.label for x in v}) > 1}
    excluded: dict[str, str] = {}
    if conflicting:
        if exact_conflict_policy == "error":
            raise LabelConflictError(
                f"{len(conflicting)} exact-duplicate request(s) carry conflicting labels "
                f"(fingerprints: {', '.join(sorted(fp[:12] for fp in conflicting))}); "
                "resolve the labels or use exact_conflict_policy='exclude'"
            )
        for fp in conflicting:
            for r in by_print[fp]:
                excluded[r.record_id] = "exact_duplicate_label_conflict"
    if drop_exact_duplicates:
        for fp, members in by_print.items():
            if fp in conflicting:
                continue
            for r in sorted(members, key=lambda x: x.record_id)[1:]:
                excluded[r.record_id] = "exact_duplicate_dropped"
    kept = [r for r in recs if r.record_id not in excluded]
    if not kept:
        raise SplitInfeasibleError("no records remain after exclusions")

    # Group by template (optionally merged by shared source id).
    group_of = _assign_groups(kept, merge_by_source)
    groups: dict[str, list[EvalRecord]] = {}
    for r in kept:
        groups.setdefault(group_of[r.record_id], []).append(r)
    mixed = sum(1 for v in groups.values() if len({x.meta.label for x in v}) > 1)

    if len(groups) < len(SPLIT_NAMES):
        raise SplitInfeasibleError(
            f"only {len(groups)} group(s) exist; at least {len(SPLIT_NAMES)} are needed so that each "
            "split can hold whole groups without sharing one"
        )

    if method == "group_random":
        split_of_group = _assign_random(groups, ratios_t, seed)  # type: ignore[arg-type]
    else:
        split_of_group = _assign_time(groups, ratios_t)

    assignments = {r.record_id: split_of_group[group_of[r.record_id]] for r in kept}

    straddling = _count_straddling(groups, split_of_group) if method == "time_forward" else 0
    report = _make_report(
        method=method, seed=seed, ratios=ratios_t, groups=groups, assignments=assignments,
        by_id=by_id, n_input=len(recs), excluded=excluded, min_per_class=min_per_class,
        ratio_tolerance=ratio_tolerance, duplicate_records=duplicate_records,
        conflicts=len(conflicting), mixed=mixed, straddling=straddling, dataset_fp=dataset_fp,
        params={
            "min_per_class": min_per_class, "drop_exact_duplicates": drop_exact_duplicates,
            "exact_conflict_policy": exact_conflict_policy, "merge_by_source": merge_by_source,
            "ratio_tolerance": ratio_tolerance, "strict": strict,
        },
    )
    result = SplitResult(assignments=assignments, group_of=dict(group_of), excluded=excluded, report=report)
    violations = verify_split_isolation(result, kept)
    if violations:
        raise IsolationViolationError("; ".join(violations))
    if strict and report.problems:
        raise SplitInfeasibleError("; ".join(report.problems))
    return SplitResult(
        assignments=result.assignments, group_of=result.group_of, excluded=result.excluded,
        report=report, manifest_sha256=_manifest_hash(result),
    )


def _check_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    try:
        values = tuple(float(x) for x in ratios)
    except (TypeError, ValueError) as exc:
        raise SplitError("ratios must be three numbers") from exc
    if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise SplitError("ratios must be three positive finite numbers")
    if abs(sum(values) - 1.0) > 1e-9:
        raise SplitError("ratios must sum to 1")
    return values  # type: ignore[return-value]


def _validate_records(records: Iterable[EvalRecord], *, require_timestamp: bool) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    seen: set[str] = set()
    for index, rec in enumerate(records):
        if not isinstance(rec, EvalRecord):
            raise InvalidRecordError(f"item {index} is not an EvalRecord")
        if rec.record_id in seen:
            raise InvalidRecordError(f"item {index}: duplicate record_id")
        seen.add(rec.record_id)
        if require_timestamp and rec.meta.timestamp is None:
            raise InvalidRecordError(f"item {index}: time-forward splitting needs a timestamp on every record")
        out.append(rec)
    if not out:
        raise InvalidRecordError("no records supplied")
    return out


def _assign_groups(records: list[EvalRecord], merge_by_source: bool) -> dict[str, str]:
    base = {r.record_id: group_id_for(r.features) for r in records}
    if not merge_by_source:
        return base
    parent: dict[str, str] = {g: g for g in set(base.values())}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_group_for_source: dict[str, str] = {}
    for r in sorted(records, key=lambda x: x.record_id):
        source = r.meta.source_id
        if source is None:
            continue
        g = base[r.record_id]
        if source in first_group_for_source:
            a, b = find(first_group_for_source[source]), find(g)
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first_group_for_source[source] = g
    return {rid: find(g) for rid, g in base.items()}


def _class_counts(members: list[EvalRecord]) -> tuple[int, int]:
    pos = sum(1 for m in members if m.meta.label == 1)
    return len(members) - pos, pos


def _assign_random(groups: dict[str, list[EvalRecord]], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    ids = sorted(groups)
    rng.shuffle(ids)
    ids.sort(key=lambda g: -len(groups[g]))  # stable: larger groups first, ties in shuffled order
    totals = [sum(_class_counts(groups[g])[c] for g in ids) for c in (0, 1)]
    targets = [[ratios[s] * totals[c] for c in (0, 1)] for s in range(3)]
    current = [[0, 0] for _ in range(3)]
    result: dict[str, str] = {}
    for gid in ids:
        neg, pos = _class_counts(groups[gid])
        best, best_cost = 0, math.inf
        for s in range(3):
            cost = 0.0
            for c, add in ((0, neg), (1, pos)):
                if totals[c] > 0:
                    # Cost *increase* of adding this group: favours the split with the largest
                    # remaining deficit against its target for each class.
                    after = (current[s][c] + add - targets[s][c]) ** 2
                    before = (current[s][c] - targets[s][c]) ** 2
                    cost += (after - before) / (totals[c] ** 2)
            if cost < best_cost - 1e-15:
                best, best_cost = s, cost
        current[best][0] += neg
        current[best][1] += pos
        result[gid] = SPLIT_NAMES[best]
    return result


def _assign_time(groups: dict[str, list[EvalRecord]], ratios: tuple[float, float, float]) -> dict[str, str]:
    total = sum(len(v) for v in groups.values())
    bound_train = ratios[0] * total
    bound_val = (ratios[0] + ratios[1]) * total
    ordered = sorted(groups, key=lambda g: (max(m.meta.timestamp for m in groups[g]), g))  # type: ignore[type-var]
    cumulative = 0
    result: dict[str, str] = {}
    for gid in ordered:
        cumulative += len(groups[gid])
        if cumulative <= bound_train:
            result[gid] = SPLIT_NAMES[0]
        elif cumulative <= bound_val:
            result[gid] = SPLIT_NAMES[1]
        else:
            result[gid] = SPLIT_NAMES[2]
    return result


def _count_straddling(groups: dict[str, list[EvalRecord]], split_of_group: dict[str, str]) -> int:
    latest: dict[str, float] = {}
    for gid, members in groups.items():
        split = split_of_group[gid]
        latest[split] = max(latest.get(split, -math.inf), max(m.meta.timestamp for m in members))  # type: ignore[type-var]
    count = 0
    for gid, members in groups.items():
        split = split_of_group[gid]
        earlier = SPLIT_NAMES[: SPLIT_NAMES.index(split)]
        if earlier and min(m.meta.timestamp for m in members) <= max(latest.get(e, -math.inf) for e in earlier):  # type: ignore[type-var]
            count += 1
    return count


def _make_report(
    *, method: str, seed: int | None, ratios: tuple[float, float, float],
    groups: dict[str, list[EvalRecord]], assignments: dict[str, str], by_id: dict[str, EvalRecord],
    n_input: int, excluded: dict[str, str], min_per_class: int, ratio_tolerance: float,
    duplicate_records: int, conflicts: int, mixed: int, straddling: int, dataset_fp: str,
    params: dict[str, Any],
) -> SplitReport:
    counts: dict[str, dict[str, int]] = {
        name: {"records": 0, "positives": 0, "negatives": 0, "groups": 0} for name in SPLIT_NAMES
    }
    for rid, split in assignments.items():
        c = counts[split]
        c["records"] += 1
        c["positives" if by_id[rid].meta.label == 1 else "negatives"] += 1
    for members in groups.values():
        for split in {assignments[m.record_id] for m in members}:
            counts[split]["groups"] += 1
    n_assigned = len(assignments)
    achieved = tuple(counts[name]["records"] / n_assigned for name in SPLIT_NAMES)
    deviation = max(abs(a - r) for a, r in zip(achieved, ratios))
    sizes = sorted(len(v) for v in groups.values())
    median = float(sizes[len(sizes) // 2]) if len(sizes) % 2 else (sizes[len(sizes) // 2 - 1] + sizes[len(sizes) // 2]) / 2

    problems: list[str] = []
    for name in SPLIT_NAMES:
        c = counts[name]
        if c["positives"] < min_per_class:
            problems.append(f"{name} has {c['positives']} positive record(s); at least {min_per_class} required")
        if c["negatives"] < min_per_class:
            problems.append(f"{name} has {c['negatives']} negative record(s); at least {min_per_class} required")
    if deviation > ratio_tolerance:
        problems.append(
            f"achieved split sizes deviate from the requested ratios by {deviation:.3f} "
            f"(tolerance {ratio_tolerance}); large groups prevent a close fit"
        )
    return SplitReport(
        protocol_version=PROTOCOL_VERSION, template_version=TEMPLATE_VERSION, method=method, seed=seed,
        ratios=ratios, parameters=params, n_input_records=n_input, n_assigned=n_assigned,
        n_excluded=len(excluded), n_groups=len(groups), group_size_max=sizes[-1], group_size_median=median,
        counts=counts, achieved_ratios=achieved,  # type: ignore[arg-type]
        max_ratio_deviation=round(deviation, 12), exact_duplicate_records=duplicate_records,
        exact_duplicate_label_conflicts=conflicts, mixed_label_templates=mixed, straddling_groups=straddling,
        problems=tuple(problems), dataset_fingerprint=dataset_fp,
    )


def _manifest_hash(result: SplitResult) -> str:
    body = json.dumps(
        {
            "report": result.report.to_dict(),
            "records": sorted([rid, result.group_of[rid], split] for rid, split in result.assignments.items()),
            "excluded": sorted(result.excluded.items()),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
