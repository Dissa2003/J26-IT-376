"""Non-model quality checks for the synthetic dataset (draft protocol heldout-eval-1.2.0-draft).

These checks inspect the generated data and its split. They train nothing and compute no accuracy,
PR-AUC or any other detection-performance number. They produce two kinds of output:

* **errors**: violations of construction rules (a class or family name inside a request field, a
  group or exact duplicate crossing splits, non-deterministic generation, template drift);
* **warnings**: *descriptive* attention flags about possible shortcuts (class-exclusive values, or an
  association with the label that exceeds label-shuffled references). They are not pass/fail
  criteria and no numeric success threshold is set here.

Passing every check does NOT show that the generator is free of leakage or shortcuts. The families
are written by the same author as the labels, so some association between request content and label
is expected by construction; the audit only makes it visible. Messages never contain raw request
text (only counts, field categories and opaque record ids).
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from typing import Any

from . import synthetic_catalog as cat
from .generator import GeneratorConfig, SyntheticDataset, generate
from .policy_sim import observe
from .splits import (
    FORBIDDEN_FEATURE_FIELD_NAMES,
    PRIMARY_SPLIT_SEED,
    SPLIT_NAMES,
    EvalRecord,
    RequestFeatures,
    SplitError,
    exact_fingerprint,
    group_id_for,
    request_template,
    split_records,
    verify_split_isolation,
)

# Protocol version this audit reports under. It must match the PROTOCOL.md header (a test checks this). It is
# unrelated to splits.PROTOCOL_VERSION, which labels the splitting code and is part of the split-manifest hash.
PROTOCOL_LABEL = "heldout-eval-1.2.0-draft"

DEFAULT_MIN_SUPPORT = 5  # legacy record-level support
DEFAULT_MIN_TEMPLATE_SUPPORT = 3  # a value must occur in at least this many distinct templates
DEFAULT_WARN_EXCLUSIVE_SHARE = 0.05  # an attention flag, NOT a pass/fail criterion
_SHUFFLES = 20
_MAX_LISTED_IDS = 3


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str  # "error" or "warning"
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "severity": self.severity, "message": self.message}


def _record_text_fields(rec: EvalRecord) -> dict[str, str]:
    """Every string a request exposes, by field category (decoded, lower-cased)."""
    f = rec.features
    decoded_path = urllib.parse.unquote_plus(f.path).lower()
    out = {
        "record_id": rec.record_id.lower(),
        "source_id": (rec.meta.source_id or "").lower(),
        "path": decoded_path,
        "body": f.body.lower(),
    }
    for index, (name, value) in enumerate(f.headers):
        out[f"header_name:{index}"] = name.lower()
        out[f"header_value:{index}"] = value.lower()
    return out


def scan_label_terms(dataset: SyntheticDataset) -> list[Finding]:
    """Class words, family names or template ids must not appear in any request field or id."""
    terms = {t.lower() for t in cat.BANNED_LABEL_WORDS} | {n.lower() for n in cat.family_names()}
    terms |= {m.template_id for m in dataset.metadata.values()}
    hits: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = {}
    for rec in dataset.records:
        for category, text in _record_text_fields(rec).items():
            category = category.split(":")[0]
            for term in terms:
                if term and term in text:
                    key = (category, term if len(term) <= 24 else term[:8] + "...")
                    hits[key] += 1
                    examples.setdefault(key, [])
                    if len(examples[key]) < _MAX_LISTED_IDS:
                        examples[key].append(rec.record_id)
    return [
        Finding("label_terms", "error",
                f"term {term!r} found in {count} record(s) in field '{category}' (examples: {', '.join(examples[(category, term)])})")
        for (category, term), count in sorted(hits.items())
    ]


def check_feature_separation(dataset: SyntheticDataset) -> list[Finding]:
    """Model-facing features must hold no label or scenario metadata, and metadata stays separate."""
    findings: list[Finding] = []
    names = {f.name for f in dataclasses.fields(RequestFeatures)}
    if names != {"method", "path", "body", "headers"}:
        findings.append(Finding("feature_separation", "error", "RequestFeatures fields differ from the documented set"))
    if names & FORBIDDEN_FEATURE_FIELD_NAMES:
        findings.append(Finding("feature_separation", "error", "RequestFeatures has a forbidden field name"))
    metadata_fields = {f.name for f in dataclasses.fields(type(next(iter(dataset.metadata.values()))))}
    if metadata_fields & names:
        findings.append(Finding("feature_separation", "error", "generation metadata shares a field name with features"))
    for rec in dataset.records:
        if not isinstance(rec.features, RequestFeatures):
            findings.append(Finding("feature_separation", "error", "a record's features are not RequestFeatures"))
            break
        if rec.record_id not in dataset.metadata:
            findings.append(Finding("feature_separation", "error", "a record has no generation metadata"))
            break
    return findings


def pool_sharing(dataset: SyntheticDataset) -> dict[str, Any]:
    """Which methods, resources, headers, roles and so on appear in only one class (descriptive)."""
    per_class: dict[str, dict[int, Counter[str]]] = {}

    def add(kind: str, label: int, value: str) -> None:
        per_class.setdefault(kind, {0: Counter(), 1: Counter()})[label][value] += 1

    for rec in dataset.records:
        label, f = rec.meta.label, rec.features
        headers = dict(f.headers)
        add("method", label, f.method)
        add("user_agent", label, headers.get("user-agent", ""))
        add("accept", label, headers.get("accept", ""))
        add("role", label, headers.get(cat.ROLE_HEADER, ""))
        add("content_type", label, headers.get("content-type", ""))
        for name, _ in f.headers:
            add("header_name", label, name)
        segments = [s for s in urllib.parse.unquote_plus(f.path.partition("?")[0]).lower().split("/") if s]
        for seg in segments:
            if seg in cat.RESOURCES:
                add("resource", label, seg)
            elif seg in cat.SUB_RESOURCES:
                add("sub_resource", label, seg)
    summary: dict[str, Any] = {}
    warnings: list[Finding] = []
    for kind, counters in sorted(per_class.items()):
        in0, in1 = set(counters[0]), set(counters[1])
        only0, only1 = sorted(in0 - in1), sorted(in1 - in0)
        summary[kind] = {"shared": len(in0 & in1), "only_in_class_0": only0, "only_in_class_1": only1,
                         "records_with_class_exclusive_value": sum(counters[0][v] for v in only0) + sum(counters[1][v] for v in only1)}
        if only0 or only1:
            warnings.append(Finding(
                "pool_sharing", "warning",
                f"{kind}: {len(only0)} value(s) only in class 0 and {len(only1)} only in class 1 "
                f"({summary[kind]['records_with_class_exclusive_value']} record(s)); may be by design for a family"))
    return {"by_kind": summary, "warnings": [w.to_dict() for w in warnings]}


_CT_MATCH = {
    "json": {"application/json"},
    "xml": {"application/xml", "text/xml"},
    "form": {cat.CONTENT_TYPE_FORM},
    "plain": {"text/plain"},
}


def actual_body_format(body: str) -> str:
    """Classify a request body by its content alone (json, xml, form, plain or none)."""
    if not body:
        return "none"
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        return "json"
    if body.lstrip().startswith("<"):
        return "xml"
    if "=" in body and " " not in body.strip():
        return "form"
    return "plain"


def check_content_type_semantics(dataset: SyntheticDataset) -> list[Finding]:
    """A content-type mismatch is incompatibility between the declared type and the actual body.

    Verified independently of the generator: for every request with a body, the declared type must match
    the body's actual format unless the request is in the mismatch family and labelled suspicious, in which
    case it must NOT match. The content-type value alone never decides the label.
    """
    findings: list[Finding] = []
    bad: list[str] = []
    for rec in dataset.records:
        body_format = actual_body_format(rec.features.body)
        if body_format == "none":
            continue
        declared = dict(rec.features.headers).get("content-type", "")
        matches = declared in _CT_MATCH[body_format]
        meta = dataset.metadata[rec.record_id]
        if meta.family == "content_type_mismatch":
            consistent = matches == (rec.meta.label == 0)
        else:
            consistent = matches
        if not consistent:
            bad.append(rec.record_id)
    if bad:
        findings.append(Finding("content_type_semantics", "error",
                                f"{len(bad)} record(s) violate the declared-type versus actual-body rule "
                                f"(examples: {', '.join(bad[:_MAX_LISTED_IDS])})"))
    return findings


def check_determinism(config: GeneratorConfig) -> list[Finding]:
    """Same config -> identical fingerprint; a different seed -> a different one."""
    findings: list[Finding] = []
    first, second = generate(config), generate(config)
    if first.fingerprint != second.fingerprint:
        findings.append(Finding("determinism", "error", "two generations with the same config differ"))
    other = generate(dataclasses.replace(config, seed=config.seed + 1))
    if other.fingerprint == first.fingerprint:
        findings.append(Finding("determinism", "error", "a different seed produced identical data"))
    return findings


def distributions(dataset: SyntheticDataset) -> dict[str, Any]:
    recs = dataset.records
    positives = sum(r.meta.label for r in recs)
    groups: dict[str, list[EvalRecord]] = {}
    for r in recs:
        groups.setdefault(group_id_for(r.features), []).append(r)
    comp = Counter()
    for members in groups.values():
        labels = {m.meta.label for m in members}
        comp["all_suspicious" if labels == {1} else "all_normal" if labels == {0} else "mixed"] += 1
    sizes = sorted(len(v) for v in groups.values())
    prints = Counter(exact_fingerprint(r.features) for r in recs)
    buckets = Counter()
    for s in sizes:
        buckets["1" if s == 1 else "2-3" if s <= 3 else "4-5" if s <= 5 else "6-8" if s <= 8 else "9+"] += 1
    n_groups = len(groups)
    return {
        "records": len(recs),
        "records_suspicious": positives,
        "records_normal": len(recs) - positives,
        "record_level_suspicious_share": positives / len(recs) if recs else 0.0,
        "groups": n_groups,
        "template_composition": {k: comp.get(k, 0) for k in ("all_normal", "all_suspicious", "mixed")},
        "template_level_all_suspicious_share": comp["all_suspicious"] / n_groups if n_groups else 0.0,
        "template_level_containing_suspicious_share": (comp["all_suspicious"] + comp["mixed"]) / n_groups if n_groups else 0.0,
        "group_size": {"min": sizes[0], "median": float(sizes[len(sizes) // 2]), "max": sizes[-1],
                       "mean": sum(sizes) / n_groups, "histogram": dict(buckets)},
        "exact_duplicate_records": sum(c - 1 for c in prints.values() if c > 1),
        "records_by_family": dict(sorted(Counter(m.family for m in dataset.metadata.values()).items())),
        "hard_negative_records": dataset.stats.hard_negative_records,
        "hard_positive_records": dataset.stats.hard_positive_records,
    }


def check_split(dataset: SyntheticDataset, seed: int = PRIMARY_SPLIT_SEED) -> tuple[dict[str, Any], list[Finding]]:
    """Group-aware split with the existing infrastructure, plus isolation verification."""
    findings: list[Finding] = []
    try:
        result = split_records(dataset.records, seed=seed, strict=False)
    except SplitError as exc:
        return {"feasible": False, "error": f"{type(exc).__name__}: {exc}"}, [
            Finding("split", "warning", f"split not feasible: {type(exc).__name__}")]
    for problem in verify_split_isolation(result, dataset.records):
        findings.append(Finding("split_isolation", "error", problem))
    for problem in result.report.problems:
        findings.append(Finding("split", "warning", problem))
    by_id = {r.record_id: r for r in dataset.records}
    per_split: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        members = result.members(name)
        groups: dict[str, set[int]] = {}
        for rid in members:
            groups.setdefault(result.group_of[rid], set()).add(by_id[rid].meta.label)
        pos = sum(by_id[rid].meta.label for rid in members)
        per_split[name] = {
            "records": len(members), "suspicious": pos, "normal": len(members) - pos,
            "suspicious_record_share": pos / len(members) if members else 0.0,
            "groups": len(groups),
            "groups_all_suspicious": sum(1 for v in groups.values() if v == {1}),
            "groups_mixed": sum(1 for v in groups.values() if v == {0, 1}),
        }
    return {
        "feasible": True, "meets_requirements": result.report.meets_requirements, "seed": seed,
        "achieved_ratios": list(result.report.achieved_ratios), "max_ratio_deviation": result.report.max_ratio_deviation,
        "per_split": per_split, "mixed_label_templates": result.report.mixed_label_templates,
        "exact_duplicate_records": result.report.exact_duplicate_records,
        "manifest_sha256": result.manifest_sha256, "problems": list(result.report.problems),
    }, findings


# -- descriptive shortcut statistics (no model) ---------------------------------------------------------


def _body_top_level_fields(body: str) -> int:
    if not body:
        return 0
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return len(parsed)
    return len(urllib.parse.parse_qsl(body, keep_blank_values=True)) if "=" in body else 0


def _feature_values(rec: EvalRecord, position: int, n: int, ts_rank: int) -> dict[str, Any]:
    f = rec.features
    headers = dict(f.headers)
    path, _, query = f.path.partition("?")
    return {
        "method": f.method,
        "path_depth": len([s for s in path.split("/") if s]),
        "path_length_bucket": len(path).bit_length(),
        "query_param_count": len(query.split("&")) if query else 0,
        "body_length_bucket": len(f.body).bit_length(),
        "body_top_level_field_count": _body_top_level_fields(f.body),
        "body_shape": request_template(f).split("|")[-1],
        "header_name_set": ",".join(sorted(name for name, _ in f.headers)),
        "content_type": headers.get("content-type", ""),
        "user_agent": headers.get("user-agent", ""),
        "role": headers.get(cat.ROLE_HEADER, ""),
        "has_authorization": "authorization" in headers,
        "has_client_note": cat.CLIENT_NOTE_HEADER in headers,
        "request_template": request_template(f),
        # Metadata-only features: these should show no association beyond chance.
        "record_id_first_char": rec.record_id[1:2],
        "source_id": rec.meta.source_id or "",
        "timestamp_decile": ts_rank * 10 // n,
        "list_position_decile": position * 10 // n,
    }


METADATA_ONLY_FEATURES = ("record_id_first_char", "source_id", "timestamp_decile", "list_position_decile")


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c) if total else 0.0


def normalized_mutual_information(values: list[Any], labels: list[int]) -> float:
    """I(feature; label) / H(label), in [0, 1]. Descriptive only; biased upward for many values."""
    h_label = _entropy(Counter(labels))
    if h_label == 0 or not values:
        return 0.0
    joint = Counter(zip(values, labels))
    n = len(values)
    px, py = Counter(values), Counter(labels)
    mi = sum((c / n) * math.log2((c / n) / ((px[x] / n) * (py[y] / n))) for (x, y), c in joint.items())
    return max(0.0, mi / h_label)


def shortcut_statistics(
    dataset: SyntheticDataset, *, min_support: int = DEFAULT_MIN_SUPPORT,
    warn_exclusive_share: float = DEFAULT_WARN_EXCLUSIVE_SHARE, seed: int = 0,
    min_template_support: int = DEFAULT_MIN_TEMPLATE_SUPPORT,
) -> dict[str, Any]:
    """Per-feature association with the label, plus label-shuffled references. Not a model.

    Two class-exclusivity measures are reported:

    * ``class_exclusive_record_share`` (legacy): records whose value occurs in >= ``min_support`` records
      and in only one class. Because one template contributes several near-identical records, this
      mostly reflects a single template's own variants.
    * ``class_exclusive_template_share``: records whose value occurs in >= ``min_template_support``
      *distinct templates* and in only one class. This is the cross-template association a model could
      carry from training templates to unseen ones, and it is the measure that raises warnings.
    """
    recs = list(dataset.records)
    n = len(recs)
    order = sorted(range(n), key=lambda i: (recs[i].meta.timestamp, i))
    ts_rank = {i: rank for rank, i in enumerate(order)}
    rows = [_feature_values(r, i, n, ts_rank[i]) for i, r in enumerate(recs)]
    labels = [r.meta.label for r in recs]
    template_of = [group_id_for(r.features) for r in recs]
    rng = random.Random(seed)
    shuffles = [rng.sample(labels, n) for _ in range(_SHUFFLES)]
    stats: dict[str, Any] = {}
    warnings: list[Finding] = []
    for name in rows[0]:
        values = [row[name] for row in rows]
        by_value: dict[Any, Counter] = {}
        for v, y in zip(values, labels):
            by_value.setdefault(v, Counter())[y] += 1
        exclusive = sum(sum(c.values()) for c in by_value.values() if len(c) == 1 and sum(c.values()) >= min_support)
        templates_of_value: dict[Any, set[str]] = {}
        for v, t in zip(values, template_of):
            templates_of_value.setdefault(v, set()).add(t)
        exclusive_t = sum(sum(c.values()) for v, c in by_value.items()
                          if len(c) == 1 and len(templates_of_value[v]) >= min_template_support)
        nmi = normalized_mutual_information(values, labels)
        ref = [normalized_mutual_information(values, s) for s in shuffles]
        entry = {
            "kind": "metadata_only" if name in METADATA_ONLY_FEATURES else "request_content",
            "distinct_values": len(by_value),
            "normalized_mi": round(nmi, 4),
            "shuffled_reference_max": round(max(ref), 4),
            "class_exclusive_record_share": round(exclusive / n, 4),
            "class_exclusive_template_share": round(exclusive_t / n, 4),
        }
        stats[name] = entry
        if entry["kind"] == "metadata_only" and nmi > max(ref):
            warnings.append(Finding("shortcut", "warning",
                                    f"metadata-only feature '{name}' associates with the label more than all {_SHUFFLES} label-shuffled references"))
        if name != "request_template" and exclusive_t / n >= warn_exclusive_share:
            # (request_template is the grouping unit: pure templates are single-class by construction.)
            warnings.append(Finding("shortcut", "warning",
                                    f"feature '{name}': {exclusive_t / n:.1%} of records have a value that occurs in at least "
                                    f"{min_template_support} distinct templates and in only one class"))
    return {"min_support": min_support, "min_template_support": min_template_support,
            "warn_exclusive_share": warn_exclusive_share,
            "note": "Descriptive attention flags, not pass/fail criteria. Request-content association is partly expected by construction.",
            "features": stats, "warnings": [w.to_dict() for w in warnings]}


OVERLAP_FEATURES = ("body_shape", "header_name_set", "content_type", "role", "path_depth", "body_top_level_field_count")


def split_feature_overlap(dataset: SyntheticDataset, seed: int = PRIMARY_SPLIT_SEED) -> dict[str, Any]:
    """How much held-out feature vocabulary also occurs in training, and how class-pure it is there.

    Descriptive only: it never predicts a held-out label. A value that is shared with training and almost
    single-class there is a possible shortcut that a model could learn on one split and reuse on another.
    """
    try:
        result = split_records(dataset.records, seed=seed, strict=False)
    except SplitError:
        return {"available": False}
    recs = list(dataset.records)
    n = len(recs)
    order = sorted(range(n), key=lambda i: (recs[i].meta.timestamp, i))
    ts_rank = {i: rank for rank, i in enumerate(order)}
    rows = {r.record_id: _feature_values(r, i, n, ts_rank[i]) for i, r in enumerate(recs)}
    labels = {r.record_id: r.meta.label for r in recs}
    out: dict[str, Any] = {"available": True, "seed": seed, "purity_cutoff_note": "single-class means 100% one class in training",
                           "features": {}}
    train_ids = result.members("train")
    for name in OVERLAP_FEATURES:
        train_counts: dict[Any, Counter] = {}
        for rid in train_ids:
            train_counts.setdefault(rows[rid][name], Counter())[labels[rid]] += 1
        entry: dict[str, Any] = {}
        for split in ("validation", "test"):
            ids = result.members(split)
            shared = [rid for rid in ids if rows[rid][name] in train_counts]
            single = [rid for rid in shared if len(train_counts[rows[rid][name]]) == 1]
            entry[split] = {
                "records": len(ids),
                "share_value_seen_in_train": round(len(shared) / len(ids), 4) if ids else 0.0,
                "share_of_shared_with_single_class_train_value": round(len(single) / len(shared), 4) if shared else 0.0,
            }
        out["features"][name] = entry
    return out


# -- pilot report -------------------------------------------------------------------------------------------


# -- structural coverage (descriptive; D4-0) -------------------------------------------------------------


_COVERAGE_LEVELS = (
    ("path_category", ("path_category",)),
    ("method_by_path_category", ("method", "path_category")),
    ("method_by_path_category_by_query", ("method", "path_category", "query")),
)


def structural_coverage(dataset: SyntheticDataset, *, min_template_support: int = DEFAULT_MIN_TEMPLATE_SUPPORT) -> dict[str, Any]:
    """Descriptive coverage of request structure by class. Not a model, not a pass/fail check.

    Counting semantics (stated once, used everywhere):

    * A request's *path category* and *method* come from ``policy_sim.observe`` (the same definition the simulated
      rule catalog uses); *query* is ``present`` when the request has a non-empty query string, else ``absent``.
    * **Record counts** count requests per class (normal = label 0, suspicious = label 1).
    * A **template** is one normalized template group (``group_id_for``). It is counted **once per cell**, never once per
      class, so a mixed-label template is a single template. Within a cell each template is classified by the labels
      its records carry *in that cell*: ``normal_only``, ``suspicious_only`` or ``mixed`` (both classes).
      ``templates_with_normal`` = normal_only + mixed and ``templates_with_suspicious`` = suspicious_only + mixed overlap
      on mixed templates and must not be added together.
    * A cell is **class-exclusive** when it holds records of one class only and at least ``min_template_support`` distinct
      templates. A cell is **sparse** when it holds fewer than ``min_template_support`` distinct templates.

    Output is deterministic (cells sorted by their key) and JSON-safe. It holds no raw request content. Class-exclusive
    cells are attention flags only: they are not added to the audit's errors or warnings and do not affect
    ``construction_checks_passed``.
    """
    observed = []
    for r in dataset.records:
        o = observe(r.features)
        observed.append((o.method, o.path_category, "present" if o.has_query else "absent", r.meta.label,
                         group_id_for(r.features)))
    keys_of = {"method": 0, "path_category": 1, "query": 2}
    out: dict[str, Any] = {}
    exclusive: dict[str, list[dict[str, Any]]] = {}
    sparse: dict[str, list[dict[str, Any]]] = {}
    for level, fields in _COVERAGE_LEVELS:
        cells: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in observed:
            key = tuple(row[keys_of[f]] for f in fields)
            cell = cells.setdefault(key, {"records": Counter(), "labels_by_template": {}})
            cell["records"][row[3]] += 1
            cell["labels_by_template"].setdefault(row[4], set()).add(row[3])
        rows, excl, spr = [], [], []
        for key in sorted(cells):
            cell = cells[key]
            sets = list(cell["labels_by_template"].values())
            n_only0 = sum(1 for s in sets if s == {0})
            n_only1 = sum(1 for s in sets if s == {1})
            n_mixed = sum(1 for s in sets if s == {0, 1})
            total = len(sets)
            entry = {
                "cell": dict(zip(fields, key)),
                "records": {"normal": cell["records"][0], "suspicious": cell["records"][1]},
                "templates": {"distinct": total, "normal_only": n_only0, "suspicious_only": n_only1, "mixed": n_mixed,
                              "with_normal": n_only0 + n_mixed, "with_suspicious": n_only1 + n_mixed},
            }
            rows.append(entry)
            if total >= min_template_support and (cell["records"][0] == 0 or cell["records"][1] == 0):
                excl.append({**entry, "exclusive_to": "normal" if cell["records"][1] == 0 else "suspicious"})
            if total < min_template_support:
                spr.append(entry)
        out[level] = rows
        exclusive[level] = excl
        sparse[level] = spr
    all_templates: dict[str, set[int]] = {}
    for row in observed:
        all_templates.setdefault(row[4], set()).add(row[3])
    return {
        "definition": ("Cells use policy_sim.observe (method, path category, query present/absent). Each normalized "
                       "template is counted once per cell; mixed-label templates are single templates, reported separately "
                       "from one-class templates. Descriptive only."),
        "min_template_support": min_template_support,
        "records": len(observed),
        "distinct_templates": len(all_templates),
        "template_composition": {
            "normal_only": sum(1 for v in all_templates.values() if v == {0}),
            "suspicious_only": sum(1 for v in all_templates.values() if v == {1}),
            "mixed": sum(1 for v in all_templates.values() if v == {0, 1}),
        },
        **out,
        "class_exclusive_cells": exclusive,
        "sparse_cells": sparse,
        "note": ("Descriptive attention output, not pass/fail criteria; it does not change the audit's errors or "
                 "warnings. Passing the audit does not show the absence of shortcut features or leakage, and this is not "
                 "an AI detection-quality evaluation."),
    }


def build_pilot_report(config: GeneratorConfig | None = None, split_seed: int = PRIMARY_SPLIT_SEED) -> dict[str, Any]:
    """Generate once and run every non-model check. JSON-safe; holds no raw request content."""
    cfg = config or GeneratorConfig()
    dataset = generate(cfg)
    errors: list[Finding] = []
    errors += scan_label_terms(dataset)
    errors += check_feature_separation(dataset)
    errors += check_content_type_semantics(dataset)
    errors += check_determinism(cfg)
    split_info, split_findings = check_split(dataset, split_seed)
    shortcuts = shortcut_statistics(dataset)
    sharing = pool_sharing(dataset)
    overlap = split_feature_overlap(dataset, split_seed)
    all_findings = errors + split_findings
    all_findings += [Finding(w["check"], w["severity"], w["message"]) for w in shortcuts["warnings"]]
    all_findings += [Finding(w["check"], w["severity"], w["message"]) for w in sharing["warnings"]]
    hard_errors = [f for f in all_findings if f.severity == "error"]
    warnings = [f for f in all_findings if f.severity == "warning"]
    stats = dataset.stats
    return {
        "protocol": PROTOCOL_LABEL,
        "config": cfg.to_dict(),
        "generation": {
            "requested_templates": stats.requested_templates, "actual_templates": stats.actual_templates,
            "shortfall": stats.shortfall, "collapsed_specs": stats.collapsed_specs, "attempts": stats.attempts,
            "records": stats.n_records, "duplicate_variants_skipped": stats.duplicate_variants_skipped,
            "templates_by_family": dict(sorted(stats.templates_by_family.items())),
            "fingerprint": dataset.fingerprint,
        },
        "distributions": distributions(dataset),
        "split": split_info,
        "pool_sharing": sharing,
        "shortcut_statistics": shortcuts,
        "split_feature_overlap": overlap,
        "structural_coverage": structural_coverage(dataset),
        "errors": [f.to_dict() for f in hard_errors],
        "warnings": [f.to_dict() for f in warnings],
        "construction_checks_passed": not hard_errors,
        "notice": ("Passing these checks does not show the generator is free of leakage or shortcuts. Labels are "
                   "defined by same-author scenario construction. No model was trained and no detection performance "
                   "was measured."),
    }
