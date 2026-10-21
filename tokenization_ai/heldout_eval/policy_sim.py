"""Simulated policy rule catalog for the future request-only versus rule-fused comparison (Arm B).

**These rules are SIMULATED. They are not Member 2 output, not formally verified, and carry no
verification status.** Nothing here imports or calls Member 2, makes a network call, or connects to
the API, the dashboard or the pipeline. It is a pure, deterministic function from a request's
*observable* properties to a list of neutral policy statements.

Why this exists and what it can show
------------------------------------
Arm B of the held-out study feeds rule tokens through the existing fusion, tokenizer and tensorizer.
Because each rule here is a deterministic function of the request itself, the rules add **no
independent security information**: any difference between the request-only arm and this arm can
only show whether the rule-token *representation* changes results on synthetic requests. It cannot
show the value of genuine, independently verified Member 2 rules, and fusion is not expected, and
not assumed, to help.

Allowed inputs (``Observables``)
--------------------------------
Only these properties of a request may decide whether a policy applies:

* the HTTP method;
* a coarse *path category* from the first path segment (admin, auth, files, documents, status or
  other resource);
* the *declared* content-type class (json, xml, form, plain, other, none), taken from the header
  value only, never from the body content;
* whether an authorization header is present (presence only, never its value);
* whether the request has a body, and whether it has a query string (presence only).

Never used: labels, scenario families, template ids, source ids, split assignments, record ids,
timestamps, generated metadata, role-header *values*, body or query *content*, or model predictions.
``simulate_rules`` accepts only ``RequestFeatures`` (and an event id), so none of those can be passed in.

Provenance
----------
The result is a ``SimulatedRuleSet``: the ``VerifiedRuleInfo`` the pipeline expects, plus explicit
``provenance="simulated"``, ``verification_status="not_verified_simulated"``, the catalog version and
its SHA-256. ``VerifiedRuleInfo`` is the pipeline's existing input type; its name must not be read
as a claim about these rules. The existing API ``RuleSource`` values are ``none``, ``mock_fixture`` and
``verified``: there is no ``simulated`` value, ``verified`` would be false, and ``mock_fixture`` is only
the nearest honest label. The simulator is therefore kept out of the API.

Catalog design
--------------
Policy ids are neutral and sequential (``SIM-P01`` ...), never named for a scenario, attack or class.
Attributes are limited to ``action``, ``resource`` and ``effect`` (fixed vocabularies), which keeps each
rule to 20 tokens in the existing tokenizer. Rule order is the catalog order. The catalog is data, so
its fingerprint covers every applicability condition.

Draft status: the catalog was written before any rule-specific diagnostics were run (the synthetic pilot's
labels had been seen in earlier generator audits, but no condition here refers to a label); the diagnostics
in ``policy_diagnostics.py`` describe it and were not used to select, add or drop a rule.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..models import VerifiedRule, VerifiedRuleInfo
from . import synthetic_catalog as cat
from .splits import EvalRecord, RequestFeatures

CATALOG_VERSION = "simpol-1"
SIMULATED_PROVENANCE = "simulated"
NOT_VERIFIED_STATUS = "not_verified_simulated"

ALLOWED_ATTRIBUTES: tuple[str, ...] = ("action", "resource", "effect")
ALLOWED_EFFECTS = frozenset({"Allow", "Deny"})
PATH_CATEGORIES = ("admin_area", "auth_area", "files_area", "documents_area", "status_area", "resource_area")
DECLARED_TYPE_CLASSES = ("json", "xml", "form", "plain", "other", "none")
_CONDITION_KEYS = frozenset({"path_category", "methods", "declared_type", "authorization", "body", "query"})
_ID_PATTERN = re.compile(r"SIM-P\d{2}")
_SAFE_VALUE = re.compile(r"[a-z][a-z-]{0,31}|[A-Z][a-z]{0,15}")

_FIRST_SEGMENT_CATEGORY = {
    "admin": "admin_area",
    "auth": "auth_area", "account": "auth_area", "portal": "auth_area",
    "files": "files_area",
    "documents": "documents_area",
    "health": "status_area", "ping": "status_area", "status": "status_area",
}
_TYPE_CLASS = {
    "application/json": "json",
    "application/xml": "xml", "text/xml": "xml",
    "application/x-www-form-urlencoded": "form",
    "text/plain": "plain",
}


class PolicySimError(ValueError):
    """The catalog or a request given to the simulator is invalid."""


@dataclass(frozen=True)
class PolicyDef:
    rule_id: str
    action: str
    resource: str
    effect: str
    applies_when: Mapping[str, Any]  # declarative conditions; every listed condition must hold


def _p(rule_id: str, action: str, resource: str, effect: str, **when: Any) -> PolicyDef:
    return PolicyDef(rule_id, action, resource, effect, when)


POLICIES: tuple[PolicyDef, ...] = (
    _p("SIM-P01", "manage", "admin-area", "Allow", path_category=("admin_area",)),
    _p("SIM-P02", "read", "resources", "Allow", path_category=("resource_area",), methods=("GET",), authorization=True),
    _p("SIM-P03", "write", "resources", "Allow", path_category=("resource_area",),
       methods=("POST", "PUT", "PATCH"), authorization=True),
    _p("SIM-P04", "delete", "resources", "Allow", path_category=("resource_area",), methods=("DELETE",)),
    _p("SIM-P05", "authenticate", "sessions", "Allow", path_category=("auth_area",), methods=("POST",)),
    _p("SIM-P06", "read", "status", "Allow", path_category=("status_area",), methods=("GET",)),
    _p("SIM-P07", "read", "files", "Allow", path_category=("files_area",), authorization=True),
    _p("SIM-P08", "read", "documents", "Allow", path_category=("documents_area",)),
    _p("SIM-P09", "submit", "json-payloads", "Allow", body=True, declared_type=("json",)),
    _p("SIM-P10", "submit", "text-payloads", "Allow", body=True, declared_type=("xml", "plain")),
    _p("SIM-P11", "submit", "form-payloads", "Allow", body=True, declared_type=("form",)),
    _p("SIM-P12", "access", "protected-resources", "Deny", authorization=False,
       path_category=("admin_area", "files_area", "documents_area", "resource_area")),
    _p("SIM-P13", "search", "query-parameters", "Allow", methods=("GET",), query=True),
)


# -- catalog validation and fingerprint ----------------------------------------------------------------------


def validate_catalog(policies: Iterable[PolicyDef]) -> None:
    """Raise :class:`PolicySimError` unless the catalog follows every rule in the module docstring."""
    policies = tuple(policies)
    if not policies:
        raise PolicySimError("the catalog is empty")
    banned = [w.lower() for w in cat.BANNED_LABEL_WORDS] + [n.lower() for n in cat.family_names()]
    seen: set[str] = set()
    for policy in policies:
        if not isinstance(policy, PolicyDef) or not _ID_PATTERN.fullmatch(str(policy.rule_id)):
            raise PolicySimError("policy ids must match SIM-Pnn")
        if policy.rule_id in seen:
            raise PolicySimError("duplicate policy id")
        seen.add(policy.rule_id)
        for name in ALLOWED_ATTRIBUTES:
            value = getattr(policy, name)
            if not isinstance(value, str) or not _SAFE_VALUE.fullmatch(value):
                raise PolicySimError(f"{policy.rule_id}: attribute '{name}' is not a safe allowlisted value")
            if any(word in value.lower() for word in banned):
                raise PolicySimError(f"{policy.rule_id}: attribute '{name}' contains a label or family word")
        if policy.effect not in ALLOWED_EFFECTS:
            raise PolicySimError(f"{policy.rule_id}: unknown effect")
        if any(word in policy.rule_id.lower() for word in banned):
            raise PolicySimError("policy id contains a label or family word")
        _validate_conditions(policy)


def _validate_conditions(policy: PolicyDef) -> None:
    when = policy.applies_when
    if not isinstance(when, Mapping) or not when:
        raise PolicySimError(f"{policy.rule_id}: applicability needs at least one condition")
    unknown = set(when) - _CONDITION_KEYS
    if unknown:
        raise PolicySimError(f"{policy.rule_id}: unknown condition key(s) {sorted(unknown)}")
    allowed_values = {
        "path_category": set(PATH_CATEGORIES), "declared_type": set(DECLARED_TYPE_CLASSES),
        "methods": {"GET", "POST", "PUT", "PATCH", "DELETE"},
    }
    for key, values in when.items():
        if key in allowed_values:
            if not isinstance(values, tuple) or not values or not set(values) <= allowed_values[key]:
                raise PolicySimError(f"{policy.rule_id}: invalid values for '{key}'")
        elif not isinstance(values, bool):
            raise PolicySimError(f"{policy.rule_id}: '{key}' must be a boolean")


def catalog_fingerprint(policies: Iterable[PolicyDef] | None = None) -> str:
    """SHA-256 of the canonical catalog (version, ids, attributes and every applicability condition)."""
    chosen = POLICIES if policies is None else tuple(policies)
    body = [CATALOG_VERSION, [
        [p.rule_id, p.action, p.resource, p.effect,
         {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(p.applies_when.items())}]
        for p in chosen]]
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


validate_catalog(POLICIES)
CATALOG_SHA256 = catalog_fingerprint()


# -- observable properties -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Observables:
    """The only request properties that may decide policy applicability."""

    method: str
    path_category: str
    declared_type: str
    has_authorization: bool
    has_body: bool
    has_query: bool


def observe(features: RequestFeatures) -> Observables:
    """Extract the allowed observable properties. Accepts ``RequestFeatures`` only."""
    if not isinstance(features, RequestFeatures):
        raise TypeError("observe accepts RequestFeatures only")
    if not isinstance(features.method, str) or not features.method.strip():
        raise PolicySimError("request method is missing or not a string")
    if not isinstance(features.path, str) or not features.path.startswith("/"):
        raise PolicySimError("request path is missing or does not start with '/'")
    if not isinstance(features.body, str):
        raise PolicySimError("request body must be a string")
    try:
        headers = {str(name).lower(): value for name, value in features.headers}
    except (TypeError, ValueError) as exc:
        raise PolicySimError("request headers are malformed") from exc
    path_part, _, query = features.path.split("#", 1)[0].partition("?")
    first = next((seg for seg in path_part.lower().split("/") if seg), "")
    declared = headers.get("content-type", "")
    declared_class = "none"
    if isinstance(declared, str) and declared.strip():
        declared_class = _TYPE_CLASS.get(declared.split(";", 1)[0].strip().lower(), "other")
    return Observables(
        method=features.method.upper(),
        path_category=_FIRST_SEGMENT_CATEGORY.get(first, "resource_area"),
        declared_type=declared_class,
        has_authorization="authorization" in headers,
        has_body=features.body != "",
        has_query=query != "",
    )


def _applies(policy: PolicyDef, obs: Observables) -> bool:
    when = policy.applies_when
    return (
        ("path_category" not in when or obs.path_category in when["path_category"])
        and ("methods" not in when or obs.method in when["methods"])
        and ("declared_type" not in when or obs.declared_type in when["declared_type"])
        and ("authorization" not in when or obs.has_authorization == when["authorization"])
        and ("body" not in when or obs.has_body == when["body"])
        and ("query" not in when or obs.has_query == when["query"])
    )


def applicable_policies(features: RequestFeatures, policies: Iterable[PolicyDef] | None = None) -> tuple[PolicyDef, ...]:
    """Policies that apply to a request, in catalog order. Depends on ``Observables`` only."""
    obs = observe(features)
    return tuple(p for p in (POLICIES if policies is None else policies) if _applies(p, obs))


@dataclass(frozen=True)
class SimulatedRuleSet:
    """Rule information for one request, with explicit simulated provenance."""

    rule_info: VerifiedRuleInfo
    policy_ids: tuple[str, ...]
    provenance: str = SIMULATED_PROVENANCE
    verification_status: str = NOT_VERIFIED_STATUS
    catalog_version: str = CATALOG_VERSION
    catalog_sha256: str = CATALOG_SHA256


def simulate_rules(event_id: str, features: RequestFeatures) -> SimulatedRuleSet:
    """Simulated rule information for one request.

    An empty rule list means only that no simulated policy applies; it says nothing about verification.
    """
    if not isinstance(event_id, str) or not event_id or event_id != event_id.strip():
        raise PolicySimError("event_id must be a non-empty string without surrounding whitespace")
    chosen = applicable_policies(features)
    rules = [VerifiedRule(rule_id=p.rule_id, attributes={"action": p.action, "resource": p.resource, "effect": p.effect})
             for p in chosen]
    return SimulatedRuleSet(rule_info=VerifiedRuleInfo(event_id=event_id, rules=rules),
                            policy_ids=tuple(p.rule_id for p in chosen))


def simulate_rules_for_record(record: EvalRecord) -> SimulatedRuleSet:
    """Convenience wrapper: uses ``record.record_id`` and ``record.features`` only (never ``record.meta``)."""
    if not isinstance(record, EvalRecord):
        raise TypeError("simulate_rules_for_record accepts an EvalRecord")
    return simulate_rules(record.record_id, record.features)
