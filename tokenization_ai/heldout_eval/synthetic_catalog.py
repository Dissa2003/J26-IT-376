"""Scenario catalog for the synthetic API request generator (draft protocol heldout-eval-1.1.0-draft).

Pure data and small text helpers: no I/O, no network, no randomness of its own (every helper takes a
``random.Random``), and no imports from the production pipeline.

Everything here is FABRICATED and harmless. The "suspicious" strings are short, generic,
illustrative patterns of the kind found in security teaching material; none is a working exploit and
none targets a real system. Words, names, tokens and credentials are invented.

How labels are defined (same-author construction)
-------------------------------------------------
A request is labelled suspicious (1) **only** by the documented construction of its scenario family,
never by a model, a detector or a real-world judgement:

* ``normal`` families only ever render ordinary values and are labelled 0.
* ``suspicious`` families always render the family's suspicious construction and are labelled 1.
* ``mixed`` families share one request template between both kinds: the *values* (or one header) of
  a given variant decide the label, by the rule stated on the family. These produce the
  mixed-label templates that the splitter keeps together.

Because the same author writes the families and the labels, a score on this data measures whether a
model learns the author's distinction on unseen templates. It is not real-world attack detection.

Label-leak rules for this catalog: family names, scenario ids and the words in ``BANNED_LABEL_WORDS``
must never appear in a request field (path, query, body, headers, ids). The shared pools below are
used by every class, so no pool item identifies a class. Words in the pools avoid the banned words
and avoid SQL-ish, markup-ish and shell-ish tokens (those appear only in the dedicated payload
helpers and in the deliberate hard-negative phrases).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

CATALOG_VERSION = "synthetic-catalog-2"

LabelKind = Literal["normal", "suspicious", "mixed"]

# Words that must never appear in any request field or identifier (labels or class names).
BANNED_LABEL_WORDS: tuple[str, ...] = (
    "attack", "malicious", "benign", "suspicious", "normal", "anomal", "threat", "exploit",
    "inject", "evil", "hostile", "intrusion", "legit", "family", "scenario", "label",
)

# -- shared pools (used by every class) -------------------------------------------------------

RESOURCES: tuple[str, ...] = (
    "orders", "invoices", "customers", "products", "shipments", "tickets", "reports", "sessions",
    "profiles", "documents", "payments", "reviews", "schedules", "inventory", "messages",
    "notifications", "accounts", "warehouses", "devices", "subscriptions", "categories", "coupons",
    "appointments", "projects", "tasks",
)

SUB_RESOURCES: tuple[str, ...] = (
    "items", "history", "summary", "details", "comments", "attachments", "status", "export",
    "preview", "settings", "lookup", "batch",
)

HTTP_METHODS: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE")

USER_AGENTS: tuple[str, ...] = (
    "ExampleClient/1.0", "SampleBrowser/5.0 (Fabricated OS)", "DemoApp/2.3 (+https://example.invalid)",
    "TestHarness/0.9", "MobileSample/3.1 (Fabricated Phone)", "BatchWorker/1.7",
)

ACCEPT_VALUES: tuple[str, ...] = ("application/json", "*/*")

ROLES_LOW: tuple[str, ...] = ("viewer", "editor", "support")
ROLE_ADMIN = "admin"

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_FORM = "application/x-www-form-urlencoded"
# Declared content types that are COMPATIBLE with each actual body format. A content-type mismatch is
# incompatibility between the declared type and the actual body, never the declared value alone: every
# value below is legitimate for its own format and wrong for the others.
CONTENT_TYPES_BY_FORMAT: dict[str, tuple[str, ...]] = {
    "json": (CONTENT_TYPE_JSON,),
    "xml": ("application/xml", "text/xml"),
    "form": (CONTENT_TYPE_FORM,),
    "plain": ("text/plain",),
}

# Short, neutral words for ordinary text values. No digits, no banned words, no SQL/markup/shell words.
WORDS: tuple[str, ...] = (
    "amber", "river", "maple", "harbor", "violet", "cedar", "lantern", "meadow", "copper", "willow",
    "orchid", "granite", "falcon", "breeze", "marble", "juniper", "saffron", "thistle", "ember", "quartz",
    "sparrow", "walnut", "pebble", "canyon", "lagoon", "tulip", "summit", "velvet", "prairie", "mosaic",
)

FIRST_NAMES: tuple[str, ...] = ("alex", "sam", "jordan", "casey", "riley", "morgan", "taylor", "avery")

FIELD_NAMES_TEXT: tuple[str, ...] = ("title", "note", "description", "name", "summary", "comment", "subject")
FIELD_NAMES_NUM: tuple[str, ...] = ("quantity", "amount", "priority", "rating", "count", "weight")
FIELD_NAMES_BOOL: tuple[str, ...] = ("active", "archived", "notify", "pinned")
FIELD_NAMES_LIST: tuple[str, ...] = ("tags", "keywords_list", "members", "options")
FIELD_NAMES_OBJECT: tuple[str, ...] = ("address", "contact", "metadata_info", "schedule_info")
QUERY_NUM: tuple[str, ...] = ("page", "limit", "offset", "year")
QUERY_TOKEN: tuple[str, ...] = ("sort", "filter", "view", "format", "include")
QUERY_TEXT: tuple[str, ...] = ("q", "term", "keyword", "search_text")

# Names that mass assignment adds to a body (privileged-looking), and keyword-free variants.
PRIVILEGED_FIELDS_OBVIOUS: tuple[str, ...] = ("is_admin", "role", "balance_override", "verified")
PRIVILEGED_FIELDS_QUIET: tuple[str, ...] = ("owner", "tier", "scope", "group_ref")

CLIENT_NOTE_HEADER = "x-client-note"
ROLE_HEADER = "x-role"

# -- ordinary text -----------------------------------------------------------------------------

# Hard-negative phrases: ordinary text that merely contains SQL-ish, markup-ish or shell-ish words.
HARD_NEGATIVE_PHRASES: tuple[str, ...] = (
    "select a seat for friday", "union station pickup schedule", "drop off time update",
    "how to insert a coin", "please update the delete policy wording", "tips & tricks; part two",
    "if a < b and b > c then a < c", "love <3 this product", "run the report; then share it",
    "compare option a && option b", "cost - discount = final price", "a note about the order -- see below",
)


def plain_phrase(rng: random.Random, min_words: int = 1, max_words: int = 4) -> str:
    """Ordinary text of ``min_words``..``max_words`` neutral words (always contains a letter)."""
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(min_words, max_words)))


def hard_negative_phrase(rng: random.Random) -> str:
    return rng.choice(HARD_NEGATIVE_PHRASES)


def fake_username(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)}_{rng.choice(WORDS)}"


def fake_secret(rng: random.Random) -> str:
    """An invented placeholder secret. Never a real credential."""
    return "Fx" + "".join(rng.choice("abcdefghijkmnpqrstuvwxyz") for _ in range(8)) + "Zq"


# -- suspicious constructions (harmless and illustrative) ------------------------------------------

_SQL_EASY = ("' OR 1=1 --", "admin'--", "1; DROP TABLE items --", "' UNION SELECT a, b FROM t --", "x' OR 'a'='a")
_CMD_EASY = ("; ls -la", "| cat /etc/hosts", "&& whoami", "; echo done")
_CMD_HARD = ("`id`", "$(whoami)", "%3B%20ls%20-la")
_MARKUP_EASY = ("<script>alert(1)</script>", "<img src=x onerror=alert(1)>", '"><svg onload=alert(1)>')
_MARKUP_HARD = ("<svg/onload=alert(1)>", "%3Cscript%3Ealert(1)%3C/script%3E", "<ScRiPt>alert(1)</sCrIpT>")


def sql_like(rng: random.Random, hard: bool) -> str:
    """SQL-injection-like text. ``hard`` obfuscates it (mixed case, comment splitting, encoding)."""
    base = rng.choice(_SQL_EASY)
    if not hard:
        return base
    style = rng.choice(("case", "comment", "encoded"))
    if style == "case":
        return "".join(ch.upper() if i % 2 else ch.lower() for i, ch in enumerate(base))
    if style == "comment":
        return base.replace(" ", "/**/")
    return base.replace("'", "%27").replace(" ", "%20")


def cmd_like(rng: random.Random, hard: bool) -> str:
    """Ordinary words followed by a command-injection-like suffix."""
    return f"{plain_phrase(rng, 1, 2)}{rng.choice(_CMD_HARD if hard else _CMD_EASY)}"


def markup_like(rng: random.Random, hard: bool) -> str:
    return rng.choice(_MARKUP_HARD if hard else _MARKUP_EASY)


TRAVERSAL_TAILS: tuple[str, ...] = (
    "etc/passwd", "windows/win.ini", "var/log/app.log", "config/secrets.cfg", "hosts", "boot.ini", "shadow", "id_rsa",
)


def traversal_segments(rng: random.Random, hard: bool) -> tuple[str, ...]:
    """Path segments of a traversal-like path (``hard`` uses encoded separators)."""
    depth = rng.randint(1, 3)
    tail = rng.choice(TRAVERSAL_TAILS)
    if hard:
        return (("..%2f" * depth) + tail.replace("/", "%2f"),)
    return tuple([".."] * depth) + tuple(tail.split("/"))


# -- scenario families -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioFamily:
    """One documented scenario family. ``name`` appears only in metadata, never in a request."""

    name: str
    label_kind: LabelKind
    description: str
    label_rule: str
    weight: int
    hard_case: str  # "", "negative" or "positive": which kind of hard case this family contributes


FAMILIES: tuple[ScenarioFamily, ...] = (
    # -- normal -------------------------------------------------------------------------------
    ScenarioFamily("crud_read", "normal", "Routine reads of a resource or nested item path, with paging and sorting.",
                   "always 0", 8, ""),
    ScenarioFamily("crud_write", "normal", "Routine create, update and delete calls with a flat JSON body of 2-7 fields.",
                   "always 0", 8, ""),
    ScenarioFamily("search_filter", "normal", "Search and filtering with ordinary text and filter values.",
                   "always 0", 7, ""),
    ScenarioFamily("auth_login", "normal", "Ordinary login-style calls with invented credentials (JSON or form).",
                   "always 0", 4, ""),
    ScenarioFamily("json_create_nested", "normal", "JSON create or update with nested objects and lists.",
                   "always 0", 6, ""),
    ScenarioFamily("admin_legit", "normal", "Administration calls on admin-only paths made with the admin role.",
                   "always 0 (role is admin)", 3, ""),
    ScenarioFamily("health_status", "normal", "Health, ping and status calls.", "always 0", 3, ""),
    ScenarioFamily("hn_keyword_text", "normal",
                   "HARD NEGATIVE: ordinary text that merely contains SQL-ish, markup-ish or shell-ish words.",
                   "always 0", 7, "negative"),
    ScenarioFamily("hn_dotted_paths", "normal",
                   "HARD NEGATIVE: ordinary file paths with dots and nested folders (names like report.final.pdf).",
                   "always 0", 6, "negative"),
    ScenarioFamily("hn_long_payload", "normal",
                   "HARD NEGATIVE: legitimately large JSON bodies with many fields and long text.",
                   "always 0", 5, "negative"),
    ScenarioFamily("alt_content_type", "normal",
                   "HARD NEGATIVE: legitimate plain-text and XML bodies sent with the matching declared type.",
                   "always 0 (declared type matches the body)", 5, "negative"),
    ScenarioFamily("admin_privileged_update", "normal",
                   "HARD NEGATIVE: administrators setting privileged-looking fields on admin-only paths.",
                   "always 0 (admin role on an admin path)", 4, "negative"),
    ScenarioFamily("hn_nested_config", "normal",
                   "HARD NEGATIVE: legitimately nested configuration bodies (3-5 levels).",
                   "always 0", 4, "negative"),
    # -- suspicious ----------------------------------------------------------------------------
    ScenarioFamily("sqli_like", "suspicious",
                   "SQL-injection-like text in a query, body or login-style text slot (ordinary and large bodies).",
                   "always 1; some variants are obfuscated (hard positives)", 5, "positive"),
    ScenarioFamily("traversal_like", "suspicious", "Path-traversal-like request paths of varying depth.",
                   "always 1; some variants use encoded separators (hard positives)", 3, "positive"),
    ScenarioFamily("cmdi_like", "suspicious", "Command-injection-like suffixes on ordinary text.",
                   "always 1; some variants use subshell forms (hard positives)", 4, "positive"),
    ScenarioFamily("markup_like", "suspicious", "Markup-style script or event-handler text in a text slot.",
                   "always 1; some variants are encoded or case-varied (hard positives)", 4, "positive"),
    ScenarioFamily("mass_assignment", "suspicious",
                   "Privileged-looking fields in a JSON body sent by a NON-admin role to a non-admin endpoint.",
                   "always 1 (non-admin role setting privileged fields); some use keyword-free names (hard positives)",
                   5, "positive"),
    ScenarioFamily("structure_anomaly", "suspicious",
                   "Nesting deeper than the legitimate depth, or a list/object in a field whose name implies a number.",
                   "always 1", 3, ""),
    # -- mixed (one shared template, label decided by the variant) --------------------------------------
    ScenarioFamily("role_action_mismatch", "mixed",
                   "One admin-only endpoint called with the admin role or with a lower role.",
                   "1 iff the role header is not admin (an unauthorized role/action combination)", 4, "positive"),
    ScenarioFamily("content_type_mismatch", "mixed",
                   "One endpoint with a JSON, XML, form or plain-text body, declared with a matching or a mismatched type.",
                   "1 iff the declared content-type is incompatible with the ACTUAL body format (the declared value "
                   "alone never decides; each type is legitimate for its own format)", 4, "positive"),
    ScenarioFamily("shared_endpoint_text", "mixed",
                   "One endpoint whose text slot holds ordinary text or an injection-, markup- or command-like value.",
                   "1 iff the slot value is a suspicious construction; ordinary values may be hard negatives", 4,
                   "positive"),
)

FAMILY_BY_NAME = {f.name: f for f in FAMILIES}

# Families that place a text value in a slot the generator may fill from a payload helper.
PAYLOAD_STYLES: tuple[str, ...] = ("sql", "cmd", "markup")


def family_names() -> tuple[str, ...]:
    return tuple(f.name for f in FAMILIES)


def default_weights() -> dict[str, int]:
    return {f.name: f.weight for f in FAMILIES}


def weight_shares() -> dict[str, float]:
    """Share of templates per label kind implied by the default weights (documentation aid)."""
    total = sum(f.weight for f in FAMILIES)
    out = {"normal": 0.0, "suspicious": 0.0, "mixed": 0.0}
    for f in FAMILIES:
        out[f.label_kind] += f.weight / total
    return out
