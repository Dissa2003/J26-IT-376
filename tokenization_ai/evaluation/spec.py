"""Independent specification of the Member 4 representation (PROTOCOL.md section 3).

Nothing in this module imports the production tokenizer, masker, normalizer,
vocabulary or tensorizer: the tag strings below are restated from the written
grammar so the oracle does not depend on the code under evaluation.

It provides:

- :func:`parse_compact` / :func:`to_compact` -- the fixtures' ``"C:[SEQ_START]"``
  token notation (``C`` control, ``A`` API literal, ``R`` rule literal, ``M`` masked).
- :func:`well_formed` -- a grammar *acceptor* for token/role sequences.
- :func:`reference_encode` -- a second, separately written encoder used only
  for seeded/designed cases whose data uses a deliberately small feature
  subset (see its docstring). It is cross-checked against the hand-written
  fixtures before scoring.
"""

from __future__ import annotations

import re
from typing import Any

ROLE_FROM_CODE = {"C": "control", "A": "literal_api", "R": "literal_rule", "M": "masked"}
CODE_FROM_ROLE = {role: code for code, role in ROLE_FROM_CODE.items()}

MASK_PLACEHOLDERS = frozenset({"[MASKED_AUTH]", "[MASKED_SECRET]", "[MASKED_PII]", "[MASKED_IP]"})

_API_PART = {
    "METHOD", "PATH_START", "PATH_SEG", "PATH_END", "HEADERS_START", "HEADER_NAME",
    "HEADER_VALUE", "HEADER_VALUE_MASKED", "HEADERS_END", "PAYLOAD_START", "PAYLOAD_END",
}
_VALUE_PART = {
    "KEY", "OBJ_START", "OBJ_END", "LIST_START", "LIST_END", "VAL_STR", "VAL_INT",
    "VAL_FLOAT", "VAL_BOOL", "VAL_NULL",
}

CONTROL_TAGS = frozenset(
    {"[SEQ_START]", "[SEQ_END]", "[RULES_START]", "[RULES_END]", "[RULE_START]", "[RULE_END]"}
    | {f"[API:{p}]" for p in _API_PART}
    | {f"[API:{p}]" for p in _VALUE_PART}
    | {"[API:VAL_MASKED_STR]"}
    | {f"[RULE:{p}]" for p in _VALUE_PART}
    | {"[RULE:ID]", "[RULE:ATTRS_START]", "[RULE:ATTRS_END]"}
)

UNK = "[UNK]"


def parse_compact(items: list[str]) -> tuple[list[str], list[str]]:
    """``["C:[SEQ_START]", "A:POST"]`` -> ``(tokens, roles)``."""
    tokens, roles = [], []
    for item in items:
        code, sep, text = item.partition(":")
        if sep != ":" or code not in ROLE_FROM_CODE:
            raise ValueError(f"bad compact token {item!r}")
        tokens.append(text)
        roles.append(ROLE_FROM_CODE[code])
    return tokens, roles


def to_compact(tokens: list[str], roles: list[str]) -> list[str]:
    return [f"{CODE_FROM_ROLE.get(role, '?')}:{token}" for token, role in zip(tokens, roles)]


# --- grammar acceptor -----------------------------------------------------


class _Reject(Exception):
    pass


class _Parser:
    def __init__(self, tokens: list[str], roles: list[str]) -> None:
        self.t, self.r, self.i = tokens, roles, 0

    def _peek_control(self) -> str | None:
        if self.i < len(self.t) and self.r[self.i] == "control":
            return self.t[self.i]
        return None

    def control(self, tag: str) -> None:
        if self._peek_control() != tag:
            got = self.t[self.i] if self.i < len(self.t) else "<end>"
            raise _Reject(f"index {self.i}: expected control {tag!r}, found {got!r}")
        self.i += 1

    def literal(self, role: str) -> str:
        if self.i >= len(self.t) or self.r[self.i] != role:
            got = self.r[self.i] if self.i < len(self.t) else "<end>"
            raise _Reject(f"index {self.i}: expected role {role!r}, found {got!r}")
        text = self.t[self.i]
        self.i += 1
        return text

    def sequence(self) -> None:
        self.control("[SEQ_START]")
        self.control("[API:METHOD]")
        self.literal("literal_api")
        self.control("[API:PATH_START]")
        while self._peek_control() == "[API:PATH_SEG]":
            self.i += 1
            self.literal("literal_api")
        self.control("[API:PATH_END]")
        self.control("[API:HEADERS_START]")
        while self._peek_control() == "[API:HEADER_NAME]":
            self.i += 1
            self.literal("literal_api")
            if self._peek_control() == "[API:HEADER_VALUE]":
                self.i += 1
                self.literal("literal_api")
            else:
                self.control("[API:HEADER_VALUE_MASKED]")
                self._masked()
        self.control("[API:HEADERS_END]")
        self.control("[API:PAYLOAD_START]")
        self._object("API")
        self.control("[API:PAYLOAD_END]")
        self.control("[RULES_START]")
        while self._peek_control() == "[RULE_START]":
            self.i += 1
            self.control("[RULE:ID]")
            self.literal("literal_rule")
            self.control("[RULE:ATTRS_START]")
            self._object("RULE")
            self.control("[RULE:ATTRS_END]")
            self.control("[RULE_END]")
        self.control("[RULES_END]")
        self.control("[SEQ_END]")

    def _masked(self) -> None:
        text = self.literal("masked")
        if text not in MASK_PLACEHOLDERS:
            raise _Reject(f"masked token {text!r} is not a placeholder")

    def _object(self, ns: str) -> None:
        lit = "literal_api" if ns == "API" else "literal_rule"
        self.control(f"[{ns}:OBJ_START]")
        while self._peek_control() == f"[{ns}:KEY]":
            self.i += 1
            self.literal(lit)
            self._value(ns)
        self.control(f"[{ns}:OBJ_END]")

    def _value(self, ns: str) -> None:
        lit = "literal_api" if ns == "API" else "literal_rule"
        tag = self._peek_control()
        if tag == f"[{ns}:OBJ_START]":
            self._object(ns)
        elif tag == f"[{ns}:LIST_START]":
            self.i += 1
            while self._peek_control() != f"[{ns}:LIST_END]":
                if self.i >= len(self.t):
                    raise _Reject("unterminated list")
                self._value(ns)
            self.i += 1
        elif tag == f"[{ns}:VAL_STR]":
            self.i += 1
            self.literal(lit)
        elif tag == "[API:VAL_MASKED_STR]" and ns == "API":
            self.i += 1
            self._masked()
        elif tag == f"[{ns}:VAL_INT]":
            self.i += 1
            if not re.fullmatch(r"-?\d+", self.literal(lit)):
                raise _Reject("VAL_INT literal is not an integer")
        elif tag == f"[{ns}:VAL_FLOAT]":
            self.i += 1
            try:
                float(self.literal(lit))
            except ValueError as exc:
                raise _Reject("VAL_FLOAT literal is not a number") from exc
        elif tag == f"[{ns}:VAL_BOOL]":
            self.i += 1
            if self.literal(lit) not in ("true", "false"):
                raise _Reject("VAL_BOOL literal is not true/false")
        elif tag == f"[{ns}:VAL_NULL]":
            self.i += 1
        else:
            got = self.t[self.i] if self.i < len(self.t) else "<end>"
            raise _Reject(f"index {self.i}: expected a value, found {got!r}")


def well_formed(tokens: list[str], roles: list[str]) -> list[str]:
    """Return a list of grammar violations (empty means the sequence is accepted)."""
    if len(tokens) != len(roles):
        return [f"{len(tokens)} tokens but {len(roles)} roles"]
    problems = []
    for token, role in zip(tokens, roles):
        if role == "control" and token not in CONTROL_TAGS:
            problems.append(f"unknown control tag {token!r}")
    parser = _Parser(tokens, roles)
    try:
        parser.sequence()
        if parser.i != len(tokens):
            problems.append(f"{len(tokens) - parser.i} trailing tokens after SEQ_END")
    except _Reject as exc:
        problems.append(str(exc))
    return problems


# --- reference encoder (restricted feature subset) --------------------------

_SECRET_PAYLOAD_KEYS = frozenset({"password"})
_AUTH_HEADERS = frozenset({"authorization"})


def reference_encode(
    method: str,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    rules: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[list[Any]]]:
    """Encode per PROTOCOL.md section 3. Returns ``(tokens, roles, masked_paths)``.

    Deliberately supports only the feature subset the seeded/designed cases
    use, and **raises** outside it rather than guessing:

    - input already normalized (upper-case method, canonical path, lower-case
      header names, no query string);
    - the only masked locations are top-level payload key ``password``
      (``[MASKED_SECRET]``) and header ``authorization`` (``[MASKED_AUTH]``);
    - string values must not be email/card/IP shaped (checked).
    """
    if method != method.strip().upper():
        raise ValueError("reference_encode expects a normalized method")
    if "?" in path or path != "/" and (not path.startswith("/") or path.endswith("/") or "//" in path):
        raise ValueError("reference_encode expects a canonical path without query")
    t: list[str] = []
    r: list[str] = []
    masked: list[list[Any]] = []

    def c(tag: str) -> None:
        t.append(tag)
        r.append("control")

    def lit(text: str, role: str) -> None:
        t.append(text)
        r.append(role)

    def value(v: Any, ns: str, path_: tuple, top: bool) -> None:
        role = "literal_api" if ns == "API" else "literal_rule"
        if isinstance(v, bool):
            c(f"[{ns}:VAL_BOOL]")
            lit("true" if v else "false", role)
        elif isinstance(v, int):
            c(f"[{ns}:VAL_INT]")
            lit(str(v), role)
        elif isinstance(v, float):
            c(f"[{ns}:VAL_FLOAT]")
            lit(str(v), role)
        elif v is None:
            c(f"[{ns}:VAL_NULL]")
        elif isinstance(v, str):
            if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v) or re.fullmatch(r"(?:\d[ -]?){13,19}", v):
                raise ValueError("reference_encode does not support PII-shaped values")
            if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", v):
                raise ValueError("reference_encode does not support IP-shaped values")
            c(f"[{ns}:VAL_STR]")
            lit(v, role)
        elif isinstance(v, dict):
            c(f"[{ns}:OBJ_START]")
            for k in sorted(v):
                c(f"[{ns}:KEY]")
                lit(k, role)
                if ns == "API" and path_ == ("payload",) and k in _SECRET_PAYLOAD_KEYS:
                    c("[API:VAL_MASKED_STR]")
                    lit("[MASKED_SECRET]", "masked")
                    masked.append(["payload", k])
                else:
                    value(v[k], ns, path_ + (k,), False)
            c(f"[{ns}:OBJ_END]")
        elif isinstance(v, list):
            c(f"[{ns}:LIST_START]")
            for item in v:
                value(item, ns, path_ + (0,), False)
            c(f"[{ns}:LIST_END]")
        else:
            raise ValueError(f"unsupported value type {type(v).__name__}")

    c("[SEQ_START]")
    c("[API:METHOD]")
    lit(method, "literal_api")
    c("[API:PATH_START]")
    for segment in [s for s in path.split("/") if s]:
        c("[API:PATH_SEG]")
        lit(segment, "literal_api")
    c("[API:PATH_END]")
    c("[API:HEADERS_START]")
    for name in sorted(headers):
        if name != name.strip().lower():
            raise ValueError("reference_encode expects lower-case header names")
        c("[API:HEADER_NAME]")
        lit(name, "literal_api")
        if name in _AUTH_HEADERS:
            c("[API:HEADER_VALUE_MASKED]")
            lit("[MASKED_AUTH]", "masked")
            masked.append(["headers", name])
        else:
            c("[API:HEADER_VALUE]")
            lit(headers[name], "literal_api")
    c("[API:HEADERS_END]")
    c("[API:PAYLOAD_START]")
    value(payload, "API", ("payload",), True)
    c("[API:PAYLOAD_END]")
    c("[RULES_START]")
    for rule in rules:
        attrs = {k: v for k, v in rule.items() if k != "rule_id"}
        c("[RULE_START]")
        c("[RULE:ID]")
        lit(rule["rule_id"], "literal_rule")
        c("[RULE:ATTRS_START]")
        value(attrs, "RULE", (), True)
        c("[RULE:ATTRS_END]")
        c("[RULE_END]")
    c("[RULES_END]")
    c("[SEQ_END]")
    return t, r, masked
