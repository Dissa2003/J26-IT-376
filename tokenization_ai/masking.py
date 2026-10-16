"""Privacy-aware masking of sensitive values before tokenization."""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import FusedInput

MASKED_AUTH = "[MASKED_AUTH]"
MASKED_SECRET = "[MASKED_SECRET]"
MASKED_PII = "[MASKED_PII]"
MASKED_IP = "[MASKED_IP]"

# A field location within a masked FusedInput's context, e.g. ("headers",
# "authorization") or ("payload", "items", 0, "ssn"). Strings name a dict
# key, ints name a list index.
FieldPath = tuple[str | int, ...]


@dataclass(frozen=True)
class MaskingResult:
    """A masked ``FusedInput`` together with which field paths were genuinely masked.

    ``masked_paths`` records *locations*, never values -- it is built only
    from the decision of whether this trusted masking stage actually
    substituted a placeholder at that position, so it never contains (and
    cannot leak) the original sensitive data. This is the provenance signal
    that lets downstream stages (the tokenizer, the vocabulary) distinguish
    a genuinely masked value from attacker-controlled text that merely
    reads like one of the placeholder constants above -- see
    ``PrivacyMasker.mask_with_provenance``.
    """

    fused: FusedInput
    masked_paths: frozenset[FieldPath]

# Header names (normalized) that carry authentication material outright.
# These also mask a *payload or query* field of the same name, as
# ``MASKED_AUTH`` (see ``_classify_key``): the same credential is just as
# sensitive in a JSON body as in a header.
_AUTH_HEADER_KEYS = {
    "authorization",
    "proxy_authorization",
    "cookie",
    "set_cookie",
    "x_api_key",
    "x_auth_token",
    "x_csrf_token",
    "x_csrf",
    "x_session_id",
    "x_session",
    "x_auth",
    "x_apikey",
    "apikey",
}

# Value-shaped credentials, detected under any key (see ``_classify_value``).
# Deliberately narrow: only the unambiguous ``Bearer <token>`` form and a
# three-part base64url JWT. ``Basic ...`` / ``Digest ...`` are NOT matched by
# value because ordinary prose ("Basic information ...") would collide.
_BEARER_VALUE = re.compile(r"^Bearer\s+[A-Za-z0-9._~+/=-]{8,}$", re.IGNORECASE)
_JWT_VALUE = re.compile(r"^eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*$")

# Key names are normalized with: strip, camelCase -> snake_case, casefold,
# then "-", "." and whitespace -> "_". ``accessToken``, ``Access-Token`` and
# ``ACCESS_TOKEN`` therefore all become ``access_token``.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SEPARATORS = re.compile(r"[-.\s]")

# Payload/header field names (normalized) that hold credentials or secrets.
# "pwd" is listed as its own exact entry (not a substring/suffix rule) because
# it is WordPress's actual wp-login.php form field name for a password --
# confirmed to otherwise pass through unmasked when real SR-BH 2020 rows
# (sourced from a WordPress honeypot) are run through this pipeline.
_SECRET_KEYS = {
    "access_token",
    "api_key",
    "auth",
    "credential",
    "credentials",
    "password",
    "passwd",
    "pwd",
    "pin",
    "private_key",
    "refresh_token",
    "secret",
    "token",
    "otp",
    # Added after the PP1 privacy audit (D3). Exact names only: broader
    # suffix rules such as "*_key" would also catch sort_key / primary_key.
    "secret_key",
    "secretkey",
    "secret_access_key",
    "access_key",
    "signing_key",
    "encryption_key",
    "apikey",
    "session",
    "session_id",
    "sessionid",
    "jwt",
    "bearer",
    "passphrase",
    "passcode",
    "csrf",
    "xsrf",
}

# Suffixes (after normalization) that mark a credential. ``_token`` and
# ``_secret`` pre-date the audit; the rest were added with it. Each is a
# *whole-word* suffix, so ``token_count`` and ``session_timeout`` are unaffected.
_SECRET_SUFFIXES = (
    "_token",
    "_secret",
    "_password",
    "_passwd",
    "_api_key",
    "_apikey",
    "_secret_key",
    "_private_key",
    "_access_key",
    "_session_id",
    "_sessionid",
)

# Payload/header field names (normalized) that hold PII rather than secrets.
_PII_KEYS = {
    "email",
    "phone",
    "phone_number",
    "mobile",
    "mobile_number",
    "ssn",
    "card",
    "card_number",
    "credit_card",
    "cvv",
    "cvc",
    "iban",
    "account_number",
    "client_ip",
    "ip_address",
    # Added after the PP1 privacy audit (D3): exact names only.
    "national_id",
    "national_identity_number",
    "nic",
    "nic_number",
    "passport",
    "passport_number",
    "tax_id",
    "drivers_license",
    "driving_license",
    "date_of_birth",
    "dob",
    "birth_date",
    "birthdate",
    "address",
    "street_address",
    "home_address",
    "billing_address",
    "shipping_address",
    "mailing_address",
    "postal_address",
    "residential_address",
    "email_address",
    "first_name",
    "last_name",
    "full_name",
    "surname",
}

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CARD = re.compile(r"^(?:\d[ -]?){13,19}$")


class PrivacyMasker:
    """Replace sensitive values with typed placeholders that preserve structure."""

    def mask(self, fused: FusedInput) -> FusedInput:
        """Return a new fused record with the API context's sensitive values masked.

        The verified rule information is carried through unchanged; the original
        ``fused`` object (and its nested context) is never mutated.

        Kept for backward compatibility with existing callers that only need
        the masked record itself. Callers that need to distinguish a
        genuinely masked value from attacker-controlled text that merely
        reads like a placeholder (e.g. the tokenizer) should call
        :meth:`mask_with_provenance` instead -- this method is a thin
        wrapper around it, so there is only one masking implementation.
        """
        return self.mask_with_provenance(fused).fused

    def mask_with_provenance(self, fused: FusedInput) -> MaskingResult:
        """Mask sensitive values and report exactly which field paths were replaced.

        ``masked_paths`` is generated entirely by this trusted masking code
        from its own substitution decisions -- never accepted as input, and
        never derived from the (attacker-controlled) field values
        themselves. It records only *locations* (field names / list
        indices), never the original sensitive values, so it cannot leak
        the data it was built from.
        """
        context = fused.context
        masked_headers, header_paths = self._mask_headers(context.headers)
        masked_payload, payload_paths = self._mask_value(
            context.payload, key_hint=None, path=("payload",)
        )
        masked_paths = set(header_paths) | set(payload_paths)

        masked_path, path_was_masked = self._mask_path(context.path)
        if path_was_masked:
            masked_paths.add(("path",))

        masked_client_ip = context.client_ip
        if context.client_ip:
            masked_client_ip = MASKED_IP
            masked_paths.add(("client_ip",))

        masked_context = context.model_copy(
            update={
                "path": masked_path,
                "headers": masked_headers,
                "payload": masked_payload,
                "client_ip": masked_client_ip,
            }
        )
        masked_fused = FusedInput(context=masked_context, rule_info=fused.rule_info)
        return MaskingResult(fused=masked_fused, masked_paths=frozenset(masked_paths))

    @classmethod
    def _mask_path(cls, path: str) -> tuple[str, bool]:
        """Mask sensitive query-string parameter values in ``path``, if any.

        ``UnifiedApiContext.path`` is a single string that may carry a query
        string (e.g. ``"/reset?token=abc123"``); the tokenizer only ever
        splits it on ``"/"`` and emits each segment as an ordinary literal
        (see ``tokenizer.py``'s ``_path_segments``) -- it has no concept of a
        "masked path segment" the way headers/payload values do. Left
        unhandled, a sensitive query parameter (``?token=``, ``?password=``,
        ``?pwd=``, ...) would reach tokenization in cleartext regardless of
        key name, even one already covered by ``_SECRET_KEYS`` -- confirmed
        before this fix via a focused test, not assumed.

        This masks query-parameter *values* by the same key/value
        classification rules used for headers and payload fields
        (:meth:`_classify_key` then :meth:`_classify_value`), entirely as a
        string rewrite of ``path`` -- it does not introduce a new token role
        or change how the tokenizer processes paths; a masked query value is
        still just an ordinary literal token afterward, the same as any
        other path segment, only with its sensitive content already gone.

        A ``path`` with no ``"?"`` is returned completely unchanged (the
        overwhelming majority of requests), so this can never affect a
        request that has no query string at all.
        """
        if "?" not in path:
            return path, False

        base, _, query = path.partition("?")
        if not query:
            return path, False

        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
        masked_pairs: list[tuple[str, str]] = []
        any_masked = False
        for key, value in pairs:
            placeholder = cls._classify_key(key)
            if placeholder is None and isinstance(value, str):
                placeholder = cls._classify_value(value)
            if placeholder is not None:
                masked_pairs.append((key, placeholder))
                any_masked = True
            else:
                masked_pairs.append((key, value))

        if not any_masked:
            return path, False
        return f"{base}?{urllib.parse.urlencode(masked_pairs)}", True

    @classmethod
    def _mask_headers(cls, headers: Mapping[str, str]) -> tuple[dict[str, str], set[FieldPath]]:
        """Mask authentication, credential, and PII header values by name.

        Returns the masked headers together with the set of header-name
        paths that were actually substituted (as opposed to passed through
        unchanged) -- this is a record of which *rule fired*, not a
        before/after comparison, so a header whose raw value already
        happens to equal its own masked form (e.g. an attacker literally
        sets ``Authorization: [MASKED_AUTH]``) is still correctly counted
        as genuinely masked.
        """
        masked: dict[str, str] = {}
        paths: set[FieldPath] = set()
        for key, value in headers.items():
            new_value, was_masked = cls._mask_header_value(key, value)
            masked[key] = new_value
            if was_masked:
                paths.add(("headers", key))
        return masked, paths

    @classmethod
    def _mask_header_value(cls, key: str, value: str) -> tuple[str, bool]:
        normalized = cls._normalize_key(key)
        if normalized in _AUTH_HEADER_KEYS or normalized.endswith("_token") or normalized.endswith("_key"):
            return MASKED_AUTH, True
        placeholder = cls._classify_key(key)
        if placeholder:
            return placeholder, True
        if isinstance(value, str):
            placeholder = cls._classify_value(value)
            if placeholder:
                return placeholder, True
        return value, False

    @classmethod
    def _mask_value(
        cls, value: Any, key_hint: str | None, path: FieldPath
    ) -> tuple[Any, set[FieldPath]]:
        """Recursively mask mappings, lists, and sensitive scalar values.

        Returns the masked value together with the set of paths that were
        actually substituted, mirroring :meth:`_mask_header_value`'s
        rule-fired semantics (not a before/after comparison).
        """
        if key_hint:
            placeholder = cls._classify_key(key_hint)
            if placeholder:
                return placeholder, {path}
        if isinstance(value, Mapping):
            masked: dict[str, Any] = {}
            paths: set[FieldPath] = set()
            for key, item in value.items():
                child_value, child_paths = cls._mask_value(item, str(key), path + (str(key),))
                masked[str(key)] = child_value
                paths |= child_paths
            return masked, paths
        if isinstance(value, (list, tuple)):
            items: list[Any] = []
            paths = set()
            for index, item in enumerate(value):
                child_value, child_paths = cls._mask_value(item, key_hint, path + (index,))
                items.append(child_value)
                paths |= child_paths
            return (type(value)(items) if isinstance(value, tuple) else items), paths
        if isinstance(value, str):
            placeholder = cls._classify_value(value)
            if placeholder:
                return placeholder, {path}
        return value, set()

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize a field name for case-, separator- and camelCase-insensitive matching.

        Surrounding whitespace is stripped first: masking runs *before*
        normalization, which later trims header names, so an un-stripped
        ``" Authorization "`` would otherwise escape masking and then be
        tokenized as ``authorization`` with its raw value.
        """
        spaced = _CAMEL_BOUNDARY.sub("_", key.strip())
        return _KEY_SEPARATORS.sub("_", spaced.casefold())

    @classmethod
    def _classify_key(cls, key: str) -> str | None:
        """Return the placeholder for a field name known to be sensitive, if any.

        Authentication-header names (``authorization``, ``x-api-key``,
        ``x-session-id``, ...) are labelled ``MASKED_AUTH`` wherever they
        appear. Names that end in ``_token`` keep their pre-audit label
        (``MASKED_SECRET``) so existing behaviour is unchanged. Then secret
        keys, then PII keys.
        """
        normalized = cls._normalize_key(key)
        if normalized in _AUTH_HEADER_KEYS and not normalized.endswith("_token"):
            return MASKED_AUTH
        if normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES):
            return MASKED_SECRET
        if normalized in _PII_KEYS or normalized.endswith("_email") or normalized.endswith("_phone"):
            return MASKED_PII
        return None

    @staticmethod
    def _classify_value(value: str) -> str | None:
        """Detect email, payment-card, bearer/JWT and IP-address string values.

        Phone numbers are masked by field name only (see ``_PII_KEYS``): their
        digit formatting is too close to dates, ids, and other benign fields
        to classify reliably from the value alone.
        """
        if _EMAIL.fullmatch(value) or _CARD.fullmatch(value):
            return MASKED_PII
        if _BEARER_VALUE.fullmatch(value) or _JWT_VALUE.fullmatch(value):
            return MASKED_AUTH
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return None
        return MASKED_IP
