from __future__ import annotations

import re
import uuid

from .models import (
    PolicyCondition,
    PolicyEffect,
    PolicyInterpretation,
    SecurityPolicy,
)


class UnsupportedPolicyError(ValueError):
    """Raised when text is outside the intentionally limited policy grammar."""


_ONLY = re.compile(
    r"^only (?P<subject>[\w -]+) can (?P<action>\w+) (?P<resource>[\w -]+)[.]?$",
    re.IGNORECASE,
)
_CAN = re.compile(
    r"^(?P<subject>[\w -]+) can (?P<action>\w+) (?P<resource>[\w -]+)[.]?$",
    re.IGNORECASE,
)
_CANNOT = re.compile(
    r"^(?P<subject>[\w -]+) cannot (?P<action>\w+) (?P<resource>[\w -]+)[.]?$",
    re.IGNORECASE,
)


def _canonical(value: str) -> str:
    return "".join(part.capitalize() for part in value.strip().split())


class PolicyParser:
    """Parser for three explicit forms; it never guesses unsupported text."""

    def parse_text(self, text: str, policy_id: str | None = None) -> SecurityPolicy:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise UnsupportedPolicyError("Policy text must not be empty.")
        if re.search(r"\b(with|level|if|when|where|whose)\b", cleaned, re.IGNORECASE):
            raise UnsupportedPolicyError(
                "Attribute conditions are outside the supported policy grammar."
            )
        match = _ONLY.match(cleaned)
        interpretation = PolicyInterpretation.ONLY_ROLE
        effect = PolicyEffect.ALLOW
        if not match:
            match = _CANNOT.match(cleaned)
            interpretation = PolicyInterpretation.ROLE_PROHIBITION
            effect = PolicyEffect.DENY
        if not match:
            match = _CAN.match(cleaned)
            interpretation = PolicyInterpretation.ROLE_PERMISSION
            effect = PolicyEffect.ALLOW
        if not match:
            raise UnsupportedPolicyError(
                "Unsupported policy. Use '<role> can <action> <resource>', "
                "'only <role> can <action> <resource>', or "
                "'<role> cannot <action> <resource>'."
            )
        values = match.groupdict()
        return SecurityPolicy(
            policy_id=policy_id or str(uuid.uuid4())[:8],
            subject=_canonical(values["subject"]),
            action=_canonical(values["action"]),
            resource=_canonical(values["resource"]),
            effect=effect,
            raw_text=text,
            interpretation=interpretation,
        )
