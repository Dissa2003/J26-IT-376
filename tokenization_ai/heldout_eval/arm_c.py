"""Arm C encoder ``armc-char3-v1``: conventional character 3-gram preprocessing for the held-out comparison.

Evaluation infrastructure only (draft protocol heldout-eval-1.2.0-draft, section 12.10). It trains and scores nothing and creates no
model. It only turns an already privacy-masked and normalized request into a ``ModelInput`` that the unchanged
``LightweightAnomalyModel`` can consume.

Input contract (privacy boundary)
---------------------------------
``encode_request`` accepts only a ``NormalizationResult``: the output of ``PrivacyMasker.mask_with_provenance`` followed by
``Normalizer.normalize_with_provenance``. It does not accept an ``EvalRecord``, a raw ``UnifiedApiContext`` or any label or metadata.
The existing stages offer no tamper-proof stage proof (``NormalizationResult`` is a plain frozen dataclass and no flag is trusted), so
the boundary is content-based: the encoder re-runs the *existing* ``PrivacyMasker`` and ``Normalizer`` on the supplied record and
refuses it if either would still change anything (a raw credential, token, e-mail or card number is therefore refused, because the
masker would replace it). It also refuses a record that carries simulated or verified rules, because Arm C is request-only.
Limit of this check: it shows that the existing masker has nothing left to mask, which is the masker's own definition of "masked"; it
cannot show that the stages were actually executed, and it does not protect values the masker does not classify as sensitive.
Reimplementing masking here is deliberately avoided.

Serialization (``armc-char3-v1``; one string, fully deterministic)
------------------------------------------------------------------
::

    <METHOD> <path-with-query>\\n
    <header-name>: <header-value>\\n        one line per header, sorted by header name (code-point order)
    \\n                                      explicit header/body separator
    <canonical JSON of the payload>        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

The path keeps its query string exactly as the production mapping supplies it: nothing is split and nothing is percent-decoded.
Raw CR and LF characters inside the method, path, header names or header values are replaced by the two-character sequences ``\\r`` and
``\\n`` so that content can never forge a line or the separator. The payload JSON escapes all non-ASCII characters. The whole string is then
case-folded with **ASCII-only** lowercasing (A-Z to a-z; other characters are left unchanged), which is independent of the Unicode tables
of a given Python build. No rule tokens, labels, family names, record ids, timestamps or source ids enter the string (the event id is
carried only as ``ModelInput.event_id``).

Character 3-grams and hashing
-----------------------------
The n-grams are all contiguous, overlapping windows of 3 code points of the serialized string, in order: ``len(text) - 2`` of them
(none for text shorter than 3 characters). Each is encoded as UTF-8 and hashed with ``hashlib.blake2b(digest_size=8)`` (no key, salt or
personalization); the 8 digest bytes are read as a big-endian unsigned integer and reduced to its low 31 bits (``& 0x7FFFFFFF``), giving
a non-negative id below 2**31. Python's built-in ``hash()`` is never used. There is no vocabulary, no unknown token, no padding and no
truncation: ``attention_mask`` is 1 for every id.

How the unchanged model consumes this
-------------------------------------
``LightweightAnomalyModel`` counts, over the valid ids, one hashed *unigram* per id and one hashed *bigram* per adjacent pair, all in
one shared ``num_buckets`` space. Here a unigram is one character 3-gram and a bigram is a pair of adjacent (overlapping) 3-grams,
which together span 4 characters. Feature mass is therefore ``2n - 1`` for ``n`` 3-grams, so the model sees roughly 3-4 times more
feature positions per request than for Arms A and B (whose positions are structural tokens), and the same ``num_buckets`` does not mean
the same information or the same collision rate. For a power-of-two ``num_buckets`` the unigram bucket depends on the low bits of the id,
which the BLAKE2b reduction mixes well. The model keeps its zero-initialised weights until ``fit`` is called (it warm-starts from existing
weights, so each future hyperparameter setting needs a fresh instance); this module never imports or builds a model.

Interpretation limits
---------------------
Arm C sees the characters of query strings and of values that Arms A and B treat as atomic tokens (and map to the unknown token when
unseen), so it is a *conventional preprocessing baseline*, not an isolated test of structure-awareness. Differences between arms
cannot be attributed to structure alone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from ..masking import PrivacyMasker
from ..models import ModelInput
from ..normalization import NormalizationResult, Normalizer

ENCODER_ID = "armc-char3-v1"
NGRAM = 3
HASH_DIGEST_SIZE = 8
ID_MASK = 0x7FFFFFFF  # ids are in [0, 2**31)

_LOWER = {c: c + 32 for c in range(ord("A"), ord("Z") + 1)}


class ArmCEncodingError(ValueError):
    """The input violates the Arm C contract. Messages never echo request content."""


def _line_safe(text: str) -> str:
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _ascii_lower(text: str) -> str:
    return text.translate(_LOWER)


def _check_contract(result: object) -> None:
    if not isinstance(result, NormalizationResult):
        raise ArmCEncodingError("Arm C accepts only the NormalizationResult of the existing masking and normalization stages")
    fused = result.fused
    if fused.rule_info.rules:
        raise ArmCEncodingError("Arm C is request-only: the record carries rules")
    if PrivacyMasker().mask_with_provenance(fused).fused != fused:
        raise ArmCEncodingError("the record is not privacy-masked: the existing masking stage would still change it")
    if Normalizer().normalize(fused) != fused:
        raise ArmCEncodingError("the record is not normalized: the existing normalization stage would still change it")


def serialize_masked_request(result: NormalizationResult) -> str:
    """The deterministic, lower-cased text form of a masked and normalized request (see module docstring)."""
    _check_contract(result)
    ctx = result.fused.context
    lines = [f"{_line_safe(ctx.method)} {_line_safe(ctx.path)}"]
    for name in sorted(ctx.headers):
        lines.append(f"{_line_safe(str(name))}: {_line_safe(str(ctx.headers[name]))}")
    try:
        body = json.dumps(ctx.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ArmCEncodingError("the payload cannot be serialized to canonical JSON") from exc
    return _ascii_lower("\n".join(lines) + "\n\n" + body)


def char_ngram_ids(text: str, n: int = NGRAM) -> list[int]:
    """Hashed ids of all overlapping character ``n``-grams of ``text`` in order (empty if ``len(text) < n``)."""
    if not isinstance(text, str):
        raise ArmCEncodingError("text must be a string")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ArmCEncodingError("n must be a positive integer")
    return [
        int.from_bytes(hashlib.blake2b(text[i:i + n].encode("utf-8"), digest_size=HASH_DIGEST_SIZE).digest(), "big") & ID_MASK
        for i in range(len(text) - n + 1)
    ]


def encode_request(result: NormalizationResult) -> ModelInput:
    """Encode one masked and normalized request as a ``ModelInput`` of hashed character 3-gram ids."""
    ids = char_ngram_ids(serialize_masked_request(result))
    if not ids:  # cannot happen for a real request (method, path and the JSON body alone exceed 3 characters)
        raise ArmCEncodingError("the serialized request is too short to form any character 3-gram")
    return ModelInput(event_id=result.fused.context.event_id, input_ids=ids, attention_mask=[1] * len(ids))


def encoder_parameters() -> Mapping[str, object]:
    """The encoder's fixed configuration, for the pre-run manifest."""
    return {"encoder_id": ENCODER_ID, "ngram": NGRAM, "hash": "blake2b", "digest_size": HASH_DIGEST_SIZE,
            "id_mask": ID_MASK, "id_range": [0, ID_MASK], "case_fold": "ascii_only", "percent_decoding": False,
            "padding": False, "truncation": False, "rules": False}
