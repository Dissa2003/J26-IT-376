"""Token-to-id vocabulary used for embedding lookup.

Design summary (Phase 7)
-------------------------
A tokenized sequence (see ``tokenizer.py``) mixes two kinds of tokens:

* **Control tokens** -- the tokenizer's fixed, finite tag vocabulary (e.g.
  ``[API:KEY]``, ``[RULE:VAL_STR]``, ``[API:OBJ_START]``) plus the privacy
  masking placeholders (e.g. ``[MASKED_PII]``). This set is known in advance
  and never grows, so every one of these tokens is permanently reserved an
  id and is **never** mapped to ``UNK_TOKEN`` -- they carry the structural
  and privacy signal the downstream model needs, so collapsing them to
  ``UNK`` would destroy exactly the information this pipeline exists to
  preserve.
* **Literal values** -- everything else (method strings, path segments,
  header names/values, payload field names/values, rule ids, rule attribute
  names/values). This is an open vocabulary: unbounded in principle, so it
  is learned from a training corpus via :meth:`Vocabulary.fit` and frozen,
  with unseen or rare values mapped to ``UNK_TOKEN`` at encode time.

Authoritative roles, not inferred ones (security): a token's role --
control tag, API literal, rule literal, or genuinely masked value -- is
read directly from ``TokenSequence.roles``, which the tokenizer populates
token-by-token as it emits each one (see ``tokenizer.TOKEN_ROLE_*``). This
``Vocabulary`` never infers a token's role from its text or from a
neighboring token's text. That distinction matters: an earlier version of
this module decided "is this token a control tag" by checking whether the
*preceding* token's text matched a known tag string. An attacker who
controls a payload key (JSON object keys have no character restrictions)
could set that key to a tag string like ``"[API:VAL_STR]"``, which caused
the *genuine* control tag immediately following it (marking that same
field's own value type) to be misclassified as an ordinary literal --
corrupting a legitimate tag's identity, not just an attacker value's.
Reading the role directly from ``roles`` eliminates this entire class of
bug: a token's role can never be changed by what text appears anywhere
else in the sequence, because it was never derived from that text in the
first place -- it is a fact the tokenizer recorded when it decided what
to emit.

Namespace separation: ``TOKEN_ROLE_LITERAL_API``/``TOKEN_ROLE_LITERAL_RULE``
route a literal into independent frequency tables and independent
``max_literal_vocab_size`` budgets and id ranges, so the same literal
string (e.g. ``"field"``) occurring in both namespaces is assigned two
distinct ids -- the namespace is therefore encoded directly in the id, not
only recoverable from context.

Fitting/freezing policy: :meth:`fit` must be called exactly once per
training corpus and only on the training split. It resets any
previously-learned literal vocabulary (so a second call never blends two
corpora's statistics) and then permanently assigns ids to literal values
observed at least ``min_frequency`` times, most-frequent first, up to
``max_literal_vocab_size`` per namespace (ties broken alphabetically for
reproducibility). :meth:`encode` is deliberately refused before ``fit`` has
run, and never adds new ids -- this is what prevents train/test leakage:
nothing seen only at evaluation time can ever influence the vocabulary.

Masking-placeholder impersonation (security): a literal value that
legitimately occurs in a value position can still be textually identical
to a masking placeholder (e.g. a user-supplied payload value of literally
``"[MASKED_PII]"`` that ``masking.py`` never actually classified as PII).
This is resolved the same authoritative way: the tokenizer only assigns a
token ``TOKEN_ROLE_MASKED`` when its caller's ``masked_paths`` (ultimately
produced by ``PrivacyMasker.mask_with_provenance``, which decides this
from its own substitution logic, never from the value's text) says this
exact field was genuinely masked. An attacker can control a field's text
but not the role the trusted tokenizer assigns to it, so this cannot be
forged from the API payload alone.

Compatibility note: :meth:`fit` and :meth:`encode` take ``TokenSequence``
objects (not bare ``list[str]``), because a token's role must always
travel with its text as one unit -- passing them as two separate
parallel lists would let a caller (accidentally) desynchronize them. This
is an intentional breaking change from the pre-role-metadata version of
this module; see the class docstring.
"""

from __future__ import annotations

from collections.abc import Iterable

from .masking import MASKED_AUTH, MASKED_IP, MASKED_PII, MASKED_SECRET
from .models import TokenSequence
from .tokenizer import (
    API_HEADER_NAME,
    API_HEADER_VALUE,
    API_HEADER_VALUE_MASKED,
    API_HEADERS_END,
    API_HEADERS_START,
    API_KEY,
    API_LIST_END,
    API_LIST_START,
    API_METHOD,
    API_OBJ_END,
    API_OBJ_START,
    API_PATH_END,
    API_PATH_SEG,
    API_PATH_START,
    API_PAYLOAD_END,
    API_PAYLOAD_START,
    API_VAL_BOOL,
    API_VAL_FLOAT,
    API_VAL_INT,
    API_VAL_MASKED_STR,
    API_VAL_NULL,
    API_VAL_STR,
    RULE_ATTRS_END,
    RULE_ATTRS_START,
    RULE_END,
    RULE_ID,
    RULE_KEY,
    RULE_LIST_END,
    RULE_LIST_START,
    RULE_OBJ_END,
    RULE_OBJ_START,
    RULE_START,
    RULE_VAL_BOOL,
    RULE_VAL_FLOAT,
    RULE_VAL_INT,
    RULE_VAL_NULL,
    RULE_VAL_STR,
    RULES_END,
    RULES_START,
    SEQ_END,
    SEQ_START,
    TOKEN_ROLE_CONTROL,
    TOKEN_ROLE_LITERAL_API,
    TOKEN_ROLE_LITERAL_RULE,
    TOKEN_ROLE_MASKED,
)

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

_API_NAMESPACE = "api"
_RULE_NAMESPACE = "rule"

# The tokenizer's complete, fixed set of structural/control tags. Sorted so
# reserved-id assignment does not depend on set/dict iteration order.
_CONTROL_TOKENS: tuple[str, ...] = tuple(
    sorted(
        {
            SEQ_START,
            SEQ_END,
            API_METHOD,
            API_PATH_START,
            API_PATH_SEG,
            API_PATH_END,
            API_HEADERS_START,
            API_HEADER_NAME,
            API_HEADER_VALUE,
            API_HEADER_VALUE_MASKED,
            API_HEADERS_END,
            API_PAYLOAD_START,
            API_PAYLOAD_END,
            API_KEY,
            API_OBJ_START,
            API_OBJ_END,
            API_LIST_START,
            API_LIST_END,
            API_VAL_STR,
            API_VAL_MASKED_STR,
            API_VAL_INT,
            API_VAL_FLOAT,
            API_VAL_BOOL,
            API_VAL_NULL,
            RULES_START,
            RULES_END,
            RULE_START,
            RULE_END,
            RULE_ID,
            RULE_ATTRS_START,
            RULE_ATTRS_END,
            RULE_KEY,
            RULE_OBJ_START,
            RULE_OBJ_END,
            RULE_LIST_START,
            RULE_LIST_END,
            RULE_VAL_STR,
            RULE_VAL_INT,
            RULE_VAL_FLOAT,
            RULE_VAL_BOOL,
            RULE_VAL_NULL,
        }
    )
)

# The masking stage's closed set of privacy placeholders. Reserved for the
# same reason as control tags: whether a value was masked, and as what kind
# of sensitive data, is itself a signal the anomaly model should always see.
_MASK_PLACEHOLDER_TOKENS: tuple[str, ...] = tuple(
    sorted({MASKED_AUTH, MASKED_SECRET, MASKED_PII, MASKED_IP})
)


class Vocabulary:
    """Bidirectional mapping between tokens and integer ids.

    See the module docstring for the full fitting/freezing and namespace
    policy. In short: construct, call :meth:`fit` once on a training
    corpus of ``TokenSequence`` objects, then call
    :meth:`encode`/:meth:`decode`.
    """

    def __init__(self, max_literal_vocab_size: int | None = None, min_frequency: int = 1) -> None:
        """Create an unfitted vocabulary with only the reserved tokens assigned.

        ``max_literal_vocab_size`` caps how many distinct literal values
        each namespace (API, rule) may keep, independently; ``None`` means
        unbounded. ``min_frequency`` drops literal values seen fewer than
        this many times in the fitting corpus. Both apply only to literal
        values -- reserved tokens are always kept regardless of either
        setting.
        """
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        if max_literal_vocab_size is not None and max_literal_vocab_size < 0:
            raise ValueError("max_literal_vocab_size must not be negative")

        self.max_literal_vocab_size = max_literal_vocab_size
        self.min_frequency = min_frequency

        # Keyed by (namespace, token): reserved tokens use namespace ``None``
        # (they are global), literals use ``"api"``/``"rule"``. This is what
        # lets the same literal string (e.g. a field named "field" in both a
        # payload and a rule's attributes) hold two distinct ids -- a plain
        # ``dict[str, int]`` would collapse them into one slot.
        self._token_to_id: dict[tuple[str | None, str], int] = {}
        self._id_to_token: dict[int, str] = {}
        self._fitted = False
        self._reset_to_reserved()

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called on this instance."""
        return self._fitted

    @property
    def pad_id(self) -> int:
        return self._token_to_id[(None, PAD_TOKEN)]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[(None, UNK_TOKEN)]

    def __len__(self) -> int:
        return len(self._token_to_id)

    def fit(self, corpus: Iterable[TokenSequence]) -> None:
        """Learn and freeze the literal-value vocabulary from a training corpus.

        ``corpus`` is an iterable of ``TokenSequence`` objects (e.g. the
        output of ``HybridTokenizer.tokenize``). Call this only on the
        training split: every id assigned here is permanent for this
        instance, and any literal value not observed here (or observed
        fewer than ``min_frequency`` times, or beyond
        ``max_literal_vocab_size`` for its namespace) maps to ``UNK_TOKEN``
        at encode time forever after -- including during evaluation. This
        is what prevents test-time values from leaking into the vocabulary.

        Calling ``fit`` again discards any previously-learned literal
        vocabulary and starts over from the reserved tokens only.

        Raises ``ValueError`` if a sequence's ``tokens`` and ``roles``
        differ in length, if a role is not one of the four recognized
        ``TOKEN_ROLE_*`` values, or if a token with role
        ``TOKEN_ROLE_CONTROL``/``TOKEN_ROLE_MASKED`` is not one of this
        vocabulary's pre-registered reserved tokens -- each indicates a
        malformed or foreign ``TokenSequence`` in the corpus, and fitting
        on it silently would risk hiding a real data-quality bug.
        """
        self._reset_to_reserved()
        api_counts: dict[str, int] = {}
        rule_counts: dict[str, int] = {}

        for sequence in corpus:
            for token, role in self._role_pairs(sequence):
                if role in (TOKEN_ROLE_CONTROL, TOKEN_ROLE_MASKED):
                    if (None, token) not in self._token_to_id:
                        raise ValueError(
                            f"token {token!r} has role {role!r} but is not a "
                            "recognized reserved token"
                        )
                elif role == TOKEN_ROLE_LITERAL_API:
                    api_counts[token] = api_counts.get(token, 0) + 1
                elif role == TOKEN_ROLE_LITERAL_RULE:
                    rule_counts[token] = rule_counts.get(token, 0) + 1
                else:
                    raise ValueError(f"unrecognized token role {role!r} for token {token!r}")

        self._assign_literal_ids(api_counts, _API_NAMESPACE)
        self._assign_literal_ids(rule_counts, _RULE_NAMESPACE)
        self._fitted = True

    def encode(self, sequence: TokenSequence) -> list[int]:
        """Map ``sequence.tokens`` to ids, using the unknown id for unseen literals.

        Raises ``RuntimeError`` if :meth:`fit` has not been called yet --
        encoding before fitting would otherwise silently map every literal
        value to ``UNK_TOKEN``, which almost always indicates a wiring bug
        rather than an intended empty vocabulary.

        Each token's role is taken directly from ``sequence.roles`` --
        never inferred from neighboring tokens -- so an attacker-controlled
        literal can only ever affect its own id, never a different,
        genuine control tag's (see the module docstring).

        Raises ``ValueError`` for an unrecognized role (not one of the
        four ``TOKEN_ROLE_*`` values), consistently with :meth:`fit`. This
        is a different case from an unseen *literal value* with a valid
        role, which maps to ``UNK_TOKEN`` as normal: an invalid role means
        the ``TokenSequence`` itself is malformed or foreign, which should
        fail loudly rather than be silently absorbed as an ordinary
        unknown-token event.
        """
        if not self._fitted:
            raise RuntimeError(
                "Vocabulary.encode() was called before fit(); call fit() on the "
                "training corpus first"
            )
        unk_id = self.unk_id
        ids: list[int] = []
        for token, role in self._role_pairs(sequence):
            if role in (TOKEN_ROLE_CONTROL, TOKEN_ROLE_MASKED):
                ids.append(self._token_to_id.get((None, token), unk_id))
            elif role == TOKEN_ROLE_LITERAL_API:
                ids.append(self._token_to_id.get((_API_NAMESPACE, token), unk_id))
            elif role == TOKEN_ROLE_LITERAL_RULE:
                ids.append(self._token_to_id.get((_RULE_NAMESPACE, token), unk_id))
            else:
                raise ValueError(f"unrecognized token role {role!r} for token {token!r}")
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        """Map ids back to their tokens.

        Raises ``KeyError`` for an id outside this vocabulary. Unlike an
        unseen literal value (an expected runtime event, mapped to
        ``UNK_TOKEN`` by ``encode``), an out-of-range id indicates a caller
        bug -- e.g. ids produced by a different ``Vocabulary`` instance or
        configuration.
        """
        return [self._id_to_token[token_id] for token_id in ids]

    @staticmethod
    def _role_pairs(sequence: TokenSequence) -> Iterable[tuple[str, str]]:
        """Zip a sequence's tokens with their roles, after validating lengths match."""
        if len(sequence.tokens) != len(sequence.roles):
            raise ValueError(
                f"TokenSequence for event {sequence.event_id!r} has "
                f"{len(sequence.tokens)} tokens but {len(sequence.roles)} roles"
            )
        return zip(sequence.tokens, sequence.roles)

    def _reset_to_reserved(self) -> None:
        self._token_to_id = {}
        self._id_to_token = {}
        for token in (PAD_TOKEN, UNK_TOKEN, *_CONTROL_TOKENS, *_MASK_PLACEHOLDER_TOKENS):
            self._add(None, token)

    def _assign_literal_ids(self, counts: dict[str, int], namespace: str) -> None:
        eligible = [token for token, count in counts.items() if count >= self.min_frequency]
        eligible.sort(key=lambda token: (-counts[token], token))
        if self.max_literal_vocab_size is not None:
            eligible = eligible[: self.max_literal_vocab_size]
        for token in eligible:
            self._add(namespace, token)

    def _add(self, namespace: str | None, token: str) -> None:
        token_id = len(self._token_to_id)
        self._token_to_id[(namespace, token)] = token_id
        self._id_to_token[token_id] = token
