"""Lightweight AI model that scores prepared inputs for anomalies.

Architecture
------------
A hashed unigram + bigram feature vector, fed into a binary logistic
regression (linear layer + sigmoid). Bigrams of *consecutive, valid*
token ids (``attention_mask == 1`` only -- padding never contributes)
are what let this otherwise bag-of-tokens model retain some local
token-order/structural sensitivity: this tokenizer's grammar packs
structural meaning into adjacent tokens (e.g. ``API_KEY, <key>,
API_VAL_STR, <value>``), so an adjacent-pair feature directly captures
"this tag was followed by this kind of value/tag" -- the kind of local
structure Member 4's hybrid tokenization is designed to preserve.

This is deliberately the smallest architecture that is still genuinely
order-sensitive and genuinely trainable (a real gradient computed from a
real loss, not a heuristic): no embedding table, no convolution, no
recurrence. A small 1D CNN or a recurrent layer would add real
expressive power at real implementation and inference-cost expense; see
the Phase 9 planning notes for why this is the recommended starting
point rather than those alternatives.

Deterministic hashing (not Python's randomized ``hash()``)
------------------------------------------------------------
Each unigram id and each consecutive id pair is mapped into one of
``num_buckets`` feature slots by explicit integer arithmetic (multiply by
a fixed odd constant, modulo ``num_buckets``), never by ``hash()`` --
whose per-type randomization (for strings; not for plain ints, but this
code never relies on that distinction) would make feature extraction,
and therefore training and inference, non-reproducible across processes.

Known limitation (hash collisions): unigrams and bigrams share the same
``num_buckets``-sized feature space, and both control tags and literal
values flow through the same id-based hashing with no special casing.
Two different ids/pairs can land in the same bucket by construction --
this is the standard, well-understood trade-off of the "hashing trick"
(as used in e.g. fastText's bag-of-n-grams classifier), traded for a
small, fixed-size feature vector regardless of vocabulary size. It is
not corrected here (e.g. via a larger table or collision-aware hashing)
because doing so is an accuracy/capacity tuning decision that has no
meaning without real data to tune against.

Untrained vs. trained state
----------------------------
A freshly constructed model has ``is_trained is False`` and zero-valued
weights. :meth:`predict` raises ``RuntimeError`` in this state rather
than returning an arbitrary score -- an untrained model has no anomaly
signal to report, and returning e.g. a constant 0.5 could be mistaken
for a real (if uncertain) prediction. Only :meth:`fit` can set
``is_trained = True``, and only by actually running gradient descent.
``is_trained`` reflects *only* that :meth:`fit` completed; it is not and
must never be read as "validated on real-world security data" -- see
the module-level and ``fit`` docstrings for the synthetic-fixture caveat.
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only when numpy is absent
    raise ImportError(
        "tokenization_ai.model requires numpy, which is a Member-4-scoped "
        "dependency kept out of the project's root requirements.txt. "
        "Install it with: pip install -r tokenization_ai/requirements-ml.txt"
    ) from exc

from .models import ModelInput

# Fixed, arbitrary odd multiplicative constants used only to disperse ids
# across buckets. Not tuned against any data -- see the module docstring's
# "Known limitation" note. ``_UNIGRAM_PRIME`` is Knuth's classic
# multiplicative hash constant; ``_BIGRAM_PRIME_B`` is a different odd
# constant so a bigram's two positions don't hash identically.
_UNIGRAM_PRIME = 2654435761
_BIGRAM_PRIME_A = 2654435761
_BIGRAM_PRIME_B = 40503


class LightweightAnomalyModel:
    """Compact, hashed-n-gram logistic regression anomaly scorer.

    See the module docstring for the architecture, hashing, and
    untrained/trained-state design. Construct, call :meth:`fit` once on
    labeled training data, then call :meth:`predict`.
    """

    def __init__(self, num_buckets: int = 1024) -> None:
        """Create an untrained model with ``num_buckets`` hashed feature slots.

        ``num_buckets`` (default 1024) is an initial, configurable size
        choice -- not a value tuned against any dataset, since none
        exists yet. Weights start at exactly zero (not random noise):
        logistic regression's loss is convex, so gradient descent
        converges to the same optimum regardless of starting point, and
        zero gives the simplest, fully deterministic starting state.
        """
        if num_buckets < 1:
            raise ValueError("num_buckets must be at least 1")
        self.num_buckets = num_buckets
        self.weights = np.zeros(num_buckets, dtype=np.float64)
        self.bias = 0.0
        self.is_trained = False
        self.training_loss_history: list[float] = []

    @property
    def model_version(self) -> str:
        """A version string that always makes the trained/untrained state visible."""
        state = "trained" if self.is_trained else "untrained"
        return f"ngram-linear-v1-{state}"

    def predict(self, model_input: ModelInput) -> float:
        """Return a bounded anomaly score in ``[0, 1]`` for ``model_input``.

        Raises ``RuntimeError`` if :meth:`fit` has not been called yet --
        an untrained model (all-zero weights) has no validated anomaly
        signal, and must never be reported as an ordinary prediction.
        Raises ``ValueError`` for a malformed ``model_input`` (see
        :meth:`_validate_model_input`). Never mutates ``model_input`` or
        this model's own state.
        """
        if not self.is_trained:
            raise RuntimeError(
                "LightweightAnomalyModel.predict() was called before fit(); "
                "an untrained model has no validated anomaly signal to report"
            )
        self._validate_model_input(model_input)
        features = self._features(model_input.input_ids, model_input.attention_mask)
        logit = float(np.dot(features, self.weights) + self.bias)
        return float(self._sigmoid(logit))

    def fit(
        self,
        inputs: list[ModelInput],
        labels: list[int],
        *,
        epochs: int,
        learning_rate: float = 0.1,
    ) -> list[float]:
        """Train by full-batch gradient descent on binary cross-entropy loss.

        ``inputs``/``labels`` must be the same, non-empty length, with
        every label exactly ``0`` or ``1``. Sets ``is_trained = True``
        and returns the per-epoch loss history (also stored as
        ``self.training_loss_history``) only after every epoch has run
        without error.

        Deterministic: full-batch gradient descent over a fixed feature
        matrix has no random element (no shuffling, no stochastic
        sampling), so identical ``inputs``/``labels``/``epochs``/
        ``learning_rate`` always produce identical final weights and an
        identical loss history.

        This method verifies the training *mechanism* is correct (the
        gradient genuinely reduces loss on whatever data it's given). It
        does not and cannot validate real-world anomaly-detection
        ability -- that requires a real labeled dataset, which must be
        supplied by the caller. Training on a small synthetic fixture
        (as this module's own tests do) verifies the code, not detection
        quality against real attacks; never report such a fixture's
        results as real-world performance.
        """
        if not inputs or not labels:
            raise ValueError("training data must be non-empty")
        if len(inputs) != len(labels):
            raise ValueError(
                f"inputs and labels must have the same length (got {len(inputs)} "
                f"and {len(labels)})"
            )
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for label in labels:
            if label not in (0, 1):
                raise ValueError(f"labels must be binary (0 or 1), got {label!r}")
        for model_input in inputs:
            self._validate_model_input(model_input)

        feature_matrix = np.vstack(
            [self._features(mi.input_ids, mi.attention_mask) for mi in inputs]
        )
        targets = np.asarray(labels, dtype=np.float64)
        sample_count = len(inputs)

        weights = self.weights.copy()
        bias = self.bias
        loss_history: list[float] = []

        for _ in range(epochs):
            logits = feature_matrix @ weights + bias
            loss_history.append(self._bce_with_logits(logits, targets))

            predictions = self._sigmoid(logits)
            gradient_logits = predictions - targets
            weight_gradient = feature_matrix.T @ gradient_logits / sample_count
            bias_gradient = float(np.mean(gradient_logits))

            weights -= learning_rate * weight_gradient
            bias -= learning_rate * bias_gradient

        self.weights = weights
        self.bias = bias
        self.training_loss_history = loss_history
        self.is_trained = True
        return loss_history

    def _features(self, input_ids: list[int], attention_mask: list[int]) -> np.ndarray:
        """Hashed unigram+bigram feature vector, built only from non-padding tokens."""
        valid_ids = [token_id for token_id, mask in zip(input_ids, attention_mask) if mask == 1]
        features = np.zeros(self.num_buckets, dtype=np.float64)
        for token_id in valid_ids:
            features[self._unigram_bucket(token_id)] += 1.0
        for first_id, second_id in zip(valid_ids, valid_ids[1:]):
            features[self._bigram_bucket(first_id, second_id)] += 1.0
        return features

    def _unigram_bucket(self, token_id: int) -> int:
        return (token_id * _UNIGRAM_PRIME) % self.num_buckets

    def _bigram_bucket(self, first_id: int, second_id: int) -> int:
        return (first_id * _BIGRAM_PRIME_A + second_id * _BIGRAM_PRIME_B) % self.num_buckets

    @staticmethod
    def _sigmoid(logits: np.ndarray | float) -> np.ndarray | float:
        """Numerically stable sigmoid, safe for arbitrarily large |logits|."""
        positive = logits >= 0
        negative_exp = np.exp(-np.abs(logits))
        return np.where(positive, 1.0 / (1.0 + negative_exp), negative_exp / (1.0 + negative_exp))

    @staticmethod
    def _bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> float:
        """Numerically stable binary cross-entropy computed directly from logits.

        Equivalent to ``-mean(y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z)))``
        but avoids ever computing ``log`` of a value that could underflow
        to exactly 0 or overflow ``exp``, via the standard
        ``max(z,0) - z*y + log(1+exp(-|z|))`` identity.
        """
        return float(np.mean(np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))))

    @staticmethod
    def _validate_model_input(model_input: ModelInput) -> None:
        ids, mask = model_input.input_ids, model_input.attention_mask
        if len(ids) != len(mask):
            raise ValueError(
                f"event {model_input.event_id!r}: input_ids and attention_mask "
                f"length mismatch ({len(ids)} vs {len(mask)})"
            )
        if len(ids) == 0:
            raise ValueError(f"event {model_input.event_id!r}: input_ids must be non-empty")
        if any(value not in (0, 1) for value in mask):
            raise ValueError(f"event {model_input.event_id!r}: attention_mask values must be 0 or 1")
