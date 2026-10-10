"""End-to-end Member 4 pipeline from fused input to anomaly prediction."""

from __future__ import annotations

from .fusion import InputFusion
from .masking import PrivacyMasker
from .model import LightweightAnomalyModel
from .models import AnomalyPrediction, RuleInfo, TokenSequence, UnifiedContext
from .normalization import Normalizer
from .tensorizer import Tensorizer
from .tokenizer import HybridTokenizer


class TokenizationAIService:
    """Run fusion, masking, normalization, tokenization, tensorization, and prediction.

    This class is pure orchestration: every security/structural property
    (trusted masking provenance, authoritative token roles, rule-preserving
    tensor truncation) is enforced inside the stages it calls, not here.
    Its one orchestration-level responsibility is to call the
    *provenance-carrying* variant of each stage --
    ``PrivacyMasker.mask_with_provenance``,
    ``Normalizer.normalize_with_provenance``,
    ``HybridTokenizer.tokenize(..., masked_paths=...)`` -- never the plain,
    backward-compatible methods that would silently drop that chain.

    Readiness: the vocabulary (owned by ``tensorizer``) and the model must
    already be fitted/trained before this service is constructed -- this
    class never fits or trains anything itself (see :meth:`__init__`).

    No cross-artifact "was this model actually trained on features
    derived from this exact vocabulary" compatibility check is performed
    or claimed: neither ``Vocabulary`` nor ``LightweightAnomalyModel`` has
    a versioned checkpoint format today, so there is nothing to compare
    them against. This constructor verifies only that each collaborator
    is individually ready (fitted / trained); that is not, and must not
    be read as, validation of production checkpoint compatibility.
    """

    def __init__(
        self,
        fusion: InputFusion,
        masker: PrivacyMasker,
        normalizer: Normalizer,
        tokenizer: HybridTokenizer,
        tensorizer: Tensorizer,
        model: LightweightAnomalyModel,
    ) -> None:
        """Wire the pipeline's collaborators, failing fast if either is not ready.

        Raises ``RuntimeError`` if ``tensorizer.vocabulary`` has not had
        :meth:`Vocabulary.fit` called on it, or if ``model`` has not had
        :meth:`LightweightAnomalyModel.fit` called on it -- using each
        component's own existing public readiness property
        (``Vocabulary.is_fitted``, ``LightweightAnomalyModel.is_trained``),
        rather than letting an unconfigured service accept requests and
        only fail deep inside ``Vocabulary.encode``/``LightweightAnomalyModel.predict``
        on the first real call.
        """
        if not tensorizer.vocabulary.is_fitted:
            raise RuntimeError(
                "TokenizationAIService requires a fitted Vocabulary; "
                "call Vocabulary.fit() on tensorizer.vocabulary before constructing this service"
            )
        if not model.is_trained:
            raise RuntimeError(
                "TokenizationAIService requires a trained LightweightAnomalyModel; "
                "call model.fit() before constructing this service"
            )
        self.fusion = fusion
        self.masker = masker
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.model = model

    def predict(self, context: UnifiedContext, rule_info: RuleInfo) -> AnomalyPrediction:
        """Return the anomaly prediction for a single event.

        Runs the full trusted pipeline: ``InputFusion.fuse ->
        PrivacyMasker.mask_with_provenance -> Normalizer.normalize_with_provenance
        -> HybridTokenizer.tokenize(masked_paths=...) -> Tensorizer.tensorize
        -> LightweightAnomalyModel.predict``.

        An empty ``rule_info.rules`` list is valid input (no verified
        rules reported for this event) -- it is not an error, and the
        resulting prediction carries no claim that formal verification
        was actually performed for this event; that fact lives with
        whoever produced (or didn't produce) ``rule_info``, not with this
        score.

        Propagates, unchanged, whatever each stage raises:
        ``InputContractError`` (event-id mismatch during fusion),
        ``ValueError`` (a tensor capacity failure -- the mandatory framing
        plus the complete verified-rule segment doesn't fit the
        tensorizer's ``max_length`` -- or a malformed intermediate
        sequence), ``RuntimeError`` (model or vocabulary not ready; should
        not occur given the constructor check, but never silently
        swallowed if it somehow does).
        """
        sequence = self._tokenize(context, rule_info)
        model_input = self.tensorizer.tensorize(sequence)
        score = self.model.predict(model_input)
        return AnomalyPrediction(
            event_id=context.event_id, score=score, model_version=self.model.model_version
        )

    def _tokenize(self, context: UnifiedContext, rule_info: RuleInfo) -> TokenSequence:
        """Run fusion through tokenization, preserving masking provenance throughout.

        Factored out from :meth:`predict` so the trusted-provenance chain
        (and its resulting token roles) can be verified directly in tests,
        independent of tensorization and model inference.
        """
        fused = self.fusion.fuse(context, rule_info)
        mask_result = self.masker.mask_with_provenance(fused)
        norm_result = self.normalizer.normalize_with_provenance(
            mask_result.fused, mask_result.masked_paths
        )
        return self.tokenizer.tokenize(norm_result.fused, norm_result.masked_paths)
