"""End-to-end LogiShield pipeline from fused input to anomaly prediction."""

from __future__ import annotations

from dataclasses import dataclass

from .fusion import InputFusion
from .masking import FieldPath, PrivacyMasker
from .model import LightweightAnomalyModel
from .models import (
    AnomalyPrediction,
    FusedInput,
    ModelInput,
    RuleInfo,
    TokenSequence,
    UnifiedApiContext,
    UnifiedContext,
    VerifiedRule,
    VerifiedRuleInfo,
)
from .normalization import NormalizationResult, Normalizer
from .tensorizer import Tensorizer
from .tokenizer import TOKEN_ROLE_MASKED, HybridTokenizer
from .tracing import (
    STAGE_FUSION,
    STAGE_MASKING,
    STAGE_MODEL_INFERENCE,
    STAGE_NORMALIZATION,
    STAGE_TENSORIZATION,
    STAGE_TOKENIZATION,
    PipelineTrace,
    PipelineTracer,
)


@dataclass(frozen=True)
class PipelineAnalysis:
    """A prediction together with the intermediates that produced it.

    ``normalized`` is the post-masking, post-normalization record (no raw
    sensitive values -- the masker has already replaced them) and
    ``masked_paths`` is the trusted provenance of which locations were
    genuinely masked. Returned by :meth:`TokenizationAIService.analyze`.
    """

    prediction: AnomalyPrediction
    normalized: FusedInput
    masked_paths: frozenset[FieldPath]
    sequence: TokenSequence
    model_input: ModelInput


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
        tracer: PipelineTracer | None = None,
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

        ``tracer`` is optional and purely additive: it supplies the
        :class:`tracing.PipelineTracer` used to start traces via
        :meth:`predict`'s ``trace`` argument (see that method and
        ``tracing.py``'s module docstring for the privacy contract on
        what a trace may record). A caller that never asks for a trace
        is completely unaffected by this parameter -- if omitted, a
        private default tracer is created, but nothing is recorded
        unless a caller explicitly passes a ``trace`` into :meth:`predict`.
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
        self.tracer = tracer if tracer is not None else PipelineTracer()

    @classmethod
    def create_default(
        cls, max_length: int = 64, num_buckets: int = 256
    ) -> "TokenizationAIService":
        """Construct a ready-to-serve pipeline instance with pre-calibrated baseline weights."""
        from .vocabulary import Vocabulary
        fusion = InputFusion()
        masker = PrivacyMasker()
        normalizer = Normalizer()
        tokenizer = HybridTokenizer()

        vocabulary = Vocabulary()
        initial_context = UnifiedApiContext(
            event_id="init-event",
            method="POST",
            path="/api/v1/init",
            client_ip="127.0.0.1",
            payload={"action": "manage", "role": "admin", "amount": 100},
            threat_score=0.1,
        )
        initial_rules = VerifiedRuleInfo(
            event_id="init-event",
            rules=[VerifiedRule(rule_id="R-INIT", attributes={"effect": "allow"})],
        )
        fused = fusion.fuse(initial_context, initial_rules)
        mask_result = masker.mask_with_provenance(fused)
        norm_result = normalizer.normalize_with_provenance(
            mask_result.fused, mask_result.masked_paths
        )
        seq = tokenizer.tokenize(norm_result.fused, norm_result.masked_paths)

        vocabulary.fit([seq])
        tensorizer = Tensorizer(vocabulary, max_length=max_length)
        model = LightweightAnomalyModel.create_calibrated(num_buckets=num_buckets)
        return cls(fusion, masker, normalizer, tokenizer, tensorizer, model)

    def predict(
        self,
        context: UnifiedContext,
        rule_info: RuleInfo,
        *,
        trace: PipelineTrace | None = None,
    ) -> AnomalyPrediction:
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

        ``trace`` is optional and purely additive (see :meth:`__init__`'s
        ``tracer`` docstring): when ``None`` (the default), this method's
        behavior, return value, and error propagation are byte-for-byte
        identical to a build of this method with no tracing support at
        all -- every timing call is skipped, not merely unused. Pass a
        trace started from ``self.tracer.start(context.event_id)`` to
        additionally record each stage's timing and a safe structural
        summary (counts only -- never raw field values) into it,
        including on failure, before the original exception is re-raised.
        """
        return self.analyze(context, rule_info, trace=trace).prediction

    def analyze(
        self,
        context: UnifiedContext,
        rule_info: RuleInfo,
        *,
        trace: PipelineTrace | None = None,
    ) -> PipelineAnalysis:
        """Run the full pipeline and return the prediction with its intermediates.

        Exactly the same stages, ordering, tracing and error propagation as
        :meth:`predict` (which delegates here); the only difference is that
        the post-masking/normalization record, the authoritative masking
        provenance, the token sequence, and the model input are returned
        alongside the prediction so an inspection layer (see ``app.py``)
        can show them without re-running or re-implementing any stage.
        """
        norm_result, sequence = self._front(context, rule_info, trace=trace)
        tokens_before_tensorize = len(sequence.tokens)
        model_input = self.tracer.run_stage(
            trace,
            STAGE_TENSORIZATION,
            lambda: self.tensorizer.tensorize(sequence),
            summary_fn=lambda _result: {
                "token_count": tokens_before_tensorize,
                "truncated": tokens_before_tensorize > self.tensorizer.max_length,
            },
        )
        score = self.tracer.run_stage(
            trace, STAGE_MODEL_INFERENCE, lambda: self.model.predict(model_input)
        )
        return PipelineAnalysis(
            prediction=AnomalyPrediction(
                event_id=context.event_id, score=score, model_version=self.model.model_version
            ),
            normalized=norm_result.fused,
            masked_paths=norm_result.masked_paths,
            sequence=sequence,
            model_input=model_input,
        )

    def _tokenize(
        self,
        context: UnifiedContext,
        rule_info: RuleInfo,
        *,
        trace: PipelineTrace | None = None,
    ) -> TokenSequence:
        """Run fusion through tokenization, preserving masking provenance throughout.

        Factored out from :meth:`predict` so the trusted-provenance chain
        (and its resulting token roles) can be verified directly in tests,
        independent of tensorization and model inference. ``trace`` is
        optional and purely additive -- see :meth:`predict`'s docstring.
        """
        return self._front(context, rule_info, trace=trace)[1]

    def _front(
        self,
        context: UnifiedContext,
        rule_info: RuleInfo,
        *,
        trace: PipelineTrace | None = None,
    ) -> tuple[NormalizationResult, TokenSequence]:
        """Fusion through tokenization; also returns the normalized record + provenance."""
        fused = self.tracer.run_stage(
            trace,
            STAGE_FUSION,
            lambda: self.fusion.fuse(context, rule_info),
            summary_fn=lambda result: {"rule_count": len(result.rule_info.rules)},
        )
        mask_result = self.tracer.run_stage(
            trace,
            STAGE_MASKING,
            lambda: self.masker.mask_with_provenance(fused),
            summary_fn=lambda result: {"masked_field_count": len(result.masked_paths)},
        )
        norm_result = self.tracer.run_stage(
            trace,
            STAGE_NORMALIZATION,
            lambda: self.normalizer.normalize_with_provenance(
                mask_result.fused, mask_result.masked_paths
            ),
            summary_fn=lambda result: {"masked_field_count": len(result.masked_paths)},
        )
        sequence = self.tracer.run_stage(
            trace,
            STAGE_TOKENIZATION,
            lambda: self.tokenizer.tokenize(norm_result.fused, norm_result.masked_paths),
            summary_fn=lambda sequence: {
                "token_count": len(sequence.tokens),
                "masked_token_count": sequence.roles.count(TOKEN_ROLE_MASKED),
            },
        )
        return norm_result, sequence
