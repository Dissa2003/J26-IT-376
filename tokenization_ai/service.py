"""End-to-end Member 4 pipeline from fused input to anomaly prediction."""

from __future__ import annotations

from .fusion import InputFusion
from .masking import PrivacyMasker
from .model import LightweightAnomalyModel
from .models import AnomalyPrediction, RuleInfo, UnifiedContext
from .normalization import Normalizer
from .tensorizer import Tensorizer
from .tokenizer import HybridTokenizer


class TokenizationAIService:
    """Run fusion, masking, normalization, tokenization, tensorization, and prediction."""

    def __init__(
        self,
        fusion: InputFusion,
        masker: PrivacyMasker,
        normalizer: Normalizer,
        tokenizer: HybridTokenizer,
        tensorizer: Tensorizer,
        model: LightweightAnomalyModel,
    ) -> None:
        self.fusion = fusion
        self.masker = masker
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.model = model

    def predict(self, context: UnifiedContext, rule_info: RuleInfo) -> AnomalyPrediction:
        """Return the anomaly prediction for a single event."""
        raise NotImplementedError
