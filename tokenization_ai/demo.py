"""Small, runnable PP1 demonstration of the LogiShield pipeline, end to end.

SYNTHETIC DEMONSTRATION ONLY -- NOT A RESEARCH RESULT.

Every event, payload, and "verified rule" in this module is a small,
hand-constructed fixture written for this demo, in the same spirit as
the synthetic fixtures already used in ``test_service.py``'s
"synthetic PP1 implementation" test and ``model.py``'s own training-fixture
caveat. In particular:

- The rule fixtures below (``rule_id="MOCK-R-..."``) are illustrative
  mock data standing in for Formal Verification's real formal-verification output.
  They are not the product of genuine rule verification, and nothing
  here exercises Formal Verification's actual rule engine.
- Runtime Verification integration is not exercised at all: ``adapters.to_runtime_verification_output``
  remains an intentional ``NotImplementedError`` pending an agreed
  contract with that team (see ``adapters.py``), and this demo does not
  attempt to work around that.
- The "benign"/"suspicious" labels used to train the small demo model
  are this module's author's own illustrative choices, not ground truth
  from any real system. The resulting scores demonstrate that the six
  LogiShield stages are wired together correctly end to end; they are not
  evidence of real-world anomaly-detection accuracy and must never be
  cited as such.

Run directly with ``python -m tokenization_ai.demo``.
"""

from __future__ import annotations

from .fusion import InputFusion
from .masking import PrivacyMasker
from .metrics import RuntimeMetrics, aggregate_runtime_metrics, runtime_metrics_from_trace
from .model import LightweightAnomalyModel
from .models import AnomalyPrediction, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from .normalization import Normalizer
from .service import TokenizationAIService
from .tensorizer import Tensorizer
from .tokenizer import HybridTokenizer
from .tracing import PipelineTrace
from .vocabulary import Vocabulary

# Explicitly synthetic demo fixtures -- hand-written for this demo, not
# sampled from real traffic and not drawn from any real dataset.
_DEMO_EVENTS: list[tuple[UnifiedApiContext, VerifiedRuleInfo]] = [
    (
        UnifiedApiContext(
            event_id="demo-benign-1",
            method="GET",
            path="/orders/123",
            payload={"amount": 25, "currency": "USD"},
        ),
        VerifiedRuleInfo(event_id="demo-benign-1", rules=[]),
    ),
    (
        UnifiedApiContext(
            event_id="demo-suspicious-1",
            method="POST",
            path="/login",
            payload={
                "username": "admin",
                "password": "hunter2",
                "q": "' OR 1=1 --",
            },
            headers={"Authorization": "Bearer demo-not-a-real-token"},
        ),
        # MOCK verified-rule fixture -- illustrative only, not genuine
        # Formal Verification output. See the module docstring.
        VerifiedRuleInfo(
            event_id="demo-suspicious-1",
            rules=[VerifiedRule(rule_id="MOCK-R-SQLI", attributes={"severity": "high"})],
        ),
    ),
]

# Illustrative labels chosen by this demo's author for the two fixtures
# above, in the same order -- not ground truth from any real system.
_DEMO_LABELS = [0, 1]


def _build_demo_service(max_length: int = 128, num_buckets: int = 256) -> TokenizationAIService:
    """Fit a vocabulary/model on the demo fixtures so the demo is self-contained.

    The model is genuinely trained (gradient descent on the two synthetic fixtures) but it is
    not a detection model: its scores only show that the stages are wired end to end.
    """
    fusion, masker, normalizer, tokenizer = (
        InputFusion(),
        PrivacyMasker(),
        Normalizer(),
        HybridTokenizer(),
    )

    def _tokenize_fixture(context: UnifiedApiContext, rule_info: VerifiedRuleInfo):
        fused = fusion.fuse(context, rule_info)
        mask_result = masker.mask_with_provenance(fused)
        norm_result = normalizer.normalize_with_provenance(
            mask_result.fused, mask_result.masked_paths
        )
        return tokenizer.tokenize(norm_result.fused, norm_result.masked_paths)

    sequences = [_tokenize_fixture(context, rule_info) for context, rule_info in _DEMO_EVENTS]

    vocabulary = Vocabulary()
    vocabulary.fit(sequences)
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    model = LightweightAnomalyModel(num_buckets=num_buckets)
    model.fit(
        [tensorizer.tensorize(sequence) for sequence in sequences],
        _DEMO_LABELS,
        epochs=50,
        learning_rate=0.5,
    )

    return TokenizationAIService(fusion, masker, normalizer, tokenizer, tensorizer, model)


def run_demo() -> list[tuple[str, AnomalyPrediction, PipelineTrace, RuntimeMetrics]]:
    """Run every demo event through the full traced pipeline.

    Returns one ``(event_id, prediction, trace, metrics)`` tuple per
    demo event. SYNTHETIC DEMONSTRATION ONLY -- see the module docstring.
    """
    service = _build_demo_service()
    results = []
    for context, rule_info in _DEMO_EVENTS:
        trace = service.tracer.start(context.event_id)
        prediction = service.predict(context, rule_info, trace=trace)
        metrics = runtime_metrics_from_trace(trace)
        results.append((context.event_id, prediction, trace, metrics))
    return results


def main() -> None:
    print("=" * 72)
    print("LogiShield PP1 demonstration -- SYNTHETIC DATA ONLY, NOT A REAL DATASET")
    print("Verified-rule fixtures are MOCK data, not genuine Formal Verification output.")
    print("Formal Verification and Runtime Verification integration is not exercised in this demo.")
    print("=" * 72)

    results = run_demo()
    for event_id, prediction, _trace, metrics in results:
        print(f"\nEvent: {event_id}")
        print(f"  Prediction: score={prediction.score:.4f} model_version={prediction.model_version}")
        print(f"  Total pipeline time: {metrics.total_elapsed_ms:.3f} ms")
        for stage, elapsed in metrics.stage_elapsed_ms.items():
            print(f"    {stage:>16}: {elapsed:.3f} ms")
        if metrics.token_count is not None:
            print(f"  Token count: {metrics.token_count} (truncated={metrics.truncated})")

    aggregate = aggregate_runtime_metrics([metrics for _, _, _, metrics in results])
    print(f"\nAggregate total-pipeline latency over {aggregate['count']} synthetic demo events:")
    print(
        f"  mean={aggregate['mean_ms']:.3f} ms  "
        f"min={aggregate['min_ms']:.3f} ms  "
        f"max={aggregate['max_ms']:.3f} ms"
    )
    print(
        "\nReminder: these are runtime timings for this machine and these "
        "two fixtures only -- not a measure of detection accuracy, and not "
        "representative of real production load."
    )


if __name__ == "__main__":
    main()
