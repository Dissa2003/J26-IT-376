"""Unit tests for Member 4's end-to-end service orchestration.

The "synthetic PP1 prototype demonstration" test at the bottom of this
file uses a small, explicitly synthetic fixture (same style as Phase 9's
model tests) solely to prove the six Member 4 stages are wired together
correctly end to end. It does not and cannot demonstrate real-world
anomaly-detection accuracy -- no labeled real dataset exists -- and its
result must never be read or cited as such.
"""

import pytest

from tokenization_ai.adapters import InputContractError
from tokenization_ai.fusion import InputFusion
from tokenization_ai.masking import PrivacyMasker
from tokenization_ai.model import LightweightAnomalyModel
from tokenization_ai.models import UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.normalization import Normalizer
from tokenization_ai.service import TokenizationAIService
from tokenization_ai.tensorizer import Tensorizer
from tokenization_ai.tokenizer import TOKEN_ROLE_LITERAL_API, TOKEN_ROLE_MASKED, HybridTokenizer
from tokenization_ai.vocabulary import Vocabulary


def _context(event_id: str = "e1", **overrides: object) -> UnifiedApiContext:
    fields: dict[str, object] = {"event_id": event_id, "method": "POST", "path": "/orders"}
    fields.update(overrides)
    return UnifiedApiContext(**fields)


def _rule_info(event_id: str = "e1", rules: list[VerifiedRule] | None = None) -> VerifiedRuleInfo:
    return VerifiedRuleInfo(event_id=event_id, rules=rules or [])


def _components() -> tuple[InputFusion, PrivacyMasker, Normalizer, HybridTokenizer]:
    return InputFusion(), PrivacyMasker(), Normalizer(), HybridTokenizer()


def _tokenize_sample(fusion, masker, normalizer, tokenizer, context, rule_info):
    """Run the same provenance-preserving chain the service uses, standalone --
    used only to build fitting/training fixtures before a service exists."""
    fused = fusion.fuse(context, rule_info)
    mask_result = masker.mask_with_provenance(fused)
    norm_result = normalizer.normalize_with_provenance(mask_result.fused, mask_result.masked_paths)
    return tokenizer.tokenize(norm_result.fused, norm_result.masked_paths)


def _build_ready_service(max_length: int = 64, num_buckets: int = 64) -> TokenizationAIService:
    """A fully-ready service: fitted Vocabulary, trained LightweightAnomalyModel,
    both fit/trained on a small, explicitly synthetic fixture."""
    fusion, masker, normalizer, tokenizer = _components()

    benign = _tokenize_sample(
        fusion, masker, normalizer, tokenizer,
        _context("fit-benign", payload={"amount": 5}),
        _rule_info("fit-benign"),
    )
    suspicious = _tokenize_sample(
        fusion, masker, normalizer, tokenizer,
        _context("fit-suspicious", payload={"q": "' OR 1=1 --"}),
        _rule_info("fit-suspicious", rules=[VerifiedRule(rule_id="R-1", attributes={"severity": "high"})]),
    )

    vocabulary = Vocabulary()
    vocabulary.fit([benign, suspicious])
    tensorizer = Tensorizer(vocabulary, max_length=max_length)

    model = LightweightAnomalyModel(num_buckets=num_buckets)
    model.fit(
        [tensorizer.tensorize(benign), tensorizer.tensorize(suspicious)],
        [0, 1],
        epochs=50,
        learning_rate=0.5,
    )

    return TokenizationAIService(fusion, masker, normalizer, tokenizer, tensorizer, model)


# -- Successful end-to-end orchestration -------------------------------------


def test_predict_runs_the_full_pipeline_successfully() -> None:
    service = _build_ready_service()

    prediction = service.predict(_context(payload={"amount": 5}), _rule_info())

    assert prediction.event_id == "e1"
    assert 0.0 <= prediction.score <= 1.0
    assert prediction.model_version == "ngram-linear-v1-trained"


def test_predict_preserves_event_id_across_context_and_prediction() -> None:
    service = _build_ready_service()

    prediction = service.predict(_context(event_id="distinct-event"), _rule_info(event_id="distinct-event"))

    assert prediction.event_id == "distinct-event"


def test_predict_score_is_always_bounded() -> None:
    service = _build_ready_service()

    for payload in [{}, {"amount": 5}, {"q": "' OR 1=1 --", "extra": "x" * 50}]:
        prediction = service.predict(_context(payload=payload), _rule_info())
        assert 0.0 <= prediction.score <= 1.0


def test_predict_is_deterministic() -> None:
    service = _build_ready_service()
    context = _context(payload={"amount": 5, "items": ["a", "b"]})
    rule_info = _rule_info(rules=[VerifiedRule(rule_id="R-1", attributes={"field": "amount"})])

    first = service.predict(context, rule_info)
    second = service.predict(context, rule_info)

    assert first == second


def test_predict_does_not_mutate_its_inputs() -> None:
    service = _build_ready_service()
    context = _context(payload={"amount": 5}, headers={"Authorization": "Bearer secret"})
    rule_info = _rule_info(rules=[VerifiedRule(rule_id="R-1", attributes={"field": "amount"})])
    before_context = context.model_dump()
    before_rule_info = rule_info.model_dump()

    service.predict(context, rule_info)

    assert context.model_dump() == before_context
    assert rule_info.model_dump() == before_rule_info


# -- Trusted masking provenance propagation -----------------------------------


def test_tokenize_step_tags_a_genuinely_masked_field_with_the_masked_role() -> None:
    service = _build_ready_service()

    sequence = service._tokenize(_context(payload={"email": "customer@example.com"}), _rule_info())

    email_key_index = sequence.tokens.index("email")
    value_role = sequence.roles[email_key_index + 2]
    assert value_role == TOKEN_ROLE_MASKED


def test_tokenize_step_leaves_an_attacker_lookalike_as_an_ordinary_literal() -> None:
    service = _build_ready_service()

    # "nickname" is never classified as sensitive -- this value merely
    # reads like a masking placeholder; it was never actually masked.
    sequence = service._tokenize(_context(payload={"nickname": "[MASKED_PII]"}), _rule_info())

    nickname_key_index = sequence.tokens.index("nickname")
    value_role = sequence.roles[nickname_key_index + 2]
    assert value_role == TOKEN_ROLE_LITERAL_API


def test_predict_does_not_leak_raw_sensitive_values_into_the_token_sequence() -> None:
    service = _build_ready_service()
    context = _context(
        payload={"email": "real.customer@example.com", "password": "super-secret-value"},
        headers={"Authorization": "Bearer real-secret-token"},
    )

    sequence = service._tokenize(context, _rule_info())

    assert "real.customer@example.com" not in sequence.tokens
    assert "super-secret-value" not in sequence.tokens
    assert "Bearer real-secret-token" not in sequence.tokens
    assert "real-secret-token" not in sequence.tokens
    # And the pipeline still runs to a bounded score using only the masked form.
    prediction = service.predict(context, _rule_info())
    assert 0.0 <= prediction.score <= 1.0


# -- Constructor readiness validation -----------------------------------------


def test_constructor_rejects_an_unfitted_vocabulary() -> None:
    fusion, masker, normalizer, tokenizer = _components()
    unfitted_vocabulary = Vocabulary()
    tensorizer = Tensorizer(unfitted_vocabulary, max_length=32)
    # A trained model, built against a *separate*, properly fitted vocabulary,
    # just to isolate "vocabulary not fitted" as the only readiness problem.
    helper_vocabulary = Vocabulary()
    sequence = _tokenize_sample(fusion, masker, normalizer, tokenizer, _context(), _rule_info())
    helper_vocabulary.fit([sequence])
    helper_tensorizer = Tensorizer(helper_vocabulary, max_length=32)
    model = LightweightAnomalyModel(num_buckets=32)
    model.fit([helper_tensorizer.tensorize(sequence)], [0], epochs=1)

    with pytest.raises(RuntimeError):
        TokenizationAIService(fusion, masker, normalizer, tokenizer, tensorizer, model)


def test_constructor_rejects_an_untrained_model() -> None:
    fusion, masker, normalizer, tokenizer = _components()
    sequence = _tokenize_sample(fusion, masker, normalizer, tokenizer, _context(), _rule_info())
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    tensorizer = Tensorizer(vocabulary, max_length=32)
    untrained_model = LightweightAnomalyModel(num_buckets=32)

    with pytest.raises(RuntimeError):
        TokenizationAIService(fusion, masker, normalizer, tokenizer, tensorizer, untrained_model)


def test_constructor_accepts_a_fully_ready_service() -> None:
    service = _build_ready_service()

    assert service.tensorizer.vocabulary.is_fitted
    assert service.model.is_trained


# -- Error propagation --------------------------------------------------------


def test_predict_propagates_input_contract_error_on_event_id_mismatch() -> None:
    service = _build_ready_service()

    with pytest.raises(InputContractError):
        service.predict(_context(event_id="e1"), _rule_info(event_id="different-event"))


def test_predict_propagates_tensor_capacity_failure() -> None:
    fusion, masker, normalizer, tokenizer = _components()
    many_rules = [VerifiedRule(rule_id=f"R-{i}", attributes={"k": "v" * 20}) for i in range(50)]
    sequence = _tokenize_sample(
        fusion, masker, normalizer, tokenizer, _context(), _rule_info(rules=many_rules)
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    # A max_length far too small for the mandatory framing plus the full
    # (undroppable) verified-rule segment -- must raise, not silently
    # drop rules or truncate past capacity.
    tiny_tensorizer = Tensorizer(vocabulary, max_length=5)
    model = LightweightAnomalyModel(num_buckets=16)
    # Train against a tensorizer with enough room, just to reach is_trained;
    # the capacity failure under test happens inside predict(), using the
    # tiny tensorizer actually wired into the service.
    roomy_tensorizer = Tensorizer(vocabulary, max_length=4096)
    model.fit([roomy_tensorizer.tensorize(sequence)], [1], epochs=1)

    service = TokenizationAIService(fusion, masker, normalizer, tokenizer, tiny_tensorizer, model)

    with pytest.raises(ValueError):
        service.predict(_context(), _rule_info(rules=many_rules))


# -- Valid empty-rule inputs ---------------------------------------------------


def test_predict_accepts_an_empty_rules_list_without_error() -> None:
    service = _build_ready_service()

    prediction = service.predict(_context(payload={"amount": 5}), _rule_info(rules=[]))

    assert 0.0 <= prediction.score <= 1.0


def test_empty_rules_prediction_carries_no_verification_claim() -> None:
    # An AnomalyPrediction has no field at all claiming "verified rules were
    # checked" -- its contract is only event_id/score/model_version, so an
    # empty rules list cannot be mistaken, even implicitly, for evidence
    # that formal verification occurred.
    service = _build_ready_service()

    prediction = service.predict(_context(), _rule_info(rules=[]))

    assert set(prediction.model_dump().keys()) == {"event_id", "score", "model_version"}


# -- Synthetic PP1 prototype demonstration ------------------------------------


def test_synthetic_pp1_prototype_end_to_end_demonstration() -> None:
    """PP1 PROTOTYPE DEMONSTRATION -- SYNTHETIC DATA ONLY.

    This test builds a Vocabulary and LightweightAnomalyModel from a
    small, hand-constructed, explicitly synthetic fixture (not real API
    traffic, not a real dataset) and runs them through the full Member 4
    service pipeline. It demonstrates that the six Member 4 stages are
    correctly wired together end to end -- fusion through masking,
    normalization, tokenization, tensorization, and model inference --
    and nothing more.

    It does NOT demonstrate, and must never be cited as demonstrating,
    real-world anomaly-detection accuracy, precision, recall, or any
    other measured security performance. No labeled real dataset exists
    for Member 4 as of this phase (see the Phase 9/10 planning reports);
    this test's "benign" and "suspicious" labels are illustrative
    fixtures chosen by this test's author, not ground truth from any
    real system.
    """
    service = _build_ready_service(max_length=128, num_buckets=256)

    benign_like = service.predict(
        _context(payload={"amount": 25, "currency": "USD"}),
        _rule_info(),
    )
    suspicious_like = service.predict(
        _context(payload={"q": "' OR 1=1 --", "note": "DROP TABLE users"}),
        _rule_info(rules=[VerifiedRule(rule_id="R-SUSPICIOUS", attributes={"severity": "high"})]),
    )

    for prediction in (benign_like, suspicious_like):
        assert 0.0 <= prediction.score <= 1.0
        assert prediction.model_version == "ngram-linear-v1-trained"
