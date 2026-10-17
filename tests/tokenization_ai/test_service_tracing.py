"""Tests for optional tracing integration in tokenization_ai.service.

These tests complement test_service.py's untraced coverage; they exist
to verify the *additional*, strictly optional tracing behavior wired
into TokenizationAIService -- that it is a no-op unless a caller
explicitly asks for a trace, that it never changes predict()'s return
value or error propagation, and that it never records sensitive input.
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
from tokenization_ai.tokenizer import HybridTokenizer
from tokenization_ai.tracing import PIPELINE_STAGES, PipelineTracer
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
    fused = fusion.fuse(context, rule_info)
    mask_result = masker.mask_with_provenance(fused)
    norm_result = normalizer.normalize_with_provenance(mask_result.fused, mask_result.masked_paths)
    return tokenizer.tokenize(norm_result.fused, norm_result.masked_paths)


def _build_ready_service(max_length: int = 64, num_buckets: int = 64) -> TokenizationAIService:
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


# -- Tracing is purely additive / opt-in ---------------------------------------


def test_predict_without_trace_returns_the_same_prediction_as_before_tracing_existed() -> None:
    service = _build_ready_service()
    context = _context(payload={"amount": 5})
    rule_info = _rule_info()

    prediction = service.predict(context, rule_info)

    assert prediction.event_id == "e1"
    assert 0.0 <= prediction.score <= 1.0
    assert prediction.model_version == "ngram-linear-v1-trained"


def test_predict_without_trace_matches_predict_with_a_trace_bit_for_bit() -> None:
    service = _build_ready_service()
    context = _context(payload={"amount": 5}, headers={"X-Test": "1"})
    rule_info = _rule_info(rules=[VerifiedRule(rule_id="R-1", attributes={"field": "amount"})])

    untraced = service.predict(context, rule_info)
    trace = service.tracer.start(context.event_id)
    traced = service.predict(context, rule_info, trace=trace)

    assert untraced == traced


def test_constructor_without_a_tracer_still_provides_a_usable_default_tracer() -> None:
    service = _build_ready_service()

    assert isinstance(service.tracer, PipelineTracer)
    trace = service.tracer.start("e1")
    assert trace.event_id == "e1"


def test_constructor_accepts_an_explicit_tracer() -> None:
    fusion, masker, normalizer, tokenizer = _components()
    sequence = _tokenize_sample(fusion, masker, normalizer, tokenizer, _context(), _rule_info())
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    tensorizer = Tensorizer(vocabulary, max_length=32)
    model = LightweightAnomalyModel(num_buckets=32)
    model.fit([tensorizer.tensorize(sequence)], [0], epochs=1)
    custom_tracer = PipelineTracer()

    service = TokenizationAIService(
        fusion, masker, normalizer, tokenizer, tensorizer, model, tracer=custom_tracer
    )

    assert service.tracer is custom_tracer


# -- All six stages are represented, in order ----------------------------------


def test_predict_with_a_trace_records_all_six_stages_in_order() -> None:
    service = _build_ready_service()
    trace = service.tracer.start("e1")

    service.predict(_context(payload={"amount": 5}), _rule_info(), trace=trace)

    assert tuple(stage.stage for stage in trace.stages) == PIPELINE_STAGES


def test_predict_with_a_trace_marks_every_stage_successful_on_a_successful_run() -> None:
    service = _build_ready_service()
    trace = service.tracer.start("e1")

    service.predict(_context(payload={"amount": 5}), _rule_info(), trace=trace)

    assert all(stage.success for stage in trace.stages)
    assert all(stage.error_type is None for stage in trace.stages)


# -- Timing values are valid ----------------------------------------------------


def test_predict_with_a_trace_records_non_negative_elapsed_times() -> None:
    service = _build_ready_service()
    trace = service.tracer.start("e1")

    service.predict(_context(payload={"amount": 5}), _rule_info(), trace=trace)

    assert all(stage.elapsed_ms >= 0.0 for stage in trace.stages)


def test_predict_with_a_trace_total_elapsed_equals_sum_of_stage_elapsed() -> None:
    service = _build_ready_service()
    trace = service.tracer.start("e1")

    service.predict(_context(payload={"amount": 5}), _rule_info(), trace=trace)

    assert trace.total_elapsed_ms == sum(stage.elapsed_ms for stage in trace.stages)


# -- Safe structural summaries, no sensitive values -----------------------------


def test_predict_with_a_trace_never_leaks_sensitive_values_into_stage_summaries() -> None:
    service = _build_ready_service()
    context = _context(
        payload={"email": "real.customer@example.com", "password": "super-secret-value"},
        headers={"Authorization": "Bearer real-secret-token"},
    )
    trace = service.tracer.start(context.event_id)

    service.predict(context, _rule_info(), trace=trace)

    trace_text = repr(trace.stages)
    assert "real.customer@example.com" not in trace_text
    assert "super-secret-value" not in trace_text
    assert "real-secret-token" not in trace_text
    # Every summary value recorded must be a small int/bool, never a string
    # (which could carry a raw field value) or a container of raw content.
    for stage in trace.stages:
        for value in stage.summary.values():
            assert isinstance(value, (int, bool))


def test_predict_with_a_trace_tokenization_summary_reports_masked_token_count() -> None:
    service = _build_ready_service()
    context = _context(payload={"email": "customer@example.com"})
    trace = service.tracer.start(context.event_id)

    service.predict(context, _rule_info(), trace=trace)

    tokenization_stage = next(stage for stage in trace.stages if stage.stage == "tokenization")
    assert tokenization_stage.summary["masked_token_count"] >= 1
    assert tokenization_stage.summary["token_count"] > 0


def test_predict_with_a_trace_tensorization_summary_reports_truncated_accurately() -> None:
    fusion, masker, normalizer, tokenizer = _components()
    many_rules = [VerifiedRule(rule_id=f"R-{i}", attributes={"k": "v"}) for i in range(20)]
    sequence = _tokenize_sample(
        fusion, masker, normalizer, tokenizer, _context(), _rule_info(rules=many_rules)
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    roomy_tensorizer = Tensorizer(vocabulary, max_length=4096)
    model = LightweightAnomalyModel(num_buckets=64)
    model.fit([roomy_tensorizer.tensorize(sequence)], [1], epochs=1)

    # A tensorizer with just enough room that the full untruncated sequence
    # fits -- truncated must be reported False, matching ground truth.
    service = TokenizationAIService(fusion, masker, normalizer, tokenizer, roomy_tensorizer, model)
    trace = service.tracer.start("e1")
    service.predict(_context(), _rule_info(rules=many_rules), trace=trace)

    tensorization_stage = next(stage for stage in trace.stages if stage.stage == "tensorization")
    assert tensorization_stage.summary["truncated"] is False


# -- Request traces remain isolated --------------------------------------------


def test_two_predict_calls_with_separate_traces_do_not_mix_their_stages() -> None:
    service = _build_ready_service()
    trace_a = service.tracer.start("event-a")
    trace_b = service.tracer.start("event-b")

    service.predict(_context(event_id="event-a", payload={"amount": 1}), _rule_info(event_id="event-a"), trace=trace_a)
    service.predict(_context(event_id="event-b", payload={"amount": 2}), _rule_info(event_id="event-b"), trace=trace_b)

    assert trace_a.event_id == "event-a"
    assert trace_b.event_id == "event-b"
    assert len(trace_a.stages) == len(PIPELINE_STAGES)
    assert len(trace_b.stages) == len(PIPELINE_STAGES)


# -- Error propagation is preserved, even with tracing --------------------------


def test_predict_with_a_trace_still_raises_input_contract_error_on_mismatch() -> None:
    service = _build_ready_service()
    trace = service.tracer.start("e1")

    with pytest.raises(InputContractError):
        service.predict(_context(event_id="e1"), _rule_info(event_id="different-event"), trace=trace)


def test_predict_with_a_trace_records_the_failing_stage_without_its_message() -> None:
    service = _build_ready_service()
    trace = service.tracer.start("e1")

    with pytest.raises(InputContractError):
        service.predict(_context(event_id="e1"), _rule_info(event_id="different-event"), trace=trace)

    assert len(trace.stages) == 1
    failed_stage = trace.stages[0]
    assert failed_stage.stage == "fusion"
    assert failed_stage.success is False
    assert failed_stage.error_type == "InputContractError"


def test_predict_with_a_trace_still_raises_tensor_capacity_failure() -> None:
    fusion, masker, normalizer, tokenizer = _components()
    many_rules = [VerifiedRule(rule_id=f"R-{i}", attributes={"k": "v" * 20}) for i in range(50)]
    sequence = _tokenize_sample(
        fusion, masker, normalizer, tokenizer, _context(), _rule_info(rules=many_rules)
    )
    vocabulary = Vocabulary()
    vocabulary.fit([sequence])
    tiny_tensorizer = Tensorizer(vocabulary, max_length=5)
    model = LightweightAnomalyModel(num_buckets=16)
    roomy_tensorizer = Tensorizer(vocabulary, max_length=4096)
    model.fit([roomy_tensorizer.tensorize(sequence)], [1], epochs=1)

    service = TokenizationAIService(fusion, masker, normalizer, tokenizer, tiny_tensorizer, model)
    trace = service.tracer.start("e1")

    with pytest.raises(ValueError):
        service.predict(_context(), _rule_info(rules=many_rules), trace=trace)

    failed_stage = trace.stages[-1]
    assert failed_stage.stage == "tensorization"
    assert failed_stage.success is False


# -- Existing privacy / token-role protections still hold with tracing ----------


def test_tokenize_step_still_tags_masked_fields_correctly_when_traced() -> None:
    from tokenization_ai.tokenizer import TOKEN_ROLE_MASKED

    service = _build_ready_service()
    trace = service.tracer.start("e1")

    sequence = service._tokenize(_context(payload={"email": "customer@example.com"}), _rule_info(), trace=trace)

    email_key_index = sequence.tokens.index("email")
    assert sequence.roles[email_key_index + 2] == TOKEN_ROLE_MASKED
    assert "customer@example.com" not in sequence.tokens
