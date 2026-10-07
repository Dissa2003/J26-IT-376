"""Unit tests for normalization."""

from tokenization_ai.masking import MASKED_AUTH, MASKED_PII, PrivacyMasker
from tokenization_ai.models import FusedInput, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo
from tokenization_ai.normalization import Normalizer


def _fused(**context_overrides: object) -> FusedInput:
    fields: dict[str, object] = {"event_id": "e1", "method": "POST", "path": "/orders"}
    fields.update(context_overrides)
    context = UnifiedApiContext(**fields)
    rule_info = VerifiedRuleInfo(
        event_id="e1",
        rules=[VerifiedRule(rule_id="R-1", attributes={"field": "amount"})],
    )
    return FusedInput(context=context, rule_info=rule_info)


def test_normalize_upper_cases_and_trims_the_method() -> None:
    fused = _fused(method=" post ")

    normalized = Normalizer().normalize(fused)

    assert normalized.context.method == "POST"


def test_normalize_collapses_slashes_and_drops_trailing_slash() -> None:
    fused = _fused(path=" /orders//items/ ")

    normalized = Normalizer().normalize(fused)

    assert normalized.context.path == "/orders/items"


def test_normalize_adds_a_leading_slash_when_missing() -> None:
    fused = _fused(path="orders")

    normalized = Normalizer().normalize(fused)

    assert normalized.context.path == "/orders"


def test_normalize_keeps_root_path_as_a_single_slash() -> None:
    fused = _fused(path="//")

    normalized = Normalizer().normalize(fused)

    assert normalized.context.path == "/"


def test_normalize_lower_cases_header_names_but_not_values() -> None:
    fused = _fused(headers={"Content-Type": "Application/JSON", "X-Request-ID": "abc"})

    normalized = Normalizer().normalize(fused)

    assert normalized.context.headers == {
        "content-type": "Application/JSON",
        "x-request-id": "abc",
    }


def test_normalize_handles_nested_dicts_and_lists_without_renaming_fields() -> None:
    fused = _fused(
        payload={
            "amount": 5,
            "customer": {"name": " Jane ", "tags": ["vip", "returning"]},
            "items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}],
        }
    )

    normalized = Normalizer().normalize(fused)
    payload = normalized.context.payload

    assert set(payload.keys()) == {"amount", "customer", "items"}
    assert payload["amount"] == 5
    assert payload["customer"]["name"] == "Jane"
    assert payload["customer"]["tags"] == ["vip", "returning"]
    assert payload["items"] == [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]


def test_normalize_preserves_int_type_exactly() -> None:
    fused = _fused(payload={"amount": 5, "nested": {"count": 3}, "values": [1, 2, 3]})

    normalized = Normalizer().normalize(fused)
    payload = normalized.context.payload

    assert payload["amount"] == 5 and isinstance(payload["amount"], int)
    assert not isinstance(payload["amount"], bool)
    assert payload["nested"]["count"] == 3 and isinstance(payload["nested"]["count"], int)
    assert all(isinstance(value, int) for value in payload["values"])


def test_normalize_preserves_float_type_exactly() -> None:
    fused = _fused(payload={"amount": 5.5, "nested": {"ratio": 0.25}})

    normalized = Normalizer().normalize(fused)
    payload = normalized.context.payload

    assert payload["amount"] == 5.5 and isinstance(payload["amount"], float)
    assert payload["nested"]["ratio"] == 0.25 and isinstance(payload["nested"]["ratio"], float)


def test_normalize_preserves_bool_and_none_types() -> None:
    fused = _fused(payload={"active": True, "deleted": False, "middle_name": None})

    normalized = Normalizer().normalize(fused)
    payload = normalized.context.payload

    assert payload["active"] is True
    assert payload["deleted"] is False
    assert payload["middle_name"] is None


def test_normalize_preserves_masking_placeholders_exactly() -> None:
    masked = PrivacyMasker().mask(
        _fused(
            client_ip="203.0.113.5",
            headers={"Authorization": "Bearer secret"},
            payload={"email": "customer@example.com", "amount": 1},
        )
    )

    normalized = Normalizer().normalize(masked)

    assert normalized.context.headers["authorization"] == MASKED_AUTH
    assert normalized.context.payload["email"] == MASKED_PII
    assert normalized.context.client_ip == masked.context.client_ip


def test_normalize_preserves_verified_rule_information_unchanged() -> None:
    fused = _fused(payload={"amount": 1})

    normalized = Normalizer().normalize(fused)

    assert normalized.rule_info == fused.rule_info
    assert normalized.rule_info.rules[0].rule_id == "R-1"
    assert normalized.rule_info.rules[0].attributes == {"field": "amount"}


def test_normalize_does_not_mutate_the_original_fused_input() -> None:
    fused = _fused(
        method=" post ",
        path=" /orders//items/ ",
        headers={"Content-Type": "Application/JSON"},
        payload={"amount": 5, "nested": {"qty": 2}},
    )
    before = fused.model_dump()

    Normalizer().normalize(fused)

    assert fused.model_dump() == before


def test_normalize_is_deterministic_for_the_same_input() -> None:
    fused = _fused(
        method=" post ",
        path=" /orders//items/ ",
        headers={"Content-Type": "Application/JSON", "Accept": "*/*"},
        payload={"b": 2, "a": {"z": 1, "y": [3, 2, 1]}},
    )

    first = Normalizer().normalize(fused)
    second = Normalizer().normalize(fused)

    assert first.model_dump() == second.model_dump()


def test_normalize_produces_the_same_result_regardless_of_payload_key_order() -> None:
    fused_a = _fused(payload={"amount": 5, "currency": "USD"})
    fused_b = _fused(payload={"currency": "USD", "amount": 5})

    normalized_a = Normalizer().normalize(fused_a)
    normalized_b = Normalizer().normalize(fused_b)

    assert list(normalized_a.context.payload.items()) == list(
        normalized_b.context.payload.items()
    )
