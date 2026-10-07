"""Unit tests for privacy-aware masking."""

from tokenization_ai.masking import MASKED_AUTH, MASKED_IP, MASKED_PII, MASKED_SECRET, PrivacyMasker
from tokenization_ai.models import FusedInput, UnifiedApiContext, VerifiedRule, VerifiedRuleInfo


def _fused(**context_overrides: object) -> FusedInput:
    fields: dict[str, object] = {"event_id": "e1", "method": "POST", "path": "/orders"}
    fields.update(context_overrides)
    context = UnifiedApiContext(**fields)
    rule_info = VerifiedRuleInfo(
        event_id="e1",
        rules=[VerifiedRule(rule_id="R-1", attributes={"field": "amount"})],
    )
    return FusedInput(context=context, rule_info=rule_info)


def test_mask_redacts_authorization_and_cookie_headers() -> None:
    fused = _fused(
        headers={
            "Authorization": "Bearer secret-value",
            "Cookie": "session=abc123",
            "X-Api-Key": "abcd1234",
            "Accept": "application/json",
        }
    )

    masked = PrivacyMasker().mask(fused)

    assert masked.context.headers["Authorization"] == MASKED_AUTH
    assert masked.context.headers["Cookie"] == MASKED_AUTH
    assert masked.context.headers["X-Api-Key"] == MASKED_AUTH
    assert masked.context.headers["Accept"] == "application/json"


def test_mask_preserves_structure_and_masks_nested_payload_fields() -> None:
    fused = _fused(
        payload={
            "amount": 149.95,
            "customer": {
                "email": "customer@example.com",
                "password": "do-not-send-me",
                "profile": {"phone": "+1 415 555 0100", "nickname": "shopper"},
            },
            "cards": [
                {"card_number": "4111 1111 1111 1111", "last_used": "2024-01-01"},
                {"card_number": "5500000000000004", "last_used": "2024-02-02"},
            ],
        }
    )

    masked = PrivacyMasker().mask(fused)
    payload = masked.context.payload

    assert set(payload.keys()) == {"amount", "customer", "cards"}
    assert payload["amount"] == 149.95
    assert payload["customer"]["email"] == MASKED_PII
    assert payload["customer"]["password"] == MASKED_SECRET
    assert payload["customer"]["profile"]["phone"] == MASKED_PII
    assert payload["customer"]["profile"]["nickname"] == "shopper"
    assert [card["card_number"] for card in payload["cards"]] == [MASKED_PII, MASKED_PII]
    assert [card["last_used"] for card in payload["cards"]] == ["2024-01-01", "2024-02-02"]


def test_mask_detects_pii_by_value_even_without_a_sensitive_key_name() -> None:
    fused = _fused(payload={"contact": "jane.doe@example.com", "note": "call me"})

    masked = PrivacyMasker().mask(fused)

    assert masked.context.payload["contact"] == MASKED_PII
    assert masked.context.payload["note"] == "call me"


def test_mask_redacts_client_ip() -> None:
    fused = _fused(client_ip="203.0.113.5")

    masked = PrivacyMasker().mask(fused)

    assert masked.context.client_ip == MASKED_IP


def test_mask_leaves_absent_client_ip_as_none() -> None:
    fused = _fused(client_ip=None)

    masked = PrivacyMasker().mask(fused)

    assert masked.context.client_ip is None


def test_mask_leaves_non_sensitive_fields_unchanged() -> None:
    fused = _fused(
        method="POST",
        path="/orders",
        content_type="application/json",
        payload={"amount": 149.95, "currency": "USD", "quantity": 2},
    )

    masked = PrivacyMasker().mask(fused)

    assert masked.context.method == "POST"
    assert masked.context.path == "/orders"
    assert masked.context.content_type == "application/json"
    assert masked.context.payload == {"amount": 149.95, "currency": "USD", "quantity": 2}


def test_mask_does_not_mutate_the_original_fused_input() -> None:
    fused = _fused(
        client_ip="203.0.113.5",
        headers={"Authorization": "Bearer secret-value"},
        payload={"email": "customer@example.com", "amount": 1.0},
    )
    before = fused.model_dump()

    PrivacyMasker().mask(fused)

    assert fused.model_dump() == before


def test_mask_preserves_verified_rule_information_unchanged() -> None:
    fused = _fused(payload={"email": "customer@example.com"})

    masked = PrivacyMasker().mask(fused)

    assert masked.rule_info == fused.rule_info
    assert masked.rule_info.rules[0].rule_id == "R-1"
    assert masked.rule_info.rules[0].attributes == {"field": "amount"}


def test_mask_preserves_event_id_across_context_and_rule_info() -> None:
    fused = _fused(client_ip="203.0.113.5")

    masked = PrivacyMasker().mask(fused)

    assert masked.event_id == "e1"
    assert masked.context.event_id == "e1"
    assert masked.rule_info.event_id == "e1"
