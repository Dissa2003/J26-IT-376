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


# -- mask_with_provenance: trusted masking provenance (security) ---------


def test_mask_is_backward_compatible_with_mask_with_provenance() -> None:
    fused = _fused(
        headers={"Authorization": "Bearer secret"},
        payload={"email": "a@b.com"},
        client_ip="203.0.113.5",
    )

    assert PrivacyMasker().mask(fused) == PrivacyMasker().mask_with_provenance(fused).fused


def test_mask_with_provenance_marks_genuine_email_as_masked() -> None:
    fused = _fused(payload={"email": "customer@example.com"})

    result = PrivacyMasker().mask_with_provenance(fused)

    assert result.fused.context.payload["email"] == MASKED_PII
    assert ("payload", "email") in result.masked_paths


def test_mask_with_provenance_marks_genuine_authorization_header_as_masked() -> None:
    fused = _fused(headers={"Authorization": "Bearer secret-value"})

    result = PrivacyMasker().mask_with_provenance(fused)

    assert result.fused.context.headers["Authorization"] == MASKED_AUTH
    assert ("headers", "Authorization") in result.masked_paths


def test_mask_with_provenance_marks_genuine_secret_field_as_masked() -> None:
    fused = _fused(payload={"password": "do-not-send-me"})

    result = PrivacyMasker().mask_with_provenance(fused)

    assert result.fused.context.payload["password"] == MASKED_SECRET
    assert ("payload", "password") in result.masked_paths


def test_mask_with_provenance_marks_genuine_client_ip_as_masked() -> None:
    fused = _fused(client_ip="203.0.113.5")

    result = PrivacyMasker().mask_with_provenance(fused)

    assert result.fused.context.client_ip == MASKED_IP
    assert ("client_ip",) in result.masked_paths


def test_mask_with_provenance_does_not_mark_attacker_lookalike_as_masked() -> None:
    # "nickname" matches no sensitive key name or value pattern, so this
    # field is passed through unchanged -- it must not be reported as
    # genuinely masked even though its text reads like a placeholder.
    fused = _fused(payload={"nickname": "[MASKED_PII]"})

    result = PrivacyMasker().mask_with_provenance(fused)

    assert result.fused.context.payload["nickname"] == "[MASKED_PII]"
    assert ("payload", "nickname") not in result.masked_paths


def test_mask_with_provenance_marks_header_even_if_raw_value_already_equals_its_masked_form() -> None:
    # The Authorization rule fires on the header *name*, independent of its
    # value, so provenance must reflect that the rule fired -- not merely
    # that the output text differs from the input text (here it doesn't).
    fused = _fused(headers={"Authorization": MASKED_AUTH})

    result = PrivacyMasker().mask_with_provenance(fused)

    assert result.fused.context.headers["Authorization"] == MASKED_AUTH
    assert ("headers", "Authorization") in result.masked_paths


def test_mask_with_provenance_tracks_paths_through_nested_objects_and_lists() -> None:
    fused = _fused(
        payload={
            "customer": {"profile": {"phone": "+1 415 555 0100"}},
            "cards": [{"card_number": "4111 1111 1111 1111"}, {"card_number": "not-a-card"}],
        }
    )

    result = PrivacyMasker().mask_with_provenance(fused)

    assert ("payload", "customer", "profile", "phone") in result.masked_paths
    assert ("payload", "cards", 0, "card_number") in result.masked_paths
    # "card_number" is a sensitive key name regardless of value, so index 1
    # is masked too even though its raw text doesn't look like a card.
    assert ("payload", "cards", 1, "card_number") in result.masked_paths


def test_mask_with_provenance_handles_mixed_genuine_and_fake_placeholders() -> None:
    fused = _fused(
        payload={
            "email": "alice@example.com",  # genuinely masked (PII value pattern)
            "nickname": "[MASKED_PII]",  # attacker-controlled lookalike
            "bio": "[MASKED_SECRET]",  # attacker-controlled lookalike
        }
    )

    result = PrivacyMasker().mask_with_provenance(fused)

    assert result.fused.context.payload == {
        "email": MASKED_PII,
        "nickname": "[MASKED_PII]",
        "bio": "[MASKED_SECRET]",
    }
    assert result.masked_paths == frozenset({("payload", "email")})


def test_mask_with_provenance_never_stores_sensitive_values_in_masked_paths() -> None:
    fused = _fused(payload={"email": "customer@example.com"}, client_ip="203.0.113.5")

    result = PrivacyMasker().mask_with_provenance(fused)

    flattened = " ".join(str(part) for path in result.masked_paths for part in path)
    assert "customer@example.com" not in flattened
    assert "203.0.113.5" not in flattened
