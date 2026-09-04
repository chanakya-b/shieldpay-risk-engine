"""
tests/test_schemas.py
─────────────────────
Unit tests for app/schemas/payload.py

Covers:
  - Valid payload acceptance (all field types)
  - payment_id prefix validator
  - amount_inr boundary checks (must be > 0, ≤ 500_000)
  - past_refund_ratio clamping (must be in [0.0, 1.0])
  - device_account_count minimum (≥ 1)
  - Strict mode — no implicit coercion (string "1" → int rejected)
  - Default values (card_network defaults to "none")
  - StrEnum membership for all decision literals
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.payload import (
    Decisions,
    DossierResponse,
    OrderStatus,
    PreFulfillmentAction,
    RefundPolicy,
    RiskScores,
    WebhookRequest,
    WebhookResponse,
)


# ---------------------------------------------------------------------------
# WebhookRequest — valid construction
# ---------------------------------------------------------------------------


class TestWebhookRequestValid:
    def test_minimal_valid_payload(self) -> None:
        req = WebhookRequest(
            payment_id="pay_Abc1234567",
            amount_inr=100.0,
            payment_method="upi",
            card_network="none",
            is_promo_applied=0,
            account_age_days=1,
            past_order_count=0,
            past_refund_ratio=0.0,
            orders_in_last_30mins=0,
            device_account_count=1,
            ip_to_delivery_dist_km=0.0,
        )
        assert req.payment_id == "pay_Abc1234567"

    def test_card_network_defaults_to_none(self) -> None:
        req = WebhookRequest(
            payment_id="pay_DefaultNet1",
            amount_inr=500.0,
            payment_method="netbanking",
            is_promo_applied=0,
            account_age_days=10,
            past_order_count=2,
            past_refund_ratio=0.0,
            orders_in_last_30mins=0,
            device_account_count=1,
            ip_to_delivery_dist_km=5.0,
        )
        assert req.card_network == "none"

    def test_boundary_amount_maximum(self) -> None:
        req = WebhookRequest(
            payment_id="pay_MaxAmount0",
            amount_inr=500_000.0,
            payment_method="credit_card",
            card_network="visa",
            is_promo_applied=0,
            account_age_days=100,
            past_order_count=10,
            past_refund_ratio=0.0,
            orders_in_last_30mins=0,
            device_account_count=1,
            ip_to_delivery_dist_km=2.0,
        )
        assert req.amount_inr == 500_000.0

    def test_refund_ratio_boundary_values(self) -> None:
        for ratio in [0.0, 0.5, 1.0]:
            req = WebhookRequest(
                payment_id="pay_RatioTest1",
                amount_inr=100.0,
                payment_method="upi",
                is_promo_applied=0,
                account_age_days=1,
                past_order_count=0,
                past_refund_ratio=ratio,
                orders_in_last_30mins=0,
                device_account_count=1,
                ip_to_delivery_dist_km=0.0,
            )
            assert req.past_refund_ratio == ratio


# ---------------------------------------------------------------------------
# WebhookRequest — validation failures
# ---------------------------------------------------------------------------


class TestWebhookRequestInvalid:
    def test_payment_id_missing_pay_prefix(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            WebhookRequest(
                payment_id="rzp_NoPrefix1",  # must start with "pay_"
                amount_inr=100.0,
                payment_method="upi",
                is_promo_applied=0,
                account_age_days=1,
                past_order_count=0,
                past_refund_ratio=0.0,
                orders_in_last_30mins=0,
                device_account_count=1,
                ip_to_delivery_dist_km=0.0,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("payment_id",) for e in errors)

    def test_amount_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WebhookRequest(
                payment_id="pay_ZeroAmt00",
                amount_inr=0.0,       # must be > 0
                payment_method="upi",
                is_promo_applied=0,
                account_age_days=1,
                past_order_count=0,
                past_refund_ratio=0.0,
                orders_in_last_30mins=0,
                device_account_count=1,
                ip_to_delivery_dist_km=0.0,
            )

    def test_amount_exceeds_maximum(self) -> None:
        with pytest.raises(ValidationError):
            WebhookRequest(
                payment_id="pay_TooBig000",
                amount_inr=500_001.0,  # exceeds ≤ 500_000 cap
                payment_method="upi",
                is_promo_applied=0,
                account_age_days=1,
                past_order_count=0,
                past_refund_ratio=0.0,
                orders_in_last_30mins=0,
                device_account_count=1,
                ip_to_delivery_dist_km=0.0,
            )

    def test_refund_ratio_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WebhookRequest(
                payment_id="pay_BadRatio0",
                amount_inr=100.0,
                payment_method="upi",
                is_promo_applied=0,
                account_age_days=1,
                past_order_count=0,
                past_refund_ratio=1.01,   # must be ≤ 1.0
                orders_in_last_30mins=0,
                device_account_count=1,
                ip_to_delivery_dist_km=0.0,
            )

    def test_device_account_count_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WebhookRequest(
                payment_id="pay_ZeroDev00",
                amount_inr=100.0,
                payment_method="upi",
                is_promo_applied=0,
                account_age_days=1,
                past_order_count=0,
                past_refund_ratio=0.0,
                orders_in_last_30mins=0,
                device_account_count=0,   # must be ≥ 1
                ip_to_delivery_dist_km=0.0,
            )

    def test_strict_mode_rejects_string_for_int(self) -> None:
        """Strict mode: '1' (str) must not be coerced into is_promo_applied (int)."""
        with pytest.raises(ValidationError):
            WebhookRequest(
                payment_id="pay_StrictMod",
                amount_inr=100.0,
                payment_method="upi",
                is_promo_applied="1",     # type: ignore[arg-type]
                account_age_days=1,
                past_order_count=0,
                past_refund_ratio=0.0,
                orders_in_last_30mins=0,
                device_account_count=1,
                ip_to_delivery_dist_km=0.0,
            )

    def test_negative_account_age_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WebhookRequest(
                payment_id="pay_NegAge000",
                amount_inr=100.0,
                payment_method="upi",
                is_promo_applied=0,
                account_age_days=-1,      # must be ≥ 0
                past_order_count=0,
                past_refund_ratio=0.0,
                orders_in_last_30mins=0,
                device_account_count=1,
                ip_to_delivery_dist_km=0.0,
            )


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestDecisionEnums:
    def test_pre_fulfillment_actions_complete(self) -> None:
        values = {a.value for a in PreFulfillmentAction}
        assert "AUTO_APPROVE" in values
        assert "STEP_UP_OTP_REQUIRED" in values
        assert "HARD_CANCEL_TRANSACTION" in values

    def test_order_statuses_complete(self) -> None:
        values = {s.value for s in OrderStatus}
        assert "DISPATCHED_TO_KITCHEN" in values
        assert "HOLD_PENDING_VERIFICATION" in values
        assert "CANCELLED_FRAUD_PREVENTION" in values
        assert "FALLBACK_SAFE_HOLD" in values

    def test_refund_policies_complete(self) -> None:
        values = {p.value for p in RefundPolicy}
        assert "INSTANT_REFUND_APPROVED" in values
        assert "REQUIRE_UNBOXING_PHOTO_PROOF" in values
        assert "DENY_AUTO_REFUND_ROUTE_TO_AGENT" in values
        assert "FALLBACK_REQUIRE_PHOTO_PROOF" in values


# ---------------------------------------------------------------------------
# Response model construction
# ---------------------------------------------------------------------------


class TestResponseModels:
    def test_webhook_response_roundtrip(self) -> None:
        resp = WebhookResponse(
            payment_id="pay_RoundTrip",
            risk_scores=RiskScores(
                p_payment_fraud=0.25,
                p_refund_abuse=0.80,
                inference_source="model",
            ),
            decisions=Decisions(
                pre_fulfillment_action=PreFulfillmentAction.STEP_UP_OTP_REQUIRED,
                order_status=OrderStatus.HOLD_PENDING_VERIFICATION,
                post_delivery_refund_policy=RefundPolicy.DENY_AUTO_REFUND_ROUTE_TO_AGENT,
            ),
        )
        data = resp.model_dump()
        assert data["payment_id"] == "pay_RoundTrip"
        assert data["risk_scores"]["inference_source"] == "model"
        assert data["decisions"]["pre_fulfillment_action"] == "STEP_UP_OTP_REQUIRED"

    def test_risk_scores_inference_source_defaults_to_model(self) -> None:
        scores = RiskScores(p_payment_fraud=0.1, p_refund_abuse=0.2)
        assert scores.inference_source == "model"
