"""
app/schemas/payload.py
──────────────────────
Pydantic v2 request and response schemas for ShieldPay.

Design principles
-----------------
- ConfigDict(strict=True) on all models — no implicit coercion (e.g., "1" → 1).
- Field-level validators enforce domain constraints (amounts > 0, ratios in [0,1]).
- Enums for all decision string literals — prevents typo drift between service
  layer and API contract.
- Separate nested models for `risk_scores` and `decisions` — keeps the response
  contract extensible without breaking existing consumers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Decision Enums — single source of truth for all action literals
# ---------------------------------------------------------------------------


class PreFulfillmentAction(StrEnum):
    AUTO_APPROVE = "AUTO_APPROVE"
    STEP_UP_OTP_REQUIRED = "STEP_UP_OTP_REQUIRED"
    HARD_CANCEL_TRANSACTION = "HARD_CANCEL_TRANSACTION"


class OrderStatus(StrEnum):
    DISPATCHED_TO_KITCHEN = "DISPATCHED_TO_KITCHEN"
    HOLD_PENDING_VERIFICATION = "HOLD_PENDING_VERIFICATION"
    CANCELLED_FRAUD_PREVENTION = "CANCELLED_FRAUD_PREVENTION"
    FALLBACK_SAFE_HOLD = "FALLBACK_SAFE_HOLD"


class RefundPolicy(StrEnum):
    INSTANT_REFUND_APPROVED = "INSTANT_REFUND_APPROVED"
    REQUIRE_UNBOXING_PHOTO_PROOF = "REQUIRE_UNBOXING_PHOTO_PROOF"
    DENY_AUTO_REFUND_ROUTE_TO_AGENT = "DENY_AUTO_REFUND_ROUTE_TO_AGENT"
    FALLBACK_REQUIRE_PHOTO_PROOF = "FALLBACK_REQUIRE_PHOTO_PROOF"


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------


class WebhookRequest(BaseModel):
    """
    Razorpay Webhook + Zomato merchant context payload.

    All fields are required; card_network defaults to "none" for non-card
    payment methods (UPI, netbanking, etc.).
    """

    model_config = ConfigDict(strict=True, frozen=True)

    payment_id: Annotated[
        str,
        Field(
            min_length=8,
            max_length=64,
            pattern=r"^[a-zA-Z0-9_\-]+$",
            description="Razorpay payment identifier (e.g. pay_Nz9K83jL01aQ)",
            examples=["pay_Nz9K83jL01aQ"],
        ),
    ]

    amount_inr: Annotated[
        float,
        Field(
            gt=0.0,
            le=500_000.0,
            description="Transaction amount in Indian Rupees (₹)",
            examples=[1299.0],
        ),
    ]

    payment_method: Annotated[
        str,
        Field(
            min_length=2,
            max_length=32,
            description="Payment method slug (credit_card, upi, netbanking, …)",
            examples=["credit_card"],
        ),
    ]

    card_network: Annotated[
        str,
        Field(
            min_length=2,
            max_length=32,
            description="Card network (visa, mastercard, rupay, none, …)",
            examples=["visa"],
        ),
    ] = "none"

    is_promo_applied: Annotated[
        int,
        Field(
            ge=0,
            le=1,
            description="1 if a discount / promo code was applied, else 0",
            examples=[1],
        ),
    ]

    account_age_days: Annotated[
        int,
        Field(
            ge=0,
            description="Number of days since the user account was created",
            examples=[14],
        ),
    ]

    past_order_count: Annotated[
        int,
        Field(
            ge=0,
            description="Total historical completed orders on the platform",
            examples=[5],
        ),
    ]

    past_refund_ratio: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Fraction of past orders that resulted in a refund",
            examples=[0.2],
        ),
    ]

    orders_in_last_30mins: Annotated[
        int,
        Field(
            ge=0,
            description="Number of orders placed in the last 30-minute rolling window",
            examples=[2],
        ),
    ]

    device_account_count: Annotated[
        int,
        Field(
            ge=1,
            description="Number of distinct accounts linked to this device fingerprint",
            examples=[1],
        ),
    ]

    ip_to_delivery_dist_km: Annotated[
        float,
        Field(
            ge=0.0,
            description="Geodesic distance (km) between IP geolocation and delivery address",
            examples=[3.5],
        ),
    ]

    @field_validator("payment_id")
    @classmethod
    def _validate_payment_id_prefix(cls, v: str) -> str:
        """Razorpay payment IDs conventionally start with 'pay_'."""
        if not v.startswith("pay_"):
            raise ValueError(
                f"payment_id must begin with 'pay_'. Received: '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# Response Sub-Models
# ---------------------------------------------------------------------------


class RiskScores(BaseModel):
    """Dual-head probability scores from ML inference."""

    model_config = ConfigDict(frozen=True)

    p_payment_fraud: Annotated[
        Optional[float],
        Field(
            ge=0.0,
            le=1.0,
            description="Probability of pre-fulfillment payment fraud (None in fallback mode)",
        ),
    ] = None
    p_refund_abuse: Annotated[
        Optional[float],
        Field(
            ge=0.0,
            le=1.0,
            description="Probability of post-delivery refund abuse (None in fallback mode)",
        ),
    ] = None
    inference_source: Annotated[
        str,
        Field(
            description="'model' if ML ran successfully, 'fallback' on safe degradation",
        ),
    ] = "model"


class Decisions(BaseModel):
    """Operational decision outputs derived from risk scores."""

    model_config = ConfigDict(frozen=True)

    pre_fulfillment_action: PreFulfillmentAction
    order_status: OrderStatus
    post_delivery_refund_policy: RefundPolicy


class WebhookResponse(BaseModel):
    """Full scoring response returned by POST /api/v1/score-webhook."""

    model_config = ConfigDict(frozen=True)

    payment_id: str
    status: str = Field(
        default="SCORED",
        description="'SCORED' for successful inference, 'SCORED_FALLBACK' for safe degradation",
    )
    risk_scores: RiskScores
    decisions: Decisions


# ---------------------------------------------------------------------------
# Dossier Response (pass-through wrapper)
# ---------------------------------------------------------------------------


class DossierResponse(BaseModel):
    """Chargeback evidence dossier response envelope."""

    model_config = ConfigDict(frozen=False)

    dispute_header: dict
    evidence_summary: dict
