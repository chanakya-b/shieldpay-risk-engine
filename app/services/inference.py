"""
app/services/inference.py
─────────────────────────
Async ML inference engine for ShieldPay's dual-head risk scoring.

Architecture
────────────
• Model artifacts are loaded ONCE at application startup via the FastAPI
  `lifespan` context manager and stored in a module-level `_artifacts`
  singleton. This avoids re-reading ~370 KB pickle files on every request.

• `run_inference()` is an async function that offloads the CPU-bound
  `sklearn` predict_proba calls to a thread pool via `asyncio.to_thread()`.
  This keeps the event loop free and satisfies the sub-50ms hot-path
  execution budget.

• Fail-safe / graceful degradation: any exception during feature engineering
  or model inference is caught, logged with full traceback, and the function
  returns a safe deterministic fallback decision (STEP_UP_OTP_REQUIRED)
  instead of propagating a 500 Internal Server Error.

Decision boundaries (from README Section 5, cost-optimised τ*)
───────────────────────────────────────────────────────────────
  Payment Fraud:
    p < 0.15  → AUTO_APPROVE / DISPATCHED_TO_KITCHEN
    0.15 ≤ p < 0.50 → STEP_UP_OTP_REQUIRED / HOLD_PENDING_VERIFICATION
    p ≥ 0.50  → HARD_CANCEL_TRANSACTION / CANCELLED_FRAUD_PREVENTION

  Refund Abuse:
    p < 0.35  → INSTANT_REFUND_APPROVED
    0.35 ≤ p < 0.65 → REQUIRE_UNBOXING_PHOTO_PROOF
    p ≥ 0.65  → DENY_AUTO_REFUND_ROUTE_TO_AGENT
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from app.config import settings
from app.schemas.payload import (
    Decisions,
    OrderStatus,
    PreFulfillmentAction,
    RefundPolicy,
    RiskScores,
    WebhookRequest,
    WebhookResponse,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Artifact paths — prefer ./models/ directory; fall back to project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _PROJECT_ROOT / "models"
_ROOT_FALLBACK = _PROJECT_ROOT


def _resolve_artifact(filename: str) -> Path:
    """Return the path to a .pkl artifact, checking models/ then root."""
    candidate = _MODEL_DIR / filename
    if candidate.exists():
        return candidate
    fallback = _ROOT_FALLBACK / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Artifact '{filename}' not found in {_MODEL_DIR} or {_ROOT_FALLBACK}"
    )


# ---------------------------------------------------------------------------
# Model artifact container
# ---------------------------------------------------------------------------
_CATEGORICAL_COLS = ["payment_method", "card_network"]


@dataclass
class ModelArtifacts:
    model_fraud: object
    model_abuse: object
    encoder: object


# Module-level singleton — populated at startup
_artifacts: Optional[ModelArtifacts] = None


def load_artifacts() -> None:
    """
    Load all pre-trained model artifacts from disk.

    Called once during application startup (FastAPI lifespan).
    Raises on failure so the process exits cleanly rather than serving
    broken predictions.
    """
    global _artifacts

    logger.info("Loading ShieldPay model artifacts from disk…")
    model_fraud = joblib.load(_resolve_artifact("model_fraud.pkl"))
    model_abuse = joblib.load(_resolve_artifact("model_abuse.pkl"))
    encoder = joblib.load(_resolve_artifact("encoder.pkl"))

    _artifacts = ModelArtifacts(
        model_fraud=model_fraud,
        model_abuse=model_abuse,
        encoder=encoder,
    )
    logger.info("ShieldPay Risk Engine initialised successfully.")


def _get_artifacts() -> ModelArtifacts:
    if _artifacts is None:
        raise RuntimeError(
            "Model artifacts have not been loaded. "
            "Ensure load_artifacts() was called during application startup."
        )
    return _artifacts


# ---------------------------------------------------------------------------
# Decision routing helpers
# ---------------------------------------------------------------------------


def _route_payment_fraud(
    p_fraud: float,
) -> tuple[PreFulfillmentAction, OrderStatus]:
    if p_fraud >= settings.fraud_threshold_hard_block:
        return (
            PreFulfillmentAction.HARD_CANCEL_TRANSACTION,
            OrderStatus.CANCELLED_FRAUD_PREVENTION,
        )
    if p_fraud >= settings.fraud_threshold_step_up:
        return (
            PreFulfillmentAction.STEP_UP_OTP_REQUIRED,
            OrderStatus.HOLD_PENDING_VERIFICATION,
        )
    return (
        PreFulfillmentAction.AUTO_APPROVE,
        OrderStatus.DISPATCHED_TO_KITCHEN,
    )


def _route_refund_abuse(p_abuse: float) -> RefundPolicy:
    if p_abuse >= settings.abuse_threshold_deny:
        return RefundPolicy.DENY_AUTO_REFUND_ROUTE_TO_AGENT
    if p_abuse >= settings.abuse_threshold_photo_proof:
        return RefundPolicy.REQUIRE_UNBOXING_PHOTO_PROOF
    return RefundPolicy.INSTANT_REFUND_APPROVED


# ---------------------------------------------------------------------------
# Core sync inference (runs in thread pool — must be pure, no coroutines)
# ---------------------------------------------------------------------------


def _sync_score(payload: WebhookRequest) -> WebhookResponse:
    """
    CPU-bound feature engineering + dual-head ML inference.

    This function is intentionally synchronous because it runs inside
    asyncio.to_thread(). Do NOT add await or async constructs here.
    """
    arts = _get_artifacts()

    input_df = pd.DataFrame(
        [
            {
                "amount_inr": payload.amount_inr,
                "payment_method": payload.payment_method,
                "card_network": payload.card_network,
                "is_promo_applied": payload.is_promo_applied,
                "account_age_days": payload.account_age_days,
                "past_order_count": payload.past_order_count,
                "past_refund_ratio": payload.past_refund_ratio,
                "orders_in_last_30mins": payload.orders_in_last_30mins,
                "device_account_count": payload.device_account_count,
                "ip_to_delivery_dist_km": payload.ip_to_delivery_dist_km,
            }
        ]
    )

    # Ordinal-encode categorical columns
    input_df[_CATEGORICAL_COLS] = arts.encoder.transform(
        input_df[_CATEGORICAL_COLS]
    )

    # Dual-head inference
    p_fraud = float(arts.model_fraud.predict_proba(input_df)[0][1])
    p_abuse = float(arts.model_abuse.predict_proba(input_df)[0][1])

    pre_action, order_status = _route_payment_fraud(p_fraud)
    refund_policy = _route_refund_abuse(p_abuse)

    return WebhookResponse(
        payment_id=payload.payment_id,
        status="SCORED",
        risk_scores=RiskScores(
            p_payment_fraud=round(p_fraud, 4),
            p_refund_abuse=round(p_abuse, 4),
            inference_source="model",
        ),
        decisions=Decisions(
            pre_fulfillment_action=pre_action,
            order_status=order_status,
            post_delivery_refund_policy=refund_policy,
        ),
    )


# ---------------------------------------------------------------------------
# Safe fallback response
# ---------------------------------------------------------------------------


def _build_fallback_response(payment_id: str) -> WebhookResponse:
    """Build a deterministic, safe fallback response when ML inference fails."""
    return WebhookResponse(
        payment_id=payment_id,
        status="SCORED_FALLBACK",
        risk_scores=RiskScores(
            p_payment_fraud=None,   # sentinel: signals degraded mode
            p_refund_abuse=None,
            inference_source="fallback",
        ),
        decisions=Decisions(
            pre_fulfillment_action=PreFulfillmentAction.STEP_UP_OTP_REQUIRED,
            order_status=OrderStatus.FALLBACK_SAFE_HOLD,
            post_delivery_refund_policy=RefundPolicy.FALLBACK_REQUIRE_PHOTO_PROOF,
        ),
    )


# ---------------------------------------------------------------------------
# Public async entrypoint
# ---------------------------------------------------------------------------


async def run_inference(payload: WebhookRequest) -> WebhookResponse:
    """
    Async wrapper: offloads CPU-bound inference to a thread pool.

    Guarantees:
      - Never raises an unhandled exception to the caller.
      - Returns a safe fallback if inference fails for any reason.
      - Logs the full exception trace for post-mortem analysis.
    """
    try:
        result: WebhookResponse = await asyncio.to_thread(_sync_score, payload)
        logger.info(
            "inference_success",
            extra={
                "payment_id": payload.payment_id,
                "p_fraud": result.risk_scores.p_payment_fraud,
                "p_abuse": result.risk_scores.p_refund_abuse,
                "pre_action": result.decisions.pre_fulfillment_action,
            },
        )
        return result
    except Exception:
        logger.exception(
            "inference_failed — activating safe fallback",
            extra={"payment_id": payload.payment_id},
        )
        return _build_fallback_response(payload.payment_id)
