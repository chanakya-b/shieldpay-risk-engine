"""
app/api/v1/router.py
────────────────────
API v1 route definitions for ShieldPay.

Route contract
──────────────
POST /api/v1/score-webhook
  - Ingests Razorpay webhook + Zomato merchant context.
  - Delegates to the async inference service.
  - Returns a fully-typed WebhookResponse.
  - Execution budget: < 50 ms P99.

GET /api/v1/generate-dispute-dossier/{payment_id}
  - Generates an automated chargeback evidence package.
  - Returns a DossierResponse.

Both endpoints benefit from the ObservabilityMiddleware which injects
X-Request-ID and X-Execution-Time-MS on every response automatically.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import JSONResponse

from app.schemas.payload import DossierResponse, WebhookRequest, WebhookResponse
from app.services.evidence import generate_chargeback_dossier
from app.services.inference import run_inference
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ShieldPay Risk Engine v1"])


# ---------------------------------------------------------------------------
# POST /api/v1/score-webhook  — HOT PATH (sub-50ms SLA)
# ---------------------------------------------------------------------------


@router.post(
    "/score-webhook",
    response_model=WebhookResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a Razorpay payment webhook",
    description=(
        "Ingests a Razorpay webhook payload enriched with Zomato merchant "
        "context (device telemetry, velocity counters, geo-discrepancy). "
        "Returns tiered risk decisions for pre-fulfillment and post-delivery "
        "refund actions. Execution budget: sub-50ms P99."
    ),
    responses={
        200: {"description": "Risk scoring completed (model or safe fallback)"},
        422: {"description": "Validation error — malformed request body"},
    },
)
async def score_webhook(payload: WebhookRequest) -> WebhookResponse:
    """
    Dual-head ML risk scoring endpoint.

    The handler is intentionally thin: validation is handled by Pydantic,
    inference is delegated to the service layer, and observability headers
    are injected by middleware. No business logic lives here.
    """
    logger.info(
        "score_webhook_received",
        extra={
            "payment_id": payload.payment_id,
            "amount_inr": payload.amount_inr,
            "payment_method": payload.payment_method,
        },
    )
    return await run_inference(payload)


# ---------------------------------------------------------------------------
# GET /api/v1/generate-dispute-dossier/{payment_id}
# ---------------------------------------------------------------------------


@router.get(
    "/generate-dispute-dossier/{payment_id}",
    response_model=DossierResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate an automated chargeback evidence dossier",
    description=(
        "Aggregates transaction logs, 2FA timestamps, GPS delivery proof, "
        "and account history into a structured representment evidence package "
        "ready for submission to Razorpay's Dispute API."
    ),
    responses={
        200: {"description": "Dossier generated successfully"},
        400: {"description": "Invalid payment_id format"},
    },
)
async def get_dispute_dossier(
    payment_id: str = Path(
        ...,
        min_length=8,
        max_length=64,
        pattern=r"^pay_[a-zA-Z0-9]+$",
        description="Razorpay payment ID (e.g. pay_Nz9K83jL01aQ)",
        examples=["pay_Nz9K83jL01aQ"],
    ),
) -> DossierResponse:
    logger.info("dispute_dossier_requested", extra={"payment_id": payment_id})
    dossier = await generate_chargeback_dossier(payment_id)
    return DossierResponse(**dossier)
