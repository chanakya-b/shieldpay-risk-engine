"""
app/services/evidence.py
────────────────────────
Async chargeback dossier generator for ShieldPay.

Migrated from the root-level evidence_generator.py module into the
service layer. The core logic is unchanged; wrapping it in an async
function allows the router to await it without blocking the event loop
(the function is I/O-bound in production when it hits a real datastore).

In this reference implementation, all data is mock/static. Replace the
body of `_fetch_transaction_telemetry()` with actual DB / cache lookups
when integrating with a live backend.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal data fetcher (stub — replace with real DB calls in production)
# ---------------------------------------------------------------------------


def _fetch_transaction_telemetry(payment_id: str) -> dict:
    """
    Retrieve raw transaction telemetry for the given payment ID.

    Production: query your PostgreSQL / Redis store here.
    Stub: returns hardcoded mock data keyed on payment_id.
    """
    return {
        "ip_address": "152.57.12.4",
        "ip_isp": "Airtel Broadband",
        "ip_geo_city": "Mumbai, MH",
        "device_fingerprint_hash": "a8f9c211e0df92b4",
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X)",
        "transaction_timestamp": "2026-08-31T13:42:10Z",
        "two_factor_auth_passed": True,
        "otp_method": "SMS_TO_REGISTERED_MOBILE",
        "otp_delivery_timestamp": "2026-08-31T13:41:55Z",
        "otp_verification_timestamp": "2026-08-31T13:42:08Z",
        "customer_phone": "+919876543210",
        "customer_email": "user@example.com",
        "delivery_status": "DELIVERED",
        "delivery_timestamp": "2026-08-31T14:05:22Z",
        "delivery_partner_id": "DP_88392",
        "gps_latitude": 19.0760,
        "gps_longitude": 72.8777,
        "address_match": True,
        "distance_to_dropoff_meters": 4.2,
        "account_created_date": "2024-03-15",
        "total_prior_successful_orders": 42,
        "total_prior_chargebacks": 0,
    }


# ---------------------------------------------------------------------------
# Sync dossier builder
# ---------------------------------------------------------------------------


def _build_dossier(payment_id: str) -> dict:
    telemetry = _fetch_transaction_telemetry(payment_id)

    return {
        "dispute_header": {
            "dispute_id": f"disp_{payment_id.replace('pay_', '')}",
            "razorpay_payment_id": payment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "merchant": "Zomato / Blinkit",
            "dispute_reason": "UNAUTHORIZED_TRANSACTION_FRAUD",
        },
        "evidence_summary": {
            "digital_footprint": {
                "ip_address": telemetry["ip_address"],
                "ip_isp": telemetry["ip_isp"],
                "ip_geo_city": telemetry["ip_geo_city"],
                "device_fingerprint_hash": telemetry["device_fingerprint_hash"],
                "user_agent": telemetry["user_agent"],
                "transaction_timestamp": telemetry["transaction_timestamp"],
            },
            "authentication_logs": {
                "two_factor_auth_passed": telemetry["two_factor_auth_passed"],
                "otp_method": telemetry["otp_method"],
                "otp_delivery_timestamp": telemetry["otp_delivery_timestamp"],
                "otp_verification_timestamp": telemetry["otp_verification_timestamp"],
                "phone_number_verified": telemetry["customer_phone"],
            },
            "fulfillment_proof": {
                "delivery_status": telemetry["delivery_status"],
                "delivery_timestamp": telemetry["delivery_timestamp"],
                "delivery_partner_id": telemetry["delivery_partner_id"],
                "gps_coordinates": {
                    "latitude": telemetry["gps_latitude"],
                    "longitude": telemetry["gps_longitude"],
                    "address_match": telemetry["address_match"],
                    "distance_to_dropoff_meters": telemetry[
                        "distance_to_dropoff_meters"
                    ],
                },
                "delivery_photo_url": (
                    f"https://cdn.zomato.com/proofs/{payment_id}_drop.jpg"
                ),
            },
            "merchant_account_history": {
                "account_created_date": telemetry["account_created_date"],
                "total_prior_successful_orders": telemetry[
                    "total_prior_successful_orders"
                ],
                "total_prior_chargebacks": telemetry["total_prior_chargebacks"],
                "customer_email": telemetry["customer_email"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Public async entrypoint
# ---------------------------------------------------------------------------


async def generate_chargeback_dossier(payment_id: str) -> dict:
    """
    Async chargeback evidence dossier generator.

    Offloads the synchronous data fetch + assembly to a thread pool,
    keeping the event loop unblocked for other concurrent requests.
    """
    logger.info("Generating chargeback dossier", extra={"payment_id": payment_id})
    dossier: dict = await asyncio.to_thread(_build_dossier, payment_id)
    logger.info(
        "Chargeback dossier ready",
        extra={
            "payment_id": payment_id,
            "dispute_id": dossier["dispute_header"]["dispute_id"],
        },
    )
    return dossier
