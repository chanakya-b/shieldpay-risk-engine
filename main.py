from typing import Optional
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

from evidence_generator import generate_chargeback_dossier

app = FastAPI(
    title="ShieldPay for Zomato - Real-Time Risk Engine", version="1.0.0"
)

# ---------------------------------------------------------
# 1. LOAD PRE-TRAINED ARTIFACTS FROM DISK
# ---------------------------------------------------------
print("Loading model artifacts from disk...")
try:
  model_fraud = joblib.load("model_fraud.pkl")
  model_abuse = joblib.load("model_abuse.pkl")
  encoder = joblib.load("encoder.pkl")
  print("ShieldPay Risk Engine Initialized Successfully!")
except Exception as e:
  print(f"Error loading model artifacts: {e}")
  print(
      "Please ensure eval_metrics.py has been executed and .pkl files exist in"
      " directory."
  )

categorical_cols = ["payment_method", "card_network"]


# ---------------------------------------------------------
# 2. INPUT PAYLOAD SCHEMA
# ---------------------------------------------------------
class RazorpayPayload(BaseModel):
  payment_id: str
  amount_inr: float
  payment_method: str
  card_network: Optional[str] = "none"
  is_promo_applied: int
  account_age_days: int
  past_order_count: int
  past_refund_ratio: float
  orders_in_last_30mins: int
  device_account_count: int
  ip_to_delivery_dist_km: float


# ---------------------------------------------------------
# 3. ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def health_check():
  return {
      "status": "online",
      "service": "ShieldPay Zomato Risk Engine",
      "version": "1.0.0",
  }


@app.post("/api/v1/score-webhook")
def score_razorpay_webhook(payload: RazorpayPayload):
  """Ingests Razorpay Webhook + Zomato User Context and outputs tiered risk decisions for pre-fulfillment shipping and post-delivery refunds."""
  # Build DataFrame from incoming JSON payload
  input_data = pd.DataFrame([{
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
  }])

  # Transform categorical columns using loaded encoder
  input_data[categorical_cols] = encoder.transform(
      input_data[categorical_cols]
  )

  # Run inference across both models
  p_fraud = float(model_fraud.predict_proba(input_data)[0][1])
  p_abuse = float(model_abuse.predict_proba(input_data)[0][1])

  # Pre-Fulfillment Action Decision
  if p_fraud >= 0.50:
    pre_fulfillment_action = "REJECT_AND_REFUND"
    order_status = "CANCELLED_FRAUD_PREVENTION"
  elif p_fraud >= 0.15:
    pre_fulfillment_action = "STEP_UP_OTP_REQUIRED"
    order_status = "HOLD_PENDING_VERIFICATION"
  else:
    pre_fulfillment_action = "AUTO_APPROVE"
    order_status = "DISPATCHED_TO_KITCHEN"

  # Post-Delivery Refund Policy Decision
  if p_abuse >= 0.50:
    refund_policy = "DENY_AUTO_REFUND_ROUTE_TO_AGENT"
  elif p_abuse >= 0.25:
    refund_policy = "REQUIRE_UNBOXING_PHOTO_PROOF"
  else:
    refund_policy = "INSTANT_TRUSTED_REFUND"

  return {
      "payment_id": payload.payment_id,
      "risk_scores": {
          "p_payment_fraud": round(p_fraud, 4),
          "p_refund_abuse": round(p_abuse, 4),
      },
      "decisions": {
          "pre_fulfillment_action": pre_fulfillment_action,
          "order_status": order_status,
          "post_delivery_refund_policy": refund_policy,
      },
  }


@app.get("/api/v1/generate-dispute-dossier/{payment_id}")
def get_dispute_dossier(payment_id: str):
  """Generates an automated chargeback defense evidence package for Razorpay Dispute API."""
  dossier = generate_chargeback_dossier(payment_id)
  return dossier