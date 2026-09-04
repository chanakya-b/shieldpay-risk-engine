"""
ShieldPay — Interactive Risk Operations Portal & Real-Time Demo
─────────────────────────────────────────────────────────────
Run with:
  streamlit run demo_app.py
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import streamlit as st

# Direct import fallback if local server is not running
from app.schemas.payload import WebhookRequest
from app.services.evidence import generate_chargeback_dossier
from app.services.inference import load_artifacts, run_inference

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ShieldPay — Real-Time Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .verdict-approved {
        background-color: #064e3b;
        color: #34d399;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        border: 1px solid #059669;
    }
    .verdict-stepup {
        background-color: #78350f;
        color: #fbbf24;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        border: 1px solid #d97706;
    }
    .verdict-blocked {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        border: 1px solid #dc2626;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize ML artifacts for fallback direct inference
@st.cache_resource
def init_artifacts() -> bool:
    try:
        load_artifacts()
        return True
    except Exception:
        return False


init_artifacts()

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.title("🛡️ ShieldPay — Dual-Head Real-Time Risk Engine")
st.caption(
    "Enterprise Fintech Risk Management Intercepting Razorpay Webhooks for Zomato & Blinkit Rails"
)

st.markdown(
    "![SLA Sub-50ms](https://img.shields.io/badge/SLA-Sub--50ms%20Hot%20Path-brightgreen.svg) "
    "![Dual-Head ML](https://img.shields.io/badge/Architecture-Dual--Head%20ML-blue.svg) "
    "![Cost Loss Optimal](https://img.shields.io/badge/Loss%20Minimization-%CF%84*%20Optimal-purple.svg) "
    "![Pydantic v2](https://img.shields.io/badge/Contract-Pydantic%20v2-teal.svg)"
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------------------------
st.sidebar.header("🕹️ Simulation Controls")

st.sidebar.subheader("💳 Gateway Parameters")
payment_id = st.sidebar.text_input("Razorpay Payment ID", value="pay_Nz9K83jL01aQ")
amount_inr = st.sidebar.number_input(
    "Amount (₹)", min_value=1.0, max_value=500000.0, value=1299.0, step=50.0
)
payment_method = st.sidebar.selectbox(
    "Payment Method", ["credit_card", "upi", "netbanking"]
)
card_network = (
    st.sidebar.selectbox("Card Network", ["visa", "mastercard", "rupay", "amex"])
    if payment_method == "credit_card"
    else "none"
)
promo_choice = st.sidebar.selectbox("Promo Code Applied", ["No", "Yes"], index=0)
is_promo_applied = 1 if promo_choice == "Yes" else 0

st.sidebar.markdown("---")
st.sidebar.subheader("👤 User Account Context")
account_age_days = st.sidebar.number_input(
    "Account Age (Days)", min_value=0, value=14
)
past_order_count = st.sidebar.number_input(
    "Past Orders", min_value=0, value=5
)
past_refund_ratio = st.sidebar.slider(
    "Past Refund Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Velocity & Hardware Telemetry")
orders_in_last_30mins = st.sidebar.number_input(
    "Orders (Last 30 mins)", min_value=0, value=2
)
device_account_count = st.sidebar.number_input(
    "Accounts on Device", min_value=1, value=1
)
ip_to_delivery_dist_km = st.sidebar.number_input(
    "IP to Delivery GPS (km)", min_value=0.0, value=3.5, step=0.5
)

st.sidebar.markdown("---")
api_mode = st.sidebar.radio(
    "Inference Connection Mode",
    ["Local REST API (http://localhost:8000)", "Direct Python Core"],
)

# ---------------------------------------------------------------------------
# Construct Payload
# ---------------------------------------------------------------------------
payload_dict = {
    "payment_id": payment_id,
    "amount_inr": amount_inr,
    "payment_method": payment_method,
    "card_network": card_network,
    "is_promo_applied": is_promo_applied,
    "account_age_days": account_age_days,
    "past_order_count": past_order_count,
    "past_refund_ratio": past_refund_ratio,
    "orders_in_last_30mins": orders_in_last_30mins,
    "device_account_count": device_account_count,
    "ip_to_delivery_dist_km": ip_to_delivery_dist_km,
}

# ---------------------------------------------------------------------------
# Main Execution Tabs
# ---------------------------------------------------------------------------
tab_score, tab_dossier, tab_contract = st.tabs(
    [
        "⚡ Real-Time Risk Scoring",
        "📄 Chargeback Dispute Dossier",
        "📐 Data Contract & API Request",
    ]
)

with tab_score:
    if st.button("🚀 Score Transaction Webhook", use_container_width=True):
        start_time = time.perf_counter()

        response_data: dict[str, Any] = {}
        exec_time_ms = 0.0
        request_id = "demo-trace-uuid-12345"

        if "REST API" in api_mode:
            try:
                resp = httpx.post(
                    "http://localhost:8000/api/v1/score-webhook",
                    json=payload_dict,
                    timeout=2.0,
                )
                exec_time_ms = float(
                    resp.headers.get(
                        "x-execution-time-ms",
                        (time.perf_counter() - start_time) * 1000,
                    )
                )
                request_id = resp.headers.get("x-request-id", request_id)
                response_data = resp.json()
            except Exception as e:
                st.warning(
                    f"Local API Server offline at localhost:8000 ({e}). Falling back to Direct Python Core."
                )
                req = WebhookRequest(**payload_dict)
                res = asyncio.run(run_inference(req))
                exec_time_ms = (time.perf_counter() - start_time) * 1000
                response_data = res.model_dump()
        else:
            req = WebhookRequest(**payload_dict)
            res = asyncio.run(run_inference(req))
            exec_time_ms = (time.perf_counter() - start_time) * 1000
            response_data = res.model_dump()

        # Render Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Execution Latency", f"{exec_time_ms:.2f} ms", delta="< 50ms SLA Target")
        m2.metric(
            "Inference Status",
            response_data.get("status", "SCORED"),
            delta=(
                "Model Active"
                if response_data.get("status") == "SCORED"
                else "Fallback Safe Hold"
            ),
        )

        p_fraud = response_data.get("risk_scores", {}).get("p_payment_fraud")
        p_abuse = response_data.get("risk_scores", {}).get("p_refund_abuse")

        m3.metric(
            "P(Payment Fraud)",
            f"{p_fraud:.4f}" if p_fraud is not None else "N/A",
            delta="Optimal Threshold τ* = 0.15",
        )
        m4.metric(
            "P(Refund Abuse)",
            f"{p_abuse:.4f}" if p_abuse is not None else "N/A",
            delta="Optimal Threshold τ* = 0.35",
        )

        st.markdown("---")
        st.subheader("🎯 Operational Decision Verdicts")

        decisions = response_data.get("decisions", {})
        pre_act = decisions.get("pre_fulfillment_action", "N/A")
        ord_stat = decisions.get("order_status", "N/A")
        ref_pol = decisions.get("post_delivery_refund_policy", "N/A")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Pre-Fulfillment Action**")
            if pre_act == "AUTO_APPROVE":
                st.markdown(
                    f'<div class="verdict-approved">✅ {pre_act}</div>',
                    unsafe_allow_html=True,
                )
            elif pre_act == "STEP_UP_OTP_REQUIRED":
                st.markdown(
                    f'<div class="verdict-stepup">⚠️ {pre_act}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict-blocked">⛔ {pre_act}</div>',
                    unsafe_allow_html=True,
                )

        with c2:
            st.markdown("**Kitchen / Fulfillment Status**")
            st.info(f"📋 `{ord_stat}`")

        with c3:
            st.markdown("**Post-Delivery Refund Policy**")
            st.warning(f"🏷️ `{ref_pol}`")

        st.markdown("---")
        st.subheader("🔍 Structured Response Inspector")
        st.json(response_data)

with tab_dossier:
    st.subheader("🛡️ Automated Chargeback Evidence Dossier")
    st.caption("Generated instantly for submission to Razorpay's Dispute API")

    if st.button("📄 Generate Dispute Evidence Dossier"):
        with st.spinner("Compiling digital footprint, 2FA logs, and GPS delivery proof..."):
            dossier_data = asyncio.run(generate_chargeback_dossier(payment_id))

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.success(
                f"**Dispute ID:** `{dossier_data['dispute_header']['dispute_id']}`"
            )
            st.info(
                f"**Reason:** `{dossier_data['dispute_header']['dispute_reason']}`"
            )
        with col_h2:
            st.write(
                f"**Timestamp:** `{dossier_data['dispute_header']['generated_at']}`"
            )
            st.write(
                f"**Merchant:** `{dossier_data['dispute_header']['merchant']}`"
            )

        st.json(dossier_data)

with tab_contract:
    st.subheader("📐 Raw Webhook Request Payload")
    st.code(json.dumps(payload_dict, indent=2), language="json")
