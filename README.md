# 🛡️ ShieldPay — Dual-Head Real-Time Risk & Fraud Engine

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)
![SLA Sub-50ms](https://img.shields.io/badge/SLA-Sub--50ms%20Hot%20Path-brightgreen.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

> **ShieldPay** is an enterprise-grade, real-time risk engine designed for high-velocity quick-commerce platforms (Zomato, Blinkit, Swiggy) operating over **Razorpay payment rails**. By bridging merchant-side physical device telemetry with gateway webhook payloads, ShieldPay scores both **Pre-Fulfillment Payment Fraud** and **Post-Delivery Refund Abuse** within a strict **sub-50ms execution SLA**.

> [!IMPORTANT]
> ### ⚡ Key Technical Highlights
> - **Sub-50ms Hot Path SLA:** Non-blocking async event loop offloading CPU-bound scikit-learn model inference via `asyncio.to_thread()`.
> - **Dual-Head Decoupled Machine Learning:** Eliminates negative task interference by isolating pre-fulfillment payment fraud (`model_fraud.pkl`) from post-delivery refund abuse (`model_abuse.pkl`).
> - **Cost-Loss Minimization Engine ($\tau^*$):** Replaces naive $0.50$ classification thresholds with asymmetric financial loss optimization, minimizing total loss under $\text{Cost}_{\text{FN}} \gg \text{Cost}_{\text{FP}}$ conditions.
> - **Zero-Touch Chargeback Dossier:** Automated representment evidence packaging (GPS proximity, 2FA logs, device fingerprints) for instant integration with the Razorpay Dispute API.

---

## 🌐 Real-World Threat Landscape & Business Need

Modern quick-commerce operating in India faces distinct financial threat vectors across payment authorization and post-delivery fulfillment. Standard rules engines and generic payment gateways fail to capture cross-domain telemetry, leaving platforms exposed to significant financial leakage.

| Threat Vector | Real-World Case Study / Industry Context | Business Financial Risk | ShieldPay Architectural Defense |
| :--- | :--- | :--- | :--- |
| **Card-Testing & Bot Velocity** | High-frequency authorization attacks using credential-stuffed cards over Razorpay payment APIs. | Severe gateway penalty fees, rate-limit throttling, and card network non-compliance fines. | **Pre-Fulfillment Scoring:** Monitors rolling 30-minute order velocity and device account multiplicity to detect bot clusters before order confirmation. |
| **CNP International Chargebacks** | Card-Not-Present (CNP) 3DS bypass exploitation using foreign-issued credit cards. | $\$15\text{--}\$25$ processor chargeback fee + 100% loss of fulfillment costs and inventory value. | **Geodesic Distance Verification:** Calculates IP-to-delivery address distance ($\text{km}$) combined with card network telemetry to identify proxy spoofing. |
| **Micro-Transaction Cyber Fraud** | Organized digital fraud networks exploiting low-value UPI transactions (RBI stats: $\sim\text{₹}22,000\text{ Cr}$ lost across digital channels). | High aggregate financial loss across millions of micro-transactions. | **Asymmetric Loss Thresholding ($\tau^* = 0.15$):** Optimizes decision boundaries to flag fraudulent micro-payments prior to kitchen dispatch. |
| **Refund & Return Abuse** | "Friendly fraud" exploiting instant refund policies (e.g., claiming missing items over offline QR payments; NRF benchmark: **13.7% fraudulent returns**). | Unrecoverable merchant inventory write-offs and platform margin erosion. | **Post-Delivery Abuse Head:** Evaluates historical refund ratios and account age, routing suspicious claims to unboxing photo proof requirements. |

---

## 🏛️ End-to-End System Architecture

```mermaid
flowchart TD
    %% Ingestion Layer
    subgraph Ingestion ["1. INGESTION & OBSERVABILITY LAYER"]
        A[Razorpay Webhook Payload] --> B[ObservabilityMiddleware]
        B -->|Inject X-Request-ID & X-Execution-Time-MS| C[Strict Pydantic v2 Contract]
    end

    %% Feature & Inference Layer
    subgraph Engine ["2. FEATURE FUSION & DUAL-HEAD ML"]
        C --> D{Feature Fusion Engine}
        D -->|Device & IP Telemetry| E["Payment Fraud Head (model_fraud.pkl)"]
        D -->|Behavioral History| F["Refund Abuse Head (model_abuse.pkl)"]
    end

    %% Cost Optimization & Decision Router
    subgraph Decision ["3. COST ENGINE & OPERATIONAL ROUTER"]
        E --> G["Financial Loss Minimizer (τ* = 0.15)"]
        F --> H["Financial Loss Minimizer (τ* = 0.35)"]
        G --> I[Operational Decision Router]
        H --> I
    end

    %% Execution & Evidence
    subgraph Output ["4. ACTION & DISPUTE EVIDENCE LAYER"]
        I -->|Low Risk| J["AUTO_APPROVE (Dispatch to Kitchen)"]
        I -->|Medium Risk| K["STEP_UP_OTP_REQUIRED (Hold Pending Verification)"]
        I -->|High Risk| L["HARD_CANCEL_TRANSACTION (Prevent Fraud)"]
        I -->|Dispute Event| M["Chargeback Evidence Generator (Razorpay Dossier)"]
    end

    %% Styling
    classDef Ingestion fill:#1e293b,stroke:#3b82f6,color:#fff
    classDef Engine fill:#0f172a,stroke:#8b5cf6,color:#fff
    classDef Decision fill:#1e1b4b,stroke:#ec4899,color:#fff
    classDef Output fill:#064e3b,stroke:#10b981,color:#fff
    class classDef
```

---

## 🧠 Deep-Dive: Core Architectural Decisions & Trade-Off Rationale

### A. Dual-Head Decoupled ML vs. Unified Single-Head Classifier

> [!NOTE]
> **Architectural Decision:** ShieldPay isolates **Payment Fraud** prediction from **Refund Abuse** prediction into two independently trained gradient-boosted classifier heads rather than forcing them into a unified single-head model.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     UNIFIED SINGLE-HEAD CLASSIFIER                        │
│                                                                           │
│  [IP Geodesic Dist + Bot Velocity] ──┐                                    │
│                                      ├───► [Unified Neural / XGB Model]   │
│  [Account Age + Refund History]    ──┘           │                        │
│                                                  ▼                        │
│                                  ❌ Negative Task Interference           │
│                                  (Conflicting Signal Gradients)           │
└───────────────────────────────────────────────────────────────────────────┘
                                     VS.
┌───────────────────────────────────────────────────────────────────────────┐
│                     DECOUPLED DUAL-HEAD ARCHITECTURE                      │
│                                                                           │
│  [IP Dist, Device Count, Velocity]  ──► 🟢 Head 1: Payment Fraud Model    │
│                                                 (model_fraud.pkl)         │
│                                                                           │
│  [Account Age, Past Refund Ratio]   ──► 🟢 Head 2: Refund Abuse Model     │
│                                                 (model_abuse.pkl)         │
└───────────────────────────────────────────────────────────────────────────┘
```

#### Signal Interference Breakdown
1. **Pre-Fulfillment Payment Fraud** signals are heavily correlated with infrastructure anomalies: high IP geodesic distance ($\text{km}$), rapid 30-minute order velocity, and multi-account device fingerprinting.
2. **Post-Delivery Refund Abuse** signals depend on long-term user behavioral patterns: account tenure in days, past refund ratio, and promotional code utilization.
3. Combining both feature vectors into a single target variable creates **negative task interference**: a high-tenure user using a VPN causes gradient conflict between the two targets, degrading overall model precision. Decoupling ensures independent retraining cycles, distinct hyperparameter tuning, and clear auditability.

---

### B. Asymmetric Cost Loss Engine ($\tau^*$) vs. Naive F1/Accuracy

Standard machine learning models default to a $0.50$ decision boundary designed to maximize raw accuracy or F1 score. In financial risk infrastructure, default $0.50$ thresholds are catastrophic due to **cost asymmetry**: a **False Negative (FN)** results in stolen goods, chargeback processor penalties ($\text{₹}2,000$), and lost merchant revenue, whereas a **False Positive (FP)** merely adds light verification friction ($\text{₹}300$).

ShieldPay formulates threshold selection as an explicit financial loss minimization problem over historical validation distributions:

$$\tau^* = \arg\min_{\tau \in [0, 1]} \sum_{i=1}^{N} \left[ \mathbf{1}_{\{y_i = 1, \hat{y}_i(\tau) = 0\}} \cdot \text{Cost}_{\text{FN}} + \mathbf{1}_{\{y_i = 0, \hat{y}_i(\tau) = 1\}} \cdot \text{Cost}_{\text{FP}} \right]$$

#### Financial Asymmetry Parameters & Derived Optimal Boundaries ($\tau^*$)

| Risk Vector | False Negative Cost ($\text{Cost}_{\text{FN}}$) | False Positive Cost ($\text{Cost}_{\text{FP}}$) | Cost Ratio ($\frac{\text{FN}}{\text{FP}}$) | Naive Threshold | Optimal Threshold ($\tau^*$) | Loss Reduction |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Payment Fraud** | **₹2,000** (Chargeback fee + stolen inventory) | **₹300** (SMS OTP cost + customer friction) | **6.67 : 1** | $0.50$ | **$\mathbf{0.15}$** | **$-64.2\%$** |
| **Refund Abuse** | **₹800** (Unrecovered food/item cost) | **₹250** (Manual support agent review fee) | **3.20 : 1** | $0.50$ | **$\mathbf{0.35}$** | **$-51.8\%$** |

---

### C. 3-Tier Operational Decision Matrix

ShieldPay translates raw probabilities $(P_{\text{fraud}}, P_{\text{abuse}})$ into deterministic operational decisions across order fulfillment and refund processing.

```
               P(Payment Fraud)
        0.00                    0.15                    0.50                    1.00
          ├───────────────────────┼───────────────────────┼───────────────────────┤
          │     AUTO_APPROVE      │ STEP_UP_OTP_REQUIRED  │ HARD_CANCEL_TRANSACT  │
          │ (Dispatch to Kitchen) │ (Hold Verification)   │ (Prevent Fraud Loss)  │
          └───────────────────────┴───────────────────────┴───────────────────────┘

               P(Refund Abuse)
        0.00                                    0.35                    0.65    1.00
          ├───────────────────────────────────────┼───────────────────────┼───────┤
          │       INSTANT_REFUND_APPROVED         │ REQUIRE_UNBOXING_PROOF│ DENY  │
          │       (Automated Payout)              │ (Photo Evidence Req.) │ (Agent│
          └───────────────────────────────────────┴───────────────────────┴───────┘
```

| Risk Vector | Probability Range | System Decision Enum | Order Status Enum | Operational Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Payment Fraud** | $P < 0.15$ | `AUTO_APPROVE` | `DISPATCHED_TO_KITCHEN` | Zero-friction instant dispatch. |
| **Payment Fraud** | $0.15 \le P < 0.50$ | `STEP_UP_OTP_REQUIRED` | `HOLD_PENDING_VERIFICATION` | Triggers 2FA/3DS OTP re-verification before order acceptance. |
| **Payment Fraud** | $P \ge 0.50$ | `HARD_CANCEL_TRANSACTION` | `CANCELLED_FRAUD_PREVENTION` | Rejects transaction; releases merchant inventory immediately. |
| **Refund Abuse** | $P < 0.35$ | `INSTANT_REFUND_APPROVED` | — | Instant automated refund payout to original payment method. |
| **Refund Abuse** | $0.35 \le P < 0.65$ | `REQUIRE_UNBOXING_PHOTO_PROOF` | — | Prompts customer to upload item photo proof before refund processing. |
| **Refund Abuse** | $P \ge 0.65$ | `DENY_AUTO_REFUND_ROUTE_TO_AGENT` | — | Blocks auto-refund; escalates ticket to senior fraud operations team. |

---

### D. Sub-50ms Production Infrastructure

```
Client Webhook ──► [FastAPI Router] ──► [asyncio.to_thread()] ──► [Thread Pool Executor] ──► [scikit-learn Predict]
                       │                                                                               │
                       ▼                                                                               ▼
              [X-Request-ID Context] ◄────────────────────── [Result Return] ───────────────────────────┘
```

> [!WARNING]
> **Hot Path Execution SLA:** High-velocity payment webhooks cannot tolerate blocking CPU calls. Running scikit-learn `predict_proba` directly inside an `async def` route freezes the Python event loop, spiking P99 tail latency.

1. **Non-Blocking Thread Offloading (`asyncio.to_thread`):** CPU-heavy matrix operations and feature encoding are offloaded to an asynchronous worker thread pool, keeping the main ASGI event loop free to handle inbound connections.
2. **Fail-Safe Graceful Degradation:** If model artifacts are unavailable or an unexpected inference exception occurs, ShieldPay catches the error and gracefully degrades to `STEP_UP_OTP_REQUIRED` with status `SCORED_FALLBACK` and sentinel `None` scores—ensuring 100% uptime with zero `500 Internal Server Errors`.
3. **Strict Pydantic v2 Data Contracts:** All request schemas use `ConfigDict(strict=True, frozen=True)` to prevent type coercion attacks (e.g., passing string numbers `"1"` into integer counts) and enforce domain bounds ($0 \le \text{ratio} \le 1$, amount $> 0$).
4. **Structured JSON Observability:** `ObservabilityMiddleware` injects UUID4 tracking IDs (`X-Request-ID`) and response headers (`X-Execution-Time-MS`) while propagating context variables to structured JSON logs.

---

### E. Zero-Touch Chargeback Dispute Dossier

When a chargeback notice is received via Razorpay, manual evidence collection takes hours and often misses critical submission deadlines. ShieldPay includes an automated dispute dossier engine (`app/services/evidence.py`) accessible via `GET /api/v1/generate-dispute-dossier/{payment_id}`.

```json
{
  "dispute_header": {
    "razorpay_payment_id": "pay_Nz9K83jL01aQ",
    "dispute_category": "FRAUDULENT_TRANSACTION",
    "representment_deadline": "2026-09-11T08:14:00Z"
  },
  "evidence_summary": {
    "digital_footprint": {
      "ip_geodesic_distance_km": 2.1,
      "device_account_count": 1,
      "risk_verdict": "LOW_PRE_FULFILLMENT_RISK"
    },
    "authentication_logs": {
      "two_factor_auth_status": "3DS_STRONG_CUSTOMER_AUTHENTICATION_SUCCESS",
      "otp_verified_at": "2026-09-04T08:14:02Z"
    },
    "fulfillment_proof": {
      "delivery_gps_coordinates": "12.9716° N, 77.5946° E",
      "proof_of_delivery_timestamp": "2026-09-04T08:32:15Z"
    }
  }
}
```

---

## 📊 Evaluation & Financial Impact Benchmarks

Evaluated over $N = 2,000$ synthetic validation transactions modeled on quick-commerce order distributions:

| Model Head | ROC-AUC | Optimal Threshold ($\tau^*$) | Precision @ $\tau^*$ | Recall @ $\tau^*$ | Baseline Loss (Default $\tau=0.50$) | ShieldPay Loss (Cost-Optimized $\tau^*$) | Financial Loss Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Payment Fraud Head** | **0.962** | **0.15** | 88.4% | 94.1% | ₹482,000 | **₹172,500** | **$-64.2\%$** |
| **Refund Abuse Head** | **0.948** | **0.35** | 84.7% | 91.8% | ₹215,000 | **₹103,600** | **$-51.8\%$** |
| **Combined System** | — | — | — | — | ₹697,000 | **₹276,100** | **$-60.4\%$** |

---

## ⚡ Quickstart & Verification Guide

### 1. Local Setup (30 Seconds)

```bash
# Clone repository
git clone https://github.com/chanakya-b/shieldpay-risk-engine.git
cd shieldpay-risk-engine

# Create virtual environment & activate
python3 -m venv venv
source venv/bin/activate

# Install locked dependencies
pip install -r requirements.txt

# Run pytest test suite (65 tests)
pytest tests/ -v
```

### 2. Start the API Server

```bash
# Run FastAPI via Uvicorn development server
uvicorn app.main:app --reload --port 8000
```

### 3. Docker Deployment

```bash
# Build multi-stage production container
docker build -t shieldpay:latest .

# Run containerized service
docker run -d -p 8000:8000 --name shieldpay_api shieldpay:latest

# Verify health check
curl -i http://localhost:8000/
```

### 4. Verification `curl` Request

Execute a scoring webhook request against the running service:

```bash
curl -i -X POST "http://localhost:8000/api/v1/score-webhook" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: test-trace-uuid-12345" \
  -d '{
    "payment_id": "pay_Nz9K83jL01aQ",
    "amount_inr": 1299.0,
    "payment_method": "credit_card",
    "card_network": "visa",
    "is_promo_applied": 1,
    "account_age_days": 14,
    "past_order_count": 5,
    "past_refund_ratio": 0.05,
    "orders_in_last_30mins": 2,
    "device_account_count": 1,
    "ip_to_delivery_dist_km": 3.5
  }'
```

#### Expected Response Body & Headers

```http
HTTP/1.1 200 OK
date: Wed, 04 Sep 2026 08:30:00 GMT
server: uvicorn
content-type: application/json
x-request-id: test-trace-uuid-12345
x-execution-time-ms: 12.45

{
  "payment_id": "pay_Nz9K83jL01aQ",
  "status": "SCORED",
  "risk_scores": {
    "p_payment_fraud": 0.0421,
    "p_refund_abuse": 0.0812,
    "inference_source": "model"
  },
  "decisions": {
    "pre_fulfillment_action": "AUTO_APPROVE",
    "order_status": "DISPATCHED_TO_KITCHEN",
    "post_delivery_refund_policy": "INSTANT_REFUND_APPROVED"
  }
}
```

---

## 📄 License & Attribution

Distributed under the **MIT License**. Created by [Chanakya Busarla](https://github.com/chanakya-b) as part of the ShieldPay Production Fintech Risk Architecture initiative.