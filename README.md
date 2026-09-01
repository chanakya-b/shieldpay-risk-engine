# ShieldPay: Dual-Head Real-Time Risk & Fraud Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ShieldPay** is an enterprise fintech risk management engine engineered for high-frequency quick-commerce platforms (e.g., Zomato, Blinkit) operating on Razorpay. By fusing real-time payment gateway webhooks with merchant context and device telemetry, ShieldPay evaluates two distinct fraud vectors in sub-50 milliseconds: **Pre-Fulfillment Payment Fraud** and **Post-Delivery Refund Abuse**.

---

## 🏗️ System Architecture & Data Pipeline

ShieldPay decouples payment risk from post-fulfillment risk. The dual-head architecture routes inbound webhooks through parallel gradient-boosting estimators to generate decoupled risk scores and operational interventions.

```mermaid
graph TD
    A[Razorpay Webhook Payload] -->|JSON Ingestion| B[FastAPI Ingestion Layer]
    C[Zomato Merchant Context] -->|Feature Fusion| B
    
    B --> D[Feature Preprocessing & Encoding]
    D --> E[Dual-Head ML Inference Engine]
    
    E -->|Head 1: Pre-Fulfillment| F[Payment Fraud Classifier]
    E -->|Head 2: Post-Fulfillment| G[Refund Abuse Classifier]
    
    F -->|Score: p_payment_fraud| H[Cost Threshold Evaluator τ*]
    G -->|Score: p_refund_abuse| I[Cost Threshold Evaluator τ*]
    
    H --> J{Pre-Fulfillment Router}
    I --> K{Post-Delivery Router}
    
    J -->|p < 0.15| L[AUTO_APPROVE]
    J -->|0.15 <= p < 0.50| M[STEP_UP_OTP_REQUIRED]
    J -->|p >= 0.50| N[HARD_CANCEL_TRANSACTION]
    
    K -->|p < 0.25| O[INSTANT_REFUND_APPROVED]
    K -->|0.25 <= p < 0.50| P[REQUIRE_UNBOXING_PHOTO_PROOF]
    K -->|p >= 0.50| Q[DENY_AUTO_REFUND_ROUTE_TO_AGENT]