# 🛡️ SYSTEM ARCHITECTURE & PROJECT MASTER SPECIFICATION (`details.md`)

> ### 🤖 MASTER SYSTEM PROMPT FOR ANTIGRAVITY (AI DEVELOPER AGENT)
> You are an elite **Principal Fintech Infrastructure Engineer & Lead Machine Learning Architect** (ex-Stripe / ex-Razorpay). 
> 
> Your sole objective is to implement, refine, and polish **ShieldPay** into an **enterprise-grade, production-ready fintech engine** that would seamlessly pass a high-bar Tier-1 tech code review (Zomato / Razorpay / Swiggy level).
> 
> #### 🛑 NON-NEGOTIABLE PRODUCTION STANDARDS:
> 1. **Zero Toy Code:** Every endpoint must feature robust typing (Pydantic v2), async I/O, custom exception handling, structured JSON logging, and graceful fallbacks.
> 2. **Sub-50ms Execution Budget:** The primary scoring hot path (`POST /api/v1/score-webhook`) must execute in under 50ms. No blocking synchronous I/O in the hot path.
> 3. **Fail-Safe / Graceful Degradation:** If model inference fails or an anomaly occurs, the system must gracefully fall back to a safe deterministic decision (e.g., `STEP_UP_OTP_REQUIRED`) without crashing or throwing a `500 Internal Server Error`.
> 4. **Production Observability:** Inject timing headers (`X-Execution-Time-MS`), request ID tracking (`X-Request-ID`), and structured metrics logging.
> 5. **Clean Code & Design Patterns:** Enforce clean separation of concerns: Service layer, Schema layer, Model inference engine, and Router layer.

---

## 📌 1. Project Overview & Hackathon Strategy

* **Project Name:** ShieldPay — Dual-Head Real-Time Risk & Fraud Engine
* **Target Domain:** Quick-Commerce & Food Delivery (Zomato, Blinkit, Swiggy) over Razorpay gateway rails.
* **Core Value Proposition:** Real-time dual-vector scoring evaluating **Pre-Fulfillment Payment Fraud** and **Post-Delivery Refund Abuse** in sub-50ms execution time.
* **Key Innovation:** Bridging merchant-side physical device telemetry with payment gateway webhook vectors, utilizing an asymmetric cost-sensitive loss engine ($\tau^*$) to minimize financial loss rather than naive accuracy metrics.

---

## 🌐 2. Threat Vector Context & Domain Logic

ShieldPay directly defends against four real-world fintech threat vectors:

1. **Automated Card-Testing & Bot Velocity:** Fraud syndicates executing rapid, low-value orders across payment rails to validate stolen card dumps ([Razorpay Fraud Analytics](https://razorpay.com/blog/what-is-fraud-analytics/)).
2. **CNP Cross-Border Chargeback Liability:** International credit cards bypassing 3D-Secure (2FA / OTP) verification, leaving quick-commerce platforms holding 100% of chargeback costs plus processor dispute fees ($15–$25) ([Razorpay International Chargebacks](https://razorpay.com/blog/international-payment-chargebacks-for-indian-businesses-how-to-win-prevent-and-handle-them/)).
3. **Escalating Micro-Transaction Fraud:** RBI cyber fraud statistics tracking over 2.4 million complaints totaling ~₹22,000 Crore in losses ([RBI Cyber Fraud Analysis](https://economictimes.indiatimes.com/industry/banking/finance/banking/rbi-is-right-to-act-on-digital-payment-fraud-but-some-safeguards-need-sharper-design/articleshow/132309496.cms)).
4. **Post-Delivery Refund Abuse:** Multi-account device farming claiming non-existent missing items or empty-box claims ([NRF Return Fraud Benchmark](https://nrf.com/media-center/press-releases/nrf-and-appriss-retail-report-743-billion-merchandise-returned-2023)).

---

## 🏛️ 3. Architecture & Data Flow

```mermaid
flowchart TD
    A["⚡ 1. Razorpay Webhook Ingestion"] --> B["🔄 2. Feature Fusion & Preprocessing"]
    B --> C["🧠 3. Dual-Head ML Inference"]
    C --> D["🧮 4. Cost Loss Engine τ*"]
    D --> E["🚦 5. Operational Decision Router"]
    E -.->|On Dispute / High Risk| F["📄 6. Chargeback Evidence Dossier"]