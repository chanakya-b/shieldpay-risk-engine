import numpy as np
import pandas as pd
import random

# Set seed for reproducible synthetic data
np.random.seed(42)
random.seed(42)

NUM_RECORDS = 10000

# ---------------------------------------------------------
# 1. GENERATE BASE FEATURES
# ---------------------------------------------------------

# Razorpay Webhook Payload Simulation
payment_methods = ['upi', 'card', 'netbanking']
card_networks = ['visa', 'mastercard', 'rupay', 'amex']

amount_inr = np.random.choice([
    np.random.randint(150, 600),       # Standard food order (Biryani, Burger)
    np.random.randint(600, 2500),      # Party / Group meal / Grocery stash
    np.random.randint(2500, 35000)     # Blinkit electronics / High-end appliances
], size=NUM_RECORDS, p=[0.60, 0.30, 0.10])

payment_method = np.random.choice(payment_methods, size=NUM_RECORDS, p=[0.65, 0.25, 0.10])
card_network = [np.random.choice(card_networks) if m == 'card' else 'none' for m in payment_method]
is_promo_applied = np.random.choice([0, 1], size=NUM_RECORDS, p=[0.70, 0.30]) # ₹150 OFF / Welcome Offer

# Zomato Merchant Context Simulation
account_age_days = np.random.exponential(scale=120, size=NUM_RECORDS).astype(int)
past_order_count = np.clip((account_age_days / 10) * np.random.uniform(0.2, 1.8, size=NUM_RECORDS), 0, 300).astype(int)

# Past refund claims frequency (Spilled food / missing item ratio)
past_refund_ratio = np.clip(np.random.beta(a=1, b=8, size=NUM_RECORDS), 0, 0.95)

# Velocity & Physical Signals
orders_in_last_30mins = np.random.poisson(lam=0.2, size=NUM_RECORDS)
device_account_count = np.random.choice([1, 2, 3, 6], size=NUM_RECORDS, p=[0.85, 0.10, 0.03, 0.02])
ip_to_delivery_dist_km = np.random.exponential(scale=3.5, size=NUM_RECORDS) # Distance between IP & delivery GPS

# ---------------------------------------------------------
# 2. INJECT FRAUD & REFUND ABUSE RULES (GROUND TRUTH LABELS)
# ---------------------------------------------------------

# Target 1: Payment Fraud (Stolen Card / ATO / Bot Chargeback)
# Drivers: High amount + New account + Velocity burst + Multiple accounts on 1 device
fraud_score = (
    (amount_inr > 3000).astype(int) * 3.0 +
    (account_age_days < 2).astype(int) * 3.5 +
    (orders_in_last_30mins > 2).astype(int) * 4.0 +
    (device_account_count > 2).astype(int) * 4.5 +
    (ip_to_delivery_dist_km > 25).astype(int) * 2.5
)
fraud_prob = 1 / (1 + np.exp(-(fraud_score - 6.0)))
is_payment_fraud = (np.random.uniform(0, 1, size=NUM_RECORDS) < fraud_prob).astype(int)

# Target 2: Post-Delivery Refund Abuse (Fake Missing Item / Spilled Food Claim)
# Drivers: High past refund ratio + Promo abuse + Account age < 30 days
abuse_score = (
    (past_refund_ratio > 0.25).astype(int) * 5.0 +
    (is_promo_applied == 1).astype(int) * 1.5 +
    (account_age_days < 15).astype(int) * 2.0 +
    (device_account_count > 1).astype(int) * 2.0 +
    (past_order_count > 2).astype(int) * 1.0
)
abuse_prob = 1 / (1 + np.exp(-(abuse_score - 4.5)))
is_refund_abuse = (np.random.uniform(0, 1, size=NUM_RECORDS) < abuse_prob).astype(int)

# Combined High Risk Indicator
is_high_risk = np.clip(is_payment_fraud + is_refund_abuse, 0, 1)

# ---------------------------------------------------------
# 3. CONSTRUCT DATAFRAME & EXPORT
# ---------------------------------------------------------

df = pd.DataFrame({
    'payment_id': [f"pay_Zom{random.randint(1000000, 9999999)}" for _ in range(NUM_RECORDS)],
    'amount_inr': amount_inr,
    'payment_method': payment_method,
    'card_network': card_network,
    'is_promo_applied': is_promo_applied,
    'account_age_days': account_age_days,
    'past_order_count': past_order_count,
    'past_refund_ratio': np.round(past_refund_ratio, 3),
    'orders_in_last_30mins': orders_in_last_30mins,
    'device_account_count': device_account_count,
    'ip_to_delivery_dist_km': np.round(ip_to_delivery_dist_km, 2),
    'is_payment_fraud': is_payment_fraud,
    'is_refund_abuse': is_refund_abuse,
    'is_high_risk': is_high_risk
})

df.to_csv("zomato_risk_dataset.csv", index=False)

print("="*50)
print(f"SUCCESS: Dataset generated with {NUM_RECORDS} rows -> 'zomato_risk_dataset.csv'")
print("="*50)
print(f"Payment Fraud Rate:   {df['is_payment_fraud'].mean()*100:.2f}%")
print(f"Refund Abuse Rate:    {df['is_refund_abuse'].mean()*100:.2f}%")
print(f"Total High Risk Rate: {df['is_high_risk'].mean()*100:.2f}%")