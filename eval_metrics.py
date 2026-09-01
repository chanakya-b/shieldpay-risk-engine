import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
df = pd.read_csv("zomato_risk_dataset.csv")

feature_cols = [
    'amount_inr',
    'payment_method',
    'card_network',
    'is_promo_applied',
    'account_age_days',
    'past_order_count',
    'past_refund_ratio',
    'orders_in_last_30mins',
    'device_account_count',
    'ip_to_delivery_dist_km',
]

X = df[feature_cols].copy()
y_fraud = df['is_payment_fraud']
y_abuse = df['is_refund_abuse']
y_high_risk = df['is_high_risk']

# ---------------------------------------------------------
# 2. ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------------
categorical_cols = ['payment_method', 'card_network']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT (80% Train, 20% Held-Out Test)
# ---------------------------------------------------------
(
    X_train,
    X_test,
    y_train_fraud,
    y_test_fraud,
    y_train_abuse,
    y_test_abuse,
) = train_test_split(
    X, y_fraud, y_abuse, test_size=0.20, random_state=42, stratify=y_high_risk
)

# ---------------------------------------------------------
# 4. TRAIN DUAL-HEAD ML MODELS
# ---------------------------------------------------------
print("Training Payment Fraud Model...")
model_fraud = HistGradientBoostingClassifier(random_state=42)
model_fraud.fit(X_train, y_train_fraud)

print("Training Refund Abuse Model...")
model_abuse = HistGradientBoostingClassifier(random_state=42)
model_abuse.fit(X_train, y_train_abuse)

# ---------------------------------------------------------
# 5. SAVE MODEL ARTIFACTS TO DISK
# ---------------------------------------------------------
joblib.dump(model_fraud, "model_fraud.pkl")
joblib.dump(model_abuse, "model_abuse.pkl")
joblib.dump(encoder, "encoder.pkl")
print("Successfully saved model_fraud.pkl, model_abuse.pkl, and encoder.pkl!")

# ---------------------------------------------------------
# 6. FINANCIAL LOSS OPTIMIZATION ENGINE
# ---------------------------------------------------------
probs_fraud = model_fraud.predict_proba(X_test)[:, 1]
probs_abuse = model_abuse.predict_proba(X_test)[:, 1]

probs_high_risk = np.maximum(probs_fraud, probs_abuse)
y_test_high_risk = np.maximum(y_test_fraud, y_test_abuse)


def evaluate_financial_loss(y_true, y_probs, cost_fn=1200, cost_fp=400):
  """Evaluates financial loss across decision thresholds."""
  thresholds = np.linspace(0.05, 0.95, 91)
  best_loss = float('inf')
  best_tau = 0.5
  best_metrics = {}

  for tau in thresholds:
    y_pred = (y_probs >= tau).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    total_loss = (fn * cost_fn) + (fp * cost_fp)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if total_loss < best_loss:
      best_loss = total_loss
      best_tau = tau
      best_metrics = {
          'precision': precision,
          'recall': recall,
          'f1': f1,
          'tp': tp,
          'fp': fp,
          'fn': fn,
          'tn': tn,
      }

  return best_tau, best_loss, best_metrics


# Execute Loss Calculations
tau_fraud, loss_fraud, metrics_fraud = evaluate_financial_loss(
    y_test_fraud, probs_fraud, cost_fn=2000, cost_fp=300
)
tau_abuse, loss_abuse, metrics_abuse = evaluate_financial_loss(
    y_test_abuse, probs_abuse, cost_fn=800, cost_fp=250
)
tau_overall, loss_overall, metrics_overall = evaluate_financial_loss(
    y_test_high_risk, probs_high_risk, cost_fn=1500, cost_fp=400
)

auc_fraud = roc_auc_score(y_test_fraud, probs_fraud)
auc_abuse = roc_auc_score(y_test_abuse, probs_abuse)
auc_overall = roc_auc_score(y_test_high_risk, probs_high_risk)

# ---------------------------------------------------------
# 7. PRINT HELD-OUT METRICS REPORT
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("             HELD-OUT TEST SET EVALUATION REPORT             ")
print("=" * 60)

print("\n[1] Payment Fraud Head")
print(f"  • ROC-AUC Score:             {auc_fraud:.4f}")
print(f"  • Optimal Threshold (τ*):    {tau_fraud:.2f}")
print(f"  • Precision @ τ*:            {metrics_fraud['precision']*100:.2f}%")
print(f"  • Recall @ τ*:               {metrics_fraud['recall']*100:.2f}%")
print(f"  • False Positives:           {metrics_fraud['fp']} users")
print(f"  • False Negatives:           {metrics_fraud['fn']} transactions")
print(f"  • Minimized Financial Loss:  ₹{loss_fraud:,.2f}")

print("\n[2] Refund Abuse Head")
print(f"  • ROC-AUC Score:             {auc_abuse:.4f}")
print(f"  • Optimal Threshold (τ*):    {tau_abuse:.2f}")
print(f"  • Precision @ τ*:            {metrics_abuse['precision']*100:.2f}%")
print(f"  • Recall @ τ*:               {metrics_abuse['recall']*100:.2f}%")
print(f"  • False Positives:           {metrics_abuse['fp']} users")
print(f"  • False Negatives:           {metrics_abuse['fn']} claims")
print(f"  • Minimized Financial Loss:  ₹{loss_abuse:,.2f}")

print("\n[3] Combined High-Risk Gatekeeper")
print(f"  • Overall ROC-AUC:           {auc_overall:.4f}")
print(f"  • Optimal Threshold (τ*):    {tau_overall:.2f}")
print(
    f"  • Precision @ τ*:            {metrics_overall['precision']*100:.2f}%"
)
print(f"  • Recall @ τ*:               {metrics_overall['recall']*100:.2f}%")
print(f"  • Total Batch Financial Loss: ₹{loss_overall:,.2f}")
print("=" * 60 + "\n")