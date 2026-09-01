import json
from datetime import datetime, timezone


def generate_chargeback_dossier(
    payment_id: str,
    customer_email: str = "user_test@domain.com",
    customer_phone: str = "+919876543210",
):
  """Generates a complete representment evidence package for bank dispute submission."""
  # Mock database telemetry retrieval for the transaction
  dossier = {
      "dispute_header": {
          "dispute_id": f"disp_{payment_id[4:]}",
          "razorpay_payment_id": payment_id,
          "generated_at": datetime.now(timezone.utc).isoformat(),
          "merchant": "Zomato / Blinkit",
          "dispute_reason": "UNAUTHORIZED_TRANSACTION_FRAUD",
      },
      "evidence_summary": {
          "digital_footprint": {
              "ip_address": "152.57.12.4",
              "ip_isp": "Airtel Broadband",
              "ip_geo_city": "Mumbai, MH",
              "device_fingerprint_hash": "a8f9c211e0df92b4",
              "user_agent": (
                  "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X)"
              ),
              "transaction_timestamp": "2026-08-31T13:42:10Z",
          },
          "authentication_logs": {
              "two_factor_auth_passed": True,
              "otp_method": "SMS_TO_REGISTERED_MOBILE",
              "otp_delivery_timestamp": "2026-08-31T13:41:55Z",
              "otp_verification_timestamp": "2026-08-31T13:42:08Z",
              "phone_number_verified": customer_phone,
          },
          "fulfillment_proof": {
              "delivery_status": "DELIVERED",
              "delivery_timestamp": "2026-08-31T14:05:22Z",
              "delivery_partner_id": "DP_88392",
              "gps_coordinates": {
                  "latitude": 19.0760,
                  "longitude": 72.8777,
                  "address_match": True,
                  "distance_to_dropoff_meters": 4.2,
              },
              "delivery_photo_url": f"https://cdn.zomato.com/proofs/{payment_id}_drop.jpg",
          },
          "merchant_account_history": {
              "account_created_date": "2024-03-15",
              "total_prior_successful_orders": 42,
              "total_prior_chargebacks": 0,
              "customer_email": customer_email,
          },
      },
  }

  return dossier


def save_dossier_to_file(dossier, filename="chargeback_evidence_package.json"):
  with open(filename, "w") as f:
    json.dump(dossier, f, indent=4)
  print(f"Chargeback Evidence Dossier exported to '{filename}' successfully.")


if __name__ == "__main__":
  # Generate sample evidence package for our tested payment ID
  test_payment_id = "pay_Nz9K83jL01aQ"
  evidence = generate_chargeback_dossier(test_payment_id)

  print("=" * 60)
  print(f"GENERATED EVIDENCE DOSSIER FOR DISPUTE: {test_payment_id}")
  print("=" * 60)
  print(json.dumps(evidence, indent=2))

  save_dossier_to_file(evidence)