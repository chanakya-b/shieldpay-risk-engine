"""
tests/conftest.py
─────────────────
Shared pytest fixtures for ShieldPay test suite.

Key fixtures
────────────
- `client`              : Synchronous HTTPX TestClient with models pre-loaded
- `mock_artifacts`      : Patches the inference singleton with deterministic mocks
- `valid_payload`       : Canonical high-risk WebhookRequest dict
- `low_risk_payload`    : Canonical low-risk WebhookRequest dict
- `frozen_artifacts`    : ModelArtifacts with fixed predict_proba outputs

Isolation strategy
──────────────────
Tests MUST NOT depend on the real .pkl files on disk. Every test that touches
inference imports goes through `mock_artifacts`, which stubs out the sklearn
models with a `MagicMock` returning deterministic probabilities. This makes
the test suite:
  - Hermetic (no file system dependency)
  - Fast (no joblib I/O)
  - Deterministic (fixed probability outputs)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.services.inference as _inf_module
from app.main import create_app
from app.services.inference import ModelArtifacts



# ---------------------------------------------------------------------------
# Module-level helpers (importable from other test modules)
# ---------------------------------------------------------------------------


def _mk_model(p0: float, p1: float) -> MagicMock:
    """Return a mock sklearn classifier whose predict_proba returns [p0, p1]."""
    model = MagicMock()
    model.predict_proba.return_value = [[p0, p1]]
    return model


def _mk_encoder() -> MagicMock:
    """Return a mock OrdinalEncoder that passes the input DataFrame through."""
    encoder = MagicMock()
    encoder.transform.side_effect = lambda x: x
    return encoder


def _mk_artifacts(p_fraud: float, p_abuse: float) -> ModelArtifacts:
    """Return a ModelArtifacts with mock models producing the given probabilities."""
    return ModelArtifacts(
        model_fraud=_mk_model(1 - p_fraud, p_fraud),
        model_abuse=_mk_model(1 - p_abuse, p_abuse),
        encoder=_mk_encoder(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def high_risk_artifacts() -> ModelArtifacts:
    """Artifacts that produce high-fraud + high-abuse probabilities."""
    return ModelArtifacts(
        model_fraud=_mk_model(p0=0.20, p1=0.80),   # p_fraud = 0.80 → HARD_CANCEL
        model_abuse=_mk_model(p0=0.10, p1=0.90),   # p_abuse = 0.90 → DENY_REFUND
        encoder=_mk_encoder(),
    )


@pytest.fixture()
def step_up_artifacts() -> ModelArtifacts:
    """Artifacts that produce mid-range fraud probability → STEP_UP_OTP."""
    return ModelArtifacts(
        model_fraud=_mk_model(p0=0.70, p1=0.30),   # p_fraud = 0.30 → STEP_UP
        model_abuse=_mk_model(p0=0.55, p1=0.45),   # p_abuse = 0.45 → PHOTO_PROOF
        encoder=_mk_encoder(),
    )


@pytest.fixture()
def low_risk_artifacts() -> ModelArtifacts:
    """Artifacts that produce low probabilities → AUTO_APPROVE + INSTANT_REFUND."""
    return ModelArtifacts(
        model_fraud=_mk_model(p0=0.95, p1=0.05),   # p_fraud = 0.05 → AUTO_APPROVE
        model_abuse=_mk_model(p0=0.88, p1=0.12),   # p_abuse = 0.12 → INSTANT_REFUND
        encoder=_mk_encoder(),
    )


@pytest.fixture()
def client(low_risk_artifacts: ModelArtifacts) -> TestClient:
    """
    FastAPI TestClient with real app factory and mocked artifact singleton.

    The `lifespan` startup call to `load_artifacts()` is bypassed by
    pre-populating `_inf_module._artifacts` before the client is created.
    """
    _inf_module._artifacts = low_risk_artifacts
    app = create_app()
    with TestClient(app) as c:
        yield c
    _inf_module._artifacts = None  # clean up after each test


# ---------------------------------------------------------------------------
# Canonical payload fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_payload() -> dict:
    """High-risk but structurally valid webhook payload."""
    return {
        "payment_id": "pay_HighRisk12345",
        "amount_inr": 3500.0,
        "payment_method": "credit_card",
        "card_network": "visa",
        "is_promo_applied": 1,
        "account_age_days": 3,
        "past_order_count": 0,
        "past_refund_ratio": 0.0,
        "orders_in_last_30mins": 5,
        "device_account_count": 3,
        "ip_to_delivery_dist_km": 85.0,
    }


@pytest.fixture()
def low_risk_payload() -> dict:
    """Structurally valid low-risk webhook payload."""
    return {
        "payment_id": "pay_LowRisk12345",
        "amount_inr": 299.0,
        "payment_method": "upi",
        "card_network": "none",
        "is_promo_applied": 0,
        "account_age_days": 365,
        "past_order_count": 42,
        "past_refund_ratio": 0.02,
        "orders_in_last_30mins": 1,
        "device_account_count": 1,
        "ip_to_delivery_dist_km": 1.2,
    }
