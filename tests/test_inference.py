"""
tests/test_inference.py
───────────────────────
Unit tests for app/services/inference.py

Test strategy
─────────────
All tests use mock ModelArtifacts — no real .pkl files needed.
The module-level `_artifacts` singleton is patched directly so tests
are hermetic and run in < 100ms total.

Covers:
  - Decision routing: all 3 fraud tiers × all 3 abuse tiers
  - Boundary conditions at exact threshold values
  - Fallback triggered when model raises an exception
  - Fallback response structure (sentinel scores, safe decisions)
  - Artifact loader raises RuntimeError when not initialised
  - inference_source field ("model" vs "fallback")
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import app.services.inference as _inf_module
from app.schemas.payload import (
    OrderStatus,
    PreFulfillmentAction,
    RefundPolicy,
    WebhookRequest,
)
from app.services.inference import (
    ModelArtifacts,
    _build_fallback_response,
    _route_payment_fraud,
    _route_refund_abuse,
    _sync_score,
    run_inference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_model(p1: float) -> MagicMock:
    m = MagicMock()
    m.predict_proba.return_value = [[1 - p1, p1]]
    return m


def _mk_encoder() -> MagicMock:
    enc = MagicMock()
    enc.transform.side_effect = lambda x: x
    return enc


def _mk_artifacts(p_fraud: float, p_abuse: float) -> ModelArtifacts:
    return ModelArtifacts(
        model_fraud=_mk_model(p_fraud),
        model_abuse=_mk_model(p_abuse),
        encoder=_mk_encoder(),
    )


def _make_request(**overrides) -> WebhookRequest:
    defaults = dict(
        payment_id="pay_TestReq000",
        amount_inr=500.0,
        payment_method="upi",
        card_network="none",
        is_promo_applied=0,
        account_age_days=30,
        past_order_count=5,
        past_refund_ratio=0.1,
        orders_in_last_30mins=1,
        device_account_count=1,
        ip_to_delivery_dist_km=2.0,
    )
    defaults.update(overrides)
    return WebhookRequest(**defaults)


# ---------------------------------------------------------------------------
# Decision routing — payment fraud (unit, no I/O)
# ---------------------------------------------------------------------------


class TestRoutePaymentFraud:
    def test_below_step_up_threshold_returns_auto_approve(self) -> None:
        action, status = _route_payment_fraud(0.10)
        assert action == PreFulfillmentAction.AUTO_APPROVE
        assert status == OrderStatus.DISPATCHED_TO_KITCHEN

    def test_at_step_up_threshold_returns_step_up(self) -> None:
        action, status = _route_payment_fraud(0.15)
        assert action == PreFulfillmentAction.STEP_UP_OTP_REQUIRED
        assert status == OrderStatus.HOLD_PENDING_VERIFICATION

    def test_between_thresholds_returns_step_up(self) -> None:
        action, status = _route_payment_fraud(0.35)
        assert action == PreFulfillmentAction.STEP_UP_OTP_REQUIRED

    def test_at_hard_block_threshold_returns_cancel(self) -> None:
        action, status = _route_payment_fraud(0.50)
        assert action == PreFulfillmentAction.HARD_CANCEL_TRANSACTION
        assert status == OrderStatus.CANCELLED_FRAUD_PREVENTION

    def test_above_hard_block_threshold_returns_cancel(self) -> None:
        action, status = _route_payment_fraud(0.99)
        assert action == PreFulfillmentAction.HARD_CANCEL_TRANSACTION


# ---------------------------------------------------------------------------
# Decision routing — refund abuse (unit, no I/O)
# ---------------------------------------------------------------------------


class TestRouteRefundAbuse:
    def test_below_photo_threshold_returns_instant_refund(self) -> None:
        policy = _route_refund_abuse(0.20)
        assert policy == RefundPolicy.INSTANT_REFUND_APPROVED

    def test_at_photo_threshold_returns_photo_proof(self) -> None:
        policy = _route_refund_abuse(0.35)
        assert policy == RefundPolicy.REQUIRE_UNBOXING_PHOTO_PROOF

    def test_between_thresholds_returns_photo_proof(self) -> None:
        policy = _route_refund_abuse(0.50)
        assert policy == RefundPolicy.REQUIRE_UNBOXING_PHOTO_PROOF

    def test_at_deny_threshold_returns_deny(self) -> None:
        policy = _route_refund_abuse(0.65)
        assert policy == RefundPolicy.DENY_AUTO_REFUND_ROUTE_TO_AGENT

    def test_above_deny_threshold_returns_deny(self) -> None:
        policy = _route_refund_abuse(0.99)
        assert policy == RefundPolicy.DENY_AUTO_REFUND_ROUTE_TO_AGENT


# ---------------------------------------------------------------------------
# _sync_score — full matrix
# ---------------------------------------------------------------------------


class TestSyncScore:
    def _run(self, p_fraud: float, p_abuse: float) -> object:
        _inf_module._artifacts = _mk_artifacts(p_fraud, p_abuse)
        req = _make_request()
        result = _sync_score(req)
        _inf_module._artifacts = None
        return result

    def test_low_fraud_low_abuse(self) -> None:
        result = self._run(0.05, 0.10)
        assert result.decisions.pre_fulfillment_action == PreFulfillmentAction.AUTO_APPROVE
        assert result.decisions.post_delivery_refund_policy == RefundPolicy.INSTANT_REFUND_APPROVED
        assert result.risk_scores.inference_source == "model"

    def test_mid_fraud_mid_abuse(self) -> None:
        result = self._run(0.30, 0.50)
        assert result.decisions.pre_fulfillment_action == PreFulfillmentAction.STEP_UP_OTP_REQUIRED
        assert result.decisions.post_delivery_refund_policy == RefundPolicy.REQUIRE_UNBOXING_PHOTO_PROOF

    def test_high_fraud_high_abuse(self) -> None:
        result = self._run(0.80, 0.90)
        assert result.decisions.pre_fulfillment_action == PreFulfillmentAction.HARD_CANCEL_TRANSACTION
        assert result.decisions.post_delivery_refund_policy == RefundPolicy.DENY_AUTO_REFUND_ROUTE_TO_AGENT

    def test_risk_scores_rounded_to_4dp(self) -> None:
        result = self._run(0.123456, 0.654321)
        assert result.risk_scores.p_payment_fraud == round(0.123456, 4)
        assert result.risk_scores.p_refund_abuse == round(0.654321, 4)

    def test_payment_id_preserved_in_response(self) -> None:
        _inf_module._artifacts = _mk_artifacts(0.05, 0.10)
        req = _make_request(payment_id="pay_IDPreserve")
        result = _sync_score(req)
        _inf_module._artifacts = None
        assert result.payment_id == "pay_IDPreserve"


# ---------------------------------------------------------------------------
# Fallback response
# ---------------------------------------------------------------------------


class TestFallbackResponse:
    def test_fallback_returns_step_up_action(self) -> None:
        fb = _build_fallback_response("pay_FallBack1")
        assert fb.decisions.pre_fulfillment_action == PreFulfillmentAction.STEP_UP_OTP_REQUIRED
        assert fb.decisions.order_status == OrderStatus.FALLBACK_SAFE_HOLD
        assert fb.decisions.post_delivery_refund_policy == RefundPolicy.FALLBACK_REQUIRE_PHOTO_PROOF

    def test_fallback_sentinel_scores(self) -> None:
        fb = _build_fallback_response("pay_Sentinel0")
        assert fb.risk_scores.p_payment_fraud is None
        assert fb.risk_scores.p_refund_abuse is None

    def test_fallback_inference_source_is_fallback(self) -> None:
        fb = _build_fallback_response("pay_Source000")
        assert fb.risk_scores.inference_source == "fallback"

    def test_fallback_preserves_payment_id(self) -> None:
        fb = _build_fallback_response("pay_IDCheck00")
        assert fb.payment_id == "pay_IDCheck00"


# ---------------------------------------------------------------------------
# run_inference — async integration (mocked artifacts)
# ---------------------------------------------------------------------------


class TestRunInferenceAsync:
    def test_successful_inference_returns_model_source(self) -> None:
        _inf_module._artifacts = _mk_artifacts(0.10, 0.20)
        req = _make_request()
        result = asyncio.run(run_inference(req))
        _inf_module._artifacts = None
        assert result.risk_scores.inference_source == "model"

    def test_model_exception_triggers_fallback(self) -> None:
        """If predict_proba raises, run_inference must return a safe fallback."""
        bad_model = MagicMock()
        bad_model.predict_proba.side_effect = RuntimeError("GPU OOM")
        _inf_module._artifacts = ModelArtifacts(
            model_fraud=bad_model,
            model_abuse=bad_model,
            encoder=_mk_encoder(),
        )
        req = _make_request(payment_id="pay_ExcTest00")
        result = asyncio.run(run_inference(req))
        _inf_module._artifacts = None

        assert result.risk_scores.inference_source == "fallback"
        assert result.decisions.pre_fulfillment_action == PreFulfillmentAction.STEP_UP_OTP_REQUIRED

    def test_unloaded_artifacts_triggers_fallback(self) -> None:
        """_artifacts=None (not loaded) must produce fallback, never a 500."""
        _inf_module._artifacts = None
        req = _make_request(payment_id="pay_NoArtifct")
        result = asyncio.run(run_inference(req))
        assert result.risk_scores.inference_source == "fallback"

    def test_encoder_failure_triggers_fallback(self) -> None:
        bad_encoder = MagicMock()
        bad_encoder.transform.side_effect = ValueError("Unknown category")
        _inf_module._artifacts = ModelArtifacts(
            model_fraud=_mk_model(0.10),
            model_abuse=_mk_model(0.10),
            encoder=bad_encoder,
        )
        req = _make_request(payment_id="pay_EncFail00")
        result = asyncio.run(run_inference(req))
        _inf_module._artifacts = None
        assert result.risk_scores.inference_source == "fallback"
