"""
tests/test_api.py
─────────────────
Integration tests for the FastAPI HTTP layer.

Uses the shared `client` fixture from conftest.py which:
  - Creates a real FastAPI app via `create_app()`
  - Pre-populates the inference singleton with low-risk mock artifacts
  - Wraps everything in a synchronous `TestClient`

Covers:
  - GET / → 200 health check structure
  - POST /api/v1/score-webhook → 200 with correct JSON shape
  - POST /api/v1/score-webhook → response headers (X-Request-ID, X-Execution-Time-MS)
  - POST /api/v1/score-webhook → X-Request-ID echoed from request header
  - POST /api/v1/score-webhook → X-Request-ID auto-generated when absent
  - POST /api/v1/score-webhook → 422 on invalid payload (missing fields)
  - POST /api/v1/score-webhook → 422 on bad payment_id prefix
  - POST /api/v1/score-webhook → 422 on strict mode type coercion
  - GET /api/v1/generate-dispute-dossier/{payment_id} → 200 dossier shape
  - GET /api/v1/generate-dispute-dossier/{payment_id} → 422 on bad payment_id
  - Fallback path via high-risk artifacts override
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

import app.services.inference as _inf_module
from app.main import create_app
from tests.conftest import _mk_artifacts


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_get_root_returns_200(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_body_structure(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert data["status"] == "online"
        assert data["version"] == "2.0.0"
        assert "docs" in data

    def test_health_includes_timing_header(self, client: TestClient) -> None:
        resp = client.get("/")
        assert "x-execution-time-ms" in resp.headers

    def test_health_includes_request_id_header(self, client: TestClient) -> None:
        resp = client.get("/")
        assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# POST /api/v1/score-webhook — success paths
# ---------------------------------------------------------------------------


class TestScoreWebhookSuccess:
    def test_returns_200(self, client: TestClient, valid_payload: dict) -> None:
        resp = client.post("/api/v1/score-webhook", json=valid_payload)
        assert resp.status_code == 200

    def test_response_contains_payment_id(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        data = client.post("/api/v1/score-webhook", json=valid_payload).json()
        assert data["payment_id"] == valid_payload["payment_id"]

    def test_response_has_risk_scores(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        data = client.post("/api/v1/score-webhook", json=valid_payload).json()
        scores = data["risk_scores"]
        assert "p_payment_fraud" in scores
        assert "p_refund_abuse" in scores
        assert "inference_source" in scores

    def test_response_has_decisions(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        data = client.post("/api/v1/score-webhook", json=valid_payload).json()
        decisions = data["decisions"]
        assert "pre_fulfillment_action" in decisions
        assert "order_status" in decisions
        assert "post_delivery_refund_policy" in decisions

    def test_x_execution_time_ms_header_present(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        resp = client.post("/api/v1/score-webhook", json=valid_payload)
        assert "x-execution-time-ms" in resp.headers
        # Must be parseable as a float
        float(resp.headers["x-execution-time-ms"])

    def test_x_request_id_echoed_from_request(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        resp = client.post(
            "/api/v1/score-webhook",
            json=valid_payload,
            headers={"X-Request-ID": "my-trace-abc-123"},
        )
        assert resp.headers["x-request-id"] == "my-trace-abc-123"

    def test_x_request_id_auto_generated_when_absent(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        resp = client.post("/api/v1/score-webhook", json=valid_payload)
        req_id = resp.headers.get("x-request-id", "")
        # Should be a valid UUID4
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            req_id,
        ), f"Expected UUID4, got: {req_id}"

    def test_low_risk_payload_returns_auto_approve(
        self, client: TestClient, low_risk_payload: dict
    ) -> None:
        # conftest `client` uses low_risk_artifacts (p_fraud=0.05, p_abuse=0.12)
        data = client.post("/api/v1/score-webhook", json=low_risk_payload).json()
        assert data["decisions"]["pre_fulfillment_action"] == "AUTO_APPROVE"
        assert data["decisions"]["post_delivery_refund_policy"] == "INSTANT_REFUND_APPROVED"

    def test_inference_source_is_model(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        data = client.post("/api/v1/score-webhook", json=valid_payload).json()
        assert data["status"] == "SCORED"
        assert data["risk_scores"]["inference_source"] == "model"


# ---------------------------------------------------------------------------
# POST /api/v1/score-webhook — fallback path
# ---------------------------------------------------------------------------


class TestScoreWebhookFallback:
    def test_broken_model_returns_200_with_fallback(
        self, valid_payload: dict
    ) -> None:
        """Even when the model blows up, the API must return 200 + safe decision."""
        from unittest.mock import MagicMock
        from tests.conftest import _mk_encoder

        app = create_app()
        with TestClient(app) as c:
            bad_model = MagicMock()
            bad_model.predict_proba.side_effect = RuntimeError("Simulated GPU OOM")
            _inf_module._artifacts = _inf_module.ModelArtifacts(
                model_fraud=bad_model,
                model_abuse=bad_model,
                encoder=_mk_encoder(),
            )
            resp = c.post("/api/v1/score-webhook", json=valid_payload)

        _inf_module._artifacts = None

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SCORED_FALLBACK"
        assert data["risk_scores"]["inference_source"] == "fallback"
        assert data["decisions"]["pre_fulfillment_action"] == "STEP_UP_OTP_REQUIRED"


# ---------------------------------------------------------------------------
# AsyncClient test execution
# ---------------------------------------------------------------------------


class TestAsyncClientExecution:
    @pytest.mark.asyncio
    async def test_async_client_post(self, valid_payload: dict) -> None:
        import httpx
        from tests.conftest import _mk_artifacts

        _inf_module._artifacts = _mk_artifacts(0.05, 0.12)
        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/v1/score-webhook", json=valid_payload)
        _inf_module._artifacts = None

        assert resp.status_code == 200
        assert resp.json()["status"] == "SCORED"
        assert "x-execution-time-ms" in resp.headers


# ---------------------------------------------------------------------------
# POST /api/v1/score-webhook — 422 validation errors
# ---------------------------------------------------------------------------


class TestScoreWebhookValidationErrors:
    def test_missing_required_fields_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/v1/score-webhook", json={})
        assert resp.status_code == 422

    def test_bad_payment_id_prefix_returns_422(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        bad = {**valid_payload, "payment_id": "rzp_NoBadPrefix"}
        resp = client.post("/api/v1/score-webhook", json=bad)
        assert resp.status_code == 422

    def test_negative_amount_returns_422(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        bad = {**valid_payload, "amount_inr": -50.0}
        resp = client.post("/api/v1/score-webhook", json=bad)
        assert resp.status_code == 422

    def test_refund_ratio_above_1_returns_422(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        bad = {**valid_payload, "past_refund_ratio": 1.5}
        resp = client.post("/api/v1/score-webhook", json=bad)
        assert resp.status_code == 422

    def test_string_for_int_field_returns_422(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        bad = {**valid_payload, "is_promo_applied": "yes"}
        resp = client.post("/api/v1/score-webhook", json=bad)
        assert resp.status_code == 422

    def test_422_response_has_detail_field(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/v1/score-webhook", json={})
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# GET /api/v1/generate-dispute-dossier/{payment_id}
# ---------------------------------------------------------------------------


class TestDisputeDossier:
    def test_valid_payment_id_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/generate-dispute-dossier/pay_TestDossier")
        assert resp.status_code == 200

    def test_dossier_has_dispute_header(self, client: TestClient) -> None:
        data = client.get(
            "/api/v1/generate-dispute-dossier/pay_TestDossier"
        ).json()
        assert "dispute_header" in data
        header = data["dispute_header"]
        assert "razorpay_payment_id" in header
        assert header["razorpay_payment_id"] == "pay_TestDossier"

    def test_dossier_has_evidence_summary(self, client: TestClient) -> None:
        data = client.get(
            "/api/v1/generate-dispute-dossier/pay_TestDossier"
        ).json()
        assert "evidence_summary" in data
        ev = data["evidence_summary"]
        assert "digital_footprint" in ev
        assert "authentication_logs" in ev
        assert "fulfillment_proof" in ev

    def test_invalid_payment_id_returns_422(self, client: TestClient) -> None:
        # payment_id must match ^pay_[a-zA-Z0-9]+$
        resp = client.get("/api/v1/generate-dispute-dossier/invalid-id")
        assert resp.status_code == 422

    def test_dossier_timing_header_present(self, client: TestClient) -> None:
        resp = client.get("/api/v1/generate-dispute-dossier/pay_TimingTest")
        assert "x-execution-time-ms" in resp.headers
