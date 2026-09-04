"""
app/main.py
───────────
ShieldPay FastAPI application factory.

Startup sequence (FastAPI lifespan)
────────────────────────────────────
1. load_artifacts()  — loads model_fraud.pkl, model_abuse.pkl, encoder.pkl
                       into the module-level inference singleton.
2. Middleware stack   — ObservabilityMiddleware (timing + request ID headers).
3. API Router        — mounts all v1 routes under /api/v1.
4. Health endpoint   — GET / returns service metadata.

Run with:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.services.inference import load_artifacts
from app.utils.logging import get_logger
from app.utils.middleware import ObservabilityMiddleware

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup:  Load pre-trained ML artifacts into memory.
    Shutdown: (Placeholder for connection pool teardown, cache flush, etc.)
    """
    logger.info("ShieldPay Risk Engine — startup initiated")
    load_artifacts()  # blocking; intentional — must succeed before serving traffic
    logger.info("ShieldPay Risk Engine — ready to serve requests")

    yield  # Application runs here

    logger.info("ShieldPay Risk Engine — graceful shutdown complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="ShieldPay — Dual-Head Real-Time Risk & Fraud Engine",
        description=(
            "Enterprise fintech risk management engine for quick-commerce "
            "platforms (Zomato, Blinkit) on Razorpay gateway rails. "
            "Evaluates pre-fulfillment payment fraud and post-delivery refund "
            "abuse in sub-50ms execution time."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Middleware stack (order matters — outermost registered = innermost exec)
    # -----------------------------------------------------------------------

    # CORS — restrict in production to your actual frontend origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Execution-Time-MS"],
    )

    # Observability — timing + request ID headers (registered last = runs first)
    app.add_middleware(ObservabilityMiddleware)

    # -----------------------------------------------------------------------
    # Routers
    # -----------------------------------------------------------------------
    app.include_router(v1_router)

    # -----------------------------------------------------------------------
    # Health / root endpoint
    # -----------------------------------------------------------------------
    @app.get(
        "/",
        tags=["Health"],
        summary="Service health check",
        response_class=JSONResponse,
    )
    async def health_check() -> dict:
        return {
            "status": "online",
            "service": "ShieldPay Dual-Head Risk Engine",
            "version": "2.0.0",
            "docs": "/docs",
        }

    return app


# ---------------------------------------------------------------------------
# ASGI entrypoint
# ---------------------------------------------------------------------------
app = create_app()
