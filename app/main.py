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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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
    # Web UI Dashboard & Health endpoints
    # -----------------------------------------------------------------------
    @app.get(
        "/dashboard",
        tags=["Dashboard"],
        summary="Interactive Operations Dashboard",
        response_class=HTMLResponse,
    )
    async def serve_dashboard():
        from pathlib import Path
        template_path = Path(__file__).parent / "templates" / "dashboard.html"
        if template_path.exists():
            return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>ShieldPay Dashboard</h1>")

    @app.get(
        "/",
        tags=["Health / Dashboard"],
        summary="Root endpoint (Dashboard / JSON Health)",
    )
    async def root_endpoint(request: Request):
        accept_header = request.headers.get("accept", "")
        # If client explicitly requests HTML or text/html (like a web browser)
        if "text/html" in accept_header and "application/json" not in accept_header:
            from pathlib import Path
            template_path = Path(__file__).parent / "templates" / "dashboard.html"
            if template_path.exists():
                return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
        
        return JSONResponse(
            content={
                "status": "online",
                "service": "ShieldPay Dual-Head Risk Engine",
                "version": "2.0.0",
                "docs": "/docs",
                "dashboard": "/dashboard",
            }
        )

    @app.get(
        "/health",
        tags=["Health"],
        summary="Service health check JSON",
        response_class=JSONResponse,
    )
    async def health_check() -> dict:
        return {
            "status": "online",
            "service": "ShieldPay Dual-Head Risk Engine",
            "version": "2.0.0",
            "docs": "/docs",
            "dashboard": "/dashboard",
        }

    return app


# ---------------------------------------------------------------------------
# ASGI entrypoint
# ---------------------------------------------------------------------------
app = create_app()
