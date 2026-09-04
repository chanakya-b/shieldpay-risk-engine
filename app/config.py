"""
app/config.py
─────────────
Centralised environment-based configuration for ShieldPay.

Uses Pydantic Settings v2 to:
  - Read values from environment variables (or a .env file at project root)
  - Validate types and apply defaults at application startup
  - Provide a single import point for all configuration consumers

Usage::

    from app.config import settings
    print(settings.model_dir)

To override at runtime::

    MODEL_DIR=/mnt/models ENVIRONMENT=production uvicorn app.main:app
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment-variable overrides.

    All fields have sensible defaults so the service boots locally
    without any .env file. Override specific values in production
    via environment variables or a secrets manager injection.
    """

    model_config = SettingsConfigDict(
        # Load from .env at project root if it exists; silently ignore if absent
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application metadata ─────────────────────────────────────────────
    app_name: str = Field(
        default="ShieldPay Risk Engine",
        description="Human-readable service name (used in logs and health checks)",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment — controls log verbosity and CORS policy",
    )
    version: str = Field(default="2.0.0")

    # ── Model artifact paths ─────────────────────────────────────────────
    model_dir: Path = Field(
        default=Path(__file__).resolve().parents[1] / "models",
        description=(
            "Directory containing model_fraud.pkl, model_abuse.pkl, encoder.pkl. "
            "Falls back to project root when artifacts are not found here."
        ),
    )

    # ── Inference decision thresholds (τ*) ──────────────────────────────
    # Derived from cost-sensitive threshold optimisation in eval_metrics.py
    fraud_threshold_step_up: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fraud probability above which STEP_UP_OTP_REQUIRED is triggered",
    )
    fraud_threshold_hard_block: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Fraud probability above which HARD_CANCEL_TRANSACTION is triggered",
    )
    abuse_threshold_photo_proof: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Abuse probability above which photo proof is required",
    )
    abuse_threshold_deny: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Abuse probability above which auto-refund is denied",
    )

    # ── Observability ────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level for structured JSON logger",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(
        default=["*"],
        description="List of allowed CORS origins. Restrict in production.",
    )

    @field_validator("fraud_threshold_step_up", "fraud_threshold_hard_block", mode="after")
    @classmethod
    def _validate_fraud_thresholds(cls, v: float) -> float:
        return v

    @field_validator("model_dir", mode="after")
    @classmethod
    def _resolve_model_dir(cls, v: Path) -> Path:
        return v.resolve()


# Module-level singleton — import this everywhere, never instantiate Settings() directly
settings = Settings()
