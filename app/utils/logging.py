"""
app/utils/logging.py
────────────────────
Structured JSON logger factory for ShieldPay.

Every log record is emitted as a single-line JSON object containing:
  - timestamp   : ISO-8601 UTC
  - level       : DEBUG / INFO / WARNING / ERROR / CRITICAL
  - service     : constant "shieldpay"
  - request_id  : propagated from middleware via contextvars
  - logger      : dotted module name
  - message     : human-readable description
  - extra       : arbitrary key/value pairs passed at call site
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, MutableMapping, Tuple

# ---------------------------------------------------------------------------
# Request-ID context variable — set by middleware, read by log handler
# ---------------------------------------------------------------------------
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


# ---------------------------------------------------------------------------
# Custom JSON formatter
# ---------------------------------------------------------------------------
class _JSONFormatter(logging.Formatter):
    """Formats a LogRecord as a single-line JSON string."""

    _RESERVED = frozenset(
        {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno",
            "message", "module", "msecs", "msg", "name", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "taskName", "thread", "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Collect any extra key/value pairs injected at call site
        extras: dict[str, Any] = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._RESERVED and not k.startswith("_")
        }

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "service": "shieldpay",
            "request_id": request_id_ctx.get(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a structured JSON logger bound to *name*.

    Usage::

        from app.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Inference complete", extra={"p_fraud": 0.12, "latency_ms": 8.4})
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when module is re-imported
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level)
    return logger
