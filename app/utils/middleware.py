"""
app/utils/middleware.py
───────────────────────
Production observability middleware for ShieldPay.

Injected on every request / response cycle:
  - X-Request-ID      : UUID4 (generated if not present in incoming headers)
  - X-Execution-Time-MS : wall-clock latency of the full request in milliseconds

The request_id is also pushed into the `request_id_ctx` ContextVar so that
all structured log records emitted during that request automatically include it.
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.utils.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Injects timing and request-ID headers on every HTTP response.

    Header contract
    ---------------
    X-Request-ID          Echoed from the incoming request or auto-generated.
    X-Execution-Time-MS   Elapsed wall-clock time in milliseconds (2 d.p.).
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # --- Resolve / generate request ID -----------------------------------
        req_id: str = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Push into ContextVar so log records pick it up automatically
        token = request_id_ctx.set(req_id)

        start_ns = time.perf_counter_ns()
        try:
            response: Response = await call_next(request)
        except Exception:  # pragma: no cover — FastAPI handles exc before here
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            request_id_ctx.reset(token)

        # --- Inject observability headers ------------------------------------
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Execution-Time-MS"] = f"{elapsed_ms:.2f}"

        logger.info(
            "request_handled",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "execution_time_ms": round(elapsed_ms, 2),
            },
        )

        return response
