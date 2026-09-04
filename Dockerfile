# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/install/pkg --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

RUN useradd --create-home --shell /bin/bash shieldpay
WORKDIR /app

COPY --from=builder /install/pkg /usr/local
COPY app/ ./app/

# Copy ML artifacts from models/ directory
COPY models/ ./models/
COPY model_fraud.pkl model_abuse.pkl encoder.pkl ./

ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    MODEL_DIR=/app/models

USER shieldpay

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--no-access-log"]
