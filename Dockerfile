# Multi-stage Dockerfile for Conformal VaR India
# Stages: builder → api (FastAPI), demo (Streamlit)

# -------------------------------------------------------------
# Stage 1: Builder — install all Python dependencies
# -------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt

# -------------------------------------------------------------
# Stage 2: api — FastAPI REST service
# -------------------------------------------------------------
FROM python:3.11-slim AS api

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

COPY pyproject.toml config.yaml ./
COPY src/ ./src/
COPY artifacts/ ./artifacts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "conformal_var.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# -------------------------------------------------------------
# Stage 3: demo — Streamlit interactive demo
# -------------------------------------------------------------
FROM python:3.11-slim AS demo

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

COPY pyproject.toml config.yaml ./
COPY src/ ./src/
COPY streamlit_app.py ./
COPY .streamlit/ ./.streamlit/
# artifacts/demo/ is mounted as a read-only volume in docker-compose

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
