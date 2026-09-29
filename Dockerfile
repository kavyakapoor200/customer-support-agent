# ==============================================================================
# Production Dockerfile for Hugging Face Spaces & Container Platforms
# ==============================================================================
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for blazing-fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up standard non-root user (Hugging Face Spaces requirement: UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    APP_ENV=production \
    DECISION_ENGINE_BACKEND=groq \
    QDRANT_URL=:memory: \
    OTEL_EXPORTER_OTLP_ENDPOINT="" \
    OTEL_TRACES_CONSOLE_ENABLED=false

WORKDIR $HOME/app

# Copy dependency specifications and install
COPY --chown=user pyproject.toml uv.lock* ./
RUN uv venv && uv pip install --no-cache -r pyproject.toml

# Copy application source code and configurations
COPY --chown=user . .

# Hugging Face Spaces standard port
EXPOSE 7860

# Run FastAPI with mounted Gradio UI on dynamic $PORT (Render / Cloud / Local)
CMD ["sh", "-c", "uv run uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

