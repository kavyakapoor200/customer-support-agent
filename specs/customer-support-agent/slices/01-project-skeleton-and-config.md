# Slice 01: Project Skeleton, Config & Docker Foundation

## 1. Contract Unlocked
Establishes the dependency environment, typed threshold configuration schema, and reproducible Docker services (PostgreSQL, Qdrant, Jaeger) running with strict memory limits for Mac M1 (8GB).

## 2. API Seam & Module Ownership
* **Package Management:** `pyproject.toml` managed via `uv`.
  * Dependencies: `fastapi`, `uvicorn`, `pydantic>=2.0`, `pydantic-settings`, `pyyaml`, `langgraph`, `qdrant-client`, `litellm`, `gradio`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `psycopg[binary]`, `asyncpg`, `pytest`, `pytest-asyncio`.
* **Config Module (`src/core/config.py`):**
  * `Settings(BaseSettings)`: loads `DATABASE_URL`, `QDRANT_URL`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `GROQ_API_KEY`, `DECISION_ENGINE_BACKEND` (`mock` | `kev` | `jev` | `groq`).
* **Thresholds Schema (`src/core/thresholds.py`):**
  * Pydantic model parsing `config/thresholds.yaml`:
    * `ActionThresholds`: `auto_execute_threshold: float`, `max_auto_amount_usd: float`, `review_threshold: float`, `deny_threshold: float`.
    * Validates that `auto_execute_threshold >= review_threshold >= deny_threshold`.
* **Docker Environment (`docker-compose.yml`):**
  * `postgres`: alpine image, port 5432, healthcheck, `mem_limit: 150m`.
  * `qdrant`: port 6333, `mem_limit: 200m`.
  * `jaeger`: `jaegertracing/all-in-one:latest`, ports 16686 (UI) & 4317 (OTLP gRPC), `mem_limit: 150m`.
  * Total idle memory overhead < 550 MB.

## 3. What the Human Can Run or See
* Run `uv sync` to install all dependencies cleanly.
* Run `docker compose up -d` and inspect Jaeger UI at `http://localhost:16686` and Qdrant at `http://localhost:6333/dashboard`.
* Inspect `config/thresholds.yaml` to view human-editable gating thresholds.

## 4. Verification Gates & Tests
* `tests/test_config.py`:
  * Validates loading of `config/thresholds.yaml` into typed Pydantic models.
  * Asserts invalid thresholds (e.g. `auto < review`) raise validation errors.
  * Asserts environment variable overrides work as expected.
* Command gate: `pytest tests/test_config.py` passes 100%.

## 5. Delegated Implementer Discretion
* Exact directory layout inside `src/core/` (e.g., separating logger setup).
* Internal helper functions for YAML parsing.

## 6. What Must Stay Green
* Tests must run completely offline without remote API keys.
* Docker Compose file must pass `docker compose config` validation.

## 7. Non-blocking Checkpoint
Review `config/thresholds.yaml` defaults. If no modifications requested within review window, proceed with defaults.
