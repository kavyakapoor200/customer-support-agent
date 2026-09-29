"""Application settings loaded from environment variables and .env file."""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application & Environment
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    PORT: int = 8000

    # Cognition & Inference
    DECISION_ENGINE_BACKEND: Literal["mock", "kev", "jev", "groq"] = "mock"
    KEV_ENDPOINT_URL: str = "http://localhost:8008"
    JEV_ENDPOINT_URL: str = "https://api.typesafe.com/v1/systemone"
    JEV_API_KEY: str | None = None
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "groq/openai/gpt-oss-120b"
    SLACK_WEBHOOK_URL: str | None = None

    # Memory & State (PostgreSQL)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/customer_support"

    # Policy Knowledge Base (Qdrant)
    QDRANT_URL: str = "http://localhost:6333"

    # Observability (OpenTelemetry)
    OTEL_SERVICE_NAME: str = "customer-support-agent"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    OTEL_TRACES_CONSOLE_ENABLED: bool = True

    # Thresholds config file location
    THRESHOLDS_PATH: str = "config/thresholds.yaml"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Returns application settings instance dynamically loading latest .env."""
    return Settings()


get_settings.cache_clear = lambda: None

