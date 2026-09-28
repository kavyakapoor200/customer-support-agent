"""Pytest configuration and test isolation fixtures."""
import pytest

from src.core.config import get_settings


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """Ensure standard unit and integration tests run deterministically with mock decision engine."""
    monkeypatch.setenv("DECISION_ENGINE_BACKEND", "mock")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
