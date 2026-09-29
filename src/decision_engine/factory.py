"""Factory function for instantiating the requested DecisionEngine backend."""
from src.core.config import get_settings
from src.decision_engine.backends.groq_baseline import GroqBaselineEngine
from src.decision_engine.backends.jev import JevDecisionEngine
from src.decision_engine.backends.kev import KevDecisionEngine
from src.decision_engine.backends.mock import MockDecisionEngine
from src.decision_engine.base import BaseDecisionEngine


def get_decision_engine(backend: str | None = None) -> BaseDecisionEngine:
    """Returns the configured DecisionEngine backend instance.

    Args:
        backend: "mock", "kev", "jev", or "groq". Defaults to Settings.DECISION_ENGINE_BACKEND.

    Returns:
        Instance conforming to BaseDecisionEngine.
    """
    settings = get_settings()
    selected_backend = (backend or settings.DECISION_ENGINE_BACKEND).lower()

    if selected_backend == "mock":
        return MockDecisionEngine()
    elif selected_backend == "kev":
        return KevDecisionEngine(
            endpoint_url=settings.KEV_ENDPOINT_URL,
            temperature=settings.TEMPERATURE,
        )
    elif selected_backend == "jev":
        return JevDecisionEngine(endpoint_url=settings.JEV_ENDPOINT_URL, api_key=settings.JEV_API_KEY)
    elif selected_backend in ("groq", "llm_baseline"):
        return GroqBaselineEngine(api_key=settings.GROQ_API_KEY)
    else:
        raise ValueError(
            f"Unknown decision engine backend: '{selected_backend}'. "
            f"Valid options are: 'mock', 'kev', 'jev', 'groq'."
        )
