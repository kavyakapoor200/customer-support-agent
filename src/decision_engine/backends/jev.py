"""Jev Decision Engine backend connecting to remote TypeSafe API."""
import logging
import time

import httpx

from src.core.config import get_settings
from src.decision_engine.backends.mock import MockDecisionEngine
from src.decision_engine.base import BaseDecisionEngine, DecisionOutput

logger = logging.getLogger(__name__)


class JevDecisionEngine(BaseDecisionEngine):
    """Decision engine client for Jev remote API."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        fallback_to_mock: bool = True,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(engine_name="jev")
        settings = get_settings()
        self.endpoint_url = endpoint_url or settings.JEV_ENDPOINT_URL
        self.api_key = api_key or settings.JEV_API_KEY
        self.fallback_to_mock = fallback_to_mock
        self.timeout = timeout
        self._mock_engine = MockDecisionEngine() if fallback_to_mock else None

    async def decide(self, text: str, candidate_actions: list[str]) -> DecisionOutput:
        start_time = time.perf_counter()
        if not self.api_key and self.fallback_to_mock and self._mock_engine:
            logger.info("JEV_API_KEY not configured. Falling back to MockDecisionEngine.")
            output = await self._mock_engine.decide(text, candidate_actions)
            return DecisionOutput(
                action=output.action,
                confidence=output.confidence,
                probabilities=output.probabilities,
                raw_scores=output.raw_scores,
                engine_name="jev-mock",
                latency_ms=output.latency_ms,
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"text": text, "options": candidate_actions}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint_url.rstrip('/')}/v1/systemone",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            latency = (time.perf_counter() - start_time) * 1000.0
            return DecisionOutput(
                action=data["action"],
                confidence=float(data["confidence"]),
                probabilities=data["probabilities"],
                raw_scores=data.get("raw_scores", {}),
                engine_name=self.engine_name,
                latency_ms=round(latency, 2),
            )
        except Exception as exc:
            if self.fallback_to_mock and self._mock_engine:
                logger.warning("Jev API request failed: %s. Using fallback.", exc)
                output = await self._mock_engine.decide(text, candidate_actions)
                return DecisionOutput(
                    action=output.action,
                    confidence=output.confidence,
                    probabilities=output.probabilities,
                    raw_scores=output.raw_scores,
                    engine_name="jev-fallback",
                    latency_ms=output.latency_ms,
                )
            raise
