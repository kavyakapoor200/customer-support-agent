"""Kev-0.8B Decision Engine backend connecting to local System 1 endpoint."""
import logging
import time

import httpx

from src.core.config import get_settings
from src.decision_engine.backends.mock import MockDecisionEngine
from src.decision_engine.base import BaseDecisionEngine, DecisionOutput

logger = logging.getLogger(__name__)


class KevDecisionEngine(BaseDecisionEngine):
    """Decision engine client for Jared Palmer's Kev-0.8B running locally."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        fallback_to_mock: bool = True,
        timeout: float = 3.0,
    ) -> None:
        super().__init__(engine_name="kev")
        self.endpoint_url = endpoint_url or get_settings().KEV_ENDPOINT_URL
        self.fallback_to_mock = fallback_to_mock
        self.timeout = timeout
        self._mock_engine = MockDecisionEngine() if fallback_to_mock else None

    async def decide(self, text: str, candidate_actions: list[str]) -> DecisionOutput:
        start_time = time.perf_counter()
        payload = {
            "text": text,
            "options": candidate_actions,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint_url.rstrip('/')}/v1/systemone",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            latency = (time.perf_counter() - start_time) * 1000.0

            action = data.get("action") or candidate_actions[0]
            confidence = float(data.get("confidence", 0.0))
            probabilities = data.get("probabilities") or {a: 1.0 / len(candidate_actions) for a in candidate_actions}

            return DecisionOutput(
                action=action,
                confidence=confidence,
                probabilities=probabilities,
                raw_scores=data.get("raw_scores", {}),
                engine_name=self.engine_name,
                latency_ms=round(latency, 2),
            )

        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if self.fallback_to_mock and self._mock_engine:
                logger.warning(
                    "Kev-0.8B endpoint %s unavailable (%s). Falling back to MockDecisionEngine.",
                    self.endpoint_url,
                    exc,
                )
                output = await self._mock_engine.decide(text, candidate_actions)
                # Retain engine_name as kev-fallback for observability
                return DecisionOutput(
                    action=output.action,
                    confidence=output.confidence,
                    probabilities=output.probabilities,
                    raw_scores=output.raw_scores,
                    engine_name="kev-fallback",
                    latency_ms=output.latency_ms,
                )
            raise ConnectionError(
                f"Failed to connect to Kev-0.8B endpoint at {self.endpoint_url}: {exc}"
            ) from exc
