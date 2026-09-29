"""Groq LLM-judge baseline decision engine for evaluation benchmarks."""
import json
import logging
import time

from src.core.config import get_settings
from src.decision_engine.backends.mock import MockDecisionEngine
from src.decision_engine.base import BaseDecisionEngine, DecisionOutput

logger = logging.getLogger(__name__)


class GroqBaselineEngine(BaseDecisionEngine):
    """LLM baseline using Groq (llama-3.3-70b-versatile via LiteLLM) for evaluation comparison."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_to_mock: bool = True,
    ) -> None:
        super().__init__(engine_name="groq-baseline")
        settings = get_settings()
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.fallback_to_mock = fallback_to_mock
        self._mock_engine = MockDecisionEngine() if fallback_to_mock else None

    async def decide(self, text: str, candidate_actions: list[str]) -> DecisionOutput:
        start_time = time.perf_counter()

        if (not self.api_key or self.api_key.startswith("gsk_your")) and self.fallback_to_mock and self._mock_engine:
            logger.info("GROQ_API_KEY not configured. Falling back to MockDecisionEngine.")
            output = await self._mock_engine.decide(text, candidate_actions)
            return DecisionOutput(
                action=output.action,
                confidence=output.confidence,
                probabilities=output.probabilities,
                raw_scores=output.raw_scores,
                engine_name="groq-baseline-mock",
                latency_ms=output.latency_ms,
            )

        prompt = (
            f"You are a ticket classifier. Given the following customer support ticket, classify it into "
            f"one of the candidate actions and provide a confidence probability (between 0.0 and 1.0).\n\n"
            f"Ticket: \"{text}\"\n"
            f"Candidates: {candidate_actions}\n\n"
            f"Respond ONLY with valid JSON in this format:\n"
            f'{{"action": "<one of candidates>", "confidence": 0.95}}'
        )

        try:
            from litellm import acompletion

            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                api_key=self.api_key,
                temperature=0.0,
            )
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)

            chosen_action = parsed.get("action")
            if chosen_action not in candidate_actions:
                chosen_action = candidate_actions[0]
            confidence = float(parsed.get("confidence", 0.8))
            confidence = min(max(confidence, 0.05), 0.99)

            # Generate synthetic normalized distribution around top choice
            remaining_prob = 1.0 - confidence
            other_actions = [a for a in candidate_actions if a != chosen_action]
            prob_per_other = round(remaining_prob / max(len(other_actions), 1), 4)

            probabilities = {a: prob_per_other for a in other_actions}
            probabilities[chosen_action] = round(1.0 - sum(probabilities.values()), 4)

            latency = (time.perf_counter() - start_time) * 1000.0

            return DecisionOutput(
                action=chosen_action,
                confidence=probabilities[chosen_action],
                probabilities=probabilities,
                raw_scores={},
                engine_name=self.engine_name,
                latency_ms=round(latency, 2),
            )
        except Exception as exc:
            if self.fallback_to_mock and self._mock_engine:
                logger.warning("Groq call failed (%s). Falling back to mock engine.", exc)
                output = await self._mock_engine.decide(text, candidate_actions)
                return DecisionOutput(
                    action=output.action,
                    confidence=output.confidence,
                    probabilities=output.probabilities,
                    raw_scores=output.raw_scores,
                    engine_name="groq-baseline-fallback",
                    latency_ms=output.latency_ms,
                )
            raise

    async def triage(self, text: str):
        """Falls back to mock triage for evaluation comparisons."""
        if self._mock_engine:
            res = await self._mock_engine.triage(text)
            res.engine_name = "groq-baseline"
            return res
        raise NotImplementedError("Triage not configured for GroqBaselineEngine without mock fallback.")
