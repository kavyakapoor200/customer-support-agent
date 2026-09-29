"""Jev Decision Engine backend connecting to remote TypeSafe API."""
import logging
import time

import httpx

from src.core.config import get_settings
from src.decision_engine.backends.mock import MockDecisionEngine
from src.decision_engine.base import BaseDecisionEngine, DecisionOutput, JevDecisionResult

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

    async def triage(self, text: str) -> JevDecisionResult:
        """Evaluates Jev primitives via remote TypeSafe API or fallback."""
        start_time = time.perf_counter()
        if not self.api_key and self.fallback_to_mock and self._mock_engine:
            res = await self._mock_engine.triage(text)
            res.engine_name = "jev-mock"
            return res

        url = f"{self.endpoint_url.rstrip('/')}/v1/systemone"
        payload = {
            "state": {"message": text},
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department should handle this customer inquiry?",
                    "criteria": {
                        "billing": "Invoice, chargeback, subscription, refund, payment issues",
                        "technical": "Software bugs, API failures, service downtime, errors",
                        "sales": "Plan upgrades, pricing inquiries, enterprise licensing",
                        "general": "General questions, documentation, feedback",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate the customer's operational urgency and impact from 0 to 3.",
                    "criteria": [
                        "0: General question, no business impact",
                        "1: Minor bug, workaround available",
                        "2: Significant blocker or time-sensitive task",
                        "3: Critical downtime, financial loss, or security issue",
                    ],
                },
                "churn_risk": {
                    "type": "noul",
                    "instructions": "Is the customer at risk of churning, cancelling, or threatening legal action?",
                },
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            answers = data.get("answers", {})
            dept_data = answers.get("department", {})
            dept = dept_data.get("choice", "general")
            dept_conf = float(dept_data.get("confidence", 0.95))
            dept_probs = dept_data.get("probabilities", {})

            urgency_data = answers.get("urgency", {})
            urgency = float(urgency_data.get("score", 1.0))
            urgency_probs = urgency_data.get("probabilities", {})

            churn = float(answers.get("churn_risk", {}).get("noul", 0.1))
            latency = (time.perf_counter() - start_time) * 1000.0

            urgency_map = {
                0: "Low (P2 - Informational)",
                1: "Normal (P2 - Minor/Standard)",
                2: "High (P1 - Imp/Urgent)",
                3: "Critical (P0 - Immediate Attention)",
            }
            level = min(3, max(0, round(urgency)))

            if churn >= 0.70 or level == 3:
                priority = "P0"
                is_escalation = True
                reasons = []
                if churn >= 0.70:
                    reasons.append(f"High Churn/Legal Risk ({churn:.1%})")
                if level == 3:
                    reasons.append(f"Critical Severity ({urgency:.1f}/3)")
                escalation_reason = " & ".join(reasons)
            elif level == 2:
                priority = "P1"
                is_escalation = False
                escalation_reason = None
            else:
                priority = "P2"
                is_escalation = False
                escalation_reason = None

            return JevDecisionResult(
                department=dept,
                department_confidence=dept_conf,
                department_probabilities=dept_probs,
                urgency_score=round(urgency, 2),
                urgency_level=level,
                urgency_description=urgency_map.get(level, "Normal"),
                urgency_probabilities=urgency_probs,
                churn_risk_probability=round(churn, 3),
                priority=priority,
                is_escalation=is_escalation,
                escalation_reason=escalation_reason,
                engine_name=self.engine_name,
                latency_ms=round(latency, 2),
                raw_answers=answers,
            )
        except Exception as exc:
            if self.fallback_to_mock and self._mock_engine:
                logger.warning("Remote Jev triage failed (%s). Falling back to mock.", exc)
                res = await self._mock_engine.triage(text)
                res.engine_name = "jev-fallback"
                return res
            raise
