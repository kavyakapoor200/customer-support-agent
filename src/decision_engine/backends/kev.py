"""Kev-0.8B Decision Engine backend connecting to local System 1 endpoint."""
import logging
import time

import httpx

from src.core.config import get_settings
from src.decision_engine.backends.mock import MockDecisionEngine
from src.decision_engine.base import BaseDecisionEngine, DecisionOutput, JevDecisionResult
from src.decision_engine.calibration import apply_temperature_scaling

logger = logging.getLogger(__name__)


class KevDecisionEngine(BaseDecisionEngine):
    """Decision engine client for Jared Palmer's Kev-0.8B running locally."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        fallback_to_mock: bool = False,
        timeout: float = 10.0,
        temperature: float | None = None,
    ) -> None:
        super().__init__(engine_name="kev")
        self.endpoint_url = endpoint_url or get_settings().KEV_ENDPOINT_URL
        self.fallback_to_mock = fallback_to_mock
        self.timeout = timeout
        self.temperature = temperature if temperature is not None else get_settings().TEMPERATURE
        self._mock_engine = MockDecisionEngine() if fallback_to_mock else None

    async def decide(self, text: str, candidate_actions: list[str]) -> DecisionOutput:
        start_time = time.perf_counter()
        criteria_descriptions = {
            "refund": "Requesting a refund, reimbursement, or charge reversal",
            "cancel_subscription": "Requesting to cancel subscription, membership, or renewal",
            "billing_dispute": "Reporting an unauthorized charge, fraud, or billing discrepancy",
            "account_escalation": "Urgent blocker, account locked out, SSO/SAML failure, or security breach",
            "general_inquiry": "General question, documentation, feature inquiry, or contact support",
        }
        criteria = {
            act: criteria_descriptions.get(act, f"Customer request regarding {act.replace('_', ' ')}")
            for act in candidate_actions
        }

        payload = {
            "model": "kev-latest",
            "state": text,
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "Which customer support action best resolves this customer query?",
                    "criteria": criteria,
                }
            },
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

            answers = data.get("answers", {})
            action_choice = answers.get("action", {})
            if isinstance(action_choice, dict) and "choice" in action_choice:
                action = action_choice["choice"]
                probabilities = action_choice.get("probabilities") or {a: 1.0 / len(candidate_actions) for a in candidate_actions}
                confidence = float(probabilities.get(action, action_choice.get("confidence", 0.0)))
            else:
                action = data.get("action") or candidate_actions[0]
                confidence = float(data.get("confidence", 0.0))
                probabilities = data.get("probabilities") or {a: 1.0 / len(candidate_actions) for a in candidate_actions}

            raw_probabilities = dict(probabilities)
            raw_confidence = confidence

            if self.temperature > 0.0 and abs(self.temperature - 1.0) > 1e-4:
                probabilities = apply_temperature_scaling(probabilities, candidate_actions, self.temperature)
                confidence = float(probabilities.get(action, confidence))

            return DecisionOutput(
                action=action,
                confidence=confidence,
                probabilities=probabilities,
                raw_scores={
                    "raw_probabilities": raw_probabilities,
                    "raw_confidence": raw_confidence,
                    **data.get("raw_scores", {}),
                },
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
                return DecisionOutput(
                    action=output.action,
                    confidence=output.confidence,
                    probabilities=output.probabilities,
                    raw_scores=output.raw_scores,
                    engine_name="kev-fallback",
                    latency_ms=output.latency_ms,
                )
            raise RuntimeError(
                f"Kev-0.8B inference call to {self.endpoint_url} failed: {exc}. "
                "Ensure local kev-serve is running or specify fallback_to_mock=True."
            ) from exc

    async def triage(self, text: str) -> JevDecisionResult:
        """Evaluates Jev primitives (Department Choice, Urgency Score, Churn Risk Noul) on Kev."""
        start_time = time.perf_counter()
        payload = {
            "model": "kev-latest",
            "state": text,
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

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint_url.rstrip('/')}/v1/systemone",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            latency = (time.perf_counter() - start_time) * 1000.0
            answers = data.get("answers", {})

            # Department Choice
            dept_ans = answers.get("department", {})
            dept = dept_ans.get("choice", "general")
            dept_conf = float(dept_ans.get("confidence", 0.9))
            dept_probs = dept_ans.get("probabilities", {dept: dept_conf})

            # Urgency Score
            urgency_ans = answers.get("urgency", {})
            urgency_score = float(urgency_ans.get("score", 1.0))
            urgency_probs = urgency_ans.get("probabilities", {})

            # Churn Risk Noul
            churn_ans = answers.get("churn_risk", {})
            churn_prob = float(churn_ans.get("noul", 0.05))

            return self._build_triage_result(
                dept=dept,
                dept_conf=dept_conf,
                dept_probs=dept_probs,
                urgency_score=urgency_score,
                urgency_probs=urgency_probs,
                churn_prob=churn_prob,
                raw_answers=answers,
                latency_ms=latency,
            )

        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if self.fallback_to_mock and self._mock_engine:
                logger.warning(
                    "Kev triage failed on %s (%s). Falling back to mock triage.",
                    self.endpoint_url,
                    exc,
                )
                res = await self._mock_engine.triage(text)
                res.engine_name = "kev-fallback"
                return res
            raise RuntimeError(
                f"Kev-0.8B triage call to {self.endpoint_url} failed: {exc}. "
                "Ensure local kev-serve is running or specify fallback_to_mock=True."
            ) from exc

    def _build_triage_result(
        self,
        dept: str,
        dept_conf: float,
        dept_probs: dict[str, float],
        urgency_score: float,
        urgency_probs: dict[int, float],
        churn_prob: float,
        raw_answers: dict,
        latency_ms: float,
    ) -> JevDecisionResult:
        urgency_map = {
            0: "Low (P2 - Informational)",
            1: "Normal (P2 - Minor/Standard)",
            2: "High (P1 - Imp/Urgent)",
            3: "Critical (P0 - Immediate Attention)",
        }
        level = min(3, max(0, round(urgency_score)))

        if churn_prob >= 0.70 or level == 3:
            priority = "P0"
            is_escalation = True
            reasons = []
            if churn_prob >= 0.70:
                reasons.append(f"High Churn/Legal Risk ({churn_prob:.1%})")
            if level == 3:
                reasons.append(f"Critical Severity ({urgency_score:.1f}/3)")
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
            department_confidence=round(dept_conf, 4),
            department_probabilities=dept_probs,
            urgency_score=round(urgency_score, 2),
            urgency_level=level,
            urgency_description=urgency_map.get(level, "Normal"),
            urgency_probabilities=urgency_probs,
            churn_risk_probability=round(churn_prob, 3),
            priority=priority,
            is_escalation=is_escalation,
            escalation_reason=escalation_reason,
            engine_name=self.engine_name,
            latency_ms=round(latency_ms, 2),
            raw_answers=raw_answers,
        )
