"""Deterministic mock decision engine for fast, zero-weight test and CI runs."""
import time
from typing import ClassVar

from src.decision_engine.base import BaseDecisionEngine, DecisionOutput


class MockDecisionEngine(BaseDecisionEngine):
    """Deterministic, keyword-grounded mock decision engine with option-order invariance."""

    KEYWORD_MAP: ClassVar[dict[str, list[str]]] = {
        "refund": [
            "refund", "money back", "reimburse", "charged twice", "double charge",
            "duplicate charge", "duplicate deduction", "chargeback", "paise wapas", "refund kardo", "wapas",
            "galti se renewal", "wrongly billed", "accidental purchase", "accidental upgrade",
            "accidental in-app", "accidental annual", "accidental subscription renewal", "रिफंड"
        ],
        "cancel_subscription": [
            "cancel", "cancellation", "unsubscribe", "stop renewal", "stop subscription",
            "stop auto renewal", "auto renewal off", "terminate membership", "terminate my team",
            "terminate subscription", "downgrade to free", "band kardo", "hata do", "radd", "रद्द",
            "समाप्त", "रद्द करें"
        ],
        "billing_dispute": [
            "dispute", "unauthorized", "fraud", "stolen", "stolen card", "overcharged", "overbilled",
            "unknown charge", "did not purchase", "never bought", "never authorized",
            "धोखाधड़ी", "अनाधिकृत", "विवाद"
        ],
        "account_escalation": [
            "locked out", "sso", "saml", "okta", "compromised", "hacked", "p0", "urgent blocker",
            "security breach", "security emergency", "security alert", "login failed", "two-factor",
            "2fa lock", "oauth token", "domain lockout", "cannot login", "admin account is locked",
            "locked and production", "अति आवश्यक", "आपातकाल", "लॉक हो गया"
        ],
        "general_inquiry": [
            "how do", "how to", "where can", "features", "pricing", "guide", "documentation",
            "inquiry", "help", "invite team", "exporting billing", "change the billing currency",
            "uptime status", "update the credit card", "कहाँ उपलब्ध", "जानकारी चाहिए"
        ]
    }

    def __init__(self) -> None:
        super().__init__(engine_name="mock")

    async def decide(self, text: str, candidate_actions: list[str]) -> DecisionOutput:
        start_time = time.perf_counter()
        normalized_text = text.lower()

        # Score each candidate independently to guarantee option-order invariance
        raw_scores: dict[str, float] = {}
        for action in candidate_actions:
            keywords = self.KEYWORD_MAP.get(action, [])
            # Neutral baseline prior
            score = 0.5
            for kw in keywords:
                if kw in normalized_text:
                    score += 3.0
            raw_scores[action] = score

        # If no keywords matched any candidate, general_inquiry gets default boost
        if all(s == 0.5 for s in raw_scores.values()) and "general_inquiry" in raw_scores:
            raw_scores["general_inquiry"] = 2.0

        # Compute normalized probabilities
        probs = self.softmax(raw_scores, temperature=0.8)

        # Normalize total sum to exactly 1.0 to avoid float rounding discrepancies
        total = sum(probs.values())
        if total > 0 and total != 1.0:
            top_key = max(probs, key=probs.get)
            diff = 1.0 - total
            probs[top_key] = round(probs[top_key] + diff, 4)

        top_action = max(probs, key=probs.get)
        confidence = probs[top_action]
        latency = (time.perf_counter() - start_time) * 1000.0

        return DecisionOutput(
            action=top_action,
            confidence=confidence,
            probabilities=probs,
            raw_scores=raw_scores,
            engine_name=self.engine_name,
            latency_ms=round(latency, 2),
        )
