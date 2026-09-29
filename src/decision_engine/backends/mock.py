"""Deterministic mock decision engine for fast, zero-weight test and CI runs."""
import time
from typing import ClassVar

from src.decision_engine.base import BaseDecisionEngine, DecisionOutput, JevDecisionResult


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
            score = 0.5
            for kw in keywords:
                if kw in normalized_text:
                    score += 3.0
            raw_scores[action] = score

        if all(s == 0.5 for s in raw_scores.values()) and "general_inquiry" in raw_scores:
            raw_scores["general_inquiry"] = 2.0

        probs = self.softmax(raw_scores, temperature=0.8)

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

    async def triage(self, text: str) -> JevDecisionResult:
        """Calibrated triage evaluation matching Jev experimentation primitives."""
        start_time = time.perf_counter()
        lower = text.lower()

        # 1. Department (Choice)
        if any(w in lower for w in ["invoice", "charge", "refund", "card", "billing", "receipt", "stripe", "vat", "payment", "$"]):
            dept = "billing"
        elif any(w in lower for w in ["bug", "error", "api", "crash", "timeout", "exception", "broken", "500", "403", "endpoint", "forbidden", "sandbox", "token"]):
            dept = "technical"
        elif any(w in lower for w in ["pricing", "enterprise", "quote", "discount", "license", "sales"]):
            dept = "sales"
        else:
            dept = "general"

        # 2. Churn risk (Noul probability)
        churn_prob = 0.05
        if any(w in lower for w in ["cancel", "switch to", "unsubscrib", "leaving", "lawyer", "legal", "sue", "breach", "chargeback", "had enough"]):
            churn_prob = 0.88
        elif any(w in lower for w in ["unhappy", "frustrated", "terrible", "disappointed", "angry"]):
            churn_prob = 0.45

        # 3. Urgency (Score 0-3)
        if any(w in lower for w in ["down", "production", "critical", "urgent", "asap", "immediately", "outage", "lost", "locked out"]):
            urgency = 3.0
        elif any(w in lower for w in ["blocked", "cannot proceed", "failing", "403", "forbidden", "error"]):
            urgency = 2.0
        elif any(w in lower for w in ["how do i", "where is", "documentation", "pdf", "invoice", "copy", "hello"]):
            urgency = 0.2
        else:
            urgency = 1.0

        urgency_map = {
            0: "Low (P2 - Informational)",
            1: "Normal (P2 - Minor/Standard)",
            2: "High (P1 - Imp/Urgent)",
            3: "Critical (P0 - Immediate Attention)",
        }
        level = min(3, max(0, round(urgency)))

        if churn_prob >= 0.70 or level == 3:
            priority = "P0"
            is_escalation = True
            reasons = []
            if churn_prob >= 0.70:
                reasons.append(f"High Churn/Legal Risk ({churn_prob:.1%})")
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

        latency = (time.perf_counter() - start_time) * 1000.0

        return JevDecisionResult(
            department=dept,
            department_confidence=0.92,
            department_probabilities={dept: 0.92},
            urgency_score=round(urgency, 2),
            urgency_level=level,
            urgency_description=urgency_map.get(level, "Normal"),
            urgency_probabilities={level: 0.9},
            churn_risk_probability=round(churn_prob, 3),
            priority=priority,
            is_escalation=is_escalation,
            escalation_reason=escalation_reason,
            engine_name=self.engine_name,
            latency_ms=round(latency, 2),
            raw_answers={"simulated": True},
        )
