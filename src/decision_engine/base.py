"""Base interfaces and data contracts for DecisionEngine backends."""
import math
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DecisionOutput(BaseModel):
    """Calibrated decision result returned by legacy candidate action classification."""
    action: str = Field(..., description="Top classified action candidate.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score of the top action.")
    probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution across all candidate actions, summing to ~1.0."
    )
    raw_scores: dict[str, Any] = Field(
        default_factory=dict,
        description="Uncalibrated model logits, raw scores, or metadata."
    )
    engine_name: str = Field(..., description="Name of the inference backend (e.g. kev, mock, groq).")
    latency_ms: float = Field(..., ge=0.0, description="Decision latency in milliseconds.")

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities_normalized(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensures all probabilities are in [0, 1] and sum to approximately 1.0."""
        if not v:
            raise ValueError("Probabilities dictionary cannot be empty.")
        total = sum(v.values())
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"Probabilities must sum to approximately 1.0, got: {total}")
        return v


class JevDecisionResult(BaseModel):
    """Calibrated decision primitives from Jev / Kev System 1: Choice, Score, and Noul."""
    department: str = Field(..., description="Department choice: billing, technical, sales, general.")
    department_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    department_probabilities: dict[str, float] = Field(default_factory=dict)
    urgency_score: float = Field(..., ge=0.0, le=3.0, description="Score primitive from 0 to 3.")
    urgency_level: int = Field(..., ge=0, le=3, description="Rounded urgency level 0, 1, 2, or 3.")
    urgency_description: str = Field(..., description="Human-readable urgency label.")
    urgency_probabilities: dict[int, float] = Field(default_factory=dict)
    churn_risk_probability: float = Field(..., ge=0.0, le=1.0, description="Noul primitive for churn risk.")
    priority: str = Field(..., description="'P0', 'P1', or 'P2'")
    is_escalation: bool = Field(..., description="True if ticket requires immediate human escalation.")
    escalation_reason: str | None = Field(default=None, description="Reason if escalated to human.")
    engine_name: str = Field(default="kev", description="Inference engine name.")
    latency_ms: float = Field(default=0.0, ge=0.0)
    raw_answers: dict[str, Any] = Field(default_factory=dict)


class BaseDecisionEngine(ABC):
    """Abstract base class that all System 1 and baseline decision engines implement."""

    def __init__(self, engine_name: str) -> None:
        self.engine_name = engine_name

    @abstractmethod
    async def decide(self, text: str, candidate_actions: list[str]) -> DecisionOutput:
        """Evaluates input text against candidate actions and returns calibrated probabilities."""

    @abstractmethod
    async def triage(self, text: str) -> JevDecisionResult:
        """Evaluates Jev primitives (Choice Department, Score Urgency, Noul Churn) to derive priority."""

    @staticmethod
    def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
        """Converts raw scores into a normalized probability distribution using softmax."""
        if not scores:
            return {}
        max_score = max(scores.values())
        exp_scores = {k: math.exp((v - max_score) / max(temperature, 1e-4)) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        return {k: round(v / total_exp, 4) for k, v in exp_scores.items()}
