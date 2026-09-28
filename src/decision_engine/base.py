"""Base interfaces and data contracts for DecisionEngine backends."""
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator


class DecisionOutput(BaseModel):
    """Calibrated decision result returned by any DecisionEngine backend."""
    action: str = Field(..., description="Top classified action candidate.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score of the top action.")
    probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution across all candidate actions, summing to ~1.0."
    )
    raw_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Uncalibrated model logits or raw score values."
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


class BaseDecisionEngine(ABC):
    """Abstract base class that all System 1 and baseline decision engines implement."""

    def __init__(self, engine_name: str) -> None:
        self.engine_name = engine_name

    @abstractmethod
    async def decide(self, text: str, candidate_actions: list[str]) -> DecisionOutput:
        """Evaluates input text against candidate actions and returns calibrated probabilities.

        Args:
            text: Customer ticket or message text.
            candidate_actions: List of valid action strings.

        Returns:
            DecisionOutput with the chosen action, confidence, and full distribution.
        """

    @staticmethod
    def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
        """Converts raw scores into a normalized probability distribution using softmax."""
        import math
        if not scores:
            return {}
        max_score = max(scores.values())
        exp_scores = {k: math.exp((v - max_score) / max(temperature, 1e-4)) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        return {k: round(v / total_exp, 4) for k, v in exp_scores.items()}
