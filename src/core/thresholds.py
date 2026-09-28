"""Deterministic gating threshold schemas and loader."""
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ActionThresholds(BaseModel):
    """Threshold settings for a specific action category."""
    auto_execute_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to auto-execute an action without human review."
    )
    max_auto_amount_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum transaction amount in USD that can be auto-executed."
    )
    review_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score below auto_execute_threshold that triggers human-in-the-loop review."
    )
    deny_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score below which the action is denied or clarification is requested."
    )
    p0_always_review: bool = Field(
        default=False,
        description="If True, this action strictly requires human review regardless of confidence."
    )

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "ActionThresholds":
        """Ensures logical hierarchy: auto >= review >= deny."""
        if self.auto_execute_threshold < self.review_threshold:
            raise ValueError(
                f"auto_execute_threshold ({self.auto_execute_threshold}) must be >= "
                f"review_threshold ({self.review_threshold})"
            )
        if self.review_threshold < self.deny_threshold:
            raise ValueError(
                f"review_threshold ({self.review_threshold}) must be >= "
                f"deny_threshold ({self.deny_threshold})"
            )
        return self


class ThresholdConfig(BaseModel):
    """Full threshold configuration file representation."""
    version: str = "1.0"
    defaults: ActionThresholds
    policies: dict[str, ActionThresholds] = Field(default_factory=dict)

    def get_policy(self, action: str) -> ActionThresholds:
        """Retrieves specific action policy or falls back to global defaults."""
        return self.policies.get(action, self.defaults)


def load_thresholds(config_path: str | Path = "config/thresholds.yaml") -> ThresholdConfig:
    """Loads and validates threshold configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Thresholds configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    return ThresholdConfig.model_validate(data)
