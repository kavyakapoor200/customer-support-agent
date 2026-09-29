"""Deterministic decision gating rules ("Model scores, code decides")."""
from pydantic import BaseModel, Field

from src.core.thresholds import ThresholdConfig, load_thresholds


class GatingDecision(BaseModel):
    """Result of evaluating a model action and confidence against YAML thresholds."""
    outcome: str = Field(..., description="'auto_execute', 'human_review', 'clarify', or 'deny'.")
    rationale: str = Field(..., description="Human-readable explanation of why this outcome was chosen.")
    requires_human: bool = Field(..., description="Whether execution must pause for human review.")
    threshold_applied: float = Field(..., description="The specific numeric threshold used.")
    priority: str = Field(default="P2", description="'P0', 'P1', or 'P2' severity level.")


def evaluate_gating(
    action: str,
    confidence: float,
    amount_usd: float | None = None,
    config: ThresholdConfig | None = None,
) -> GatingDecision:
    """Evaluates an action and confidence score against YAML threshold rules.

    Args:
        action: The classified candidate action.
        confidence: The calibrated probability score (0.0 to 1.0).
        amount_usd: Financial amount if applicable.
        config: Optional ThresholdConfig instance; loads config/thresholds.yaml by default.

    Returns:
        GatingDecision containing the outcome, rationale, human-review flag, and severity priority.
    """
    threshold_config = config or load_thresholds()
    policy = threshold_config.get_policy(action)

    # 1. Mandatory P0 Review Rule (Emergency/Security)
    if policy.p0_always_review or policy.priority == "P0":
        return GatingDecision(
            outcome="human_review",
            rationale=f"[P0] Action '{action}' is classified as P0 Critical Emergency and strictly requires human sign-off.",
            requires_human=True,
            threshold_applied=policy.auto_execute_threshold,
            priority="P0",
        )

    # 2. Financial Amount Limit Rule (P1 High Severity)
    if policy.max_auto_amount_usd > 0.0 and amount_usd is not None and amount_usd > policy.max_auto_amount_usd:
        return GatingDecision(
            outcome="human_review",
            rationale=(
                f"[P1] Requested amount ${amount_usd:.2f} exceeds auto-approval ceiling "
                f"of ${policy.max_auto_amount_usd:.2f}; routed to High-Priority Supervisor Queue."
            ),
            requires_human=True,
            threshold_applied=policy.auto_execute_threshold,
            priority="P1",
        )

    # Determine base priority for remaining paths
    base_priority = "P1" if policy.priority == "P1" else "P2"

    # 3. High-Confidence Auto-Execute Rule
    if confidence >= policy.auto_execute_threshold:
        return GatingDecision(
            outcome="auto_execute",
            rationale=(
                f"[{base_priority}] Confidence {confidence:.2f} meets or exceeds auto-execution threshold "
                f"({policy.auto_execute_threshold:.2f})."
            ),
            requires_human=False,
            threshold_applied=policy.auto_execute_threshold,
            priority=base_priority,
        )

    # 4. Borderline Human Review Rule
    if confidence >= policy.review_threshold:
        return GatingDecision(
            outcome="human_review",
            rationale=(
                f"[{base_priority}] Confidence {confidence:.2f} falls into review window "
                f"[{policy.review_threshold:.2f}, {policy.auto_execute_threshold:.2f})."
            ),
            requires_human=True,
            threshold_applied=policy.review_threshold,
            priority=base_priority,
        )

    # 5. Low-Confidence Clarification Rule
    if confidence >= policy.deny_threshold:
        return GatingDecision(
            outcome="clarify",
            rationale=f"[{base_priority}] Confidence {confidence:.2f} is ambiguous. Requesting customer clarification.",
            requires_human=False,
            threshold_applied=policy.deny_threshold,
            priority=base_priority,
        )

    # 6. Fallback Deny Rule
    return GatingDecision(
        outcome="deny",
        rationale=f"[{base_priority}] Confidence {confidence:.2f} is below minimum threshold ({policy.deny_threshold:.2f}).",
        requires_human=False,
        threshold_applied=policy.deny_threshold,
        priority=base_priority,
    )
