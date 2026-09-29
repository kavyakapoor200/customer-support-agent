"""State definition for LangGraph customer support workflow."""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """LangGraph state representation consisting exclusively of JSON-serializable primitives."""
    ticket_id: str
    customer_id: str
    raw_text: str
    detected_language: str
    detected_script: str
    language_confidence: float
    is_supported_primary: bool

    # Jev Decision Primitives
    department: str
    department_confidence: float
    department_probabilities: dict[str, float]
    urgency_score: float
    urgency_level: int
    urgency_description: str
    urgency_probabilities: dict[int, float]
    churn_risk_probability: float

    # Priority & Triage Gate
    priority: str                # "P0" | "P1" | "P2"
    is_escalation: bool
    escalation_reason: str | None
    triage_action: str          # "ESCALATE_HUMAN" | "AUTOMATED_LLM_RESPONSE"

    # Response & Review
    draft_reply: str | None
    reply: str | None
    review_status: str | None   # "pending" | "approved" | "rejected"
    reviewer_notes: str | None
    requires_human_review: bool

    # Legacy fields preserved for backward compatibility
    extracted_amount: float | None
    candidate_actions: list[str]
    decision_action: str
    decision_confidence: float
    probabilities: dict[str, float]
    retrieved_policies: list[dict[str, Any]]
    gating_outcome: str
    verification_passed: bool
    final_action_taken: str | None
    tool_result: dict[str, Any] | None

    trajectory: list[dict[str, Any]]
