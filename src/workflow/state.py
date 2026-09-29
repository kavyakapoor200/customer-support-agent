"""State definition for LangGraph customer support workflow."""
from typing import Any, TypedDict


class AgentState(TypedDict):
    """LangGraph state representation consisting exclusively of JSON-serializable primitives."""
    ticket_id: str
    customer_id: str
    raw_text: str
    detected_language: str       # "english" | "hinglish" | "hindi"
    detected_script: str         # "latin" | "devanagari"
    extracted_amount: float | None
    candidate_actions: list[str]
    decision_action: str
    decision_confidence: float
    probabilities: dict[str, float]
    retrieved_policies: list[dict[str, Any]]
    gating_outcome: str          # "auto_execute" | "human_review" | "clarify" | "deny"
    review_status: str | None    # "pending" | "approved" | "rejected"
    reviewer_notes: str | None
    draft_reply: str | None
    verification_passed: bool
    final_action_taken: str | None
    tool_result: dict[str, Any] | None
    priority: str                # "P0" | "P1" | "P2"
    trajectory: list[dict[str, Any]]
