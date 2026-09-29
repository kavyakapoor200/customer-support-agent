"""API request and response models."""
from typing import Any

from pydantic import BaseModel, Field


class TicketIntakeRequest(BaseModel):
    """Payload for submitting a new customer support ticket."""
    text: str = Field(..., min_length=1, description="Customer message or support query.")
    customer_id: str = Field(default="CUST-DEFAULT", description="Unique customer ID.")
    candidate_actions: list[str] | None = Field(
        default=None,
        description="Optional list of action candidates. Defaults to standard taxonomy."
    )


class TicketResponse(BaseModel):
    """Standardized response from ticket processing."""
    ticket_id: str
    status: str = Field(..., description="'completed' or 'needs_review'.")
    priority: str = Field(default="P2", description="'P0', 'P1', or 'P2' severity level.")
    department: str | None = Field(default=None, description="Classified department (billing, technical, sales, general).")
    urgency_score: float | None = Field(default=None, description="Calibrated urgency score (0 to 3).")
    urgency_description: str | None = Field(default=None, description="Urgency label.")
    churn_risk_probability: float | None = Field(default=None, description="Churn risk probability (0.0 to 1.0).")
    triage_action: str | None = Field(default=None, description="'ESCALATE_HUMAN' or 'AUTOMATED_LLM_RESPONSE'.")
    reply: str | None = None
    detected_language: str
    requires_human_review: bool
    tool_result: dict[str, Any] | None = None
    gating_outcome: str
    decision_action: str = "general"
    decision_confidence: float = 1.0
    trajectory: list[dict[str, Any]] = Field(default_factory=list)


class ReviewActionRequest(BaseModel):
    """Payload for approving, rejecting, or editing an interrupted ticket."""
    approved: bool = Field(default=True, description="Whether the human reviewer approves the action.")
    edited_reply: str | None = Field(default=None, description="Optional edited response to the customer.")
    notes: str | None = Field(default=None, description="Supervisor or reviewer justification notes.")


class PendingTicketItem(BaseModel):
    """Item representation in the human review queue."""
    ticket_id: str
    customer_id: str
    text: str
    detected_language: str
    priority: str = Field(default="P0", description="'P0', 'P1', or 'P2' severity level.")
    department: str = "general"
    urgency_score: float = 0.0
    churn_risk: float = 0.0
    action: str = "escalate"
    confidence: float = 1.0
    amount: float | None = None
    reason: str = "P0 Escalation"
    draft_reply: str | None = None
    retrieved_policies: list[dict[str, Any]] = Field(default_factory=list)
