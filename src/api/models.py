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
    decision_action: str
    decision_confidence: float
    reply: str | None = None
    tool_result: dict[str, Any] | None = None
    gating_outcome: str
    detected_language: str
    requires_human_review: bool


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
    action: str
    confidence: float
    amount: float | None
    reason: str
    draft_reply: str | None
    retrieved_policies: list[dict[str, Any]]
