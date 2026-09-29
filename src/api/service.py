"""Service layer orchestrating the LangGraph workflow for API and UI consumers."""
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.api.models import (
    PendingTicketItem,
    ReviewActionRequest,
    TicketIntakeRequest,
    TicketResponse,
)
from src.workflow.graph import create_customer_support_graph


class TicketWorkflowService:
    """Manages LangGraph execution, thread state tracking, and human review queues."""

    def __init__(self) -> None:
        self.checkpointer = MemorySaver()
        self.graph = create_customer_support_graph(checkpointer=self.checkpointer)
        self._threads: dict[str, str] = {}
        self._pending_tickets: dict[str, PendingTicketItem] = {}

    async def intake_ticket(self, request: TicketIntakeRequest) -> TicketResponse:
        """Processes a new ticket through the LangGraph state machine."""
        ticket_id = f"TK-{uuid.uuid4().hex[:6].upper()}"
        thread_id = ticket_id
        self._threads[ticket_id] = thread_id

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "ticket_id": ticket_id,
            "customer_id": request.customer_id,
            "raw_text": request.text,
            "candidate_actions": request.candidate_actions,
            "trajectory": [],
        }

        # Run stream until either completion or interrupt
        async for _ in self.graph.astream(initial_state, config=config):
            pass

        state = await self.graph.aget_state(config)
        values = state.values

        priority = values.get("priority", "P2")
        dept = values.get("department", "general")
        urgency = values.get("urgency_score", 0.0)
        churn = values.get("churn_risk_probability", 0.0)
        triage_action = values.get("triage_action", "AUTOMATED_LLM_RESPONSE")
        lang = values.get("detected_language", "english")

        # Check if paused at human review interrupt
        if state.next and "human_review" in state.next:
            interrupt_val = {}
            if state.tasks and state.tasks[0].interrupts:
                interrupt_val = state.tasks[0].interrupts[0].value or {}

            pending_item = PendingTicketItem(
                ticket_id=ticket_id,
                customer_id=request.customer_id,
                text=request.text,
                detected_language=lang,
                priority=priority,
                department=dept,
                urgency_score=urgency,
                churn_risk=churn,
                action=values.get("decision_action", dept),
                confidence=values.get("decision_confidence", 1.0),
                amount=values.get("extracted_amount"),
                reason=interrupt_val.get("reason", values.get("reviewer_notes", "P0 Escalation")),
                draft_reply=values.get("draft_reply"),
                retrieved_policies=values.get("retrieved_policies", []),
            )
            self._pending_tickets[ticket_id] = pending_item

            return TicketResponse(
                ticket_id=ticket_id,
                status="needs_review",
                priority=priority,
                department=dept,
                urgency_score=urgency,
                urgency_description=values.get("urgency_description"),
                churn_risk_probability=churn,
                triage_action=triage_action,
                decision_action=values.get("decision_action", dept),
                decision_confidence=values.get("decision_confidence", 1.0),
                reply=values.get("draft_reply") or "Your ticket has been escalated to our human specialist desk.",
                tool_result=None,
                gating_outcome=values.get("gating_outcome", "human_review"),
                detected_language=lang,
                requires_human_review=True,
                trajectory=values.get("trajectory", []),
            )

        # Completed automatically without interrupt
        return TicketResponse(
            ticket_id=ticket_id,
            status="completed",
            priority=priority,
            department=dept,
            urgency_score=urgency,
            urgency_description=values.get("urgency_description"),
            churn_risk_probability=churn,
            triage_action=triage_action,
            decision_action=values.get("decision_action", dept),
            decision_confidence=values.get("decision_confidence", 1.0),
            reply=values.get("draft_reply"),
            tool_result=values.get("tool_result"),
            gating_outcome=values.get("gating_outcome", "auto_execute"),
            detected_language=lang,
            requires_human_review=False,
            trajectory=values.get("trajectory", []),
        )

    async def list_pending(self) -> list[PendingTicketItem]:
        """Returns all tickets currently paused in the human review queue."""
        return list(self._pending_tickets.values())

    async def review_ticket(self, ticket_id: str, review: ReviewActionRequest) -> TicketResponse:
        """Resumes an interrupted ticket with the human reviewer's verdict."""
        if ticket_id not in self._threads:
            raise KeyError(f"Ticket ID '{ticket_id}' not found in active session threads.")

        thread_id = self._threads[ticket_id]
        config = {"configurable": {"thread_id": thread_id}}

        resume_command = Command(resume=review.model_dump())
        async for _ in self.graph.astream(resume_command, config=config):
            pass

        final_state = await self.graph.aget_state(config)
        values = final_state.values

        self._pending_tickets.pop(ticket_id, None)

        dept = values.get("department", "general")
        priority = values.get("priority", "P0")

        return TicketResponse(
            ticket_id=ticket_id,
            status="completed",
            priority=priority,
            department=dept,
            urgency_score=values.get("urgency_score"),
            urgency_description=values.get("urgency_description"),
            churn_risk_probability=values.get("churn_risk_probability"),
            triage_action=values.get("triage_action"),
            decision_action=values.get("decision_action", dept),
            decision_confidence=values.get("decision_confidence", 1.0),
            reply=values.get("draft_reply"),
            tool_result=values.get("tool_result"),
            gating_outcome=values.get("gating_outcome", "human_review"),
            detected_language=values.get("detected_language", "english"),
            requires_human_review=False,
            trajectory=values.get("trajectory", []),
        )


# Singleton workflow service
workflow_service = TicketWorkflowService()
