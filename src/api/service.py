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

        # Check if paused at human review interrupt
        if state.next and "human_review" in state.next:
            interrupt_val = {}
            if state.tasks and state.tasks[0].interrupts:
                interrupt_val = state.tasks[0].interrupts[0].value or {}

            pending_item = PendingTicketItem(
                ticket_id=ticket_id,
                customer_id=request.customer_id,
                text=request.text,
                detected_language=values.get("detected_language", "english"),
                action=values.get("decision_action", "unknown"),
                confidence=values.get("decision_confidence", 0.0),
                amount=values.get("extracted_amount"),
                reason=interrupt_val.get("reason", values.get("reviewer_notes", "Requires human review")),
                priority=values.get("priority", "P2"),
                draft_reply=values.get("draft_reply"),
                retrieved_policies=values.get("retrieved_policies", []),
            )
            self._pending_tickets[ticket_id] = pending_item

            lang = values.get("detected_language", "english")
            if lang == "hindi":
                review_reply = "नमस्ते, आपके अनुरोध के लिए सुपरवाइजर सत्यापन की आवश्यकता है। इसे हमारे सपोर्ट डेस्क पर भेज दिया गया है, जल्द ही समाधान मिलेगा।"
            elif lang == "hinglish":
                review_reply = "Hi, aapki request ke liye supervisor verification ki zaroorat hai. Humne isse support desk par forward kar diya hai, jald update milega."
            elif lang == "french":
                review_reply = "Votre demande nécessite la validation d'un superviseur et a été transmise à notre équipe de support."
            elif lang == "spanish":
                review_reply = "Su solicitud requiere verificación por parte de un supervisor y ha sido transferida a nuestro equipo de soporte."
            elif lang == "german":
                review_reply = "Ihre Anfrage erfordert eine Überprüfung durch einen Supervisor und wurde an unser Support-Team weitergeleitet."
            else:
                review_reply = "Your request requires supervisor verification and has been routed to our human support desk."

            return TicketResponse(
                ticket_id=ticket_id,
                status="needs_review",
                decision_action=values.get("decision_action", "unknown"),
                decision_confidence=values.get("decision_confidence", 0.0),
                reply=review_reply,
                tool_result=None,
                gating_outcome=values.get("gating_outcome", "human_review"),
                detected_language=lang,
                requires_human_review=True,
                priority=values.get("priority", "P2"),
                trajectory=values.get("trajectory", []),
            )

        # Completed automatically without interrupt
        return TicketResponse(
            ticket_id=ticket_id,
            status="completed",
            decision_action=values.get("decision_action", "unknown"),
            decision_confidence=values.get("decision_confidence", 0.0),
            reply=values.get("draft_reply"),
            tool_result=values.get("tool_result"),
            gating_outcome=values.get("gating_outcome", "auto_execute"),
            detected_language=values.get("detected_language", "english"),
            requires_human_review=False,
            priority=values.get("priority", "P2"),
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

        # Remove from pending queue
        self._pending_tickets.pop(ticket_id, None)

        return TicketResponse(
            ticket_id=ticket_id,
            status="completed",
            decision_action=values.get("decision_action", "unknown"),
            decision_confidence=values.get("decision_confidence", 0.0),
            reply=values.get("draft_reply"),
            tool_result=values.get("tool_result"),
            gating_outcome=values.get("gating_outcome", "human_review"),
            detected_language=values.get("detected_language", "english"),
            requires_human_review=False,
            priority=values.get("priority", "P2"),
            trajectory=values.get("trajectory", []),
        )


# Singleton workflow service
workflow_service = TicketWorkflowService()
