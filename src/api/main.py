"""FastAPI application entrypoint mounting REST endpoints and Gradio Dual-Portal."""
from contextlib import asynccontextmanager
from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException, status

from src.api.models import (
    PendingTicketItem,
    ReviewActionRequest,
    TicketIntakeRequest,
    TicketResponse,
)
from src.api.service import workflow_service
from src.core.config import get_settings
from src.ui.gradio_app import create_gradio_ui


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Ensure Qdrant policies are ingested on startup
    from src.kb.store import PolicyStore
    try:
        store = PolicyStore(url=":memory:")
        store.ingest_markdown_policies("data/policies")
    except Exception:
        pass
    yield


app = FastAPI(
    title="Customer Support Agent API",
    description="Enterprise-grade support ticket triage, System 1 decision engine, and LangGraph HITL supervision.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint validating service readiness."""
    settings = get_settings()
    return {
        "status": "healthy",
        "app_env": settings.APP_ENV,
        "decision_engine_backend": settings.DECISION_ENGINE_BACKEND,
        "gradio_ui_path": "/ui",
    }


@app.post(
    "/api/v1/tickets",
    response_model=TicketResponse,
    status_code=status.HTTP_200_OK,
    tags=["Tickets"],
)
async def submit_ticket(payload: TicketIntakeRequest) -> TicketResponse:
    """Intakes a customer support ticket, executes the LangGraph state machine,

    and returns the resolution or flags human review.
    """
    try:
        return await workflow_service.intake_ticket(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ticket intake failed: {exc}",
        ) from exc


@app.get(
    "/api/v1/tickets/pending",
    response_model=list[PendingTicketItem],
    tags=["Review Queue"],
)
async def get_pending_review_tickets() -> list[PendingTicketItem]:
    """Lists all support tickets currently paused at the human review checkpoint."""
    return await workflow_service.list_pending()


@app.post(
    "/api/v1/tickets/{ticket_id}/review",
    response_model=TicketResponse,
    tags=["Review Queue"],
)
async def review_ticket_action(ticket_id: str, review: ReviewActionRequest) -> TicketResponse:
    """Resumes an interrupted ticket with supervisor approval, rejection, or edited response."""
    try:
        return await workflow_service.review_ticket(ticket_id, review)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active review thread for ticket '{ticket_id}' not found.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit ticket review: {exc}",
        ) from exc


# Mount Gradio Dual-Portal at /ui
gradio_interface = create_gradio_ui()
app = gr.mount_gradio_app(app, gradio_interface, path="/ui")
