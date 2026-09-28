"""Integration tests for FastAPI endpoints and review workflows."""
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Validates that /health returns HTTP 200 with service readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["gradio_ui_path"] == "/ui"


def test_ticket_intake_auto_execute():
    """Validates that an eligible ticket is automatically resolved."""
    payload = {
        "text": "I was charged twice, duplicate accidental renewal, please refund my money back for $45",
        "customer_id": "CUST-API-01",
    }
    response = client.post("/api/v1/tickets", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "completed"
    assert data["requires_human_review"] is False
    assert data["decision_action"] == "refund"
    assert data["gating_outcome"] == "auto_execute"
    assert data["reply"] is not None
    assert data["tool_result"]["success"] is True


def test_ticket_intake_and_human_review_lifecycle():
    """Validates end-to-end lifecycle for high-value ticket requiring human review."""
    # 1. Intake $180 ticket (exceeds $50 auto-execute limit)
    payload = {
        "text": "Please refund my company card $180 for annual license dispute",
        "customer_id": "CUST-API-02",
    }
    intake_resp = client.post("/api/v1/tickets", json=payload)
    assert intake_resp.status_code == 200
    intake_data = intake_resp.json()

    assert intake_data["status"] == "needs_review"
    assert intake_data["requires_human_review"] is True
    ticket_id = intake_data["ticket_id"]

    # 2. Check that ticket is listed in /api/v1/tickets/pending
    pending_resp = client.get("/api/v1/tickets/pending")
    assert pending_resp.status_code == 200
    pending_list = pending_resp.json()
    matching = [t for t in pending_list if t["ticket_id"] == ticket_id]
    assert len(matching) == 1
    assert matching[0]["amount"] == 180.0

    # 3. Submit human reviewer approval
    review_payload = {
        "approved": True,
        "edited_reply": "Supervisor verified: Your $180 refund is approved.",
        "notes": "Approved annual license exception.",
    }
    review_resp = client.post(f"/api/v1/tickets/{ticket_id}/review", json=review_payload)
    assert review_resp.status_code == 200
    review_data = review_resp.json()

    assert review_data["status"] == "completed"
    assert review_data["requires_human_review"] is False
    assert "Supervisor verified" in review_data["reply"]
    assert review_data["tool_result"]["success"] is True

    # 4. Verify ticket is removed from pending queue
    pending_after_resp = client.get("/api/v1/tickets/pending")
    pending_after = pending_after_resp.json()
    assert not any(t["ticket_id"] == ticket_id for t in pending_after)


def test_gradio_ui_mount():
    """Validates that Gradio portal is mounted at /ui and returns HTML."""
    response = client.get("/ui/")
    assert response.status_code in (200, 307) # 200 OK or 307 redirect to trailing slash
