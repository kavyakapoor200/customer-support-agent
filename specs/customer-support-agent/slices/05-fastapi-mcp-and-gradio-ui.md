# Slice 05: Interfaces (FastAPI, MCP Server & Gradio Dual-Portal)

## 1. Contract Unlocked
Exposes the multi-channel interface layer: production REST API (FastAPI), Model Context Protocol server (MCP), and interactive Gradio Dual-Portal mounted at `/ui` for customer chat and agent human-in-the-loop review.

## 2. API Seam & Module Ownership
* **Module:** `src/api/` and `src/ui/`
* **FastAPI Endpoints (`src/api/main.py`):**
  * `POST /api/v1/tickets`: Accepts customer ticket; executes LangGraph; returns immediately with completed resolution or `status: "needs_review"`.
  * `GET /api/v1/tickets/pending`: Lists all tickets currently paused at the human review interrupt.
  * `POST /api/v1/tickets/{ticket_id}/review`: Resumes graph execution with reviewer approval, rejection, or edited response.
  * `GET /health`: Healthcheck endpoint reporting readiness of PostgreSQL, Qdrant, and DecisionEngine.
* **MCP Server (`src/mcp/server.py`):**
  * Standard MCP protocol server exposing:
    * `classify_ticket(text: str) -> dict`
    * `verify_reply(draft: str, policy_snippet: str) -> dict`
    * `gate_action(action: str, confidence: float, amount: float | None) -> dict`
* **Gradio Dual-Portal (`src/ui/gradio_app.py`):**
  * Mounted at `/ui` on the FastAPI server using `gr.mount_gradio_app`.
  * **Tab 1: Customer Support Portal**
    * Interactive chat widget where users test tickets in English, Hinglish, or Hindi.
    * Shows instant response or "Sent to human support team" notification.
  * **Tab 2: Agent Review Desk**
    * Live queue of pending tickets.
    * Card view showing:
      * Ticket ID, customer text, and language/script tag.
      * Model decision confidence bar and probability breakdown.
      * Retrieved policy citation from Qdrant.
      * Editable response text area.
      * Action buttons: `[Approve & Execute]`, `[Edit & Send]`, `[Reject / Escalate]`.

## 3. What the Human Can Run or See
* Run the server:
  `uvicorn src.api.main:app --port 8000 --reload`
* Visit `http://localhost:8000/docs` for interactive Swagger OpenAPI documentation.
* Visit `http://localhost:8000/ui` to interact with both the Customer Chat and Agent Review tabs.

## 4. Verification Gates & Tests
* `tests/test_api.py`:
  * Tests `POST /api/v1/tickets` using `TestClient`.
  * Verifies auto-resolve returns HTTP 200 with final response.
  * Verifies interrupt returns `status: "needs_review"` and appears in `/api/v1/tickets/pending`.
  * Verifies `POST /api/v1/tickets/{id}/review` resumes execution and transitions status to `completed`.
* `tests/test_mcp.py`:
  * Tests MCP tool schemas and local invocations.
* Command gate: `pytest tests/test_api.py tests/test_mcp.py` passes 100%.

## 5. Delegated Implementer Discretion
* CSS styling and layout spacing inside Gradio blocks.
* Exact polling interval for Gradio pending queue refresh.

## 6. What Must Stay Green
* FastAPI tests must run using FastAPI `TestClient` without external networking.
* Gradio mount must not block or interfere with `/api/v1/` routes.

## 7. Non-blocking Checkpoint
Review Gradio layout and color scheme. If no feedback received within review window, proceed with default clean theme.
