# Slice 03: Policy KB (Qdrant) & Sandboxed Operational Tools

## 1. Contract Unlocked
Implements vector search over company support policies using Qdrant (supporting both local Docker and embedded in-memory mode) and establishes sandbox mock tools for safe side-effect simulation.

## 2. API Seam & Module Ownership
* **Policy KB Module (`src/kb/`):**
  * `src/kb/store.py`: `PolicyStore` class wrapping `QdrantClient`.
    * Supports `QdrantClient(":memory:")` for CI/testing without Docker.
    * Method: `search_policies(query: str, limit: int = 3) -> list[PolicySnippet]`.
    * Method: `ingest_markdown_policies(directory: Path) -> int`.
  * `data/policies/`:
    * `01_refund_sla.md`: 14-day SLA, <$50 auto-approval rule, annual renewal dispute exception.
    * `02_subscription_cancellation.md`: Billing cycle termination, mid-cycle prorating rules.
    * `03_security_escalation.md`: P0 credentials, SSO lockout, unauthorized access protocol.
* **Mock Tools Module (`src/tools/`):**
  * Sandbox tools returning typed execution results with zero real external side effects:
    * `execute_refund(ticket_id: str, amount_usd: float, reason: str) -> ToolResult`
    * `cancel_subscription(ticket_id: str, customer_id: str, immediate: bool) -> ToolResult`
    * `escalate_to_team(ticket_id: str, target_team: str, priority: str, notes: str) -> ToolResult`
  * Each execution produces an `AuditEntry` logged with timestamp, simulated transaction ID, and status.

## 3. What the Human Can Run or See
* Run the ingestion & query probe:
  `python -m src.kb.cli --query "Can I get a refund after 20 days?"`
  Returns matched policy markdown chunk, score, and citation header.
* Run mock tool probe:
  `python -m src.tools.cli --tool refund --amount 49.00`
  Returns simulated JSON audit log entry with transaction hash.

## 4. Verification Gates & Tests
* `tests/test_kb.py`:
  * Tests policy ingestion and semantic retrieval using in-memory Qdrant.
  * Asserts refund policy chunk is top-1 match for refund-related queries.
* `tests/test_mock_tools.py`:
  * Tests that mock tools correctly validate amounts (rejecting negative numbers).
  * Verifies audit record formatting.
* Command gate: `pytest tests/test_kb.py tests/test_mock_tools.py` passes 100%.

## 5. Delegated Implementer Discretion
* Embedding model choice for local search: `fastembed` (lightweight ONNX runtime) or sentence-transformers.
* Internal document chunking strategy (e.g. Markdown header split vs character window).

## 6. What Must Stay Green
* Tests must run in CI using in-memory Qdrant (`":memory:"`) without requiring an active Docker daemon.
* Mock tools must never attempt external network calls.

## 7. Non-blocking Checkpoint
Review policy markdown text in `data/policies/`. If no modifications submitted, proceed with default SLA rules.
