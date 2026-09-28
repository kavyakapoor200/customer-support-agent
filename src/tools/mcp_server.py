"""Standard Model Context Protocol (MCP) server exposing customer support tools."""
import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.decision_engine.factory import get_decision_engine
from src.kb.store import PolicyStore
from src.tools.mock_tools import cancel_subscription, escalate_to_team, execute_refund

# Initialize official MCPServer
mcp_server = MCPServer(name="customer-support-mcp-server")


@mcp_server.tool()
def refund_action(ticket_id: str, amount_usd: float, reason: str) -> dict[str, Any]:
    """Issues a financial refund for an eligible customer support ticket.

    Args:
        ticket_id: Support ticket ID (e.g. TK-1029).
        amount_usd: Amount in USD to refund (must be > 0.00).
        reason: Explanation for the refund (e.g. accidental renewal within 14-day SLA).
    """
    res = execute_refund(ticket_id=ticket_id, amount_usd=amount_usd, reason=reason)
    return res.model_dump()


@mcp_server.tool()
def cancel_subscription_action(ticket_id: str, customer_id: str, immediate: bool = False) -> dict[str, Any]:
    """Cancels a customer's recurring SaaS subscription.

    Args:
        ticket_id: Support ticket ID.
        customer_id: Customer or workspace identifier.
        immediate: If True, terminates immediately; otherwise cancels at end of current cycle.
    """
    res = cancel_subscription(ticket_id=ticket_id, customer_id=customer_id, immediate=immediate)
    return res.model_dump()


@mcp_server.tool()
def escalate_ticket_action(ticket_id: str, target_team: str, priority: str, notes: str) -> dict[str, Any]:
    """Escalates a support ticket to internal teams (e.g. SecOps, BillingOps, Tier2).

    Args:
        ticket_id: Support ticket ID.
        target_team: Destination team (e.g. 'SecOps', 'BillingOps', 'Tier-2').
        priority: Priority level: 'P0' (emergency on-call page), 'P1', 'P2', or 'P3'.
        notes: Critical context and blocker details for the escalation.
    """
    res = escalate_to_team(ticket_id=ticket_id, target_team=target_team, priority=priority, notes=notes)
    return res.model_dump()


@mcp_server.tool()
def search_policy_kb(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Searches company customer support policy documents in the Qdrant knowledge base.

    Args:
        query: Search query (e.g. 'Can customer get refund after 20 days?').
        limit: Maximum number of policy snippets to retrieve (default: 3).
    """
    store = PolicyStore(url=":memory:")
    store.ingest_markdown_policies("data/policies")
    results = store.search_policies(query=query, limit=limit)
    return [r.model_dump() for r in results]


@mcp_server.tool()
async def classify_ticket(ticket_text: str) -> dict[str, Any]:
    """Classifies a customer support ticket using the calibrated DecisionEngine.

    Args:
        ticket_text: The customer's message or inquiry.
    """
    engine = get_decision_engine("mock")
    candidates = ["refund", "cancel_subscription", "billing_dispute", "account_escalation", "general_inquiry"]
    decision = await engine.decide(ticket_text, candidates)
    return decision.model_dump()


def run_server() -> None:
    """Entry point for running the MCP server over stdio."""
    asyncio.run(mcp_server.run_stdio_async())


if __name__ == "__main__":
    run_server()
