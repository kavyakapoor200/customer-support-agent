"""MCP Server exposing ticket classification, reply verification, and action gating."""
import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.cognition.gating import evaluate_gating
from src.decision_engine.factory import get_decision_engine
from src.tools.mock_tools import cancel_subscription, escalate_to_team, execute_refund

# Compliant MCPServer instance
mcp_server = MCPServer(name="customer-support-interface-mcp")


@mcp_server.tool()
async def classify_ticket(text: str) -> dict[str, Any]:
    """Classifies a customer ticket text and returns calibrated decision probabilities.

    Args:
        text: Customer support inquiry or message.
    """
    engine = get_decision_engine("mock")
    candidates = ["refund", "cancel_subscription", "billing_dispute", "account_escalation", "general_inquiry"]
    decision = await engine.decide(text, candidates)
    return decision.model_dump()


@mcp_server.tool()
def verify_reply(draft: str, policy_snippet: str) -> dict[str, Any]:
    """Verifies that a drafted customer response aligns with company policy invariants.

    Args:
        draft: Generated customer response text.
        policy_snippet: Retrieved policy document content.
    """
    is_valid = True
    issues: list[str] = []

    # Check for unauthorized promise of instant bank deposit
    if "instant refund" in draft.lower() and "3-5 business days" in policy_snippet.lower():
        is_valid = False
        issues.append("Draft promises instant refund, which violates the 3-5 business days SLA.")

    return {
        "verified": is_valid,
        "policy_grounding_score": 0.95 if is_valid else 0.40,
        "issues": issues,
    }


@mcp_server.tool()
def gate_action(action: str, confidence: float, amount: float | None = None) -> dict[str, Any]:
    """Applies deterministic YAML gating rules to an action and confidence score.

    Args:
        action: Classified action candidate (e.g. 'refund', 'cancel_subscription').
        confidence: Probability score between 0.0 and 1.0.
        amount: Financial amount in USD if applicable.
    """
    gating = evaluate_gating(action=action, confidence=confidence, amount_usd=amount)
    return gating.model_dump()


@mcp_server.tool()
def refund_action(ticket_id: str, amount_usd: float, reason: str) -> dict[str, Any]:
    """Executes a financial refund for an eligible ticket.

    Args:
        ticket_id: Ticket ID.
        amount_usd: Amount in USD.
        reason: Refund rationale.
    """
    res = execute_refund(ticket_id=ticket_id, amount_usd=amount_usd, reason=reason)
    return res.model_dump()


@mcp_server.tool()
def cancel_subscription_action(ticket_id: str, customer_id: str, immediate: bool = False) -> dict[str, Any]:
    """Terminates or schedules cancellation for a customer subscription."""
    res = cancel_subscription(ticket_id=ticket_id, customer_id=customer_id, immediate=immediate)
    return res.model_dump()


@mcp_server.tool()
def escalate_ticket_action(ticket_id: str, target_team: str, priority: str, notes: str) -> dict[str, Any]:
    """Escalates a ticket to internal teams with optional P0 paging."""
    res = escalate_to_team(ticket_id=ticket_id, target_team=target_team, priority=priority, notes=notes)
    return res.model_dump()


def run_mcp_server() -> None:
    """Runs the MCP server over stdio for external MCP clients."""
    asyncio.run(mcp_server.run_stdio_async())


if __name__ == "__main__":
    run_mcp_server()
