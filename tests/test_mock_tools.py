"""Tests for tools and compliant MCP server."""
import pytest

from src.tools.mcp_server import mcp_server
from src.tools.mock_tools import cancel_subscription, escalate_to_team, execute_refund


def test_refund_tool_valid():
    """Validates successful refund execution with valid amount and audit log."""
    res = execute_refund(ticket_id="TK-101", amount_usd=45.00, reason="Within 14-day SLA")
    assert res.success is True
    assert res.tool_name == "refund"
    assert res.data["refund_amount_usd"] == 45.00
    assert res.data["status"] == "COMPLETED"
    assert res.transaction_id.startswith("tx_")
    assert res.audit_entry["action_type"] == "FINANCIAL_REIMBURSEMENT"


def test_refund_tool_invalid_amount():
    """Validates that negative or zero refund amounts are rejected."""
    res = execute_refund(ticket_id="TK-102", amount_usd=-10.00, reason="Invalid negative")
    assert res.success is False
    assert "strictly greater than $0.00" in res.error

    res_zero = execute_refund(ticket_id="TK-103", amount_usd=0.00, reason="Zero amount")
    assert res_zero.success is False


def test_cancel_subscription_tool():
    """Validates subscription cancellation (immediate vs standard end-of-cycle)."""
    res_default = cancel_subscription(ticket_id="TK-201", customer_id="CUST-99", immediate=False)
    assert res_default.success is True
    assert "end of current billing cycle" in res_default.data["effective_date"]

    res_imm = cancel_subscription(ticket_id="TK-202", customer_id="CUST-99", immediate=True)
    assert res_imm.success is True
    assert res_imm.data["effective_date"] == "Immediately"


def test_escalate_to_team_tool():
    """Validates team escalation and P0 oncall paging."""
    res_p0 = escalate_to_team(ticket_id="TK-301", target_team="SecOps", priority="P0", notes="SSO down")
    assert res_p0.success is True
    assert res_p0.data["pager_status"] == "PAGED_ONCALL"

    res_p2 = escalate_to_team(ticket_id="TK-302", target_team="BillingOps", priority="P2", notes="Clarification")
    assert res_p2.success is True
    assert res_p2.data["pager_status"] == "NOT_PAGED"

    res_invalid = escalate_to_team(ticket_id="TK-303", target_team="Tier2", priority="INVALID", notes="Bad")
    assert res_invalid.success is False
    assert "Invalid priority" in res_invalid.error


@pytest.mark.asyncio
async def test_mcp_server_protocol_execution():
    """Validates that the official MCPServer registers tools and executes them via protocol."""
    tools = await mcp_server.list_tools()
    tool_names = [t.name for t in tools]
    assert "refund_action" in tool_names
    assert "cancel_subscription_action" in tool_names
    assert "escalate_ticket_action" in tool_names
    assert "search_policy_kb" in tool_names
    assert "classify_ticket" in tool_names

    # Call tool via standard MCP server API
    call_res = await mcp_server.call_tool(
        "refund_action",
        {"ticket_id": "TK-MCP-99", "amount_usd": 35.50, "reason": "MCP call test"}
    )
    assert not call_res.is_error
    assert len(call_res.content) > 0
    text_content = call_res.content[0].text
    assert '"success": true' in text_content
    assert '"refund_amount_usd": 35.5' in text_content
