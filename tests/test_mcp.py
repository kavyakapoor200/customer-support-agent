"""Tests for the MCP interface server."""
import json

import pytest

from src.mcp.server import mcp_server


@pytest.mark.asyncio
async def test_mcp_tools_listing():
    """Validates that all expected tools are registered on the interface MCP server."""
    tools = await mcp_server.list_tools()
    names = [t.name for t in tools]

    expected = [
        "classify_ticket",
        "search_policy_kb",
        "verify_reply",
        "gate_action",
        "refund_action",
        "cancel_subscription_action",
        "escalate_ticket_action",
    ]
    for exp in expected:
        assert exp in names, f"Expected tool '{exp}' not found in MCP server."


@pytest.mark.asyncio
async def test_mcp_classify_ticket_tool():
    """Validates execution of classify_ticket MCP tool."""
    res = await mcp_server.call_tool("classify_ticket", {"text": "Cancel my membership"})
    assert not res.is_error
    data = json.loads(res.content[0].text)
    assert data["action"] == "cancel_subscription"
    assert data["confidence"] > 0.50


@pytest.mark.asyncio
async def test_mcp_verify_reply_tool():
    """Validates execution of verify_reply MCP tool."""
    res_valid = await mcp_server.call_tool(
        "verify_reply",
        {"draft": "Your refund will take 3-5 business days.", "policy_snippet": "Takes 3-5 business days."}
    )
    assert not res_valid.is_error
    data_valid = json.loads(res_valid.content[0].text)
    assert data_valid["verified"] is True

    res_invalid = await mcp_server.call_tool(
        "verify_reply",
        {"draft": "You get an instant refund right now!", "policy_snippet": "Standard bank takes 3-5 business days."}
    )
    data_invalid = json.loads(res_invalid.content[0].text)
    assert data_invalid["verified"] is False
    assert len(data_invalid["issues"]) > 0


@pytest.mark.asyncio
async def test_mcp_gate_action_tool():
    """Validates execution of gate_action MCP tool."""
    res = await mcp_server.call_tool(
        "gate_action",
        {"action": "refund", "confidence": 0.95, "amount": 40.0}
    )
    assert not res.is_error
    data = json.loads(res.content[0].text)
    assert data["outcome"] == "auto_execute"
    assert data["requires_human"] is False

    res_high = await mcp_server.call_tool(
        "gate_action",
        {"action": "refund", "confidence": 0.95, "amount": 150.0}
    )
    data_high = json.loads(res_high.content[0].text)
    assert data_high["outcome"] == "human_review"
    assert data_high["requires_human"] is True


@pytest.mark.asyncio
async def test_mcp_search_policy_kb_tool():
    """Validates execution of search_policy_kb MCP tool."""
    res = await mcp_server.call_tool(
        "search_policy_kb",
        {"query": "Can customer get refund after 14 days?", "limit": 2}
    )
    items = [json.loads(c.text) for c in res.content]
    assert len(items) > 0
    assert "policy_id" in items[0]

