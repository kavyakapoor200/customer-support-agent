"""Typed data models and audit schemas for MCP tools."""
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized response from tool execution."""
    success: bool = Field(..., description="Whether the tool execution succeeded.")
    tool_name: str = Field(..., description="Name of the executed tool.")
    transaction_id: str = Field(..., description="Unique transaction/audit identifier.")
    data: dict[str, Any] = Field(default_factory=dict, description="Result payload.")
    audit_entry: dict[str, Any] = Field(..., description="Audit record for governance logging.")
    error: str | None = Field(default=None, description="Error message if execution failed.")


class AuditEntry(BaseModel):
    """Immutable audit record for every tool execution."""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp."
    )
    tool_name: str
    action_type: str
    actor: str = "customer_support_agent"
    ticket_id: str
    input_parameters: dict[str, Any]
    output_status: str
    transaction_hash: str
