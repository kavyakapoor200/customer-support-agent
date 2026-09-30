"""Standard Model Context Protocol (MCP) server exposing customer support tools."""
from src.mcp.server import mcp_server, run_mcp_server

run_server = run_mcp_server

__all__ = ["mcp_server", "run_server"]

