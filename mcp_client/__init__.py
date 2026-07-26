"""Multi-server MCP client that adapts remote tools into the local ToolRegistry."""

from mcp_client.config import McpServerConfig, load_mcp_config
from mcp_client.hub import MultiServerMcpClient
from mcp_client.tool import McpTool

__all__ = [
    "McpServerConfig",
    "McpTool",
    "MultiServerMcpClient",
    "load_mcp_config",
]
