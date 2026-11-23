"""Adapter that turns an MCP tool into a local Tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tools.base import Tool

if TYPE_CHECKING:
    from mcp_client.hub import MultiServerMcpClient


class McpTool(Tool):
    """Thin wrapper: LLM tool call → MultiServerMcpClient.call_tool."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        client: MultiServerMcpClient,
        capability: str | None = None,
        capability_description: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._client = client
        self.capability = capability
        self.capability_description = capability_description

    def execute(self, **kwargs: Any) -> Any:
        return self._client.call_tool(self.name, kwargs)
