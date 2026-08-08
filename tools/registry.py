"""Tool registry for the agent loop."""

from __future__ import annotations

import json
from typing import Any

from tools.base import Tool


class ToolRegistry:
    """Register tools and expose OpenAI specs + execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def local_names(self) -> list[str]:
        """Names of non-MCP (locally defined) tools only."""
        from mcp_client.tool import McpTool

        return sorted(
            name for name, tool in self._tools.items() if not isinstance(tool, McpTool)
        )

    def count(self) -> int:
        return len(self._tools)

    def local_count(self) -> int:
        return len(self.local_names())

    def specs(self) -> list[dict[str, Any]]:
        return [tool.to_openai_spec() for tool in self._tools.values()]

    def execute(
        self,
        name: str,
        arguments: str | dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> str:
        """Look up and run a tool. Returns a string result (or error message)."""
        from tools.session_context import reset_session_id, set_session_id

        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        if arguments is None or arguments == "":
            kwargs: dict[str, Any] = {}
        elif isinstance(arguments, dict):
            kwargs = arguments
        else:
            try:
                kwargs = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return f"Invalid tool arguments JSON: {exc}"

        token = set_session_id(session_id) if session_id else None
        try:
            try:
                result = tool.execute(**kwargs)
            except TypeError as exc:
                return f"Tool call error: {exc}"
            except Exception as exc:  # noqa: BLE001 — surface tool failures to the model
                return f"Tool execution error: {exc}"
        finally:
            if token is not None:
                reset_session_id(token)

        if isinstance(result, str):
            return result
        return json.dumps(result)
