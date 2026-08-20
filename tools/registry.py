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

    def _iter_tools(self, *, chat_agent: bool | None = None) -> list[Tool]:
        tools = [tool for tool in self._tools.values() if tool.is_enabled()]
        if chat_agent is True:
            return [tool for tool in tools if tool.is_chat_agent_tool()]
        if chat_agent is False:
            return [tool for tool in tools if not tool.is_chat_agent_tool()]
        return tools

    def names(self, *, chat_agent: bool | None = None) -> list[str]:
        return sorted(tool.name for tool in self._iter_tools(chat_agent=chat_agent))

    def local_names(self, *, chat_agent: bool | None = None) -> list[str]:
        """Names of non-MCP (locally defined) tools only."""
        from mcp_client.tool import McpTool

        return sorted(
            tool.name
            for tool in self._iter_tools(chat_agent=chat_agent)
            if not isinstance(tool, McpTool)
        )

    def count(self, *, chat_agent: bool | None = None) -> int:
        return len(self._iter_tools(chat_agent=chat_agent))

    def local_count(self, *, chat_agent: bool | None = None) -> int:
        return len(self.local_names(chat_agent=chat_agent))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def catalog_text(self, *, chat_agent: bool | None = None) -> str:
        """Human-readable name + description list for a system prompt."""
        tools = sorted(
            self._iter_tools(chat_agent=chat_agent),
            key=lambda tool: tool.name,
        )
        if not tools:
            return "- (none registered)"
        lines: list[str] = []
        for tool in tools:
            description = " ".join((tool.description or "").split())
            if len(description) > 220:
                description = description[:217] + "..."
            lines.append(f"- {tool.name}: {description}")
        return "\n".join(lines)

    def view(
        self,
        names: list[str],
        *,
        worker_only: bool = True,
    ) -> ToolRegistry:
        """A registry containing only the named tools (order preserved)."""
        subset = ToolRegistry()
        seen: set[str] = set()
        for name in names:
            if not name or name in seen:
                continue
            tool = self._tools.get(name)
            if tool is None:
                continue
            if worker_only and tool.is_chat_agent_tool():
                continue
            seen.add(name)
            subset.register(tool)
        return subset

    def specs(self, *, chat_agent: bool | None = None) -> list[dict[str, Any]]:
        """OpenAI tool specs, optionally filtered for ChatAgent vs worker."""
        return [tool.to_openai_spec() for tool in self._iter_tools(chat_agent=chat_agent)]

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
