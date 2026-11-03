from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agents.loop import run_tool_loop
from connectors.registry import ConnectorRegistry
from connectors.whoop import WhoopConnector
from llm.base import LLMClient
from mcp_client.token_store import FileTokenStore
from tools.base import Tool
from tools.events import connector_tools
from tools.registry import ToolRegistry
from tools.snaptrade import (
    LinkSnaptradeTool,
    SnaptradeStatusTool,
    UnlinkSnaptradeTool,
)


class CountingTool(Tool):
    name = "counter"
    description = "Counts calls."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return f"count={self.calls}"


class ScriptedLLM(LLMClient):
    """Return the scripted tool calls in order, then a final text reply."""

    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._calls = list(calls)
        self.prompts: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, temperature=0.7, timeout=None) -> dict[str, Any]:
        self.prompts.append(messages)
        if not self._calls:
            return {"role": "assistant", "content": "done"}
        name, args = self._calls.pop(0)
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call{len(self.prompts)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }],
        }


class RepeatedToolCallTests(unittest.TestCase):
    def _run(self, calls: list[tuple[str, dict[str, Any]]], max_iterations: int = 5):
        tool = CountingTool()
        registry = ToolRegistry()
        registry.register(tool)
        llm = ScriptedLLM(calls)
        history: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        replies = list(run_tool_loop(
            llm, history, registry, session_id="+15550000000",
            max_iterations=max_iterations,
        ))
        return tool, history, replies

    def test_identical_repeat_is_not_run_and_model_is_told(self) -> None:
        tool, history, replies = self._run([
            ("counter", {"n": 1}),
            ("counter", {"n": 1}),
        ])

        self.assertEqual(tool.calls, 1)
        self.assertEqual(replies, ["done"])
        tool_results = [m["content"] for m in history if m["role"] == "tool"]
        self.assertEqual(tool_results[0], "count=1")
        self.assertIn("already called counter", tool_results[1])

    def test_different_arguments_still_run(self) -> None:
        tool, _, replies = self._run([
            ("counter", {"n": 1}),
            ("counter", {"n": 2}),
        ])

        self.assertEqual(tool.calls, 2)
        self.assertEqual(replies, ["done"])


class ConnectorToolScopeTests(unittest.TestCase):
    def test_account_tools_are_chat_agent_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            connectors = ConnectorRegistry()
            connectors.register(WhoopConnector(
                client_id="client",
                client_secret="secret",
                redirect_url="https://agent.example/oauth/whoop/callback",
                token_store=FileTokenStore(Path(temp) / "oauth"),
            ))
            registry = ToolRegistry()
            for tool in connector_tools(repository=None, connectors=connectors):  # type: ignore[arg-type]
                registry.register(tool)

        self.assertEqual(registry.names(chat_agent=False), ["query_events"])
        self.assertEqual(
            registry.names(chat_agent=True),
            ["link_whoop", "unlink_whoop", "whoop_status"],
        )
        for cls in (LinkSnaptradeTool, SnaptradeStatusTool, UnlinkSnaptradeTool):
            self.assertTrue(cls(client=None).is_chat_agent_tool())  # type: ignore[arg-type]
