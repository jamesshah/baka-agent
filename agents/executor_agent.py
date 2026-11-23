"""Worker agent that runs delegated tasks with a granted tool subset."""

from __future__ import annotations

import logging
from typing import Any

from agents.loop import run_tool_loop
from llm.base import LLMClient
from tools.registry import ToolRegistry
from tools.session_context import get_turn_id
from tracing import trace_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a worker agent. Use ONLY the tools provided in this request — do not assume others exist. Return a concise, factual result for the chat agent to relay. No greetings, chit-chat, or mentioning that you are a worker.

Never claim an action succeeded or an item was created/found unless a tool call in this turn actually returned that result. If a tool fails or is missing, report the exact error plainly.

When using Composio tools:
1. Call COMPOSIO_SEARCH_TOOLS first to discover exact tool slugs, input schemas, and connection status.
2. If a toolkit is not connected, call COMPOSIO_MANAGE_CONNECTIONS (when granted) and include the Connect Link URL in your result.
3. Call COMPOSIO_MULTI_EXECUTE_TOOL with the discovered tool_slug and schema-compliant arguments.
"""


class ExecutorAgent:
    """Ephemeral tool-calling worker. Fresh history per task."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_agent_iterations: int = 8,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_agent_iterations = max_agent_iterations

    def available_tool_names(self) -> list[str]:
        """Worker tools that ChatAgent may grant on spawn."""
        return self._tools.names(chat_agent=False)

    def available_capabilities(self) -> list[str]:
        """Capability group names exposed to ChatAgent for spawn_agent.tools."""
        return self._tools.capability_names(chat_agent=False)

    def run_task(
        self,
        session_id: str,
        task: str,
        *,
        tool_names: list[str] | None = None,
    ) -> str:
        """Run one delegated task with only the granted worker tools."""
        requested = tool_names or []
        resolved_names = self._tools.resolve_worker_tools(requested)
        granted = self._tools.view(resolved_names, worker_only=True)
        valid_groups = {g.lower() for g in self._tools.capability_names(chat_agent=False)}
        valid_names = set(self._tools.names(chat_agent=False))
        skipped = [
            name
            for name in requested
            if name not in valid_names and name.lower() not in valid_groups
        ]
        turn_id = get_turn_id()
        logger.info(
            "executor start session=%s turn=%s granted=%s skipped=%s",
            session_id,
            turn_id or "-",
            granted.names(),
            skipped,
        )
        history: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task},
        ]
        result = ""
        with trace_span(
            "executor_task",
            span_type="AGENT",
            inputs={
                "task": task,
                "requested_tools": requested,
            },
            attributes={
                "turn_id": turn_id,
                "granted_tools": granted.names(),
                "skipped_tools": skipped,
                "max_iterations": self._max_agent_iterations,
            },
        ) as span:
            for text in run_tool_loop(
                self._llm,
                history,
                granted,
                session_id=session_id,
                max_iterations=self._max_agent_iterations,
                tool_specs=granted.specs(),
                label="executor",
                turn_id=turn_id,
            ):
                result = text
            span.set_outputs(
                {
                    "result": result,
                    "message_count": len(history),
                }
            )
        return result
