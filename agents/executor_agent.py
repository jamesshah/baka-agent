"""Worker agent that runs delegated tasks with a granted tool subset."""

from __future__ import annotations

import logging
from typing import Any

from agents.loop import run_tool_loop
from llm.base import LLMClient
from tools.registry import ToolRegistry
from tools.session_context import get_turn_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a worker agent. You only have the tools provided in this request — do not assume others exist. Complete the assigned task using those tools if needed. Return a concise factual result for the chat agent to relay to the user. Do not greet the user, do not add chit-chat, and do not mention that you are a worker.

You are only allowed to use the tools provided in this request - do not assume others exist. If you cannot perform the task with the tools provided, return "I cannot perform this task with the tools provided."

SnapTrade portfolio tools are read-only and require a one-time link:
if the task is about balances/positions/orders and SnapTrade is not linked, call link_snaptrade and include the verification URL in your result.
Brokerage connection links from SnapTrade expire in about 5 minutes.
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

    def run_task(
        self,
        session_id: str,
        task: str,
        *,
        tool_names: list[str] | None = None,
    ) -> str:
        """Run one delegated task with only the granted worker tools."""
        requested = tool_names or []
        granted = self._tools.view(requested, worker_only=True)
        skipped = [name for name in requested if name not in granted.names()]
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
        return result
