"""Class-based agent loop — no agent SDK."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from llm.base import LLMClient
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ChatAgent(Agent):
    """Tool-calling chat agent with per-session history."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        system_prompt: str,
        max_history_messages: int = 40,
        max_agent_iterations: int = 5,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_history_messages = max_history_messages
        self._max_agent_iterations = max_agent_iterations
        self._histories: dict[str, list[dict[str, Any]]] = {}

    def _get_history(self, session_id: str) -> list[dict[str, Any]]:
        if session_id not in self._histories:
            self._histories[session_id] = [
                {"role": "system", "content": self._system_prompt},
            ]
        return self._histories[session_id]

    def _trim_history(self, history: list[dict[str, Any]]) -> None:
        """Keep system prompt + the most recent N messages."""
        max_msgs = self._max_history_messages
        if len(history) <= max_msgs:
            return
        system = history[0]
        rest = history[1:]
        history[:] = [system, *rest[-(max_msgs - 1) :]]

    def clear_history(self, session_id: str | None = None) -> None:
        """Clear one conversation or all conversations."""
        if session_id is None:
            self._histories.clear()
        else:
            self._histories.pop(session_id, None)

    def run_turn(self, session_id: str, user_text: str) -> str:
        """
        Run one user turn through the agent loop.

        1. Append the user message.
        2. Call the LLM (up to max_agent_iterations).
        3. If tool_calls are present, execute them and loop.
        4. Otherwise return the assistant text.
        """
        history = self._get_history(session_id)
        history.append({"role": "user", "content": user_text})
        self._trim_history(history)

        tool_specs = self._tools.specs()

        for iteration in range(self._max_agent_iterations):
            logger.info("agent iteration %s for %s", iteration + 1, session_id)
            message = self._llm.chat(history, tools=tool_specs)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                # Persist the assistant message that requested tools.
                history.append(message)
                for call in tool_calls:
                    fn = call.get("function") or {}
                    name = fn.get("name") or ""
                    arguments = fn.get("arguments") or "{}"
                    call_id = call.get("id") or name
                    logger.info("tool call: %s(%s)", name, arguments)
                    result = self._tools.execute(name, arguments)
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result,
                        }
                    )
                self._trim_history(history)
                continue

            content = (message.get("content") or "").strip()
            if not content:
                content = "(empty model response)"
            history.append({"role": "assistant", "content": content})
            self._trim_history(history)
            return content

        fallback = (
            "Sorry, I hit my tool-call limit before finishing. "
            "Try asking again more simply."
        )
        history.append({"role": "assistant", "content": fallback})
        return fallback
