"""From-scratch agent loop — no agent SDK."""

from __future__ import annotations

import logging
from typing import Any

import llm
from config import get_settings
from tools import TOOL_SPECS, execute_tool

logger = logging.getLogger(__name__)

# Per-conversation history keyed by E.164 phone number.
_histories: dict[str, list[dict[str, Any]]] = {}


def _get_history(number: str) -> list[dict[str, Any]]:
    settings = get_settings()
    if number not in _histories:
        _histories[number] = [
            {"role": "system", "content": settings.system_prompt},
        ]
    return _histories[number]


def _trim_history(history: list[dict[str, Any]]) -> None:
    """Keep system prompt + the most recent N messages."""
    settings = get_settings()
    max_msgs = settings.max_history_messages
    if len(history) <= max_msgs:
        return
    system = history[0]
    rest = history[1:]
    history[:] = [system, *rest[-(max_msgs - 1) :]]


def clear_history(number: str | None = None) -> None:
    """Clear one conversation or all conversations."""
    if number is None:
        _histories.clear()
    else:
        _histories.pop(number, None)


def run_turn(number: str, user_text: str) -> str:
    """
    Run one user turn through the agent loop.

    1. Append the user message.
    2. Call the LLM (up to MAX_AGENT_ITERATIONS).
    3. If tool_calls are present, execute them and loop.
    4. Otherwise return the assistant text.
    """
    settings = get_settings()
    history = _get_history(number)
    history.append({"role": "user", "content": user_text})
    _trim_history(history)

    for iteration in range(settings.max_agent_iterations):
        logger.info("agent iteration %s for %s", iteration + 1, number)
        message = llm.chat(history, tools=TOOL_SPECS)

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
                result = execute_tool(name, arguments)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result,
                    }
                )
            _trim_history(history)
            continue

        content = (message.get("content") or "").strip()
        if not content:
            content = "(empty model response)"
        history.append({"role": "assistant", "content": content})
        _trim_history(history)
        return content

    fallback = (
        "Sorry, I hit my tool-call limit before finishing. "
        "Try asking again more simply."
    )
    history.append({"role": "assistant", "content": fallback})
    return fallback
