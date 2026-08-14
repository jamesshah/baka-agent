"""Shared tool-calling loop used by ChatAgent and ExecutorAgent."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from llm.base import LLMClient
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

BeforeTool = Callable[[str, dict[str, Any]], Iterator[str]]
OrderToolCalls = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
TrimHistory = Callable[[list[dict[str, Any]]], None]

_EMPTY_RESPONSE = "(empty model response)"
_ITERATION_LIMIT = (
    "Sorry, I hit my tool-call limit before finishing. "
    "Try asking again more simply."
)


_TURN_ID_KEY = "turn_id"


def run_tool_loop(
    llm: LLMClient,
    history: list[dict[str, Any]],
    tools: ToolRegistry,
    *,
    session_id: str,
    max_iterations: int,
    tool_specs: list[dict[str, Any]] | None = None,
    timeout: float | None = None,
    label: str = "agent",
    turn_id: str | None = None,
    order_tool_calls: OrderToolCalls | None = None,
    before_tool: BeforeTool | None = None,
    trim_history: TrimHistory | None = None,
) -> Iterator[str]:
    """
    Call the LLM until it returns text (no tool_calls) or hits max_iterations.

    Yields any strings from ``before_tool``, then the final assistant text.
    ``turn_id`` is stored on history messages and stripped before LLM calls.
    """
    specs = tool_specs if tool_specs is not None else tools.specs()
    for iteration in range(max_iterations):
        logger.info(
            "%s iteration %s session=%s turn=%s",
            label,
            iteration + 1,
            session_id,
            turn_id or "-",
        )
        message = llm.chat(_for_llm(history), tools=specs, timeout=timeout)

        tool_calls = list(message.get("tool_calls") or [])
        if tool_calls:
            history.append(_with_turn_id(message, turn_id))
            if order_tool_calls is not None:
                tool_calls = order_tool_calls(tool_calls)
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                arguments = fn.get("arguments") or "{}"
                call_id = call.get("id") or name
                logger.info("tool call: %s(%s)", name, arguments)
                if before_tool is not None:
                    yield from before_tool(name, _parse_arguments(arguments))
                result = tools.execute(
                    name, arguments, session_id=session_id
                )
                history.append(
                    _with_turn_id(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result,
                        },
                        turn_id,
                    )
                )
            if trim_history is not None:
                trim_history(history)
            continue

        content = (message.get("content") or "").strip() or _EMPTY_RESPONSE
        history.append(
            _with_turn_id({"role": "assistant", "content": content}, turn_id)
        )
        if trim_history is not None:
            trim_history(history)
        yield content
        return

    history.append(
        _with_turn_id({"role": "assistant", "content": _ITERATION_LIMIT}, turn_id)
    )
    if trim_history is not None:
        trim_history(history)
    yield _ITERATION_LIMIT


def _with_turn_id(message: dict[str, Any], turn_id: str | None) -> dict[str, Any]:
    stored = dict(message)
    if turn_id:
        stored[_TURN_ID_KEY] = turn_id
    return stored


def _for_llm(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop turn_id so the model API only sees OpenAI chat fields."""
    return [
        {key: value for key, value in message.items() if key != _TURN_ID_KEY}
        for message in history
    ]


def _parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
