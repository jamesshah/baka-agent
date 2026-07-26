"""Tiny tool registry for the from-scratch agent loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable


def get_current_time() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


TOOLS: dict[str, Callable[..., Any]] = {
    "get_current_time": get_current_time,
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool(name: str, arguments: str | dict[str, Any] | None = None) -> str:
    """Look up and run a tool. Returns a string result (or error message)."""
    fn = TOOLS.get(name)
    if fn is None:
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

    try:
        result = fn(**kwargs)
    except TypeError as exc:
        return f"Tool call error: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface tool failures to the model
        return f"Tool execution error: {exc}"

    if isinstance(result, str):
        return result
    return json.dumps(result)
