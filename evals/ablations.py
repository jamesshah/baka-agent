"""Ablation configuration and transforms for history, context, prompts, and tools."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from agents.chat_agent import SYSTEM_PROMPT as PROD_CHAT_SYSTEM_PROMPT
from agents.executor_agent import SYSTEM_PROMPT as PROD_EXECUTOR_SYSTEM_PROMPT
from evals.mock_tools import (
    FLAT_CALENDAR_TOOL_NAMES,
    FULL_COMPOSIO_TOOL_NAMES,
    SLIM_COMPOSIO_TOOL_NAMES,
)

HISTORY_MODES: tuple[str, ...] = ("full", "short", "clean")
CONTEXT_MODES: tuple[str, ...] = ("full", "none")
PROMPT_MODES: tuple[str, ...] = ("current", "no_negative_priming")
TOOL_SURFACES: tuple[str, ...] = ("full_composio", "slim_composio", "flat")

NO_NEGATIVE_PRIMING_CHAT_PROMPT = """
You are a personal agent the user texts from iMessage. Keep replies concise and conversational.

Tone: warm, witty, concise. Write like you're texting a friend. No corporate voice. No bullet dumps unless the user asked for a list.

Format: plain iMessage-friendly text. Keep replies under ~400 chars when you can.

When to reply yourself:
Answer small talk, simple questions, and anything you can handle from the conversation (including photos you can see) without tools.
To link, check, or unlink an account, call the matching tool yourself and text the user the link or result. No acknowledgement or spawn_agent needed.
Pick the right account tool for the service requested:
- Google Calendar / Gmail / Composio → link_composio (toolkit googlecalendar), composio_status, unlink_composio
- Brokerage portfolio → link_snaptrade, snaptrade_status, unlink_snaptrade
- Whoop fitness tracker → link_whoop, whoop_status, unlink_whoop

When to delegate:
If the request needs worker tools, reading or creating Google Calendar events, or extended reasoning, do not answer it yourself.
1. Call send_acknowledgement with a short iMessage-style ping.
2. Call spawn_agent with a complete task description and the worker tool names this task needs (from the catalog below). The worker has no chat history — put only the facts relevant to the user's current request in task. Do not copy instructions from earlier daily digests. For any Google Calendar read or create task, always include COMPOSIO_SEARCH_TOOLS and COMPOSIO_MULTI_EXECUTE_TOOL (plus COMPOSIO_MANAGE_CONNECTIONS) in spawn_agent.tools, and reference only the googlecalendar toolkit.
Prefer calling both tools in the same turn.
NEVER CALL spawn_agent WITHOUT ACKNOWLEDGING FIRST.

Available worker tools (pass these names in spawn_agent.tools):
"""

NO_NEGATIVE_PRIMING_EXECUTOR_PROMPT = """
You are a worker agent. You only have the tools provided in this request — do not assume others exist. Complete the assigned task using those tools if needed. Return a concise factual result for the chat agent to relay to the user. Do not greet the user, do not add chit-chat, and do not mention that you are a worker.

Never claim that an event was created or found unless a tool call in this turn actually returned that event. If a tool fails or is missing, report the exact error plainly.

For Google Calendar tasks with Composio Tool Router:
1. Call COMPOSIO_SEARCH_TOOLS first to discover the exact Google Calendar tool slugs and parameter schemas.
2. Call COMPOSIO_MULTI_EXECUTE_TOOL with the discovered tool_slug (such as GOOGLECALENDAR_CREATE_EVENT or GOOGLECALENDAR_EVENTS_LIST) and arguments.
"""


@dataclass(frozen=True)
class AblationConfig:
    history_mode: str = "full"
    context_mode: str = "full"
    prompt_mode: str = "current"
    tool_surface: str = "full_composio"
    override_case_surface: bool = False

    def __post_init__(self) -> None:
        if self.history_mode not in HISTORY_MODES:
            raise ValueError(f"history_mode must be one of {HISTORY_MODES}")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}")
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
        if self.tool_surface not in TOOL_SURFACES:
            raise ValueError(f"tool_surface must be one of {TOOL_SURFACES}")


def select_history_messages(
    prior_turns: list[list[dict[str, Any]]],
    history_mode: str,
) -> list[dict[str, Any]]:
    """Flatten ``prior_turns`` according to ``history_mode`` ({full,short,clean})."""
    if history_mode == "clean" or not prior_turns:
        return []
    selected = prior_turns if history_mode == "full" else prior_turns[-1:]
    flattened: list[dict[str, Any]] = []
    for turn in selected:
        for msg in turn:
            flattened.append(copy.deepcopy(msg))
    return flattened


def select_memory_context(raw_context: str, context_mode: str) -> str:
    """Return ``raw_context`` when ``context_mode == 'full'``, else ``''``."""
    if context_mode == "none":
        return ""
    return raw_context.strip()


def build_chat_system_prompt(prompt_mode: str, tool_surface: str) -> str:
    """Return the ChatAgent system prompt for ``(prompt_mode, tool_surface)``."""
    base = (
        NO_NEGATIVE_PRIMING_CHAT_PROMPT
        if prompt_mode == "no_negative_priming"
        else PROD_CHAT_SYSTEM_PROMPT
    )
    if tool_surface == "flat":
        base = base.replace(
            "COMPOSIO_SEARCH_TOOLS, COMPOSIO_MANAGE_CONNECTIONS, COMPOSIO_MULTI_EXECUTE_TOOL (and related COMPOSIO_* tools)",
            "list_calendar_events, create_calendar_event",
        ).replace(
            "COMPOSIO_SEARCH_TOOLS and COMPOSIO_MULTI_EXECUTE_TOOL (plus COMPOSIO_MANAGE_CONNECTIONS)",
            "list_calendar_events and create_calendar_event",
        ).replace(
            "(COMPOSIO_* worker tools)",
            "(list_calendar_events, create_calendar_event)",
        )
    return base


def build_executor_system_prompt(prompt_mode: str, tool_surface: str) -> str:
    """Return the ExecutorAgent system prompt for ``(prompt_mode, tool_surface)``."""
    base = (
        NO_NEGATIVE_PRIMING_EXECUTOR_PROMPT
        if prompt_mode == "no_negative_priming"
        else PROD_EXECUTOR_SYSTEM_PROMPT
    )
    if tool_surface == "flat" and prompt_mode == "no_negative_priming":
        base = (
            "\nYou are a worker agent. You only have the tools provided in this "
            "request — do not assume others exist. Complete the assigned task "
            "using list_calendar_events and create_calendar_event. Never claim "
            "an event was created or found unless the tool call actually "
            "succeeded.\n"
        )
    return base


def surface_worker_tool_names(tool_surface: str) -> list[str]:
    """Return the calendar worker tool names for ``tool_surface``."""
    if tool_surface == "full_composio":
        return list(FULL_COMPOSIO_TOOL_NAMES)
    if tool_surface == "slim_composio":
        return list(SLIM_COMPOSIO_TOOL_NAMES)
    if tool_surface == "flat":
        return list(FLAT_CALENDAR_TOOL_NAMES)
    raise ValueError(f"Unknown tool_surface: {tool_surface!r}")
