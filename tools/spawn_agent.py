"""Delegate a task to the worker ExecutorAgent."""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any

from tools.base import Tool
from tools.session_context import get_turn_id, require_session_id

if TYPE_CHECKING:
    from agents.executor_agent import ExecutorAgent
    from memory.retrieval import ContextBuilder

logger = logging.getLogger(__name__)


class SpawnAgentTool(Tool):
    """ChatAgent-only: run a worker agent with a chosen subset of capabilities/tools."""

    name = "spawn_agent"
    description = (
        "Delegate a task that needs external apps, lookups, or extended reasoning "
        "to a worker agent. Include a short iMessage acknowledgement to send "
        "immediately, a self-contained task description with all needed facts, "
        "and the capability names from the catalog (empty list for reasoning-only)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "acknowledgement": {
                "type": "string",
                "description": (
                    "Short, natural one-line iMessage ping texted immediately "
                    "before the worker starts, e.g. 'Checking your calendar…'"
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "Complete instructions for the worker focused strictly on "
                    "the user's current request, including relevant facts from "
                    "the conversation."
                ),
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Capability names (from the catalog) this task needs. "
                    "Use [] if no tools are needed."
                ),
            },
        },
        "required": ["acknowledgement", "task", "tools"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        executor: ExecutorAgent,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._executor = executor
        # Kept for signature compatibility; ChatAgent already receives <context>
        # and passes relevant facts in ``task`` so ExecutorAgent stays focused
        # and preserves KV-cache reuse without duplicate skill/event injection.
        self._context_builder = context_builder

    def is_chat_agent_tool(self) -> bool:
        return True

    def to_openai_spec(self) -> dict[str, Any]:
        spec = copy.deepcopy(super().to_openai_spec())
        capabilities = self._executor.available_capabilities()
        if capabilities:
            spec["function"]["parameters"]["properties"]["tools"]["items"] = {
                "type": "string",
                "enum": capabilities,
            }
        return spec

    def execute(self, **kwargs: Any) -> str:
        task = (kwargs.get("task") or "").strip()
        if not task:
            return "spawn_agent error: task is required"
        requested = _as_name_list(kwargs.get("tools"))
        session_id = require_session_id()
        logger.info(
            "spawning executor session=%s turn=%s requested_tools=%s",
            session_id,
            get_turn_id() or "-",
            requested,
        )
        return self._executor.run_task(
            session_id, task, tool_names=requested
        )


def _as_name_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                return _as_name_list(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        names: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
        return names
    return []
