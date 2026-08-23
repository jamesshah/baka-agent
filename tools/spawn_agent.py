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
    """ChatAgent-only: run a worker agent with a chosen subset of tools."""

    name = "spawn_agent"
    description = (
        "Delegate a task that needs tools, lookups, or extended reasoning "
        "to a worker agent. Include all conversation context the worker needs "
        "in task. Pass tools as the worker tool names from the catalog that "
        "this task actually needs (empty list for reasoning-only). "
        "Call send_acknowledgement first."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Full instructions for the worker, including any context "
                    "from the conversation."
                ),
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Worker tool names to grant this executor. Only names from "
                    "the available-tools catalog. Use [] if no tools are needed."
                ),
            },
        },
        "required": ["task", "tools"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        executor: ExecutorAgent,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._executor = executor
        self._context_builder = context_builder

    def is_chat_agent_tool(self) -> bool:
        return True

    def to_openai_spec(self) -> dict[str, Any]:
        spec = copy.deepcopy(super().to_openai_spec())
        worker_names = self._executor.available_tool_names()
        if worker_names:
            spec["function"]["parameters"]["properties"]["tools"]["items"] = {
                "type": "string",
                "enum": worker_names,
            }
        return spec

    def execute(self, **kwargs: Any) -> str:
        task = (kwargs.get("task") or "").strip()
        if not task:
            return "spawn_agent error: task is required"
        requested = _as_name_list(kwargs.get("tools"))
        session_id = require_session_id()
        if self._context_builder is not None:
            context = self._context_builder.render(session_id, task)
            if context:
                task = f"{task}\n\n{context}"
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
