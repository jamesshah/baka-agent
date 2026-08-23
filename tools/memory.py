"""Explicit inspect, remember, correct, and forget operations."""

from __future__ import annotations

from typing import Any

from memory.embeddings import EmbeddingClient
from memory.repository import SqlAlchemyMemoryRepository
from tools.base import Tool
from tools.session_context import require_session_id


class ManageMemoryTool(Tool):
    name = "manage_memory"
    description = (
        "Inspect, explicitly remember/correct, or forget durable personal "
        "memory. Use when the user asks what you remember, says to remember "
        "something, corrects a remembered fact, or asks you to forget it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "remember", "forget"],
            },
            "key": {
                "type": "string",
                "description": "Stable snake_case memory key.",
            },
            "value": {
                "type": "string",
                "description": "Fact or preference to remember.",
            },
            "kind": {
                "type": "string",
                "description": "Memory category, such as preference or fact.",
                "default": "fact",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        repository: SqlAlchemyMemoryRepository,
        embeddings: EmbeddingClient | None = None,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        session_id = require_session_id()
        operation = str(kwargs.get("operation") or "")
        key = str(kwargs.get("key") or "").strip()
        if operation == "list":
            memories = self._repository.list_memories(session_id)
            if not memories:
                return "No durable memories are stored."
            return "\n".join(
                f"{item.key} [{item.kind}]: {item.value}"
                for item in memories
            )
        if operation == "forget":
            if not key:
                return "manage_memory error: key is required for forget"
            forgotten = self._repository.forget_memory(session_id, key)
            return "Memory forgotten." if forgotten else "Memory key not found."
        if operation == "remember":
            value = str(kwargs.get("value") or "").strip()
            if not key or not value:
                return (
                    "manage_memory error: key and value are required "
                    "for remember"
                )
            kind = str(kwargs.get("kind") or "fact").strip() or "fact"
            memory = self._repository.upsert_memory(
                session_id,
                kind=kind,
                key=key,
                value=value,
                confidence=1.0,
            )
            if self._embeddings is not None:
                content = f"{kind} {key}: {value}"
                vector = self._embeddings.embed_documents([content])[0]
                self._repository.put_embedding(
                    entity_type="memory",
                    entity_id=memory.id,
                    model_id=self._embeddings.model_id,
                    content=content,
                    vector=vector,
                )
            return "Memory saved."
        return "manage_memory error: unsupported operation"
