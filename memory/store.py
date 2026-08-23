"""Storage contracts and retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class MemoryHit:
    id: str
    kind: str
    key: str
    value: str
    confidence: float
    score: float


@dataclass(frozen=True)
class SkillHit:
    id: str
    name: str
    description: str
    body: str
    tools: list[str]
    score: float


class MemoryStore(Protocol):
    def load_history(
        self, principal_external_id: str, *, limit: int
    ) -> list[dict[str, Any]]: ...

    def begin_turn(
        self,
        principal_external_id: str,
        turn_id: str,
        user_message: dict[str, Any],
    ) -> None: ...

    def complete_turn(
        self,
        turn_id: str,
        messages: list[dict[str, Any]],
        *,
        status: str = "completed",
    ) -> list[int]: ...

    def clear_history(self, principal_external_id: str | None = None) -> None: ...
