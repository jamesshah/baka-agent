"""Storage contracts and retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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


@dataclass(frozen=True)
class DeliveryRecord:
    id: str
    principal_external_id: str
    source: str
    event_type: str
    dedupe_key: str
    payload: dict[str, Any]
    status: str
    attempts: int


@dataclass(frozen=True)
class EventRecord:
    id: str
    principal_external_id: str
    source: str
    event_type: str
    dedupe_key: str
    status: str
    occurred_at: datetime
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    decision: str | None = None
    decided_by: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class EventRuleRecord:
    rule_id: str
    match: dict[str, Any]
    when: str | None
    action: str
    urgent: bool
    enabled: bool
    note: str
    created_at: datetime


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
