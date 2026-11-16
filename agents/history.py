"""Chat history alignment helpers (agent-side only)."""

from __future__ import annotations

from typing import Any


def align_messages_to_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop leading messages until the first is role=user (or empty)."""
    aligned = list(messages)
    while aligned and aligned[0].get("role") != "user":
        aligned.pop(0)
    return aligned