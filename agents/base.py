"""Abstract agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Agent(ABC):
    """Multi-turn agent that can use tools."""

    @abstractmethod
    def run_turn(self, session_id: str, user_text: str) -> str:
        """Process one user message and return the assistant reply."""

    @abstractmethod
    def clear_history(self, session_id: str | None = None) -> None:
        """Clear one conversation or all conversations."""
