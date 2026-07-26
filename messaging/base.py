"""Abstract messaging client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal


class MessagingClient(ABC):
    """Outbound messaging (iMessage/SMS, etc.)."""

    @abstractmethod
    def send_message(self, number: str, content: str) -> Any:
        """Send a message to ``number``."""

    @abstractmethod
    def send_typing_indicator(
        self,
        number: str,
        state: Literal["start", "stop"] = "start",
        *,
        max_duration_ms: int | None = None,
    ) -> Any:
        """Show or clear typing indicators for ``number``."""
