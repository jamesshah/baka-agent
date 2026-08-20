"""Abstract agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from messaging.media import DownloadedMedia, MediaKind


class Agent(ABC):
    """Multi-turn agent that can use tools."""

    def has_multimodal(self) -> bool:
        """True if the served model accepts any image/audio/video input."""
        return False

    def supports_media(self, kind: MediaKind) -> bool:
        """True if the served model accepts this media kind."""
        return False

    @abstractmethod
    def run_turn(
        self,
        session_id: str,
        user_text: str,
        media: DownloadedMedia | None = None,
    ) -> Iterator[str]:
        """Process one user message; yield user-facing replies in order."""

    @abstractmethod
    def clear_history(self, session_id: str | None = None) -> None:
        """Clear one conversation or all conversations."""
