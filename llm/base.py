"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """OpenAI-compatible chat completions client."""

    def has_multimodal(self) -> bool:
        """True if the served model accepts any image/audio/video input."""
        return bool(self.input_modalities())

    def supports_media(self, kind: str) -> bool:
        """True if the served model accepts this media kind (image/audio/video)."""
        return kind in self.input_modalities()

    def input_modalities(self) -> set[str]:
        """Served input kinds: a subset of ``image``, ``audio``, ``video``."""
        return set()

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Return choices[0].message (may include content and/or tool_calls)."""
