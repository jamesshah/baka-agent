"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """OpenAI-compatible chat completions client."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Return choices[0].message (may include content and/or tool_calls)."""
