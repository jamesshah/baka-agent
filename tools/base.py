"""Abstract tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """A single callable tool exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Run the tool with keyword arguments from the model."""

    def to_openai_spec(self) -> dict[str, Any]:
        """Build an OpenAI-style function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
