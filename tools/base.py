"""Abstract tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """A single callable tool exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    capability: str | None = None
    capability_description: str | None = None

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Run the tool with keyword arguments from the model."""

    def is_enabled(self) -> bool:
        """True if the tool is enabled."""
        return True

    def is_chat_agent_tool(self) -> bool:
        """True if the tool is exposed to the user-facing ChatAgent."""
        return False

    def capability_group(self) -> str:
        """Group name shown to ChatAgent when delegating (defaults to tool name)."""
        return (self.capability or self.name).strip() or self.name

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
