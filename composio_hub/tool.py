"""Adapter that turns a Composio session tool into a local Tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tools.base import Tool

if TYPE_CHECKING:
    from composio_hub.hub import ComposioHub


class ComposioTool(Tool):
    """Thin wrapper: LLM tool call → ComposioHub.execute for the current phone."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        hub: ComposioHub,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._hub = hub

    def execute(self, **kwargs: Any) -> Any:
        return self._hub.execute(self.name, kwargs)
