"""Send a short iMessage acknowledgement before spawning a worker."""

from __future__ import annotations

from typing import Any

from tools.base import Tool


class SendAcknowledgementTool(Tool):
    """ChatAgent-only: ping the user before a longer delegated task."""

    name = "send_acknowledgement"
    description = (
        "Text the user a short acknowledgement before starting a longer task. "
        "Call this immediately before spawn_agent. Keep message to one short "
        "iMessage line."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "Short iMessage ping, e.g. 'Checking your portfolio…'"
                ),
            },
        },
        "required": ["message"],
        "additionalProperties": False,
    }

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        del kwargs
        return "sent"
