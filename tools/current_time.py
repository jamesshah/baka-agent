"""Current-time tool adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tools.base import Tool


class GetCurrentTimeTool(Tool):
    """Return the current UTC time as an ISO-8601-ish string."""

    name = "get_current_time"
    description = "Get the current date and time in UTC."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
