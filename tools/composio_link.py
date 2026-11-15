"""Local tools for Composio Connect Link / status / unlink."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tools.base import Tool
from tools.session_context import require_session_id

if TYPE_CHECKING:
    from composio_hub.hub import ComposioHub

logger = logging.getLogger(__name__)


class _ChatComposioTool(Tool):
    """Fast, single-call Composio account tools the chat agent runs without a worker."""

    def is_chat_agent_tool(self) -> bool:
        return True


class LinkComposioTool(_ChatComposioTool):
    """Start a Composio Connect Link for an allowlisted toolkit."""

    name = "link_composio"
    description = (
        "Link a Composio app (Google Calendar, etc.) for the current iMessage "
        "user. Returns a Connect Link URL the user must open in a browser. "
        "Use for Google Calendar / Gmail / Composio — NEVER for SnapTrade "
        "portfolio or Whoop (those have their own link_* tools). "
        "Pass toolkit='googlecalendar' when the user asks to connect Calendar."
    )

    def __init__(self, hub: ComposioHub) -> None:
        self._hub = hub
        slugs = hub.toolkit_slugs
        properties: dict[str, Any] = {
            "toolkit": {
                "type": "string",
                "description": (
                    "Toolkit slug to connect "
                    f"(allowlist: {', '.join(slugs) or 'any'})."
                ),
            }
        }
        if slugs:
            properties["toolkit"]["enum"] = slugs
        self.parameters = {
            "type": "object",
            "properties": properties,
            "required": ["toolkit"],
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> Any:
        toolkit = str(kwargs.get("toolkit") or "").strip()
        if not toolkit:
            return {"status": "error", "detail": "toolkit is required"}
        phone = require_session_id()
        status = self._hub.link_status(phone, toolkit)
        if status.get("status") == "linked":
            return {
                "status": "already_linked",
                "toolkit": toolkit,
                "detail": f"{toolkit} is already linked for this user.",
                **status,
            }
        try:
            return self._hub.authorize(phone, toolkit)
        except Exception as exc:  # noqa: BLE001
            logger.exception("link_composio failed")
            return {"status": "error", "toolkit": toolkit, "detail": str(exc)}


class ComposioStatusTool(_ChatComposioTool):
    """Report Composio connection status for the current user."""

    name = "composio_status"
    description = (
        "Check whether Composio apps (e.g. googlecalendar) are connected for "
        "the current iMessage user. Not for SnapTrade or Whoop."
    )

    def __init__(self, hub: ComposioHub) -> None:
        self._hub = hub
        slugs = hub.toolkit_slugs
        properties: dict[str, Any] = {
            "toolkit": {
                "type": "string",
                "description": (
                    "Optional toolkit slug to check. Omit to list allowlisted "
                    f"toolkits ({', '.join(slugs) or 'all'})."
                ),
            }
        }
        if slugs:
            properties["toolkit"]["enum"] = slugs
        self.parameters = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> Any:
        toolkit = kwargs.get("toolkit")
        toolkit_str = str(toolkit).strip() if toolkit else None
        phone = require_session_id()
        return self._hub.link_status(phone, toolkit_str)


class UnlinkComposioTool(_ChatComposioTool):
    """Revoke Composio connected accounts for the current user."""

    name = "unlink_composio"
    description = (
        "Unlink a Composio app (e.g. googlecalendar) for the current iMessage "
        "user. Not for SnapTrade or Whoop."
    )

    def __init__(self, hub: ComposioHub) -> None:
        self._hub = hub
        slugs = hub.toolkit_slugs
        properties: dict[str, Any] = {
            "toolkit": {
                "type": "string",
                "description": (
                    "Optional toolkit slug to unlink. Omit to unlink all "
                    f"Composio connections for this user "
                    f"({', '.join(slugs) or 'all'})."
                ),
            }
        }
        if slugs:
            properties["toolkit"]["enum"] = slugs
        self.parameters = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> Any:
        toolkit = kwargs.get("toolkit")
        toolkit_str = str(toolkit).strip() if toolkit else None
        phone = require_session_id()
        try:
            return self._hub.unlink(phone, toolkit_str)
        except Exception as exc:  # noqa: BLE001
            logger.exception("unlink_composio failed")
            return {"status": "error", "detail": str(exc)}
