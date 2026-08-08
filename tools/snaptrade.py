"""Local tools for SnapTrade MCP OAuth link / status / unlink."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tools.base import Tool
from tools.session_context import require_session_id

if TYPE_CHECKING:
    from mcp_client.hub import MultiServerMcpClient

logger = logging.getLogger(__name__)

SNAPTRADE_SERVER = "snaptrade"


class LinkSnaptradeTool(Tool):
    """Start SnapTrade device OAuth and return the verification URL."""

    name = "link_snaptrade"
    description = (
        "Start a one-time SnapTrade OAuth link for the current iMessage user. "
        "Returns a verification URL the user must open in a browser to approve "
        "read access. After they approve, SnapTrade portfolio tools become "
        "available automatically. Call this when the user asks to link SnapTrade "
        "or when portfolio tools are needed but SnapTrade is not linked."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, client: MultiServerMcpClient) -> None:
        self._client = client

    def execute(self, **kwargs: Any) -> Any:
        del kwargs
        phone = require_session_id()
        if not self._client.has_server(SNAPTRADE_SERVER):
            return {
                "status": "error",
                "detail": (
                    "SnapTrade is not configured in mcp.json. "
                    "Add a snaptrade server with auth=oauth."
                ),
            }
        status = self._client.link_status(SNAPTRADE_SERVER, phone)
        if status.get("status") == "linked":
            return {
                "status": "already_linked",
                "detail": "SnapTrade is already linked for this user.",
                **status,
            }
        try:
            result = self._client.begin_device_link(SNAPTRADE_SERVER, phone)
        except Exception as exc:  # noqa: BLE001
            logger.exception("link_snaptrade failed")
            return {"status": "error", "detail": str(exc)}

        url = result.get("verification_uri_complete") or result.get("verification_uri")
        return {
            **result,
            "message": (
                f"Ask the user to open this link and approve read access to SnapTrade: {url}. "
                "After they finish, you will get a confirmation message automatically."
            ),
        }


class SnaptradeStatusTool(Tool):
    """Report whether SnapTrade is linked for the current user."""

    name = "snaptrade_status"
    description = (
        "Check whether SnapTrade MCP is linked and connected for the current "
        "iMessage user. Use before portfolio questions if unsure."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, client: MultiServerMcpClient) -> None:
        self._client = client

    def execute(self, **kwargs: Any) -> Any:
        del kwargs
        phone = require_session_id()
        if not self._client.has_server(SNAPTRADE_SERVER):
            return {"status": "not_configured", "server": SNAPTRADE_SERVER}
        return self._client.link_status(SNAPTRADE_SERVER, phone)


class UnlinkSnaptradeTool(Tool):
    """Revoke SnapTrade OAuth and drop MCP tools."""

    name = "unlink_snaptrade"
    description = (
        "Unlink SnapTrade for the current iMessage user: revoke the OAuth token "
        "(best effort), delete local tokens, and disconnect SnapTrade MCP tools. "
        "Call when the user asks to disconnect or unlink SnapTrade."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, client: MultiServerMcpClient) -> None:
        self._client = client

    def execute(self, **kwargs: Any) -> Any:
        del kwargs
        phone = require_session_id()
        if not self._client.has_server(SNAPTRADE_SERVER):
            return {"status": "not_configured", "server": SNAPTRADE_SERVER}
        try:
            return self._client.unlink(SNAPTRADE_SERVER, phone)
        except Exception as exc:  # noqa: BLE001
            logger.exception("unlink_snaptrade failed")
            return {"status": "error", "detail": str(exc)}
