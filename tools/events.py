"""Read stored connector events, and link/unlink OAuth connectors."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from tools.base import Tool
from tools.session_context import require_session_id

if TYPE_CHECKING:
    from connectors.base import OAuthConnector
    from connectors.registry import ConnectorRegistry
    from memory.repository import SqlAlchemyMemoryRepository

logger = logging.getLogger(__name__)

_MAX_EVENTS = 100


class QueryEventsTool(Tool):
    name = "query_events"
    description = (
        "Look up the user's stored events from connected services (e.g. Whoop "
        "sleep, recovery, workouts). Use for questions like 'how did I sleep "
        "this week' or 'what was my recovery yesterday'. Returns newest first "
        "with summaries and numeric metrics."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Connector name, e.g. whoop. Omit for all.",
            },
            "type": {
                "type": "string",
                "description": (
                    "Event type, e.g. recovery.updated or sleep.*. Omit for all."
                ),
            },
            "days": {
                "type": "integer",
                "description": "How many days back to look (default 7, max 90).",
            },
            "limit": {
                "type": "integer",
                "description": f"Max events to return (default 20, max {_MAX_EVENTS}).",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, repository: SqlAlchemyMemoryRepository) -> None:
        self._repository = repository

    def execute(self, **kwargs: Any) -> Any:
        phone = require_session_id()
        days = _clamp(kwargs.get("days"), default=7, low=1, high=90)
        limit = _clamp(kwargs.get("limit"), default=20, low=1, high=_MAX_EVENTS)
        since = datetime.now(timezone.utc) - timedelta(days=days)
        events = self._repository.recent_events(
            phone,
            since=since,
            source=str(kwargs.get("source") or "").strip() or None,
            event_type=str(kwargs.get("type") or "").strip() or None,
            exclude_statuses=("ignored",),
            limit=limit,
        )
        return {
            "count": len(events),
            "events": [
                {
                    "occurred_at": event.occurred_at.astimezone().isoformat(
                        timespec="minutes"
                    ),
                    "source": event.source,
                    "type": event.event_type,
                    "summary": event.summary,
                    "metrics": event.metrics,
                }
                for event in events
            ],
        }


class _ChatConnectorTool(Tool):
    """Fast, single-call account tools the chat agent runs without a worker."""

    def is_chat_agent_tool(self) -> bool:
        return True


class LinkConnectorTool(_ChatConnectorTool):
    def __init__(self, connector: OAuthConnector) -> None:
        self._connector = connector
        self.name = f"link_{connector.name}"
        self.description = (
            f"Start linking {connector.title} for the current iMessage user. "
            "Returns a URL the user must open to approve read access; they get "
            f"a confirmation text when done. Call when the user asks to connect "
            f"{connector.title}."
        )
        self.parameters = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> Any:
        phone = require_session_id()
        if not self._connector.is_configured():
            return {
                "status": "error",
                "detail": f"{self._connector.title} is not configured on the server.",
            }
        if self._connector.status(phone).get("linked"):
            return {"status": "already_linked"}
        try:
            url = self._connector.authorization_url(phone)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", self.name)
            return {"status": "error", "detail": str(exc)}
        return {
            "status": "pending",
            "url": url,
            "message": (
                f"Ask the user to open this link and approve {self._connector.title} "
                f"access: {url} The link expires in 15 minutes."
            ),
        }


class ConnectorStatusTool(_ChatConnectorTool):
    def __init__(self, connector: OAuthConnector) -> None:
        self._connector = connector
        self.name = f"{connector.name}_status"
        self.description = (
            f"Check whether {connector.title} is linked for the current user."
        )
        self.parameters = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> Any:
        return self._connector.status(require_session_id())


class UnlinkConnectorTool(_ChatConnectorTool):
    def __init__(self, connector: OAuthConnector) -> None:
        self._connector = connector
        self.name = f"unlink_{connector.name}"
        self.description = (
            f"Disconnect {connector.title} for the current user: revoke access "
            "and delete stored credentials. Call when the user asks to unlink it."
        )
        self.parameters = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def execute(self, **kwargs: Any) -> Any:
        try:
            return self._connector.unlink(require_session_id())
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", self.name)
            return {"status": "error", "detail": str(exc)}


def connector_tools(
    repository: SqlAlchemyMemoryRepository, connectors: ConnectorRegistry
) -> list[Tool]:
    tools: list[Tool] = [QueryEventsTool(repository)]
    for connector in connectors.oauth():
        tools += [
            LinkConnectorTool(connector),
            ConnectorStatusTool(connector),
            UnlinkConnectorTool(connector),
        ]
    return tools


def _clamp(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))
