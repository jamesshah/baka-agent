"""Deterministic offline tool mocks for ChatAgent and ExecutorAgent evals."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from evals.fixtures import (
    DEFAULT_CALENDAR_EVENTS,
    DEFAULT_WHOOP_EVENTS,
    EVAL_SESSION_ID,
    FIXED_NOW_UTC,
)
from tools.base import Tool
from tools.registry import ToolRegistry
from tools.send_acknowledgement import SendAcknowledgementTool

FULL_COMPOSIO_TOOL_NAMES: tuple[str, ...] = (
    "COMPOSIO_SEARCH_TOOLS",
    "COMPOSIO_GET_TOOL_SCHEMAS",
    "COMPOSIO_MANAGE_CONNECTIONS",
    "COMPOSIO_MULTI_EXECUTE_TOOL",
    "COMPOSIO_REMOTE_BASH_TOOL",
    "COMPOSIO_REMOTE_WORKBENCH",
)

SLIM_COMPOSIO_TOOL_NAMES: tuple[str, ...] = (
    "COMPOSIO_SEARCH_TOOLS",
    "COMPOSIO_MANAGE_CONNECTIONS",
    "COMPOSIO_MULTI_EXECUTE_TOOL",
)

FLAT_CALENDAR_TOOL_NAMES: tuple[str, ...] = (
    "list_calendar_events",
    "create_calendar_event",
)

_GOOGLECALENDAR_SCHEMAS: dict[str, dict[str, Any]] = {
    "GOOGLECALENDAR_CREATE_EVENT": {
        "slug": "GOOGLECALENDAR_CREATE_EVENT",
        "toolkit": "googlecalendar",
        "description": "Create a new event on a Google Calendar.",
        "input_parameters": {
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "default": "primary",
                    "description": "Calendar identifier (default 'primary').",
                },
                "summary": {
                    "type": "string",
                    "description": "Title / summary of the calendar event.",
                },
                "start_datetime": {
                    "type": "string",
                    "description": "ISO-8601 start datetime, e.g. '2026-11-11T22:30:00-06:00'.",
                },
                "end_datetime": {
                    "type": "string",
                    "description": "ISO-8601 end datetime, e.g. '2026-11-11T23:00:00-06:00'.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional event description.",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone, e.g. 'America/Chicago'.",
                },
            },
            "required": ["summary", "start_datetime"],
        },
    },
    "GOOGLECALENDAR_EVENTS_LIST": {
        "slug": "GOOGLECALENDAR_EVENTS_LIST",
        "toolkit": "googlecalendar",
        "description": "List events on a Google Calendar within an optional time window.",
        "input_parameters": {
            "type": "object",
            "properties": {
                "calendarId": {
                    "type": "string",
                    "default": "primary",
                    "description": "Calendar identifier (default 'primary').",
                },
                "timeMin": {
                    "type": "string",
                    "description": "Lower bound (inclusive) as an RFC3339 timestamp.",
                },
                "timeMax": {
                    "type": "string",
                    "description": "Upper bound (exclusive) as an RFC3339 timestamp.",
                },
                "singleEvents": {"type": "boolean", "default": True},
                "maxResults": {"type": "integer", "default": 25},
            },
        },
    },
    "GOOGLECALENDAR_FIND_EVENT": {
        "slug": "GOOGLECALENDAR_FIND_EVENT",
        "toolkit": "googlecalendar",
        "description": "Find events on Google Calendar matching a text query or date window.",
        "input_parameters": {
            "type": "object",
            "properties": {
                "calendar_id": {"type": "string", "default": "primary"},
                "query": {"type": "string", "description": "Free-text search query."},
                "timeMin": {"type": "string"},
                "timeMax": {"type": "string"},
            },
        },
    },
}

_CREATE_SLUGS = {
    "GOOGLECALENDAR_CREATE_EVENT",
    "GOOGLECALENDAR_INSERT_EVENT",
    "GOOGLECALENDAR_QUICK_ADD",
    "GOOGLECALENDAR_ADD_EVENT",
}
_LIST_SLUGS = {
    "GOOGLECALENDAR_EVENTS_LIST",
    "GOOGLECALENDAR_FIND_EVENT",
    "GOOGLECALENDAR_LIST_EVENTS",
    "GOOGLECALENDAR_GET_EVENTS",
}


@dataclass
class MockWorld:
    """Per-case deterministic state and tool execution log."""

    composio_linked: bool = True
    snaptrade_linked: bool = False
    whoop_linked: bool = True
    calendar_events: list[dict[str, Any]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_CALENDAR_EVENTS)
    )
    whoop_events: list[dict[str, Any]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_WHOOP_EVENTS)
    )
    created_events: list[dict[str, Any]] = field(default_factory=list)
    listed_events_count: int = 0
    searched_composio: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def record_call(
        self,
        agent: str,
        name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        self.tool_calls.append(
            {
                "agent": agent,
                "name": name,
                "arguments": copy.deepcopy(arguments),
                "result": copy.deepcopy(result),
            }
        )
        return result

    def create_calendar_event(
        self,
        *,
        summary: str,
        start_datetime: str,
        end_datetime: str | None = None,
        description: str = "",
        timezone: str = "America/Chicago",
    ) -> dict[str, Any]:
        event = {
            "id": f"evt_created_{len(self.created_events) + 1}",
            "summary": summary or "Untitled Event",
            "start": {"dateTime": start_datetime, "timeZone": timezone},
            "end": {
                "dateTime": end_datetime or start_datetime,
                "timeZone": timezone,
            },
            "description": description,
            "status": "confirmed",
            "htmlLink": f"https://calendar.google.com/calendar/event?eid=mock_{len(self.created_events) + 1}",
        }
        self.created_events.append(event)
        self.calendar_events.append(event)
        return event

    def list_calendar_events(self, query: str | None = None) -> list[dict[str, Any]]:
        self.listed_events_count += 1
        if not query:
            return list(self.calendar_events)
        needle = query.strip().lower()
        return [
            evt
            for evt in self.calendar_events
            if needle in str(evt.get("summary") or "").lower()
            or needle in str(evt.get("description") or "").lower()
        ]


class StaticContextBuilder:
    """Stand-in for ``memory.retrieval.ContextBuilder`` returning a fixed string."""

    def __init__(self, rendered: str) -> None:
        self._rendered = rendered

    def render(self, principal_external_id: str, query: str) -> str:
        del principal_external_id, query
        return self._rendered


class RecordingSendAckTool(SendAcknowledgementTool):
    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> str:
        res = super().execute(**kwargs)
        self._world.record_call("chat", self.name, kwargs, res)
        return res


class MockLinkComposioTool(Tool):
    name = "link_composio"
    description = (
        "Link a Composio app (Google Calendar, etc.) for the current iMessage "
        "user. Returns a Connect Link URL the user must open in a browser. "
        "Use for Google Calendar / Gmail / Composio — NEVER for SnapTrade "
        "portfolio or Whoop (those have their own link_* tools). "
        "Pass toolkit='googlecalendar' when the user asks to connect Calendar."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "toolkit": {
                "type": "string",
                "description": "Toolkit slug to connect (allowlist: googlecalendar).",
                "enum": ["googlecalendar"],
            }
        },
        "required": ["toolkit"],
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        toolkit = str(kwargs.get("toolkit") or "").strip().lower()
        if not toolkit:
            res: dict[str, Any] = {"status": "error", "detail": "toolkit is required"}
        elif toolkit != "googlecalendar":
            res = {
                "status": "error",
                "toolkit": toolkit,
                "detail": f"Toolkit {toolkit!r} is not in COMPOSIO_TOOLKITS (googlecalendar).",
            }
        elif self._world.composio_linked:
            res = {
                "status": "already_linked",
                "toolkit": toolkit,
                "phone": EVAL_SESSION_ID,
                "detail": f"{toolkit} is already linked for this user.",
                "toolkits": [{"toolkit": "googlecalendar", "connected": True}],
            }
        else:
            url = "https://connect.composio.dev/link/ca_mock_gcal_123"
            res = {
                "status": "pending",
                "toolkit": toolkit,
                "connection_id": "ca_mock_gcal_123",
                "redirect_url": url,
                "message": (
                    f"Ask the user to open this link and connect {toolkit}: {url}. "
                    "After they finish, they can ask again and the tools will work."
                ),
            }
        return self._world.record_call("chat", self.name, kwargs, res)


class MockComposioStatusTool(Tool):
    name = "composio_status"
    description = (
        "Check whether Composio apps (e.g. googlecalendar) are connected for "
        "the current iMessage user. Not for SnapTrade or Whoop."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "toolkit": {
                "type": "string",
                "description": "Optional toolkit slug to check. Omit to list allowlisted toolkits (googlecalendar).",
                "enum": ["googlecalendar"],
            }
        },
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        connected = self._world.composio_linked
        res = {
            "status": "linked" if connected else "not_linked",
            "phone": EVAL_SESSION_ID,
            "toolkit": "googlecalendar",
            "toolkits": [{"toolkit": "googlecalendar", "connected": connected}],
        }
        return self._world.record_call("chat", self.name, kwargs, res)


class MockUnlinkComposioTool(Tool):
    name = "unlink_composio"
    description = (
        "Unlink a Composio app (e.g. googlecalendar) for the current iMessage "
        "user. Not for SnapTrade or Whoop."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "toolkit": {
                "type": "string",
                "enum": ["googlecalendar"],
            }
        },
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        self._world.composio_linked = False
        res = {"status": "unlinked", "toolkit": kwargs.get("toolkit") or "googlecalendar"}
        return self._world.record_call("chat", self.name, kwargs, res)


class MockLinkSnaptradeTool(Tool):
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

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        if self._world.snaptrade_linked:
            res: dict[str, Any] = {
                "status": "already_linked",
                "detail": "SnapTrade is already linked for this user.",
            }
        else:
            url = "https://app.snaptrade.com/oauth/device?user_code=MOCK-1234"
            res = {
                "status": "pending",
                "verification_uri": url,
                "verification_uri_complete": url,
                "user_code": "MOCK-1234",
                "message": (
                    f"Ask the user to open this link and approve read access to SnapTrade: {url}. "
                    "After they finish, you will get a confirmation message automatically."
                ),
            }
        return self._world.record_call("chat", self.name, kwargs, res)


class MockSnaptradeStatusTool(Tool):
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

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        res = {
            "status": "linked" if self._world.snaptrade_linked else "not_linked",
            "server": "snaptrade",
            "connected": self._world.snaptrade_linked,
        }
        return self._world.record_call("chat", self.name, kwargs, res)


class MockUnlinkSnaptradeTool(Tool):
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

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        self._world.snaptrade_linked = False
        res = {"status": "unlinked", "server": "snaptrade"}
        return self._world.record_call("chat", self.name, kwargs, res)


class MockWhoopLinkTool(Tool):
    name = "link_whoop"
    description = (
        "Start linking Whoop for the current iMessage user. Returns a URL the "
        "user must open to approve read access; they get a confirmation text "
        "when done. Call when the user asks to connect Whoop."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        res = (
            {"status": "already_linked"}
            if self._world.whoop_linked
            else {
                "status": "pending",
                "url": "https://api.prod.whoop.com/oauth/oauth2/auth?state=mock",
            }
        )
        return self._world.record_call("chat", self.name, kwargs, res)


class MockWhoopStatusTool(Tool):
    name = "whoop_status"
    description = "Check whether Whoop is linked for the current user."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        res = {"linked": self._world.whoop_linked, "connector": "whoop"}
        return self._world.record_call("chat", self.name, kwargs, res)


class MockWhoopUnlinkTool(Tool):
    name = "unlink_whoop"
    description = (
        "Disconnect Whoop for the current user: revoke access and delete "
        "stored credentials. Call when the user asks to unlink it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        self._world.whoop_linked = False
        res = {"status": "unlinked", "connector": "whoop"}
        return self._world.record_call("chat", self.name, kwargs, res)


class MockGetCurrentTimeTool(Tool):
    name = "get_current_time"
    description = "Get the current date and time in UTC."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> str:
        self._world.record_call("executor", self.name, kwargs, FIXED_NOW_UTC)
        return FIXED_NOW_UTC


class MockQueryEventsTool(Tool):
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
            "source": {"type": "string", "description": "Connector name, e.g. whoop."},
            "type": {"type": "string", "description": "Event type, e.g. recovery.updated."},
            "days": {"type": "integer", "description": "How many days back to look."},
            "limit": {"type": "integer", "description": "Max events to return."},
        },
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        res = {
            "count": len(self._world.whoop_events),
            "events": list(self._world.whoop_events),
        }
        return self._world.record_call("executor", self.name, kwargs, res)


class MockComposioSearchTools(Tool):
    name = "COMPOSIO_SEARCH_TOOLS"
    description = (
        "Search and discover Composio tools by use case across enabled toolkits "
        "(e.g. Google Calendar). Returns matching tool slugs, connection status, "
        "and parameter schemas ready for COMPOSIO_MULTI_EXECUTE_TOOL."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": "List of tool search queries by use case.",
                "items": {
                    "type": "object",
                    "properties": {
                        "use_case": {
                            "type": "string",
                            "description": "What you want to accomplish, e.g. 'create a Google Calendar event' or 'list calendar events'.",
                        },
                        "known_fields": {
                            "type": "string",
                            "description": "Optional known field names or hints.",
                        },
                    },
                    "required": ["use_case"],
                },
            },
            "use_case": {
                "type": "string",
                "description": "Single use-case query string (shorthand if queries is omitted).",
            },
        },
        "additionalProperties": True,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        self._world.searched_composio = True
        raw_text = json.dumps(kwargs).lower()
        if "snaptrade" in raw_text and "calendar" not in raw_text:
            res: dict[str, Any] = {
                "successful": False,
                "error": "Toolkit 'snaptrade' is not available in Composio. Allowed toolkits: ['googlecalendar'].",
                "tools": [],
            }
            return self._world.record_call("executor", self.name, kwargs, res)

        res = {
            "successful": True,
            "session": {
                "id": "trs_mock_session_01",
                "current_utc_time": FIXED_NOW_UTC,
                "user_timezone": "America/Chicago",
            },
            "toolkit_connection_statuses": [
                {
                    "toolkit": "googlecalendar",
                    "status": "ACTIVE" if self._world.composio_linked else "INACTIVE",
                    "connected": self._world.composio_linked,
                }
            ],
            "tools": list(_GOOGLECALENDAR_SCHEMAS.values()),
        }
        return self._world.record_call("executor", self.name, kwargs, res)


class MockComposioGetToolSchemas(Tool):
    name = "COMPOSIO_GET_TOOL_SCHEMAS"
    description = (
        "Fetch full input parameter schemas for specific Composio tool slugs "
        "discovered via COMPOSIO_SEARCH_TOOLS."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "tool_slugs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact Composio tool slugs, e.g. ['GOOGLECALENDAR_CREATE_EVENT'].",
            }
        },
        "required": ["tool_slugs"],
        "additionalProperties": True,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        raw_slugs = (
            kwargs.get("tool_slugs")
            or kwargs.get("tools")
            or kwargs.get("slugs")
            or []
        )
        if isinstance(raw_slugs, str):
            slugs = [s.strip() for s in raw_slugs.split(",") if s.strip()]
        elif isinstance(raw_slugs, list):
            slugs = [str(s).strip() for s in raw_slugs if str(s).strip()]
        else:
            slugs = []

        found: dict[str, Any] = {}
        unknown: list[str] = []
        for slug in slugs:
            upper = slug.upper()
            if upper in _GOOGLECALENDAR_SCHEMAS:
                found[upper] = _GOOGLECALENDAR_SCHEMAS[upper]
            else:
                unknown.append(slug)

        if unknown and not found:
            res: dict[str, Any] = {
                "successful": False,
                "error": (
                    f"Unknown tool slug(s): {unknown}. Call COMPOSIO_SEARCH_TOOLS "
                    "first to discover valid Google Calendar tool slugs."
                ),
                "schemas": {},
            }
        else:
            res = {
                "successful": True,
                "schemas": found,
                "unknown_slugs": unknown,
            }
        return self._world.record_call("executor", self.name, kwargs, res)


class MockComposioManageConnections(Tool):
    name = "COMPOSIO_MANAGE_CONNECTIONS"
    description = (
        "Check or initiate OAuth connections for Composio toolkits "
        "(e.g. 'googlecalendar')."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "toolkits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Toolkit slugs to check or connect, e.g. ['googlecalendar'].",
            },
            "action": {
                "type": "string",
                "enum": ["check", "initiate"],
                "default": "check",
            },
        },
        "additionalProperties": True,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        if self._world.composio_linked:
            res: dict[str, Any] = {
                "successful": True,
                "connections": [
                    {
                        "toolkit": "googlecalendar",
                        "status": "ACTIVE",
                        "connected": True,
                    }
                ],
            }
        else:
            res = {
                "successful": True,
                "connections": [
                    {
                        "toolkit": "googlecalendar",
                        "status": "INACTIVE",
                        "connected": False,
                        "redirect_url": "https://connect.composio.dev/link/ca_mock_gcal_123",
                    }
                ],
            }
        return self._world.record_call("executor", self.name, kwargs, res)


class MockComposioMultiExecuteTool(Tool):
    name = "COMPOSIO_MULTI_EXECUTE_TOOL"
    description = (
        "Execute one or more Composio tools by tool_slug with their JSON arguments."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "description": "List of tool invocations to execute.",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_slug": {
                            "type": "string",
                            "description": "Exact tool slug, e.g. 'GOOGLECALENDAR_CREATE_EVENT' or 'GOOGLECALENDAR_EVENTS_LIST'.",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments matching the tool's input_parameters schema.",
                        },
                    },
                    "required": ["tool_slug", "arguments"],
                },
            }
        },
        "required": ["tools"],
        "additionalProperties": True,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        invocations = (
            kwargs.get("tools")
            or kwargs.get("executions")
            or kwargs.get("tool_calls")
        )
        if not invocations and (kwargs.get("tool_slug") or kwargs.get("tool_name")):
            invocations = [
                {
                    "tool_slug": kwargs.get("tool_slug") or kwargs.get("tool_name"),
                    "arguments": kwargs.get("arguments") or kwargs.get("params") or {},
                }
            ]
        if not isinstance(invocations, list) or not invocations:
            res: dict[str, Any] = {
                "successful": False,
                "error": "No tools provided in 'tools' array.",
            }
            return self._world.record_call("executor", self.name, kwargs, res)

        results: list[dict[str, Any]] = []
        for item in invocations:
            if not isinstance(item, dict):
                continue
            slug = str(
                item.get("tool_slug") or item.get("name") or item.get("slug") or ""
            ).strip().upper()
            raw_args = item.get("arguments") or item.get("params") or {}
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            if "SNAPTRADE" in slug:
                results.append(
                    {
                        "tool_slug": slug,
                        "successful": False,
                        "error": "SnapTrade is not a Composio toolkit.",
                    }
                )
            elif slug in _CREATE_SLUGS:
                summary = str(
                    args.get("summary")
                    or args.get("title")
                    or args.get("text")
                    or "Ideal Bedtime"
                )
                start_dt = str(
                    args.get("start_datetime")
                    or (args.get("start") or {}).get("dateTime")
                    if isinstance(args.get("start"), dict)
                    else args.get("start_datetime")
                    or args.get("start")
                    or "2026-11-11T22:30:00-06:00"
                )
                end_dt = str(
                    args.get("end_datetime")
                    or (args.get("end") or {}).get("dateTime")
                    if isinstance(args.get("end"), dict)
                    else args.get("end_datetime")
                    or args.get("end")
                    or "2026-11-11T23:00:00-06:00"
                )
                event = self._world.create_calendar_event(
                    summary=summary,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    description=str(args.get("description") or ""),
                    timezone=str(args.get("timezone") or "America/Chicago"),
                )
                results.append(
                    {
                        "tool_slug": slug,
                        "successful": True,
                        "data": {"event": event},
                    }
                )
            elif slug in _LIST_SLUGS:
                events = self._world.list_calendar_events(
                    query=str(args.get("query") or args.get("q") or "") or None
                )
                results.append(
                    {
                        "tool_slug": slug,
                        "successful": True,
                        "data": {"items": events, "count": len(events)},
                    }
                )
            else:
                results.append(
                    {
                        "tool_slug": slug,
                        "successful": False,
                        "error": (
                            f"Unknown tool_slug {slug!r}. Use COMPOSIO_SEARCH_TOOLS "
                            "to find valid Google Calendar tools."
                        ),
                    }
                )

        overall_ok = bool(results) and all(r.get("successful") for r in results)
        res = {"successful": overall_ok, "results": results}
        return self._world.record_call("executor", self.name, kwargs, res)


class MockComposioRemoteBashTool(Tool):
    name = "COMPOSIO_REMOTE_BASH_TOOL"
    description = "Execute bash commands inside a remote Composio sandbox."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."}
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        res = {
            "successful": False,
            "error": "Remote bash sandbox is not needed for Google Calendar API calls. Use COMPOSIO_SEARCH_TOOLS and COMPOSIO_MULTI_EXECUTE_TOOL.",
        }
        return self._world.record_call("executor", self.name, kwargs, res)


class MockComposioRemoteWorkbench(Tool):
    name = "COMPOSIO_REMOTE_WORKBENCH"
    description = "Run Python scripts in a remote Composio workbench for bulk data processing."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute."}
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        res = {
            "successful": False,
            "error": "Remote workbench is not needed for direct Google Calendar calls. Use COMPOSIO_SEARCH_TOOLS and COMPOSIO_MULTI_EXECUTE_TOOL.",
        }
        return self._world.record_call("executor", self.name, kwargs, res)


class MockFlatListCalendarEventsTool(Tool):
    name = "list_calendar_events"
    description = (
        "List events on the user's Google Calendar for a date range or search query."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date to check in YYYY-MM-DD format (default today, 2026-11-11).",
            },
            "query": {
                "type": "string",
                "description": "Optional text filter for event titles.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        events = self._world.list_calendar_events(
            query=str(kwargs.get("query") or "") or None
        )
        res = {"status": "ok", "count": len(events), "events": events}
        return self._world.record_call("executor", self.name, kwargs, res)


class MockFlatCreateCalendarEventTool(Tool):
    name = "create_calendar_event"
    description = "Create a new event on the user's Google Calendar."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Event title, e.g. 'Ideal Bedtime'.",
            },
            "start_datetime": {
                "type": "string",
                "description": "ISO-8601 start datetime, e.g. '2026-11-11T22:30:00-06:00'.",
            },
            "end_datetime": {
                "type": "string",
                "description": "ISO-8601 end datetime, e.g. '2026-11-11T23:00:00-06:00'.",
            },
            "description": {
                "type": "string",
                "description": "Optional event description.",
            },
        },
        "required": ["summary", "start_datetime"],
        "additionalProperties": False,
    }

    def __init__(self, world: MockWorld) -> None:
        self._world = world

    def execute(self, **kwargs: Any) -> Any:
        event = self._world.create_calendar_event(
            summary=str(kwargs.get("summary") or "Ideal Bedtime"),
            start_datetime=str(
                kwargs.get("start_datetime") or "2026-11-11T22:30:00-06:00"
            ),
            end_datetime=str(
                kwargs.get("end_datetime") or "2026-11-11T23:00:00-06:00"
            ),
            description=str(kwargs.get("description") or ""),
        )
        res = {"status": "created", "event": event}
        return self._world.record_call("executor", self.name, kwargs, res)


def register_worker_tools(
    registry: ToolRegistry,
    world: MockWorld,
    *,
    tool_surface: str = "full_composio",
) -> list[str]:
    """Register local + calendar worker tools for the selected tool surface."""
    registry.register(MockGetCurrentTimeTool(world))
    registry.register(MockQueryEventsTool(world))

    if tool_surface == "full_composio":
        tools: list[Tool] = [
            MockComposioSearchTools(world),
            MockComposioGetToolSchemas(world),
            MockComposioManageConnections(world),
            MockComposioMultiExecuteTool(world),
            MockComposioRemoteBashTool(world),
            MockComposioRemoteWorkbench(world),
        ]
    elif tool_surface == "slim_composio":
        tools = [
            MockComposioSearchTools(world),
            MockComposioManageConnections(world),
            MockComposioMultiExecuteTool(world),
        ]
    elif tool_surface == "flat":
        tools = [
            MockFlatListCalendarEventsTool(world),
            MockFlatCreateCalendarEventTool(world),
        ]
    else:
        raise ValueError(f"Unknown tool_surface: {tool_surface!r}")

    for tool in tools:
        registry.register(tool)
    return [tool.name for tool in tools]


def register_chat_tools(
    registry: ToolRegistry,
    world: MockWorld,
) -> None:
    """Register ChatAgent-facing account and acknowledgement tools."""
    registry.register(RecordingSendAckTool(world))
    registry.register(MockLinkComposioTool(world))
    registry.register(MockComposioStatusTool(world))
    registry.register(MockUnlinkComposioTool(world))
    registry.register(MockLinkSnaptradeTool(world))
    registry.register(MockSnaptradeStatusTool(world))
    registry.register(MockUnlinkSnaptradeTool(world))
    registry.register(MockWhoopLinkTool(world))
    registry.register(MockWhoopStatusTool(world))
    registry.register(MockWhoopUnlinkTool(world))
