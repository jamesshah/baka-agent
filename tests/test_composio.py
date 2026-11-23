"""Unit tests for Composio adapters (no live API)."""

from __future__ import annotations

from types import SimpleNamespace

from composio_hub.hub import (
    ComposioHub,
    _compact_meta_tool_spec,
    _compact_runtime_result,
    _openai_spec_parts,
    _serialize_result,
    _toolkit_item_connected,
)
from composio_hub.tool import ComposioTool
from tools.registry import ToolRegistry


def test_toolkit_item_connected_reads_nested_is_active() -> None:
    # Shape returned by Composio session.toolkits() today.
    active = SimpleNamespace(
        slug="googlecalendar",
        connection=SimpleNamespace(
            is_active=True,
            connected_account=SimpleNamespace(id="ca_x", status="ACTIVE"),
        ),
    )
    inactive = SimpleNamespace(
        slug="googlecalendar",
        connection=SimpleNamespace(is_active=False, connected_account=None),
    )
    assert _toolkit_item_connected(active) is True
    assert _toolkit_item_connected(inactive) is False
    assert _toolkit_item_connected({"slug": "googlecalendar", "connected": True}) is True
    assert (
        _toolkit_item_connected(
            {
                "slug": "googlecalendar",
                "connection": {
                    "is_active": True,
                    "connected_account": {"id": "ca_x", "status": "ACTIVE"},
                },
            }
        )
        is True
    )
    # Missing nested connection must not silently count as linked.
    assert _toolkit_item_connected(SimpleNamespace(slug="googlecalendar")) is False


def test_openai_spec_parts_from_function_tool() -> None:
    raw = {
        "type": "function",
        "function": {
            "name": "COMPOSIO_SEARCH_TOOLS",
            "description": "Search tools",
            "parameters": {
                "type": "object",
                "properties": {"queries": {"type": "array"}},
            },
        },
    }
    parts = _openai_spec_parts(raw)
    assert parts is not None
    name, description, parameters = parts
    assert name == "COMPOSIO_SEARCH_TOOLS"
    assert description == "Search tools"
    assert parameters["type"] == "object"


def test_serialize_result_primitives_and_model_dump() -> None:
    assert _serialize_result({"ok": True}) == {"ok": True}
    assert _serialize_result("done") == "done"

    class Fake:
        def model_dump(self, mode: str = "python", exclude_none: bool = False):
            del mode, exclude_none
            return {"result_type": "completed", "data": {"n": 1}}

    assert _serialize_result(Fake()) == {"result_type": "completed", "data": {"n": 1}}


def test_composio_tool_excluded_from_local_names() -> None:
    class StubHub:
        def execute(self, tool_slug: str, arguments: dict | None = None):
            del tool_slug, arguments
            return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ComposioTool(
            name="COMPOSIO_SEARCH_TOOLS",
            description="Search",
            parameters={"type": "object", "properties": {}},
            hub=StubHub(),  # type: ignore[arg-type]
        )
    )
    assert registry.names(chat_agent=False) == ["COMPOSIO_SEARCH_TOOLS"]
    assert registry.local_names(chat_agent=False) == []


def test_composio_hub_disables_sandbox_by_default() -> None:
    hub = ComposioHub(api_key="ak_test", toolkits=["googlecalendar", "gmail"])
    kwargs = hub._session_create_kwargs("+15550001234")
    assert kwargs["toolkits"] == ["googlecalendar", "gmail"]
    assert kwargs["sandbox"] == {"enable": False}

    hub_sb = ComposioHub(api_key="ak_test", toolkits=["googlecalendar"], sandbox_enabled=True)
    assert "sandbox" not in hub_sb._session_create_kwargs("+15550001234")


def test_compact_meta_tool_spec_trims_bloat_and_keeps_required() -> None:
    raw_params = {
        "type": "object",
        "required": ["tools"],
        "properties": {
            "tools": {
                "type": "array",
                "title": "Tools",
                "examples": [[{"tool_slug": "GMAIL_SEND_EMAIL"}]],
                "description": "Execute tools in parallel.",
                "items": {"type": "object", "properties": {"tool_slug": {"type": "string"}}},
            },
            "thought": {"type": "string", "description": "Internal reasoning."},
            "current_step": {"type": "string"},
            "current_step_metric": {"type": "string"},
            "session_id": {"type": "string", "title": "Session ID"},
        },
    }
    desc, params = _compact_meta_tool_spec(
        "COMPOSIO_MULTI_EXECUTE_TOOL",
        "X" * 3000,
        raw_params,
        ["googlecalendar", "gmail"],
    )
    assert len(desc) < 250
    assert "tools" in params["properties"]
    assert "session_id" in params["properties"]
    assert "thought" not in params["properties"]
    assert "current_step" not in params["properties"]
    assert "title" not in params["properties"]["tools"]
    assert "examples" not in params["properties"]["tools"]


def test_compact_runtime_result_collapses_deep_optional_subobjects() -> None:
    payload = {
        "successful": True,
        "schemas": {
            "GOOGLECALENDAR_CREATE_EVENT": {
                "description": "Create an event " + ("extra " * 80),
                "input_schema": {
                    "type": "object",
                    "required": ["summary"],
                    "properties": {
                        "summary": {"type": "string", "title": "Summary", "description": "Title"},
                        "workingLocationProperties": {
                            "type": "object",
                            "description": "Working location details.",
                            "properties": {
                                "homeOffice": {"type": "object", "properties": {"a": {"type": "string"}}}
                            },
                        },
                    },
                },
            }
        },
    }
    compacted = _compact_runtime_result("COMPOSIO_GET_TOOL_SCHEMAS", payload)
    schema = compacted["schemas"]["GOOGLECALENDAR_CREATE_EVENT"]["input_schema"]
    assert "title" not in schema["properties"]["summary"]
    wl = schema["properties"]["workingLocationProperties"]
    assert wl["type"] == "object"
    assert "properties" not in wl
