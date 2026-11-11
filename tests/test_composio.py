"""Unit tests for Composio adapters (no live API)."""

from __future__ import annotations

from composio_hub.hub import _openai_spec_parts, _serialize_result
from composio_hub.tool import ComposioTool
from tools.registry import ToolRegistry


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
