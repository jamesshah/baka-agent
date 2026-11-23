"""Per-phone Composio Sessions hub (Tool Router)."""

from __future__ import annotations

import logging
import threading
from typing import Any

from composio import Composio

from composio_hub.tool import ComposioTool
from tools.session_context import require_session_id

logger = logging.getLogger(__name__)


def _serialize_result(result: Any) -> Any:
    """Turn a Composio execute response into a JSON-friendly value."""
    if result is None:
        return None
    if isinstance(result, (str, int, float, bool, list, dict)):
        return result
    if hasattr(result, "model_dump"):
        try:
            return result.model_dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001
            pass
    data: dict[str, Any] = {}
    for key in ("result_type", "data", "error", "log_id", "successful", "message"):
        if hasattr(result, key):
            value = getattr(result, key)
            if value is not None:
                data[key] = value
    if data:
        return data
    return str(result)


def _toolkit_item_connected(item: Any) -> bool:
    """Whether a ``session.toolkits()`` row represents an active connection.

    Composio's Tool Router returns ``ToolkitConnectionState`` with nested
    ``connection.is_active`` (and optionally ``connection.connected_account``).
    Older/dict shapes may use top-level ``is_connected`` / ``connected``.
    """
    if isinstance(item, dict):
        connection = item.get("connection")
        if isinstance(connection, dict):
            if "is_active" in connection:
                return bool(connection.get("is_active"))
            account = connection.get("connected_account")
            if isinstance(account, dict) and account.get("status"):
                return str(account.get("status")).upper() == "ACTIVE"
            if account:
                return True
        for key in ("is_connected", "connected", "is_active"):
            if key in item and item.get(key) is not None:
                return bool(item.get(key))
        return False

    connection = getattr(item, "connection", None)
    if connection is not None:
        is_active = getattr(connection, "is_active", None)
        if is_active is not None:
            return bool(is_active)
        account = getattr(connection, "connected_account", None)
        if account is not None:
            status = getattr(account, "status", None)
            if status is not None:
                return str(status).upper() == "ACTIVE"
            return True

    for key in ("is_connected", "connected", "is_active"):
        value = getattr(item, key, None)
        if value is not None:
            return bool(value)
    return False


def _openai_spec_parts(raw: Any) -> tuple[str, str, dict[str, Any]] | None:
    """Extract name/description/parameters from an OpenAI-style tool dict."""
    if not isinstance(raw, dict):
        # TypedDict / MappingProxy / pydantic-ish
        try:
            raw = dict(raw)
        except Exception:  # noqa: BLE001
            return None
    function = raw.get("function") if raw.get("type") == "function" else raw
    if not isinstance(function, dict):
        try:
            function = dict(function)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return None
    name = function.get("name") or raw.get("name")
    if not name or not isinstance(name, str):
        return None
    description = function.get("description") or raw.get("description") or ""
    parameters = function.get("parameters") or raw.get("parameters") or {
        "type": "object",
        "properties": {},
    }
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    return name, str(description), parameters


_META_OPTIONAL_DROPS: dict[str, frozenset[str]] = {
    "COMPOSIO_SEARCH_TOOLS": frozenset({"model", "search_strategy"}),
    "COMPOSIO_MULTI_EXECUTE_TOOL": frozenset(
        {"thought", "current_step", "current_step_metric"}
    ),
    "COMPOSIO_GET_TOOL_SCHEMAS": frozenset({"include"}),
    "COMPOSIO_MANAGE_CONNECTIONS": frozenset({"reinitiate_all"}),
}
_SCHEMA_NOISE_KEYS = frozenset(
    {"title", "examples", "deprecated", "$schema", "additionalProperties"}
)
_SCHEMA_TOOLS_TO_COMPACT = frozenset(
    {"COMPOSIO_SEARCH_TOOLS", "COMPOSIO_GET_TOOL_SCHEMAS"}
)


def _truncate_text(value: str, max_chars: int) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _compact_json_schema(
    node: Any,
    *,
    depth: int = 0,
    is_required: bool = True,
) -> Any:
    """Recursively shrink verbose JSON Schema definitions without breaking types."""
    if isinstance(node, list):
        return [
            _compact_json_schema(item, depth=depth + 1, is_required=is_required)
            for item in node
        ]
    if not isinstance(node, dict):
        return node

    # Collapse deeply nested optional object schemas (e.g. Google Calendar
    # workingLocationProperties / outOfOfficeProperties / focusTimeProperties).
    if (
        depth >= 2
        and not is_required
        and node.get("type") == "object"
        and "properties" in node
    ):
        summary: dict[str, Any] = {"type": "object"}
        if isinstance(node.get("description"), str) and node["description"].strip():
            summary["description"] = _truncate_text(node["description"], 120)
        return summary

    out: dict[str, Any] = {}
    required_set = set(node.get("required") or []) if isinstance(node.get("required"), list) else set()
    for key, value in node.items():
        if key in _SCHEMA_NOISE_KEYS:
            continue
        if key == "description" and isinstance(value, str):
            out[key] = _truncate_text(value, 160)
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {
                prop_name: _compact_json_schema(
                    prop_schema,
                    depth=depth + 1,
                    is_required=(prop_name in required_set),
                )
                for prop_name, prop_schema in value.items()
            }
            continue
        out[key] = _compact_json_schema(value, depth=depth + 1, is_required=is_required)
    return out


def _compact_meta_tool_spec(
    name: str,
    description: str,
    parameters: dict[str, Any],
    toolkits: list[str],
) -> tuple[str, dict[str, Any]]:
    """Trim verbose Composio meta-tool essays & telemetry params for small LLMs."""
    scope = ", ".join(toolkits) if toolkits else "all enabled apps"
    compact_descriptions = {
        "COMPOSIO_SEARCH_TOOLS": (
            f"Discover available Composio tools for {scope}. Call first to get "
            "exact tool slugs, parameter schemas, and connection status. Pass "
            "queries=[{use_case: '...'}] and session={generate_id: true} (or "
            "{id: '...'} to continue)."
        ),
        "COMPOSIO_GET_TOOL_SCHEMAS": (
            "Fetch input parameter schemas for exact tool slugs returned by "
            "COMPOSIO_SEARCH_TOOLS (never guess slugs)."
        ),
        "COMPOSIO_MULTI_EXECUTE_TOOL": (
            "Execute one or more discovered Composio tools by exact tool_slug "
            "and schema-compliant arguments. Only batch independent calls."
        ),
        "COMPOSIO_MANAGE_CONNECTIONS": (
            f"Check or initiate OAuth Connect Links for Composio toolkits ({scope}). "
            "Call if a toolkit is not connected and return the Connect Link URL."
        ),
    }
    compact_desc = compact_descriptions.get(name, _truncate_text(description, 260))

    drop_props = _META_OPTIONAL_DROPS.get(name, frozenset())
    required_set = set(parameters.get("required") or [])
    raw_props = parameters.get("properties")
    pruned_params = dict(parameters)
    if isinstance(raw_props, dict) and drop_props:
        pruned_params["properties"] = {
            k: v
            for k, v in raw_props.items()
            if k not in drop_props or k in required_set
        }
    compact_params = _compact_json_schema(pruned_params, depth=0, is_required=True)
    if not isinstance(compact_params, dict):
        compact_params = {"type": "object", "properties": {}}
    return compact_desc, compact_params


def _compact_runtime_result(tool_slug: str, payload: Any) -> Any:
    """Compact runtime schema payloads returned by SEARCH_TOOLS / GET_TOOL_SCHEMAS."""
    if tool_slug not in _SCHEMA_TOOLS_TO_COMPACT:
        return payload

    def _walk(obj: Any) -> Any:
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        if not isinstance(obj, dict):
            return obj
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in {"input_schema", "input_parameters", "parameters"} and isinstance(value, dict):
                out[key] = _compact_json_schema(value, depth=1, is_required=True)
                continue
            if key == "description" and isinstance(value, str):
                out[key] = _truncate_text(value, 220)
                continue
            if key in {"plan", "pitfalls", "recommended_plan", "instructions"} and isinstance(value, str):
                out[key] = _truncate_text(value, 600)
                continue
            out[key] = _walk(value)
        return out

    return _walk(payload)


class ComposioHub:
    """Owns a Composio client and one Tool Router session per phone number."""

    def __init__(
        self,
        *,
        api_key: str,
        toolkits: list[str] | None = None,
        sandbox_enabled: bool = False,
        compact_schemas: bool = True,
    ) -> None:
        self._api_key = api_key.strip()
        self._toolkits = list(toolkits or [])
        self._sandbox_enabled = sandbox_enabled
        self._compact_schemas = compact_schemas
        self._client: Composio | None = None
        self._sessions: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._tools: list[ComposioTool] = []
        self._started = False

    def _session_create_kwargs(self, user_id: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"user_id": user_id}
        if self._toolkits:
            kwargs["toolkits"] = self._toolkits
        if not self._sandbox_enabled:
            kwargs["sandbox"] = {"enable": False}
        return kwargs

    @property
    def tools(self) -> list[ComposioTool]:
        return list(self._tools)

    @property
    def toolkit_slugs(self) -> list[str]:
        return list(self._toolkits)

    def start(self) -> None:
        """Create the SDK client and discover Tool Router meta-tools."""
        if self._started:
            return
        if not self._api_key:
            raise ValueError("COMPOSIO_API_KEY is required")

        client = Composio(api_key=self._api_key)
        # Bootstrap session only to learn meta-tool schemas (same for every user).
        bootstrap_user = "baka-agent-bootstrap"
        session = client.create(**self._session_create_kwargs(bootstrap_user))
        raw_tools = session.tools()
        scope_label = ", ".join(self._toolkits) if self._toolkits else "all configured apps"
        cap_desc = (
            f"Search, connect, and run actions in connected Composio apps ({scope_label})."
        )
        adapters: list[ComposioTool] = []
        for raw in raw_tools:
            parts = _openai_spec_parts(raw)
            if parts is None:
                logger.warning("Skipping unrecognized Composio tool shape: %r", raw)
                continue
            name, description, parameters = parts
            if self._compact_schemas:
                description, parameters = _compact_meta_tool_spec(
                    name, description, parameters, self._toolkits
                )
            adapters.append(
                ComposioTool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    hub=self,
                    capability="composio",
                    capability_description=cap_desc,
                )
            )
        if not adapters:
            raise RuntimeError("Composio session.tools() returned no tools")

        with self._lock:
            self._client = client
            # Keep the bootstrap session in the cache so we do not leak orphans
            # on every restart; real phones get their own sessions.
            self._sessions[bootstrap_user] = session
            self._tools = adapters
            self._started = True
        logger.info(
            "Composio ready: %d meta tool(s), toolkits=%s, sandbox=%s, compact=%s",
            len(adapters),
            self._toolkits or "all",
            self._sandbox_enabled,
            self._compact_schemas,
        )

    def stop(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._tools = []
            self._client = None
            self._started = False

    def status(self) -> dict[str, Any]:
        if not self._started or self._client is None:
            return {
                "status": "disabled",
                "toolkits": self._toolkits,
                "tool_count": 0,
                "session_count": 0,
            }
        return {
            "status": "ok",
            "toolkits": self._toolkits,
            "tool_count": len(self._tools),
            "session_count": len(self._sessions),
            "tools": [tool.name for tool in self._tools],
        }

    def _require_client(self) -> Composio:
        if self._client is None:
            raise RuntimeError("Composio hub is not started")
        return self._client

    def session_for(self, phone: str) -> Any:
        """Return (creating if needed) the Tool Router session for ``phone``."""
        phone = phone.strip()
        if not phone:
            raise ValueError("phone is required")
        with self._lock:
            existing = self._sessions.get(phone)
            if existing is not None:
                return existing
            client = self._require_client()
            session = client.create(**self._session_create_kwargs(phone))
            self._sessions[phone] = session
            logger.info("Created Composio session for %s (%s)", phone, session.session_id)
            return session

    def execute(self, tool_slug: str, arguments: dict[str, Any] | None = None) -> Any:
        phone = require_session_id()
        session = self.session_for(phone)
        try:
            result = session.execute(tool_slug, arguments=arguments or {})
        except Exception:
            logger.exception("Composio execute failed: %s", tool_slug)
            raise
        serialized = _serialize_result(result)
        if self._compact_schemas:
            return _compact_runtime_result(tool_slug, serialized)
        return serialized

    def authorize(self, phone: str, toolkit: str) -> dict[str, Any]:
        toolkit = toolkit.strip().lower()
        allowed = {slug.lower() for slug in self._toolkits}
        if allowed and toolkit not in allowed:
            return {
                "status": "error",
                "detail": (
                    f"Toolkit {toolkit!r} is not in COMPOSIO_TOOLKITS "
                    f"({', '.join(self._toolkits)})."
                ),
            }
        session = self.session_for(phone)
        try:
            request = session.authorize(toolkit)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Composio authorize failed for %s", toolkit)
            return {"status": "error", "toolkit": toolkit, "detail": str(exc)}
        return {
            "status": "pending",
            "toolkit": toolkit,
            "connection_id": getattr(request, "id", None),
            "redirect_url": getattr(request, "redirect_url", None),
            "message": (
                f"Ask the user to open this link and connect {toolkit}: "
                f"{getattr(request, 'redirect_url', None)}. "
                "After they finish, they can ask again and the tools will work."
            ),
        }

    def link_status(self, phone: str, toolkit: str | None = None) -> dict[str, Any]:
        client = self._require_client()
        try:
            session = self.session_for(phone)
            kwargs: dict[str, Any] = {}
            if toolkit:
                kwargs["toolkits"] = [toolkit.strip().lower()]
            elif self._toolkits:
                kwargs["toolkits"] = list(self._toolkits)
            details = session.toolkits(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Composio toolkit status failed")
            # Fall back to connected_accounts.list
            try:
                listed = client.connected_accounts.list(
                    user_ids=[phone],
                    statuses=["ACTIVE"],
                )
                items = getattr(listed, "items", None) or getattr(listed, "data", None) or []
                accounts = []
                for item in items:
                    slug = (
                        getattr(item, "toolkit", None)
                        or getattr(item, "appName", None)
                        or getattr(item, "app_name", None)
                    )
                    if hasattr(slug, "slug"):
                        slug = slug.slug
                    accounts.append(
                        {
                            "id": getattr(item, "id", None),
                            "toolkit": slug,
                            "status": getattr(item, "status", None),
                        }
                    )
                if toolkit:
                    toolkit = toolkit.strip().lower()
                    accounts = [
                        a
                        for a in accounts
                        if str(a.get("toolkit") or "").lower() == toolkit
                    ]
                return {
                    "status": "ok",
                    "phone": phone,
                    "connected": accounts,
                    "detail": f"toolkits() failed: {exc}",
                }
            except Exception as exc2:  # noqa: BLE001
                return {"status": "error", "detail": f"{exc}; fallback: {exc2}"}

        items = getattr(details, "items", None) or getattr(details, "toolkits", None)
        if items is None and isinstance(details, dict):
            items = details.get("items") or details.get("toolkits") or []
        if items is None:
            items = []

        connected: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                slug = item.get("slug") or item.get("toolkit") or item.get("name")
                connected.append(
                    {
                        "toolkit": slug,
                        "connected": _toolkit_item_connected(item),
                        "raw": item,
                    }
                )
            else:
                slug = getattr(item, "slug", None) or getattr(item, "toolkit", None)
                connected.append(
                    {
                        "toolkit": slug,
                        "connected": _toolkit_item_connected(item),
                    }
                )

        if toolkit:
            toolkit_l = toolkit.strip().lower()
            matched = [
                row
                for row in connected
                if str(row.get("toolkit") or "").lower() == toolkit_l
            ]
            linked = any(row.get("connected") for row in matched)
            return {
                "status": "linked" if linked else "not_linked",
                "phone": phone,
                "toolkit": toolkit_l,
                "toolkits": matched or connected,
            }

        any_connected = any(row.get("connected") for row in connected)
        return {
            "status": "linked" if any_connected else "not_linked",
            "phone": phone,
            "toolkits": connected,
        }

    def unlink(self, phone: str, toolkit: str | None = None) -> dict[str, Any]:
        """Delete connected accounts for ``phone``, optionally filtered by toolkit."""
        client = self._require_client()
        toolkit_filter = toolkit.strip().lower() if toolkit else None
        try:
            listed = client.connected_accounts.list(user_ids=[phone])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Composio list connected accounts failed")
            return {"status": "error", "detail": str(exc)}

        items = getattr(listed, "items", None) or getattr(listed, "data", None) or []
        deleted: list[str] = []
        errors: list[str] = []
        for item in items:
            item_id = getattr(item, "id", None)
            slug = getattr(item, "toolkit", None)
            if hasattr(slug, "slug"):
                slug = slug.slug
            if slug is None:
                slug = getattr(item, "appName", None) or getattr(item, "app_name", None)
            slug_str = str(slug or "").lower()
            if toolkit_filter and slug_str and slug_str != toolkit_filter:
                continue
            if not item_id:
                continue
            try:
                client.connected_accounts.delete(item_id, revoke_on_delete=True)
                deleted.append(item_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to delete connected account %s", item_id)
                errors.append(f"{item_id}: {exc}")

        with self._lock:
            self._sessions.pop(phone, None)

        if errors and not deleted:
            return {"status": "error", "deleted": deleted, "errors": errors}
        return {
            "status": "unlinked",
            "deleted": deleted,
            "errors": errors or None,
            "toolkit": toolkit_filter,
        }
