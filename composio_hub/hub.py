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


class ComposioHub:
    """Owns a Composio client and one Tool Router session per phone number."""

    def __init__(
        self,
        *,
        api_key: str,
        toolkits: list[str] | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._toolkits = list(toolkits or [])
        self._client: Composio | None = None
        self._sessions: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._tools: list[ComposioTool] = []
        self._started = False

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
        create_kwargs: dict[str, Any] = {"user_id": bootstrap_user}
        if self._toolkits:
            create_kwargs["toolkits"] = self._toolkits
        session = client.create(**create_kwargs)
        raw_tools = session.tools()
        adapters: list[ComposioTool] = []
        for raw in raw_tools:
            parts = _openai_spec_parts(raw)
            if parts is None:
                logger.warning("Skipping unrecognized Composio tool shape: %r", raw)
                continue
            name, description, parameters = parts
            adapters.append(
                ComposioTool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    hub=self,
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
            "Composio ready: %d meta tool(s), toolkits=%s",
            len(adapters),
            self._toolkits or "all",
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
            create_kwargs: dict[str, Any] = {"user_id": phone}
            if self._toolkits:
                create_kwargs["toolkits"] = self._toolkits
            session = client.create(**create_kwargs)
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
        return _serialize_result(result)

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
