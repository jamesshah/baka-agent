"""Background multi-server MCP connection hub."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, Tool as McpSdkTool

from mcp_client.config import McpServerConfig
from mcp_client.oauth_device import (
    BearerTokenAuth,
    DeviceLinkSession,
    OAuthDeviceClient,
    OAuthDeviceError,
)
from mcp_client.token_store import FileTokenStore
from mcp_client.tool import McpTool

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")

ToolsChangedCallback = Callable[[list[McpTool], list[str]], None]
LinkedCallback = Callable[[str, str], None]


def sanitize_name(value: str) -> str:
    """Make a name safe for OpenAI-style function tool identifiers."""
    cleaned = _SAFE_NAME_RE.sub("_", value.strip()).strip("_")
    return cleaned or "mcp"


class MultiServerMcpClient:
    """
    Connect to many MCP servers at once and expose their tools.

    Owns a dedicated asyncio event loop on a background thread so the rest of
    the (sync) agent can call tools safely via ``call_tool``.

    OAuth servers connect lazily once tokens exist (or after device-code link).
    """

    def __init__(
        self,
        servers: dict[str, McpServerConfig],
        *,
        token_store: FileTokenStore | None = None,
        oauth_owner_phone: str = "",
        on_tools_changed: ToolsChangedCallback | None = None,
        on_linked: LinkedCallback | None = None,
    ) -> None:
        self._servers = servers
        self._token_store = token_store or FileTokenStore(".data/mcp-oauth")
        self._oauth_owner_phone = oauth_owner_phone.strip()
        self._oauth = OAuthDeviceClient(self._token_store)
        self._on_tools_changed = on_tools_changed
        self._on_linked = on_linked

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._shutdown: asyncio.Event | None = None

        self._sessions: dict[str, ClientSession] = {}
        self._server_stacks: dict[str, AsyncExitStack] = {}
        self._tool_index: dict[str, tuple[str, str]] = {}
        self._tools: list[McpTool] = []
        self._pending_auth: set[str] = set()
        self._link_sessions: dict[str, DeviceLinkSession] = {}
        self._link_tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def tools(self) -> list[McpTool]:
        return list(self._tools)

    @property
    def token_store(self) -> FileTokenStore:
        return self._token_store

    @property
    def oauth_owner_phone(self) -> str:
        return self._oauth_owner_phone

    def oauth_server_names(self) -> list[str]:
        return sorted(
            n for n, c in self._servers.items() if c.auth == "oauth" and c.enabled
        )

    def has_server(self, name: str) -> bool:
        config = self._servers.get(name)
        return config is not None and config.enabled

    def server_config(self, name: str) -> McpServerConfig | None:
        return self._servers.get(name)

    def status(self) -> dict[str, Any]:
        """Snapshot of configured vs connected MCP servers and tools."""
        configured = sorted(n for n, c in self._servers.items() if c.enabled)
        disabled = sorted(n for n, c in self._servers.items() if not c.enabled)
        connected = sorted(self._sessions)
        pending = sorted(self._pending_auth)
        running = self._loop is not None and self._loop.is_running()
        if not configured:
            overall = "disabled"
        elif not running:
            overall = "error"
        elif set(connected) == set(configured):
            overall = "ok"
        elif connected or pending:
            overall = "degraded"
        else:
            overall = "error"
        return {
            "status": overall,
            "running": running,
            "configured": configured,
            "disabled": disabled,
            "connected": connected,
            "pending_auth": pending,
            "tool_count": len(self._tools),
        }

    def start(self, timeout: float = 60.0) -> None:
        """Spawn the background loop, connect servers, and discover tools."""
        if not any(config.enabled for config in self._servers.values()):
            logger.info("No MCP servers configured")
            return
        if self._thread is not None:
            raise RuntimeError("MultiServerMcpClient already started")

        self._thread = threading.Thread(
            target=self._thread_main,
            name="mcp-client-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            self.stop()
            raise TimeoutError("Timed out connecting to MCP servers")
        if self._start_error is not None:
            err = self._start_error
            self.stop()
            raise RuntimeError(f"Failed to start MCP client: {err}") from err

    def stop(self, timeout: float = 15.0) -> None:
        """Disconnect all servers and stop the background loop."""
        loop = self._loop
        shutdown = self._shutdown
        thread = self._thread
        if loop is not None and shutdown is not None and loop.is_running():
            loop.call_soon_threadsafe(shutdown.set)
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._loop = None
        self._shutdown = None
        self._sessions.clear()
        self._server_stacks.clear()
        self._tool_index.clear()
        self._tools.clear()
        self._pending_auth.clear()
        self._link_sessions.clear()
        self._link_tasks.clear()
        self._ready.clear()
        self._start_error = None

    def call_tool(self, qualified_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Synchronously invoke an MCP tool by its qualified registry name."""
        if self._loop is None:
            raise RuntimeError("MCP client is not running")
        mapping = self._tool_index.get(qualified_name)
        if mapping is None:
            raise KeyError(f"Unknown MCP tool: {qualified_name}")
        server_name, tool_name = mapping
        session = self._sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP server '{server_name}' is not connected")

        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(tool_name, arguments or {}),
            self._loop,
        )
        result: CallToolResult = future.result(timeout=120)
        return _format_call_result(result)

    def begin_device_link(self, server: str, phone: str) -> dict[str, Any]:
        """Start device OAuth and return verification URL details (sync)."""
        if self._loop is None:
            raise RuntimeError("MCP client is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._begin_device_link(server, phone),
            self._loop,
        )
        return future.result(timeout=60)

    def unlink(self, server: str, phone: str) -> dict[str, Any]:
        """Revoke tokens, disconnect server, drop tools (sync)."""
        if self._loop is None:
            raise RuntimeError("MCP client is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._unlink(server, phone),
            self._loop,
        )
        return future.result(timeout=60)

    def link_status(self, server: str, phone: str) -> dict[str, Any]:
        """Return link / connection status for one oauth server (sync)."""
        config = self._servers.get(server)
        if config is None:
            return {"status": "unknown_server", "server": server}
        if not config.enabled:
            return {"status": "disabled", "server": server}
        if config.auth != "oauth":
            return {
                "status": "connected" if server in self._sessions else "configured",
                "server": server,
                "auth": "none",
            }
        tokens = self._token_store.load(server, phone)
        if server in self._sessions:
            return {
                "status": "linked",
                "server": server,
                "phone": phone,
                "connected": True,
                "tool_count": sum(
                    1 for _, (s, _) in self._tool_index.items() if s == server
                ),
            }
        if server in self._link_sessions or server in self._link_tasks:
            return {
                "status": "pending",
                "server": server,
                "phone": phone,
                "detail": "Waiting for browser authorization",
            }
        if tokens and tokens.access_token:
            return {
                "status": "token_present",
                "server": server,
                "phone": phone,
                "connected": False,
                "detail": "Tokens on disk but MCP not connected",
            }
        return {
            "status": "unlinked",
            "server": server,
            "phone": phone,
        }

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._run())
        except BaseException as exc:  # noqa: BLE001 — surface startup failures
            self._start_error = exc
            logger.exception("MCP client loop crashed")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            loop.close()
            if self._loop is loop:
                self._loop = None
            self._ready.set()

    async def _run(self) -> None:
        self._shutdown = asyncio.Event()
        try:
            for name, config in self._servers.items():
                if not config.enabled:
                    logger.info(
                        "MCP server '%s' is disabled — not connecting", name
                    )
                    continue
                try:
                    await self._maybe_connect_on_startup(name, config)
                except Exception:  # noqa: BLE001 — keep other servers alive
                    logger.exception("Failed to connect MCP server '%s'", name)
                    if config.auth == "oauth":
                        self._pending_auth.add(name)

            await self._rebuild_all_tools()
            self._ready.set()
            assert self._shutdown is not None
            await self._shutdown.wait()
        finally:
            for name in list(self._link_tasks):
                task = self._link_tasks.pop(name, None)
                if task is not None:
                    task.cancel()
            for name in list(self._server_stacks):
                await self._disconnect_server(name, notify=False)

    async def _maybe_connect_on_startup(
        self, name: str, config: McpServerConfig
    ) -> None:
        if config.auth == "oauth":
            phone = self._oauth_owner_phone
            tokens = self._token_store.find_for_server(name, preferred_phone=phone)
            if tokens is None or not tokens.access_token:
                self._pending_auth.add(name)
                logger.info(
                    "MCP server '%s' waiting for OAuth link (pending_auth)", name
                )
                return
            phone = tokens.phone or phone
            if not phone:
                self._pending_auth.add(name)
                logger.info(
                    "MCP server '%s' has tokens but no owner phone; pending_auth",
                    name,
                )
                return
            try:
                await self._oauth.ensure_fresh_tokens(
                    server=name,
                    phone=phone,
                    resource_url=config.url or "",
                )
            except OAuthDeviceError:
                logger.exception(
                    "OAuth refresh failed for '%s' — pending re-link", name
                )
                self._pending_auth.add(name)
                return
            await self._connect_named(name, phone=phone)
            return

        await self._connect_named(name)

    async def _connect_named(self, name: str, *, phone: str = "") -> None:
        config = self._servers[name]
        if not config.enabled:
            logger.info("MCP server '%s' is disabled — not connecting", name)
            return
        if name in self._sessions:
            return

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            session = await self._connect_server(stack, config, phone=phone)
            self._server_stacks[name] = stack
            self._sessions[name] = session
            self._pending_auth.discard(name)
            logger.info("Connected to MCP server '%s'", name)
        except Exception:
            await stack.__aexit__(None, None, None)
            raise

    async def _disconnect_server(self, name: str, *, notify: bool = True) -> list[str]:
        removed_tools = [
            q for q, (s, _) in list(self._tool_index.items()) if s == name
        ]
        self._sessions.pop(name, None)
        stack = self._server_stacks.pop(name, None)
        if stack is not None:
            try:
                await stack.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.exception("Error closing MCP server '%s'", name)

        if removed_tools:
            self._tool_index = {
                q: v for q, v in self._tool_index.items() if v[0] != name
            }
            self._tools = [t for t in self._tools if t.name not in set(removed_tools)]

        if notify and removed_tools and self._on_tools_changed is not None:
            self._on_tools_changed([], removed_tools)
        return removed_tools

    async def _connect_server(
        self,
        stack: AsyncExitStack,
        config: McpServerConfig,
        *,
        phone: str = "",
    ) -> ClientSession:
        auth: httpx.Auth | None = None
        if config.auth == "oauth":
            assert config.url is not None
            owner = phone or self._oauth_owner_phone
            if not owner:
                tokens = self._token_store.find_for_server(config.name)
                owner = tokens.phone if tokens else ""
            if not owner:
                raise OAuthDeviceError(
                    f"No phone for OAuth MCP server '{config.name}'"
                )
            auth = BearerTokenAuth(
                oauth=self._oauth,
                server=config.name,
                phone=owner,
                resource_url=config.url,
            )

        if config.transport == "stdio":
            assert config.command is not None
            env = {**os.environ, **config.env} if config.env else None
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=env,
                cwd=config.cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        elif config.transport == "sse":
            assert config.url is not None
            read, write = await stack.enter_async_context(
                sse_client(config.url, headers=config.headers or None, auth=auth)
            )
        else:
            assert config.url is not None
            read, write, _get_session_id = await stack.enter_async_context(
                streamablehttp_client(
                    config.url,
                    headers=config.headers or None,
                    auth=auth,
                )
            )

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _begin_device_link(self, server: str, phone: str) -> dict[str, Any]:
        config = self._servers.get(server)
        if config is None:
            raise KeyError(f"Unknown MCP server: {server}")
        if not config.enabled:
            raise RuntimeError(f"Server '{server}' is disabled")
        if config.auth != "oauth" or not config.url:
            raise RuntimeError(f"Server '{server}' is not configured for OAuth")

        # Cancel any in-flight link for this server.
        existing = self._link_tasks.pop(server, None)
        if existing is not None:
            existing.cancel()

        link = await self._oauth.begin_device_authorization(
            server=server,
            phone=phone,
            resource_url=config.url,
            scopes=config.scopes or ["read"],
        )
        self._link_sessions[server] = link
        self._pending_auth.add(server)

        task = asyncio.create_task(
            self._poll_and_connect(link),
            name=f"mcp-oauth-poll-{server}",
        )
        self._link_tasks[server] = task

        return {
            "status": "pending",
            "server": server,
            "phone": phone,
            "verification_uri": link.device.verification_uri,
            "verification_uri_complete": link.device.verification_uri_complete,
            "user_code": link.device.user_code,
            "expires_in": link.device.expires_in,
            "interval": link.device.interval,
        }

    async def _poll_and_connect(self, link: DeviceLinkSession) -> None:
        server = link.server
        try:
            await self._oauth.poll_until_tokens(link)
            await self._disconnect_server(server, notify=True)
            await self._connect_named(server, phone=link.phone)
            added = await self._discover_tools_for(server)
            if self._on_tools_changed is not None and added:
                self._on_tools_changed(added, [])
            if self._on_linked is not None:
                self._on_linked(server, link.phone)
            logger.info("OAuth link complete for MCP server '%s'", server)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("OAuth device poll failed for '%s'", server)
            self._pending_auth.add(server)
        finally:
            self._link_sessions.pop(server, None)
            self._link_tasks.pop(server, None)

    async def _unlink(self, server: str, phone: str) -> dict[str, Any]:
        config = self._servers.get(server)
        if config is None:
            raise KeyError(f"Unknown MCP server: {server}")

        task = self._link_tasks.pop(server, None)
        if task is not None:
            task.cancel()
        self._link_sessions.pop(server, None)

        if config.auth == "oauth" and config.url:
            await self._oauth.revoke(
                server=server, phone=phone, resource_url=config.url
            )
        else:
            self._token_store.delete(server, phone)

        removed = await self._disconnect_server(server, notify=True)
        if config.auth == "oauth":
            self._pending_auth.add(server)
        return {
            "status": "unlinked",
            "server": server,
            "phone": phone,
            "removed_tools": removed,
        }

    async def _discover_tools_for(self, server_name: str) -> list[McpTool]:
        session = self._sessions.get(server_name)
        if session is None:
            return []
        try:
            listed = await session.list_tools()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to list tools for MCP server '%s'", server_name)
            return []

        drop_names = {
            q for q, (s, _) in self._tool_index.items() if s == server_name
        }
        self._tool_index = {
            q: v for q, v in self._tool_index.items() if v[0] != server_name
        }
        self._tools = [t for t in self._tools if t.name not in drop_names]

        added: list[McpTool] = []
        for tool in listed.tools:
            adapter = self._make_adapter(server_name, tool)
            self._tools.append(adapter)
            self._tool_index[adapter.name] = (server_name, tool.name)
            added.append(adapter)
            logger.debug(
                "Registered MCP tool %s → %s:%s",
                adapter.name,
                server_name,
                tool.name,
            )
        return added

    async def _rebuild_all_tools(self) -> None:
        self._tools = []
        self._tool_index = {}
        for server_name in list(self._sessions):
            await self._discover_tools_for(server_name)

    def _make_adapter(self, server_name: str, tool: McpSdkTool) -> McpTool:
        qualified = f"mcp__{sanitize_name(server_name)}__{sanitize_name(tool.name)}"
        base = qualified
        suffix = 2
        while qualified in self._tool_index:
            qualified = f"{base}_{suffix}"
            suffix += 1

        description = tool.description or f"MCP tool '{tool.name}' from '{server_name}'"
        description = f"[{server_name}] {description}"
        parameters = tool.inputSchema or {"type": "object", "properties": {}}
        return McpTool(
            name=qualified,
            description=description,
            parameters=parameters,
            client=self,
        )


def _format_call_result(result: CallToolResult) -> Any:
    if result.isError:
        texts = [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text" and hasattr(block, "text")
        ]
        message = "\n".join(texts) if texts else "MCP tool returned an error"
        raise RuntimeError(message)

    if result.structuredContent is not None:
        return result.structuredContent

    texts = [
        block.text
        for block in result.content
        if getattr(block, "type", None) == "text" and hasattr(block, "text")
    ]
    if len(texts) == 1:
        return texts[0]
    if texts:
        return "\n".join(texts)

    return [block.model_dump(mode="json") for block in result.content]
