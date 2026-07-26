"""Background multi-server MCP connection hub."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, Tool as McpSdkTool

from mcp_client.config import McpServerConfig
from mcp_client.tool import McpTool

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def sanitize_name(value: str) -> str:
    """Make a name safe for OpenAI-style function tool identifiers."""
    cleaned = _SAFE_NAME_RE.sub("_", value.strip()).strip("_")
    return cleaned or "mcp"


class MultiServerMcpClient:
    """
    Connect to many MCP servers at once and expose their tools.

    Owns a dedicated asyncio event loop on a background thread so the rest of
    the (sync) agent can call tools safely via ``call_tool``.
    """

    def __init__(self, servers: dict[str, McpServerConfig]) -> None:
        self._servers = servers
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._shutdown: asyncio.Event | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tool_index: dict[str, tuple[str, str]] = {}
        self._tools: list[McpTool] = []

    @property
    def tools(self) -> list[McpTool]:
        return list(self._tools)

    def status(self) -> dict[str, Any]:
        """Snapshot of configured vs connected MCP servers and tools."""
        configured = sorted(self._servers)
        connected = sorted(self._sessions)
        running = self._loop is not None and self._loop.is_running()
        if not configured:
            overall = "disabled"
        elif not running:
            overall = "error"
        elif set(connected) == set(configured):
            overall = "ok"
        elif connected:
            overall = "degraded"
        else:
            overall = "error"
        return {
            "status": overall,
            "running": running,
            "configured": configured,
            "connected": connected,
            "tool_count": len(self._tools),
        }

    def start(self, timeout: float = 60.0) -> None:
        """Spawn the background loop, connect servers, and discover tools."""
        if not self._servers:
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
        self._tool_index.clear()
        self._tools.clear()
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
        async with AsyncExitStack() as stack:
            for name, config in self._servers.items():
                try:
                    session = await self._connect_server(stack, config)
                    self._sessions[name] = session
                    logger.info("Connected to MCP server '%s'", name)
                except Exception:  # noqa: BLE001 — keep other servers alive
                    logger.exception("Failed to connect MCP server '%s'", name)

            await self._discover_tools()
            self._ready.set()
            assert self._shutdown is not None
            await self._shutdown.wait()

    async def _connect_server(
        self,
        stack: AsyncExitStack,
        config: McpServerConfig,
    ) -> ClientSession:
        if config.transport == "stdio":
            assert config.command is not None
            # Merge process env so tools like npx still work; overlay config env.
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
                sse_client(config.url, headers=config.headers or None)
            )
        else:
            assert config.url is not None
            read, write, _get_session_id = await stack.enter_async_context(
                streamablehttp_client(
                    config.url, headers=config.headers or None)
            )

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _discover_tools(self) -> None:
        collected: list[tuple[str, McpSdkTool]] = []
        for server_name, session in self._sessions.items():
            try:
                listed = await session.list_tools()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to list tools for MCP server '%s'", server_name)
                continue
            for tool in listed.tools:
                collected.append((server_name, tool))

        tools: list[McpTool] = []
        index: dict[str, tuple[str, str]] = {}
        for server_name, tool in collected:
            # Always: mcp__<server>__<tool>
            qualified = (
                f"mcp__{sanitize_name(server_name)}__{sanitize_name(tool.name)}"
            )

            # Avoid collisions after sanitization.
            base = qualified
            suffix = 2
            while qualified in index:
                qualified = f"{base}_{suffix}"
                suffix += 1

            description = tool.description or f"MCP tool '{tool.name}' from '{server_name}'"
            description = f"[{server_name}] {description}"

            parameters = tool.inputSchema or {
                "type": "object",
                "properties": {},
            }
            adapter = McpTool(
                name=qualified,
                description=description,
                parameters=parameters,
                client=self,
            )
            tools.append(adapter)
            index[qualified] = (server_name, tool.name)
            logger.debug(
                "Registered MCP tool %s → %s:%s",
                qualified,
                server_name,
                tool.name,
            )

        self._tools = tools
        self._tool_index = index


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

    # Fall back to raw model dump for non-text content (images, etc.).
    return [block.model_dump(mode="json") for block in result.content]
