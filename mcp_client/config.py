"""Load Cursor-compatible mcp.json configuration."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

TransportKind = Literal["stdio", "sse", "streamable_http"]
AuthKind = Literal["none", "oauth"]

_ENV_VAR_RE = re.compile(r"\$\{env:([^}]+)\}")
_VAR_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class McpServerConfig:
    """One MCP server entry from mcp.json."""

    name: str
    transport: TransportKind
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    auth: AuthKind = "none"
    scopes: list[str] = field(default_factory=list)


def _resolve_string(value: str, *, workspace: Path) -> str:
    """Resolve Cursor-style ${env:NAME} / ${workspaceFolder} placeholders."""

    def env_sub(match: re.Match[str]) -> str:
        key = match.group(1)
        return os.environ.get(key, "")

    value = _ENV_VAR_RE.sub(env_sub, value)

    replacements = {
        "workspaceFolder": str(workspace),
        "userHome": str(Path.home()),
    }

    def var_sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key.startswith("env:"):
            return os.environ.get(key[4:], "")
        return replacements.get(key, match.group(0))

    return _VAR_RE.sub(var_sub, value)


def _resolve_value(value: Any, *, workspace: Path) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, workspace=workspace)
    if isinstance(value, list):
        return [_resolve_value(v, workspace=workspace) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_value(v, workspace=workspace) for k, v in value.items()}
    return value


def _detect_transport(raw: dict[str, Any]) -> TransportKind:
    explicit = (raw.get("type") or raw.get("transport") or "").strip().lower()
    if explicit in {"stdio", "sse"}:
        return explicit  # type: ignore[return-value]
    if explicit in {"http", "streamablehttp", "streamable_http", "streamable-http"}:
        return "streamable_http"
    if raw.get("url"):
        # Cursor defaults remote URLs to Streamable HTTP; SSE is opt-in via type.
        return "streamable_http"
    if raw.get("command"):
        return "stdio"
    raise ValueError("MCP server config needs either 'command' (stdio) or 'url' (remote)")


def _detect_auth(raw: dict[str, Any]) -> AuthKind:
    explicit = (raw.get("auth") or "").strip().lower()
    if explicit == "oauth":
        return "oauth"
    if explicit in {"none", ""}:
        return "none"
    raise ValueError(f"Unknown auth mode '{explicit}' (expected 'none' or 'oauth')")


def _parse_server(name: str, raw: dict[str, Any], *, workspace: Path) -> McpServerConfig:
    data = _resolve_value(raw, workspace=workspace)
    if not isinstance(data, dict):
        raise ValueError(f"Server '{name}' config must be an object")

    transport = _detect_transport(data)
    auth = _detect_auth(data)
    command = data.get("command")
    url = data.get("url")
    args = data.get("args") or []
    env = data.get("env") or {}
    headers = data.get("headers") or {}
    cwd = data.get("cwd")
    scopes_raw = data.get("scopes") or []

    if not isinstance(args, list):
        raise ValueError(f"Server '{name}': 'args' must be a list")
    if not isinstance(env, dict):
        raise ValueError(f"Server '{name}': 'env' must be an object")
    if not isinstance(headers, dict):
        raise ValueError(f"Server '{name}': 'headers' must be an object")
    if not isinstance(scopes_raw, list):
        raise ValueError(f"Server '{name}': 'scopes' must be a list")

    if transport == "stdio":
        if not command or not isinstance(command, str):
            raise ValueError(f"Server '{name}': stdio transport requires 'command'")
        if auth == "oauth":
            raise ValueError(f"Server '{name}': oauth auth requires a remote 'url' transport")
    else:
        if not url or not isinstance(url, str):
            raise ValueError(f"Server '{name}': remote transport requires 'url'")

    scopes = [str(s) for s in scopes_raw]
    if auth == "oauth" and not scopes:
        scopes = ["read"]

    return McpServerConfig(
        name=name,
        transport=transport,
        command=command if isinstance(command, str) else None,
        args=[str(a) for a in args],
        env={str(k): str(v) for k, v in env.items()},
        cwd=str(cwd) if cwd else None,
        url=url if isinstance(url, str) else None,
        headers={str(k): str(v) for k, v in headers.items()},
        auth=auth,
        scopes=scopes,
    )


def load_mcp_config(
    path: str | Path | None = None,
    *,
    workspace: str | Path | None = None,
) -> dict[str, McpServerConfig]:
    """
    Load a Cursor-compatible mcp.json file.

    Expected shape:
      {
        "mcpServers": {
          "server-name": { "command": "...", "args": [...], "env": {...} },
          "remote": { "url": "https://...", "headers": {...} },
          "snaptrade": { "url": "https://mcp.snaptrade.com/mcp", "auth": "oauth" }
        }
      }
    """
    workspace_path = Path(workspace or Path.cwd()).resolve()
    config_path = Path(path) if path else workspace_path / "mcp.json"

    if not config_path.is_file():
        logger.info("No MCP config at %s — skipping MCP tools", config_path)
        return {}

    with config_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"MCP config root must be an object: {config_path}")

    servers_raw = payload.get("mcpServers")
    if servers_raw is None:
        raise ValueError(f"MCP config missing 'mcpServers': {config_path}")
    if not isinstance(servers_raw, dict):
        raise ValueError(f"'mcpServers' must be an object: {config_path}")

    servers: dict[str, McpServerConfig] = {}
    for name, raw in servers_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Server '{name}' config must be an object")
        servers[name] = _parse_server(str(name), raw, workspace=workspace_path)

    logger.info("Loaded %d MCP server(s) from %s", len(servers), config_path)
    return servers
