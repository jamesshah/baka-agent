"""FastAPI server for Sendblue ↔ local agent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from agents import ChatAgent
from config import get_settings
from llm import LlamaServerAdapter
from mcp_client import MultiServerMcpClient, load_mcp_config
from messaging import SendblueAdapter
from tools import GetCurrentTimeTool, ToolRegistry
from webhooks import SendblueWebhookHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_settings = get_settings()
_llm = LlamaServerAdapter(
    base_url=_settings.llama_base_url,
    model=_settings.llama_model,
)
_tools = ToolRegistry()
_tools.register(GetCurrentTimeTool())
_mcp: MultiServerMcpClient | None = None
_agent = ChatAgent(
    llm=_llm,
    tools=_tools,
    system_prompt=_settings.system_prompt,
    max_history_messages=_settings.max_history_messages,
    max_agent_iterations=_settings.max_agent_iterations,
)
_messaging = SendblueAdapter(
    api_key=_settings.sendblue_api_key,
    api_secret=_settings.sendblue_api_secret,
    from_number=_settings.sendblue_from_number,
)
_webhook = SendblueWebhookHandler(agent=_agent, messaging=_messaging)


def _start_mcp() -> MultiServerMcpClient | None:
    if not _settings.mcp_enabled:
        logger.info("MCP disabled via MCP_ENABLED=false")
        return None

    servers = load_mcp_config(_settings.mcp_config_path)
    if not servers:
        return None

    client = MultiServerMcpClient(servers)
    client.start()
    for tool in client.tools:
        _tools.register(tool)
    logger.info("Registered %d MCP tool(s)", len(client.tools))
    return client


def _mcp_health() -> dict[str, Any]:
    if not _settings.mcp_enabled:
        return {
            "status": "disabled",
            "configured": [],
            "connected": [],
            "tool_count": 0,
            "detail": "MCP_ENABLED=false",
        }
    if _mcp is None:
        return {
            "status": "disabled",
            "configured": [],
            "connected": [],
            "tool_count": 0,
            "detail": "No MCP servers configured or startup failed",
        }
    return _mcp.status()


def _overall_status(checks: dict[str, dict[str, Any]]) -> str:
    statuses = [check.get("status", "error") for check in checks.values()]
    if any(s == "error" for s in statuses):
        return "error"
    if any(s in {"degraded", "unconfigured"} for s in statuses):
        return "degraded"
    return "ok"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _mcp
    try:
        _mcp = _start_mcp()
    except Exception:  # noqa: BLE001 — app should still serve without MCP
        logger.exception("MCP startup failed; continuing without MCP tools")
        _mcp = None
    yield
    if _mcp is not None:
        _mcp.stop()
        _mcp = None


app = FastAPI(title="baka-agent", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    local_names = _tools.local_names()
    tools_check = {
        "status": "ok" if local_names else "error",
        "count": len(local_names),
        "names": local_names,
    }
    mcp_check = _mcp_health()
    model_check = _llm.health_check()
    webhook_check = _messaging.webhook_health_check(
        secret_configured=bool(_settings.sendblue_global_webhook_secret),
    )

    checks = {
        "tools": tools_check,
        "mcp": mcp_check,
        "model": model_check,
        "sendblue_webhook": webhook_check,
    }
    overall = _overall_status(checks)
    body = {"status": overall, "checks": checks}
    return JSONResponse(body, status_code=200 if overall != "error" else 503)


@app.post("/webhooks/receive")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    return await _webhook.handle_receive(request, background_tasks)
