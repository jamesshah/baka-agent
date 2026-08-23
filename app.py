"""FastAPI server for Sendblue ↔ local agent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from agents import ChatAgent, ExecutorAgent
from config import get_settings
from llm import LlamaServerAdapter
from mcp_client import MultiServerMcpClient, load_mcp_config
from mcp_client.token_store import FileTokenStore
from memory import (
    ContextBuilder,
    Database,
    HybridRetriever,
    LlamaEmbeddingClient,
    MemoryConsolidator,
    SkillIndexer,
    SqlAlchemyMemoryRepository,
)
from memory.migrate import upgrade as upgrade_database
from messaging import SendblueAdapter
from tools import (
    GetCurrentTimeTool,
    LinkSnaptradeTool,
    ManageMemoryTool,
    SendAcknowledgementTool,
    SnaptradeStatusTool,
    SpawnAgentTool,
    ToolRegistry,
    UnlinkSnaptradeTool,
)
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
_mcp: MultiServerMcpClient | None = None
_database = Database(_settings.database_path) if _settings.memory_enabled else None
_memory_repository = (
    SqlAlchemyMemoryRepository(_database) if _database is not None else None
)
_embeddings = (
    LlamaEmbeddingClient(
        base_url=_settings.embedding_base_url,
        model=_settings.embedding_model,
        dimensions=_settings.embedding_dimensions,
        timeout=_settings.embedding_timeout_seconds,
    )
    if _memory_repository is not None and _settings.embedding_enabled
    else None
)
_retriever = (
    HybridRetriever(
        _memory_repository,
        embeddings=_embeddings,
        lexical_weight=_settings.memory_lexical_weight,
        vector_weight=_settings.memory_vector_weight,
        minimum_vector_score=_settings.memory_minimum_vector_score,
    )
    if _memory_repository is not None
    else None
)
_context_builder = (
    ContextBuilder(
        _retriever,
        max_memory_chars=_settings.memory_max_chars,
        max_skill_chars=_settings.skill_max_chars,
    )
    if _retriever is not None
    else None
)
_memory_consolidator = (
    MemoryConsolidator(
        _memory_repository,
        _llm,
        embeddings=_embeddings,
        enabled=_settings.memory_consolidation_enabled,
        summary_every_turns=_settings.memory_summary_every_turns,
    )
    if _memory_repository is not None
    else None
)
_skill_indexer = (
    SkillIndexer(
        _memory_repository,
        _settings.skills_dir,
        embeddings=_embeddings,
    )
    if _memory_repository is not None
    else None
)
_executor = ExecutorAgent(
    llm=_llm,
    tools=_tools,
    max_agent_iterations=_settings.max_executor_iterations,
)

_tools.register(GetCurrentTimeTool())
_tools.register(SendAcknowledgementTool())
_tools.register(SpawnAgentTool(_executor, _context_builder))
if _memory_repository is not None:
    _tools.register(ManageMemoryTool(_memory_repository, _embeddings))

_agent = ChatAgent(
    llm=_llm,
    tools=_tools,
    max_history_messages=_settings.max_history_messages,
    max_agent_iterations=_settings.max_agent_iterations,
    memory_repository=_memory_repository,
    context_builder=_context_builder,
    memory_consolidator=_memory_consolidator,
)
_messaging = SendblueAdapter(
    api_key=_settings.sendblue_api_key,
    api_secret=_settings.sendblue_api_secret,
    from_number=_settings.sendblue_from_number,
)
_webhook = SendblueWebhookHandler(agent=_agent, messaging=_messaging)


def _on_mcp_tools_changed(added: list[Any], removed: list[str]) -> None:
    for name in removed:
        _tools.unregister(name)
    for tool in added:
        _tools.register(tool)
    if added or removed:
        logger.info(
            "MCP tools updated: +%d -%d (registry=%d)",
            len(added),
            len(removed),
            _tools.count(),
        )


def _on_mcp_linked(server: str, phone: str) -> None:
    message = (
        f"{server.title()} is linked. You can ask about balances, positions, "
        "and orders."
    )
    try:
        _messaging.send_message(phone, message)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to send %s linked confirmation to %s", server, phone)


def _register_oauth_local_tools(client: MultiServerMcpClient) -> None:
    if client.has_server("snaptrade"):
        _tools.register(LinkSnaptradeTool(client))
        _tools.register(SnaptradeStatusTool(client))
        _tools.register(UnlinkSnaptradeTool(client))
        logger.info("Registered SnapTrade link/status/unlink tools")


def _start_mcp() -> MultiServerMcpClient | None:
    if not _settings.mcp_enabled:
        logger.info("MCP disabled via MCP_ENABLED=false")
        return None

    servers = load_mcp_config(_settings.mcp_config_path)
    if not servers:
        return None

    client = MultiServerMcpClient(
        servers,
        token_store=FileTokenStore(_settings.mcp_oauth_data_dir),
        oauth_owner_phone=_settings.resolved_oauth_owner_number,
        on_tools_changed=_on_mcp_tools_changed,
        on_linked=_on_mcp_linked,
    )
    client.start()
    for tool in client.tools:
        _tools.register(tool)
    _register_oauth_local_tools(client)
    logger.info("Registered %d MCP tool(s)", len(client.tools))
    return client


def _mcp_health() -> dict[str, Any]:
    if not _settings.mcp_enabled:
        return {
            "status": "disabled",
            "configured": [],
            "disabled": [],
            "connected": [],
            "pending_auth": [],
            "tool_count": 0,
            "detail": "MCP_ENABLED=false",
        }
    if _mcp is None:
        return {
            "status": "disabled",
            "configured": [],
            "disabled": [],
            "connected": [],
            "pending_auth": [],
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
    if _settings.memory_enabled:
        upgrade_database(_settings.database_path)
        indexed = _skill_indexer.sync() if _skill_indexer is not None else 0
        logger.info("Memory database ready; indexed %d skill(s)", indexed)
    try:
        _mcp = _start_mcp()
    except Exception:  # noqa: BLE001 — app should still serve without MCP
        logger.exception("MCP startup failed; continuing without MCP tools")
        _mcp = None

    try:
        result = _messaging.ensure_receive_webhook(
            _settings.sendblue_webhook_url,
            global_secret=_settings.sendblue_global_webhook_secret,
        )
        status = result.get("status")
        detail = result.get("detail")
        if status == "ok":
            logger.info("Sendblue webhook: %s", detail)
        elif status == "missing":
            logger.warning("Sendblue webhook: %s", detail)
        elif status == "skipped":
            logger.info("Sendblue webhook: %s", detail)
        else:
            logger.error("Sendblue webhook: %s", detail)
    except Exception:  # noqa: BLE001
        logger.exception("Sendblue webhook check/register failed")

    yield
    if _mcp is not None:
        _mcp.stop()
        _mcp = None
    if _memory_consolidator is not None:
        _memory_consolidator.close()
    if _database is not None:
        _database.close()


app = FastAPI(title="baka-agent", version="0.1.0", lifespan=lifespan)


def _agents_health() -> dict[str, Any]:
    chat_names = _tools.names(chat_agent=True)
    executor_names = _tools.names(chat_agent=False)
    executor_local = _tools.local_names(chat_agent=False)
    chat_ok = bool(chat_names)
    executor_ok = bool(executor_names)
    if chat_ok and executor_ok:
        status = "ok"
    else:
        status = "error"
    return {
        "status": status,
        "chat": {
            "class": "ChatAgent",
            "role": "user-facing",
            "status": "ok" if chat_ok else "error",
            "tool_count": len(chat_names),
            "tools": chat_names,
        },
        "executor": {
            "class": "ExecutorAgent",
            "role": "worker",
            "status": "ok" if executor_ok else "error",
            "tool_count": len(executor_names),
            "local_tool_count": len(executor_local),
            "granted": "per spawn",
            "tools": executor_names,
        },
    }


def _memory_health() -> dict[str, Any]:
    if _database is None:
        return {"status": "disabled"}
    try:
        with _database.engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    result: dict[str, Any] = {
        "status": "ok",
        "database": _settings.database_path,
        "semantic_retrieval": "hybrid" if _embeddings is not None else "fts5",
    }
    if _embeddings is not None:
        result["embeddings"] = _embeddings.health_check()
        if result["embeddings"].get("status") != "ok":
            result["status"] = "degraded"
    return result


@app.get("/health")
def health() -> JSONResponse:
    agents_check = _agents_health()
    memory_check = _memory_health()
    mcp_check = _mcp_health()
    model_check = _llm.health_check()
    webhook_check = _messaging.webhook_health_check(
        secret_configured=bool(_settings.sendblue_global_webhook_secret),
        expected_url=_settings.sendblue_webhook_url,
    )

    checks = {
        "agents": agents_check,
        "memory": memory_check,
        "mcp": mcp_check,
        "model": model_check,
        "sendblue_webhook": webhook_check,
    }
    overall = _overall_status(checks)
    body = {"status": overall, "checks": checks}
    return JSONResponse(body, status_code=200 if overall != "error" else 503)


@app.post("/webhooks/register")
async def register_webhook(request: Request) -> JSONResponse:
    """
    Register (or confirm) the Sendblue receive webhook.

    Body is optional JSON: ``{"url": "https://.../webhooks/receive"}``.
    Falls back to ``SENDBLUE_WEBHOOK_URL`` when ``url`` is omitted.
    """
    url = _settings.sendblue_webhook_url
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        provided = (body.get("url") or "").strip()
        if provided:
            url = provided

    if not url:
        return JSONResponse(
            {
                "error": "url_required",
                "detail": (
                    "Provide {\"url\": \"https://.../webhooks/receive\"} or set "
                    "SENDBLUE_WEBHOOK_URL"
                ),
            },
            status_code=400,
        )

    result = _messaging.ensure_receive_webhook(
        url,
        global_secret=_settings.sendblue_global_webhook_secret,
    )
    status = result.get("status")
    http_status = 200 if status == "ok" else 502 if status == "error" else 400
    return JSONResponse(result, status_code=http_status)


@app.post("/webhooks/receive")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    return await _webhook.handle_receive(request, background_tasks)
