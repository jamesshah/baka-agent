"""FastAPI server for Sendblue ↔ local agent."""

from __future__ import annotations

import logging

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from agents import ChatAgent
from config import get_settings
from llm import LlamaServerAdapter
from messaging import SendblueAdapter
from tools import GetCurrentTimeTool, ToolRegistry
from webhooks import SendblueWebhookHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="baka-agent", version="0.1.0")

# Wire adapters once at startup.
_settings = get_settings()
_llm = LlamaServerAdapter(
    base_url=_settings.llama_base_url,
    model=_settings.llama_model,
)
_tools = ToolRegistry()
_tools.register(GetCurrentTimeTool())
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhooks/receive")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    return await _webhook.handle_receive(request, background_tasks)
