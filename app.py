"""FastAPI webhook server for Sendblue ↔ local agent."""

from __future__ import annotations

import hmac
import logging
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

import agent
import sendblue_client
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="baka-agent", version="0.1.0")

# Header Sendblue sends when a webhook / global secret is configured.
_SENDBLUE_SIGNING_HEADER = "sb-signing-secret"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _verify_webhook_secret(request: Request) -> JSONResponse | None:
    """
    Verify the Sendblue ``sb-signing-secret`` header against
    ``SENDBLUE_GLOBAL_WEBHOOK_SECRET``.

    Returns a 401 JSONResponse if verification fails, otherwise None.
    """
    settings = get_settings()
    expected = settings.sendblue_global_webhook_secret
    if not expected:
        logger.error(
            "SENDBLUE_GLOBAL_WEBHOOK_SECRET is not set — rejecting webhook"
        )
        return JSONResponse(
            {"error": "unauthorized"},
            status_code=401,
        )

    provided = request.headers.get(_SENDBLUE_SIGNING_HEADER) or ""
    if not hmac.compare_digest(provided, expected):
        logger.warning(
            "webhook rejected: invalid or missing %s from %s",
            _SENDBLUE_SIGNING_HEADER,
            request.client.host if request.client else "unknown",
        )
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def _typing(number: str, state: str) -> None:
    """Send a typing indicator and log Sendblue's result (SENT vs ERROR)."""
    try:
        resp = sendblue_client.send_typing_indicator(number, state)  # type: ignore[arg-type]
        payload = resp.model_dump() if hasattr(resp, "model_dump") else resp
        logger.info("typing %s -> %s: %s", state, number, payload)
        if getattr(resp, "status", None) == "ERROR":
            logger.warning(
                "typing indicator error for %s: %s",
                number,
                getattr(resp, "error_message", None),
            )
    except Exception:
        logger.exception("typing indicator %s failed for %s", state, number)


def _process_inbound(from_number: str, content: str) -> None:
    """Run the agent and reply via Sendblue (background)."""
    try:
        # Start typing while the local model thinks (iMessage only).
        _typing(from_number, "start")

        reply = agent.run_turn(from_number, content)
        logger.info("replying to %s: %s", from_number, reply[:200])

        _typing(from_number, "stop")
        sendblue_client.send_message(from_number, reply)
    except Exception:
        logger.exception("failed to process message from %s", from_number)
        _typing(from_number, "stop")
        try:
            sendblue_client.send_message(
                from_number,
                "Sorry, something went wrong processing your message.",
            )
        except Exception:
            logger.exception(
                "also failed to send error reply to %s", from_number)


@app.post("/webhooks/receive")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Sendblue inbound-message webhook.

    Always returns 200 quickly so Sendblue does not retry. Agent work
    (and the outbound reply) runs in a background task.
    """
    auth_error = _verify_webhook_secret(request)
    if auth_error is not None:
        return auth_error

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        logger.warning("webhook with non-JSON body")
        return JSONResponse({"received": True, "ignored": "invalid_json"})

    if body.get("is_outbound") is True:
        return JSONResponse({"received": True, "ignored": "outbound"})

    content = (body.get("content") or "").strip()
    from_number = (body.get("from_number") or body.get("number") or "").strip()

    if not from_number:
        logger.warning("webhook missing from_number: %s", body)
        return JSONResponse({"received": True, "ignored": "no_from_number"})

    if not content:
        logger.info("empty content from %s — ignoring", from_number)
        return JSONResponse({"received": True, "ignored": "empty_content"})

    settings = get_settings()
    allowed = settings.allowed_numbers_set
    if allowed and from_number not in allowed:
        logger.info("number %s not in allowlist — ignoring", from_number)
        return JSONResponse({"received": True, "ignored": "not_allowed"})

    service = (body.get("service") or "").strip()
    logger.info(
        "inbound from %s service=%s: %s",
        from_number,
        service or "unknown",
        content[:200],
    )
    if service and service.lower() != "imessage":
        logger.warning(
            "typing indicators only work on iMessage; this chat is %s",
            service,
        )

    background_tasks.add_task(_process_inbound, from_number, content)
    return JSONResponse({"received": True})
