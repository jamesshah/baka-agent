"""Sendblue inbound-message webhook handler."""

from __future__ import annotations

import hmac
import logging
from typing import Any, Literal

from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse

from agents.base import Agent
from config import get_settings
from messaging.base import MessagingClient
from messaging.format import chunk_text, strip_markdown

logger = logging.getLogger(__name__)

# Header Sendblue sends when a webhook / global secret is configured.
_SENDBLUE_SIGNING_HEADER = "sb-signing-secret"


class SendblueWebhookHandler:
    """Verify, filter, and process Sendblue receive webhooks."""

    def __init__(self, agent: Agent, messaging: MessagingClient) -> None:
        self._agent = agent
        self._messaging = messaging

    def verify_secret(self, request: Request) -> JSONResponse | None:
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

    def _typing(
        self, number: str, state: Literal["start", "stop"]
    ) -> None:
        """Send a typing indicator and log Sendblue's result (SENT vs ERROR)."""
        try:
            resp = self._messaging.send_typing_indicator(number, state)
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

    def process_inbound(self, from_number: str, content: str) -> None:
        """Run the agent and reply via messaging (background)."""
        try:
            # Start typing while the local model thinks (iMessage only).
            self._typing(from_number, "start")

            reply = self._agent.run_turn(from_number, content)
            plain = strip_markdown(reply)
            chunks = chunk_text(plain)
            if not chunks:
                logger.warning("empty reply after format for %s — not sending", from_number)
                self._typing(from_number, "stop")
                return

            logger.info(
                "replying to %s (%d chunk(s)): %s",
                from_number,
                len(chunks),
                chunks[0][:200],
            )

            self._typing(from_number, "stop")
            for chunk in chunks:
                self._messaging.send_message(from_number, chunk)
        except Exception:
            logger.exception("failed to process message from %s", from_number)
            self._typing(from_number, "stop")
            try:
                self._messaging.send_message(
                    from_number,
                    "Sorry, something went wrong processing your message.",
                )
            except Exception:
                logger.exception(
                    "also failed to send error reply to %s", from_number
                )

    async def handle_receive(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> JSONResponse:
        """
        Sendblue inbound-message webhook.

        Always returns 200 quickly so Sendblue does not retry. Agent work
        (and the outbound reply) runs in a background task.
        """
        auth_error = self.verify_secret(request)
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
        from_number = (
            body.get("from_number") or body.get("number") or ""
        ).strip()

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

        background_tasks.add_task(self.process_inbound, from_number, content)
        return JSONResponse({"received": True})
