"""Sendblue inbound-message webhook handler."""

from __future__ import annotations

import hmac
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Literal

from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse

from agents.base import Agent
from config import get_settings
from messaging.base import MessagingClient
from messaging.format import chunk_text, strip_markdown
from messaging.media import (
    MULTIMODAL_REFUSAL,
    MediaDownloadError,
    UnsupportedAttachment,
    download_media,
)

logger = logging.getLogger(__name__)

# Header Sendblue sends when a webhook / global secret is configured.
_SENDBLUE_SIGNING_HEADER = "sb-signing-secret"
_GENERIC_ERROR = "Sorry, something went wrong processing your message."
_TYPING_PULSE_INTERVAL_S = 5.0
_SEEN_HANDLE_MAX = 10_000


class _TypingPulse:
    """Resend iMessage typing ``start`` on an interval until ``stop``."""

    def __init__(
        self,
        emit: Callable[[Literal["start", "stop"]], None],
        *,
        interval: float = _TYPING_PULSE_INTERVAL_S,
    ) -> None:
        self._emit = emit
        self._interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._active = True
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name="sendblue-typing-pulse",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        thread: threading.Thread | None
        with self._lock:
            self._active = False
            thread = self._thread
            self._thread = None
            self._stop.set()
        if thread is not None:
            thread.join(timeout=self._interval + 1.0)
        with self._lock:
            self._emit("stop")

    def __enter__(self) -> _TypingPulse:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _emit_start(self) -> None:
        with self._lock:
            if not self._active:
                return
        self._emit("start")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._emit_start()
            if self._stop.wait(self._interval):
                break


class SendblueWebhookHandler:
    """Verify, filter, and process Sendblue receive webhooks."""

    def __init__(self, agent: Agent, messaging: MessagingClient) -> None:
        self._agent = agent
        self._messaging = messaging
        self._seen_handles: OrderedDict[str, None] = OrderedDict()
        self._seen_lock = threading.Lock()

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

    def _is_duplicate_handle(self, message_handle: str) -> bool:
        """Claim ``message_handle``; return True if it was already processed."""
        if not message_handle:
            return False
        with self._seen_lock:
            if message_handle in self._seen_handles:
                return True
            self._seen_handles[message_handle] = None
            while len(self._seen_handles) > _SEEN_HANDLE_MAX:
                self._seen_handles.popitem(last=False)
            return False

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
            logger.exception(
                "typing indicator %s failed for %s", state, number)

    def process_inbound(
        self,
        from_number: str,
        content: str,
        media_url: str = "",
    ) -> None:
        """Run the agent and reply via messaging (background)."""
        pulse = _TypingPulse(lambda state: self._typing(from_number, state))
        try:
            with pulse:
                for reply in self._iter_inbound(from_number, content, media_url):
                    plain = strip_markdown(reply)
                    chunks = chunk_text(plain)
                    if not chunks:
                        logger.warning(
                            "empty reply after format for %s — not sending",
                            from_number,
                        )
                        continue

                    logger.info(
                        "replying to %s (%d chunk(s)): %s",
                        from_number,
                        len(chunks),
                        chunks[0][:200],
                    )

                    pulse.stop()
                    for chunk in chunks:
                        self._messaging.send_message(from_number, chunk)
                    pulse.start()
        except Exception:
            logger.exception("failed to process message from %s", from_number)
            try:
                self._messaging.send_message(from_number, _GENERIC_ERROR)
            except Exception:
                logger.exception(
                    "also failed to send error reply to %s", from_number
                )

    def _iter_inbound(
        self,
        from_number: str,
        content: str,
        media_url: str,
    ) -> Iterator[str]:
        media = None
        if media_url:
            if not self._agent.has_multimodal():
                logger.info(
                    "refusing media from %s: model has no multimodal capability",
                    from_number,
                )
                yield MULTIMODAL_REFUSAL
                return
            try:
                media = download_media(media_url)
            except UnsupportedAttachment:
                logger.info(
                    "non-media attachment from %s — using caption only",
                    from_number,
                )
                if not content:
                    logger.info(
                        "ignoring unusable attachment with empty caption from %s",
                        from_number,
                    )
                    return
            except MediaDownloadError:
                logger.exception(
                    "failed to download media for %s", from_number)
                yield _GENERIC_ERROR
                return
            else:
                if not self._agent.supports_media(media.kind):
                    logger.info(
                        "refusing %s from %s: model lacks %s input",
                        media.kind,
                        from_number,
                        media.kind,
                    )
                    yield MULTIMODAL_REFUSAL
                    return

        yield from self._agent.run_turn(from_number, content, media=media)

    async def handle_receive(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> JSONResponse:
        """
        Sendblue inbound-message webhook.

        Always returns 200 quickly so Sendblue does not retry. Duplicate
        ``message_handle`` deliveries are ignored. Agent work (and the
        outbound reply) runs in a background task.
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
        media_url = (body.get("media_url") or "").strip()
        from_number = (
            body.get("from_number") or body.get("number") or ""
        ).strip()

        if not from_number:
            logger.warning("webhook missing from_number: %s", body)
            return JSONResponse({"received": True, "ignored": "no_from_number"})

        if not content and not media_url:
            logger.info("empty content from %s — ignoring", from_number)
            return JSONResponse({"received": True, "ignored": "empty_content"})

        settings = get_settings()
        allowed = settings.allowed_numbers_set
        if allowed and from_number not in allowed:
            logger.info("number %s not in allowlist — ignoring", from_number)
            return JSONResponse({"received": True, "ignored": "not_allowed"})

        message_handle = (body.get("message_handle") or "").strip()
        if not message_handle:
            logger.warning(
                "webhook missing message_handle from %s — cannot dedupe",
                from_number,
            )
        elif self._is_duplicate_handle(message_handle):
            logger.info(
                "duplicate message_handle %s from %s — ignoring",
                message_handle,
                from_number,
            )
            return JSONResponse({"received": True, "ignored": "duplicate"})

        service = (body.get("service") or "").strip()
        preview = content[:200] if content else f"[media] {media_url[:120]}"
        logger.info(
            "inbound from %s handle=%s service=%s: %s",
            from_number,
            message_handle or "-",
            service or "unknown",
            preview,
        )
        if service and service.lower() != "imessage":
            logger.warning(
                "typing indicators only work on iMessage; this chat is %s",
                service,
            )

        background_tasks.add_task(
            self.process_inbound, from_number, content, media_url
        )
        return JSONResponse({"received": True})
