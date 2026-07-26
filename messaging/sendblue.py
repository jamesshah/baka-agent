"""Sendblue messaging adapter."""

from __future__ import annotations

from typing import Any, Literal

from sendblue_api import SendblueAPI
from sendblue_api.types.webhook_configuration import WebhookConfiguration

from messaging.base import MessagingClient


class SendblueAdapter(MessagingClient):
    """Outbound iMessage/SMS via the official Sendblue SDK."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        from_number: str,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._from_number = from_number
        self._client: SendblueAPI | None = None

    def _get_client(self) -> SendblueAPI:
        if self._client is None:
            if not self._api_key or not self._api_secret:
                raise RuntimeError(
                    "SENDBLUE_API_KEY and SENDBLUE_API_SECRET must be set in .env"
                )
            self._client = SendblueAPI(
                api_key=self._api_key,
                api_secret=self._api_secret,
            )
        return self._client

    def webhook_health_check(self, *, secret_configured: bool) -> dict[str, Any]:
        """Check API auth and whether a receive webhook is registered."""
        if not self._api_key or not self._api_secret:
            return {
                "status": "unconfigured",
                "secret_configured": secret_configured,
                "receive_urls": [],
                "detail": "SENDBLUE_API_KEY / SENDBLUE_API_SECRET not set",
            }

        try:
            response = self._get_client().webhooks.list()
        except Exception as exc:  # noqa: BLE001 — surface API failures in health
            return {
                "status": "error",
                "secret_configured": secret_configured,
                "receive_urls": [],
                "detail": str(exc),
            }

        receive = []
        if response.webhooks is not None and response.webhooks.receive:
            receive = response.webhooks.receive

        urls = [_webhook_url(entry) for entry in receive]
        urls = [u for u in urls if u]

        if not urls:
            return {
                "status": "error",
                "secret_configured": secret_configured,
                "receive_urls": [],
                "detail": "No receive webhook registered in Sendblue",
            }

        status = "ok" if secret_configured else "degraded"
        detail = None
        if not secret_configured:
            detail = "Receive webhook registered, but SENDBLUE_GLOBAL_WEBHOOK_SECRET is unset"

        return {
            "status": status,
            "secret_configured": secret_configured,
            "receive_urls": urls,
            "detail": detail,
        }

    def _require_from_number(self) -> str:
        if not self._from_number:
            raise RuntimeError("SENDBLUE_FROM_NUMBER must be set in .env")
        return self._from_number

    def send_message(self, number: str, content: str) -> Any:
        """Send an iMessage/SMS via Sendblue."""
        from_number = self._require_from_number()
        return self._get_client().messages.send(
            number=number,
            from_number=from_number,
            content=content,
        )

    def send_typing_indicator(
        self,
        number: str,
        state: Literal["start", "stop"] = "start",
        *,
        max_duration_ms: int | None = None,
    ) -> Any:
        """
        Show or clear the iMessage typing dots.

        Sendblue only delivers these for iMessage (not SMS/RCS), on a best-effort
        basis. For AI replies, pass a generous max_duration_ms on start, then stop
        when the reply is ready.
        """
        from_number = self._require_from_number()
        kwargs: dict[str, Any] = {
            "from_number": from_number,
            "number": number,
            "state": state,
        }
        # Only meaningful for start; default 60s is often too short for local LLMs.
        if state == "start":
            kwargs["max_duration_ms"] = (
                max_duration_ms if max_duration_ms is not None else 180_000  # 3 minutes
            )

        return self._get_client().typing_indicators.send(**kwargs)


def _webhook_url(entry: str | WebhookConfiguration) -> str:
    if isinstance(entry, str):
        return entry
    return entry.url
