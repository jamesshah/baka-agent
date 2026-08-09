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

    def list_receive_webhook_urls(self) -> list[str]:
        """Return registered Sendblue receive webhook URLs."""
        response = self._get_client().webhooks.list()
        receive: list[Any] = []
        if response.webhooks is not None and response.webhooks.receive:
            receive = list(response.webhooks.receive)
        return [u for u in (_webhook_url(entry) for entry in receive) if u]

    def register_receive_webhook(
        self,
        url: str,
        *,
        global_secret: str = "",
    ) -> Any:
        """Append a receive webhook URL (and optional global signing secret)."""
        normalized = _normalize_webhook_url(url)
        kwargs: dict[str, Any] = {
            "webhooks": [normalized],
            "type": "receive",
        }
        if global_secret:
            kwargs["global_secret"] = global_secret
        return self._get_client().webhooks.create(**kwargs)

    def ensure_receive_webhook(
        self,
        url: str = "",
        *,
        global_secret: str = "",
    ) -> dict[str, Any]:
        """
        Ensure a receive webhook is registered with Sendblue.

        If ``url`` is set and not already registered, registers it.
        Returns a small status dict for logging / health.
        """
        if not self._api_key or not self._api_secret:
            return {
                "status": "skipped",
                "detail": "SENDBLUE_API_KEY / SENDBLUE_API_SECRET not set",
                "receive_urls": [],
            }

        try:
            urls = self.list_receive_webhook_urls()
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "detail": f"Failed to list webhooks: {exc}",
                "receive_urls": [],
            }

        normalized_existing = {_normalize_webhook_url(u) for u in urls}
        target = _normalize_webhook_url(url) if url else ""

        if target and target in normalized_existing:
            return {
                "status": "ok",
                "detail": f"Receive webhook already registered: {target}",
                "receive_urls": urls,
                "registered": False,
            }

        if target:
            try:
                self.register_receive_webhook(target, global_secret=global_secret)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "detail": f"Failed to register webhook {target}: {exc}",
                    "receive_urls": urls,
                    "registered": False,
                }
            urls = [*urls, target]
            return {
                "status": "ok",
                "detail": f"Registered receive webhook: {target}",
                "receive_urls": urls,
                "registered": True,
            }

        if urls:
            return {
                "status": "ok",
                "detail": "Receive webhook(s) present; set SENDBLUE_WEBHOOK_URL to auto-manage",
                "receive_urls": urls,
                "registered": False,
            }

        return {
            "status": "missing",
            "detail": (
                "No receive webhook registered. Set SENDBLUE_WEBHOOK_URL to your "
                "public /webhooks/receive URL (e.g. cloudflared tunnel) and restart, "
                "or POST /webhooks/register with {\"url\": \"https://.../webhooks/receive\"}."
            ),
            "receive_urls": [],
            "registered": False,
        }

    def webhook_health_check(
        self,
        *,
        secret_configured: bool,
        expected_url: str = "",
    ) -> dict[str, Any]:
        """Check API auth and whether a receive webhook is registered."""
        if not self._api_key or not self._api_secret:
            return {
                "status": "unconfigured",
                "secret_configured": secret_configured,
                "receive_urls": [],
                "detail": "SENDBLUE_API_KEY / SENDBLUE_API_SECRET not set",
            }

        try:
            urls = self.list_receive_webhook_urls()
        except Exception as exc:  # noqa: BLE001 — surface API failures in health
            return {
                "status": "error",
                "secret_configured": secret_configured,
                "receive_urls": [],
                "detail": str(exc),
            }

        if not urls:
            return {
                "status": "error",
                "secret_configured": secret_configured,
                "receive_urls": [],
                "detail": "No receive webhook registered in Sendblue",
            }

        expected = _normalize_webhook_url(expected_url) if expected_url else ""
        if expected and expected not in {_normalize_webhook_url(u) for u in urls}:
            return {
                "status": "degraded",
                "secret_configured": secret_configured,
                "receive_urls": urls,
                "detail": (
                    f"Receive webhooks exist, but expected URL is not registered: {expected}"
                ),
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


def _normalize_webhook_url(url: str) -> str:
    return url.strip().rstrip("/")
