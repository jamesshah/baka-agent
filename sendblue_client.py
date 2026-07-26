from typing import Literal

from sendblue_api import SendblueAPI

from config import get_settings

_client: SendblueAPI | None = None


def get_client() -> SendblueAPI:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.sendblue_api_key or not settings.sendblue_api_secret:
            raise RuntimeError(
                "SENDBLUE_API_KEY and SENDBLUE_API_SECRET must be set in .env"
            )
        _client = SendblueAPI(
            api_key=settings.sendblue_api_key,
            api_secret=settings.sendblue_api_secret,
        )
    return _client


def send_message(number: str, content: str):
    """Send an iMessage/SMS via Sendblue."""
    settings = get_settings()
    if not settings.sendblue_from_number:
        raise RuntimeError("SENDBLUE_FROM_NUMBER must be set in .env")
    return get_client().messages.send(
        number=number,
        from_number=settings.sendblue_from_number,
        content=content,
    )


def send_typing_indicator(
    number: str,
    state: Literal["start", "stop"] = "start",
    *,
    max_duration_ms: int | None = None,
):
    """
    Show or clear the iMessage typing dots.

    Sendblue only delivers these for iMessage (not SMS/RCS), on a best-effort
    basis. For AI replies, pass a generous max_duration_ms on start, then stop
    when the reply is ready.
    """
    settings = get_settings()
    if not settings.sendblue_from_number:
        raise RuntimeError("SENDBLUE_FROM_NUMBER must be set in .env")

    kwargs: dict = {
        "from_number": settings.sendblue_from_number,
        "number": number,
        "state": state,
    }
    # Only meaningful for start; default 60s is often too short for local LLMs.
    if state == "start":
        kwargs["max_duration_ms"] = (
            max_duration_ms
            if max_duration_ms is not None
            else 180_000  # 3 minutes
        )

    return get_client().typing_indicators.send(**kwargs)
