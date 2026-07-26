"""Raw OpenAI-compatible client for llama.cpp's llama-server."""

from __future__ import annotations

from typing import Any

import httpx

from config import get_settings


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    POST to {LLAMA_BASE_URL}/chat/completions and return choices[0].message.

    The returned message may include ``content`` and/or ``tool_calls``.
    """
    settings = get_settings()
    url = settings.llama_base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": settings.llama_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM returned no choices: {data}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"LLM returned invalid message: {data}")
    return message
