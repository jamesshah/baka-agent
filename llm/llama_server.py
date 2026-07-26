"""OpenAI-compatible LLM adapter for llama.cpp's llama-server."""

from __future__ import annotations

from typing import Any

import httpx

from llm.base import LLMClient


class LlamaServerAdapter(LLMClient):
    """POST to {base_url}/chat/completions and return choices[0].message."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        with httpx.Client(timeout=self._timeout) as client:
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
