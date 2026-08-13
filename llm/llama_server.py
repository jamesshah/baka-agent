"""OpenAI-compatible LLM adapter for llama.cpp's llama-server."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

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

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    def health_check(self, *, timeout: float = 5.0) -> dict[str, Any]:
        """Probe the model server and fetch the served model name."""
        last_error = "no health endpoints tried"
        with httpx.Client(timeout=timeout) as client:
            ready = False
            checked_url: str | None = None
            for url in self._health_urls():
                try:
                    response = client.get(url)
                    if response.status_code == 200:
                        ready = True
                        checked_url = url
                        break
                    last_error = f"{url} → HTTP {response.status_code}"
                except httpx.HTTPError as exc:
                    last_error = f"{url} → {exc}"

            if not ready:
                return {
                    "status": "error",
                    "base_url": self._base_url,
                    "model": self._model,
                    "detail": last_error,
                }

            served = self._fetch_served_model(client)
            prompt_cache = self._fetch_prompt_cache_stats(client)
            return {
                "status": "ok",
                "base_url": self._base_url,
                "model": self._model,
                "checked_url": checked_url,
                **served,
                **prompt_cache,
            }

    def _server_root(self) -> str:
        parts = urlsplit(self._base_url)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            root_path = path[: -len("/v1")] or ""
            return urlunsplit((parts.scheme, parts.netloc, root_path, "", ""))
        return urlunsplit((parts.scheme, parts.netloc, path, "", "")) or self._base_url

    def _fetch_served_model(self, client: httpx.Client) -> dict[str, Any]:
        """Read model id/alias/path from /v1/models and /props."""
        info: dict[str, Any] = {
            "served_model": None,
            "model_alias": None,
            "model_path": None,
        }

        models_url = f"{self._base_url}/models"
        try:
            response = client.get(models_url)
            if response.status_code == 200:
                data = response.json()
                entries = data.get("data") or []
                if entries and isinstance(entries[0], dict):
                    info["served_model"] = entries[0].get("id")
                elif data.get("models"):
                    first = data["models"][0]
                    if isinstance(first, dict):
                        info["served_model"] = first.get("name") or first.get("model")
        except (httpx.HTTPError, ValueError, TypeError):
            pass

        props_url = f"{self._server_root()}/props"
        try:
            response = client.get(props_url)
            if response.status_code == 200:
                props = response.json()
                info["model_alias"] = props.get("model_alias")
                info["model_path"] = props.get("model_path")
                if not info["served_model"]:
                    info["served_model"] = props.get("model_alias")
        except (httpx.HTTPError, ValueError, TypeError):
            pass

        return info

    def _fetch_prompt_cache_stats(self, client: httpx.Client) -> dict[str, Any]:
        """Aggregate prompt-cache hit ratio from llama-server GET /slots."""
        slots_url = f"{self._server_root()}/slots"
        try:
            response = client.get(slots_url)
        except httpx.HTTPError as exc:
            return {
                "prompt_cache": {
                    "status": "unavailable",
                    "detail": f"{slots_url} → {exc}",
                }
            }

        if response.status_code != 200:
            return {
                "prompt_cache": {
                    "status": "unavailable",
                    "detail": f"{slots_url} → HTTP {response.status_code}",
                }
            }

        try:
            slots = response.json()
        except ValueError:
            return {
                "prompt_cache": {
                    "status": "unavailable",
                    "detail": "invalid JSON from /slots",
                }
            }

        if not isinstance(slots, list):
            return {
                "prompt_cache": {
                    "status": "unavailable",
                    "detail": "unexpected /slots shape",
                }
            }

        cached_total = 0
        prompt_total = 0
        per_slot: list[dict[str, Any]] = []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            n_cache = slot.get("n_prompt_tokens_cache")
            n_prompt = slot.get("n_prompt_tokens")
            n_processed = slot.get("n_prompt_tokens_processed")
            if n_cache is None and n_prompt is None and n_processed is None:
                continue

            cache_tokens = int(n_cache or 0)
            if n_prompt is not None:
                prompt_tokens = int(n_prompt)
            else:
                prompt_tokens = cache_tokens + int(n_processed or 0)

            if prompt_tokens <= 0 and cache_tokens <= 0:
                continue

            cached_total += cache_tokens
            prompt_total += prompt_tokens
            ratio = (
                round(cache_tokens / prompt_tokens, 4) if prompt_tokens > 0 else None
            )
            per_slot.append(
                {
                    "id": slot.get("id"),
                    "n_prompt_tokens_cache": cache_tokens,
                    "n_prompt_tokens": prompt_tokens,
                    "cache_hit_ratio": ratio,
                }
            )

        if prompt_total <= 0:
            return {
                "prompt_cache": {
                    "status": "ok",
                    "cache_hit_ratio": None,
                    "n_prompt_tokens_cache": 0,
                    "n_prompt_tokens": 0,
                    "detail": "no prompt token stats yet",
                    "slots": per_slot,
                }
            }

        return {
            "prompt_cache": {
                "status": "ok",
                "cache_hit_ratio": round(cached_total / prompt_total, 4),
                "n_prompt_tokens_cache": cached_total,
                "n_prompt_tokens": prompt_total,
                "slots": per_slot,
            }
        }

    def _health_urls(self) -> list[str]:
        """Prefer native /health, then OpenAI-compatible /v1/health and /models."""
        root = self._server_root()
        urls = [
            f"{root}/health",
            f"{root}/v1/health",
            f"{self._base_url}/models",
        ]
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        unique: list[str] = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

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
