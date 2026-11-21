"""OpenAI-compatible eval client and per-call usage recording."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from llm.base import LLMClient
from llm.perf import extract_metrics


@dataclass(frozen=True)
class SamplingParams:
    """Sampling knobs. ``None`` means "omit from the payload" (server default)."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    max_tokens: int | None = None

    def payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class OpenAICompatClient(LLMClient):
    """POST {base_url}/chat/completions with explicit sampling params.

    Unlike ``LlamaServerAdapter`` this never forces ``temperature``: the tool
    loop's default ``temperature=0.7`` argument is ignored so that unset knobs
    fall back to the server's (model card / ``--temp``) defaults. It also does
    not write ``.run/logs/llm-perf.jsonl``; usage lands on ``last_metrics``.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        sampling: SamplingParams | None = None,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._sampling = sampling or SamplingParams()
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self.last_metrics: dict[str, Any] | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def sampling(self) -> SamplingParams:
        return self._sampling

    def close(self) -> None:
        self._client.close()

    def served_model(self) -> str | None:
        """First model id from ``GET /models`` (None when unreachable)."""
        try:
            response = self._client.get(f"{self._base_url}/models", timeout=10.0)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        entries = data.get("data") if isinstance(data, dict) else None
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            ident = entries[0].get("id")
            return str(ident) if ident else None
        return None

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            **self._sampling.payload(),
        }
        if tools:
            payload["tools"] = tools
        return payload

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del temperature  # sampling comes from SamplingParams only
        self.last_metrics = None
        payload = self.build_payload(messages, tools)
        started = time.perf_counter()
        response = self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            timeout=timeout if timeout is not None else self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        wall_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(data, dict):
            raise RuntimeError(f"LLM returned non-object JSON: {data!r}")
        self.last_metrics = extract_metrics(data, wall_ms=wall_ms, model=self._model)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned no choices: {data}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"LLM returned invalid message: {data}")
        return message


class RecordingLLM(LLMClient):
    """Wrap a client; append one usage/trace record per call to ``calls``.

    Reads ``inner.last_metrics`` (set by ``OpenAICompatClient`` or test fakes)
    for prompt/completion/cache token counts.
    """

    def __init__(self, inner: LLMClient, agent: str, calls: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._agent = agent
        self._calls = calls

    def input_modalities(self) -> set[str]:
        return self._inner.input_modalities()

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if hasattr(self._inner, "last_metrics"):
            self._inner.last_metrics = None  # type: ignore[attr-defined]
        started = time.perf_counter()
        message = self._inner.chat(
            messages, tools=tools, temperature=temperature, timeout=timeout
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        metrics = getattr(self._inner, "last_metrics", None) or {}
        proposed = [
            {
                "name": (call.get("function") or {}).get("name"),
                "arguments": (call.get("function") or {}).get("arguments"),
            }
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict)
        ]
        self._calls.append(
            {
                "agent": self._agent,
                "n_messages": len(messages),
                "n_tools": len(tools or []),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "cache_tokens": metrics.get("cache_tokens"),
                "finish_reason": metrics.get("finish_reason"),
                "wall_ms": round(metrics.get("wall_ms") or wall_ms, 2),
                "tool_calls": proposed,
                "content": message.get("content") or "",
            }
        )
        return message


def usage_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum token usage over recorded calls; cache ratio over calls that report both."""
    prompt = sum(int(call.get("prompt_tokens") or 0) for call in calls)
    completion = sum(int(call.get("completion_tokens") or 0) for call in calls)
    cache_pairs = [
        (int(call["cache_tokens"]), int(call["prompt_tokens"]))
        for call in calls
        if call.get("cache_tokens") is not None and call.get("prompt_tokens")
    ]
    cached = sum(pair[0] for pair in cache_pairs)
    cache_den = sum(pair[1] for pair in cache_pairs)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_tokens": cached,
        "cache_hit_ratio": round(cached / cache_den, 4) if cache_den else None,
        "max_prompt_tokens": max(
            (int(call.get("prompt_tokens") or 0) for call in calls), default=0
        ),
        "llm_calls": len(calls),
        "chat_llm_calls": sum(1 for call in calls if call.get("agent") == "chat"),
        "executor_llm_calls": sum(
            1 for call in calls if call.get("agent") == "executor"
        ),
    }
