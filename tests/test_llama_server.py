"""Tests for LlamaServerAdapter sampling payloads and /props reporting."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from config import Settings
from llm.llama_server import LlamaServerAdapter, runtime_from_props

BASE = "http://llama.test:8080/v1"
PROPS = {
    "model_alias": "lfm",
    "model_path": "/m/lfm.gguf",
    "chat_template": "{% if tools %}<tool_call>{% endif %}",
    "default_generation_settings": {
        "n_ctx": 8192,
        "params": {"temperature": 0.0, "top_k": 1, "top_p": 0.95,
                   "repeat_penalty": 1.0, "seed": -1},
    },
}


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"payloads": [], "gets": []}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST":
            state["payloads"].append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "hi"}}]})
        state["gets"].append(path)
        if path.endswith("/props"):
            return httpx.Response(200, json=PROPS)
        if path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "lfm"}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    return state


def _chat(adapter: LlamaServerAdapter, server: dict[str, Any], **kw: Any) -> dict:
    adapter.chat([{"role": "user", "content": "x"}], **kw)
    return server["payloads"][-1]


def test_temperature_omitted_when_none(server):
    payload = _chat(LlamaServerAdapter(BASE, "m"), server)
    for key in ("temperature", "top_p", "top_k", "repeat_penalty"):
        assert key not in payload


def test_explicit_zero_temperature_sent(server):
    payload = _chat(LlamaServerAdapter(BASE, "m"), server, temperature=0.0)
    assert payload["temperature"] == 0.0


def test_configured_default_used_and_call_overrides(server):
    adapter = LlamaServerAdapter(BASE, "m", temperature=0.3)
    assert _chat(adapter, server)["temperature"] == 0.3
    assert _chat(adapter, server, temperature=0.0)["temperature"] == 0.0


def test_optional_sampling_overrides(server):
    adapter = LlamaServerAdapter(BASE, "m", top_k=50, repeat_penalty=1.1)
    payload = _chat(adapter, server, top_p=0.9)
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 50
    assert payload["repeat_penalty"] == 1.1
    assert "temperature" not in payload
    payload = _chat(adapter, server, top_k=1)
    assert payload["top_k"] == 1
    assert "top_p" not in payload


def test_health_check_reports_props_runtime(server):
    adapter = LlamaServerAdapter(BASE, "m")
    health = adapter.health_check()
    runtime = health["runtime"]
    assert runtime["n_ctx"] == 8192
    assert runtime["sampling"]["temperature"] == 0.0
    assert runtime["sampling"]["top_k"] == 1
    assert runtime["chat_template_has_tool_call"] is True
    assert adapter.runtime_config == runtime


def test_chat_uses_cached_n_ctx_without_extra_props_call(server):
    adapter = LlamaServerAdapter(BASE, "m")
    adapter.health_check()
    before = server["gets"].count("/props")
    _chat(adapter, server)
    assert server["gets"].count("/props") == before


def test_runtime_from_props_handles_missing():
    assert runtime_from_props(None) == {}
    assert runtime_from_props({"n_ctx": 4096})["n_ctx"] == 4096


def test_settings_client_sampling_defaults(monkeypatch):
    for k in ("LLAMA_CLIENT_TEMP", "LLAMA_CLIENT_TOP_K"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.llama_temperature is None and s.llama_top_k is None
    monkeypatch.setenv("LLAMA_CLIENT_TEMP", "0.2")
    monkeypatch.setenv("LLAMA_CLIENT_TOP_K", "")
    s = Settings(_env_file=None)
    assert s.llama_temperature == 0.2
    assert s.llama_top_k is None
