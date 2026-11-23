"""Tests for local-first MLflow tracing and PII redaction."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import httpx
import mlflow

from agents.chat_agent import ChatAgent
from agents.executor_agent import ExecutorAgent
from llm.llama_server import LlamaServerAdapter
from log_redact import EMAIL, PHONE, SSN
from memory.database import Database
from memory.migrate import upgrade as upgrade_database
from memory.repository import SqlAlchemyMemoryRepository
from memory.retrieval import ContextBuilder, HybridRetriever
from tools.base import Tool
from tools.registry import ToolRegistry
from tools.send_acknowledgement import SendAcknowledgementTool
from tools.spawn_agent import SpawnAgentTool
from tracing import (
    configure_tracing,
    flush_traces,
    prompt_breakdown,
    reset_tracing_for_tests,
    sanitize_for_trace,
    session_hash,
    tracing_health,
)


class MockCalendarTool(Tool):
    name = "COMPOSIO_MULTI_EXECUTE_TOOL"
    description = "Execute a calendar action."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        return json.dumps(
            {
                "successful": True,
                "owner_phone": "+15559876543",
                "owner_email": "alice@example.com",
                "event": "Ideal Bedtime at 10:30 PM",
            }
        )


class FailingTool(Tool):
    name = "broken_tool"
    description = "Always raises."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("boom for +15551112222")


class _FakeHttpClient:
    """Drop-in replacement for httpx.Client serving scripted llama-server responses."""

    def __init__(self, responses: list[dict[str, Any]], **kwargs: Any) -> None:
        self._responses = responses

    def __enter__(self) -> _FakeHttpClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def get(self, url: str) -> httpx.Response:
        if url.endswith("/props"):
            return httpx.Response(
                200,
                json={"default_generation_settings": {"n_ctx": 32768}},
                request=httpx.Request("GET", url),
            )
        return httpx.Response(404, request=httpx.Request("GET", url))

    def post(self, url: str, json: dict[str, Any]) -> httpx.Response:  # noqa: A002
        payload = self._responses.pop(0)
        return httpx.Response(
            200,
            json=payload,
            request=httpx.Request("POST", url),
        )


def _completion_payload(
    *,
    content: str = "",
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 120,
    completion_tokens: int = 25,
    cache_tokens: int = 80,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "model": "LiquidAI/LFM3-1.7B-GGUF:Q4_K_M",
        "choices": [
            {
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": msg,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timings": {
            "cache_n": cache_tokens,
            "prompt_n": prompt_tokens - cache_tokens,
            "prompt_ms": 120.0,
            "prompt_per_second": 950.0,
            "predicted_n": completion_tokens,
            "predicted_ms": 400.0,
            "predicted_per_token_ms": 16.0,
            "predicted_per_second": 62.5,
        },
    }


class TracingSanitizationTests(unittest.TestCase):
    def test_sanitize_redacts_pii_secrets_and_base64_media(self) -> None:
        raw = {
            "phone": "+15551234567",
            "email": "ada@example.com",
            "ssn": "123-45-6789",
            "api_key": "sk-super-secret",
            "nested": {
                "token": "bearer-xyz",
                "image_url": {"url": "data:image/jpeg;base64," + ("A" * 2048)},
                "input_audio": {"format": "wav", "data": "B" * 1024},
            },
        }
        cleaned = sanitize_for_trace(raw)
        serialized = json.dumps(cleaned)
        self.assertNotIn("15551234567", serialized)
        self.assertNotIn("ada@example.com", serialized)
        self.assertNotIn("123-45-6789", serialized)
        self.assertNotIn("sk-super-secret", serialized)
        self.assertNotIn("bearer-xyz", serialized)
        self.assertNotIn("A" * 100, serialized)
        self.assertNotIn("B" * 100, serialized)
        self.assertEqual(cleaned["phone"], PHONE)
        self.assertEqual(cleaned["email"], EMAIL)
        self.assertEqual(cleaned["ssn"], SSN)
        self.assertEqual(cleaned["api_key"], "<REDACTED>")
        self.assertEqual(cleaned["nested"]["token"], "<REDACTED>")
        self.assertIn("<media-base64 mime=image/jpeg chars=2048>", cleaned["nested"]["image_url"]["url"])
        self.assertIn("<audio-base64 format=wav chars=1024>", cleaned["nested"]["input_audio"]["data"])

    def test_prompt_breakdown_splits_sections(self) -> None:
        messages = [
            {"role": "system", "content": "System rules."},
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Old reply"},
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
            {
                "role": "user",
                "content": "<context>\nMemory fact\n</context>\n\nAdd bedtime",
            },
        ]
        tools = [{"type": "function", "function": {"name": "spawn_agent", "parameters": {}}}]
        bd = prompt_breakdown(messages, tools)
        self.assertEqual(bd["system_chars"], len("System rules."))
        self.assertEqual(bd["tool_count"], 1)
        self.assertEqual(bd["tool_names"], ["spawn_agent"])
        self.assertGreater(bd["tools_chars"], 0)
        self.assertEqual(bd["history_messages"], 2)
        self.assertEqual(bd["tool_result_messages"], 1)
        self.assertEqual(bd["context_block_chars"], len("<context>\nMemory fact\n</context>"))
        self.assertGreater(bd["estimated_tokens"], 0)

    def test_session_hash_is_stable_and_hides_phone(self) -> None:
        hashed = session_hash("+15551234567")
        self.assertIsNotNone(hashed)
        assert hashed is not None
        self.assertTrue(hashed.startswith("sess_"))
        self.assertNotIn("555", hashed)
        self.assertEqual(hashed, session_hash("+15551234567"))


class MlflowEndToEndTracingTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_tracing_for_tests()

    def test_disabled_tracing_reports_disabled_and_is_noop(self) -> None:
        ok = configure_tracing(enabled=False)
        self.assertFalse(ok)
        health = tracing_health()
        self.assertEqual(health["status"], "disabled")
        self.assertEqual(health["backend"], "mlflow")

    def test_full_chat_and_executor_turn_persists_nested_redacted_trace(self) -> None:
        with TemporaryDirectory() as tmp:
            mlflow_db = Path(tmp) / "mlflow.db"
            memory_db = Path(tmp) / "baka.db"
            perf_log = Path(tmp) / "llm-perf.jsonl"

            ok = configure_tracing(
                enabled=True,
                tracking_uri=f"sqlite:///{mlflow_db}",
                experiment="baka-agent-test",
            )
            self.assertTrue(ok)
            self.assertEqual(tracing_health()["status"], "ok")

            upgrade_database(str(memory_db))
            database = Database(str(memory_db))
            repo = SqlAlchemyMemoryRepository(database)
            repo.upsert_memory(
                "+15551234567",
                kind="preference",
                key="bedtime",
                value="Prefers 10:30 PM; contact alice@example.com",
                confidence=1.0,
                source_message_ids=[],
            )
            context_builder = ContextBuilder(HybridRetriever(repo))

            scripted = [
                # 1. ChatAgent delegates to ExecutorAgent
                _completion_payload(
                    reasoning_content="Need to delegate calendar creation.",
                    tool_calls=[
                        {
                            "id": "call_spawn_1",
                            "type": "function",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": json.dumps(
                                    {
                                        "task": "Add Ideal Bedtime at 10:30 PM",
                                        "tools": ["COMPOSIO_MULTI_EXECUTE_TOOL"],
                                    }
                                ),
                            },
                        }
                    ],
                    prompt_tokens=640,
                    completion_tokens=45,
                    cache_tokens=400,
                ),
                # 2. ExecutorAgent calls Composio tool
                _completion_payload(
                    reasoning_content="Calling calendar execute tool.",
                    tool_calls=[
                        {
                            "id": "call_exec_1",
                            "type": "function",
                            "function": {
                                "name": "COMPOSIO_MULTI_EXECUTE_TOOL",
                                "arguments": json.dumps({"action": "create_event"}),
                            },
                        }
                    ],
                    prompt_tokens=510,
                    completion_tokens=30,
                    cache_tokens=0,
                ),
                # 3. ExecutorAgent finishes
                _completion_payload(
                    content="Created Ideal Bedtime at 10:30 PM.",
                    prompt_tokens=590,
                    completion_tokens=15,
                    cache_tokens=500,
                ),
                # 4. ChatAgent replies to user
                _completion_payload(
                    content="Done! Added Ideal Bedtime at 10:30 PM.",
                    prompt_tokens=720,
                    completion_tokens=14,
                    cache_tokens=640,
                ),
            ]

            llm = LlamaServerAdapter("http://127.0.0.1:8080/v1", "local")
            tools = ToolRegistry()
            tools.register(SendAcknowledgementTool())
            tools.register(MockCalendarTool())
            tools.register(FailingTool())
            executor = ExecutorAgent(llm=llm, tools=tools, max_agent_iterations=4)
            tools.register(SpawnAgentTool(executor, context_builder))
            agent = ChatAgent(
                llm=llm,
                tools=tools,
                memory_repository=repo,
                context_builder=context_builder,
            )

            with (
                patch("llm.llama_server.httpx.Client", side_effect=lambda **kw: _FakeHttpClient(scripted, **kw)),
                patch("llm.perf.PERF_LOG_PATH", perf_log),
            ):
                replies = list(
                    agent.run_turn(
                        "+15551234567",
                        "Add ideal bedtime on my calendar; my email is bob@example.com",
                    )
                )

            trace_id = mlflow.get_last_active_trace_id()
            flush_traces()
            database.close()

            self.assertTrue(replies)
            self.assertIn("Added Ideal Bedtime", replies[-1])
            self.assertIsNotNone(trace_id)

            trace = mlflow.get_trace(trace_id)
            self.assertIsNotNone(trace)
            assert trace is not None
            spans = trace.data.spans
            span_names = [s.name for s in spans]

            # Verify full nested hierarchy: ChatAgent -> Retriever + LLM + spawn_agent -> Executor -> LLM + Composio
            self.assertIn("chat_turn", span_names)
            self.assertIn("context_builder.render", span_names)
            self.assertIn("llm:chat", span_names)
            self.assertIn("tool:spawn_agent", span_names)
            self.assertIn("executor_task", span_names)
            self.assertIn("llm:executor", span_names)
            self.assertIn("tool:COMPOSIO_MULTI_EXECUTE_TOOL", span_names)

            # Verify ExecutorAgent is nested under tool:spawn_agent
            by_name = {s.name: s for s in spans}
            self.assertEqual(
                by_name["executor_task"].parent_id,
                by_name["tool:spawn_agent"].span_id,
            )

            # Verify LLM span captured token usage, reasoning, and prompt breakdown
            chat_llm_spans = [s for s in spans if s.name == "llm:chat"]
            self.assertEqual(len(chat_llm_spans), 2)
            first_chat_llm = chat_llm_spans[0]
            self.assertEqual(
                first_chat_llm.attributes["mlflow.chat.tokenUsage"],
                {
                    "input_tokens": 640,
                    "output_tokens": 45,
                    "total_tokens": 685,
                    "cache_read_input_tokens": 400,
                },
            )
            self.assertEqual(first_chat_llm.attributes["agent_label"], "chat")
            self.assertEqual(first_chat_llm.attributes["iteration"], 1)
            self.assertIn("prompt.system_chars", first_chat_llm.attributes)
            self.assertEqual(
                first_chat_llm.outputs["reasoning_content"],
                "Need to delegate calendar creation.",
            )

            # Verify perf JSONL now gets real labels ("chat" and "executor")
            perf_rows = [
                json.loads(line)
                for line in perf_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["label"] for row in perf_rows],
                ["chat", "executor", "executor", "chat"],
            )

            # Verify PII is completely absent across all persisted spans and trace metadata
            all_trace_blob = json.dumps(
                {
                    "tags": trace.info.tags,
                    "metadata": trace.info.trace_metadata,
                    "spans": [
                        {
                            "inputs": s.inputs,
                            "outputs": s.outputs,
                            "attributes": s.attributes,
                        }
                        for s in spans
                    ],
                },
                default=str,
            )
            self.assertNotIn("15551234567", all_trace_blob)
            self.assertNotIn("15559876543", all_trace_blob)
            self.assertNotIn("alice@example.com", all_trace_blob)
            self.assertNotIn("bob@example.com", all_trace_blob)
            self.assertIn(PHONE, all_trace_blob)
            self.assertIn(EMAIL, all_trace_blob)
            self.assertEqual(
                trace.info.trace_metadata.get("mlflow.trace.session"),
                session_hash("+15551234567"),
            )


if __name__ == "__main__":
    unittest.main()
