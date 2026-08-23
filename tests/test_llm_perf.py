"""Tests for llama.cpp completion performance metrics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm.perf import (
    extract_metrics,
    format_perf_message,
    log_completion,
    n_ctx_from_props,
    prompt_cache_health,
    reset_prompt_cache_stats,
)


class ExtractMetricsTests(unittest.TestCase):
    def test_llama_server_timings(self) -> None:
        data = {
            "id": "chatcmpl-abc",
            "model": "local",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            "usage": {
                "prompt_tokens": 417,
                "completion_tokens": 1885,
                "total_tokens": 2302,
            },
            "timings": {
                "cache_n": 0,
                "prompt_n": 417,
                "prompt_ms": 1634.87,
                "prompt_per_second": 255.07,
                "predicted_n": 1885,
                "predicted_ms": 63170.63,
                "predicted_per_token_ms": 33.51,
                "predicted_per_second": 29.84,
            },
        }
        metrics = extract_metrics(data, wall_ms=65012.4, n_ctx=100096, session_id="+1")
        self.assertEqual(metrics["prompt_tokens"], 417)
        self.assertEqual(metrics["completion_tokens"], 1885)
        self.assertEqual(metrics["tokens_in_context"], 2302)
        self.assertEqual(metrics["n_ctx"], 100096)
        self.assertEqual(metrics["context_used_pct"], 2.3)
        self.assertEqual(metrics["cache_tokens"], 0)
        self.assertAlmostEqual(metrics["prefill_tok_s"], 255.07)
        self.assertAlmostEqual(metrics["decode_tok_s"], 29.84)
        self.assertAlmostEqual(metrics["ttft_ms"], 1634.87 + 33.51)
        self.assertEqual(metrics["finish_reason"], "stop")
        self.assertEqual(metrics["tool_calls"], 0)
        line = format_perf_message(metrics)
        self.assertIn("prompt=417", line)
        self.assertIn("gen=1885", line)
        self.assertIn("ctx=2302/100096 (2.3%)", line)
        self.assertIn("decode=29.8 tok/s", line)

    def test_cache_hit_and_usage_fallback(self) -> None:
        data = {
            "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{}, {}]}}],
            "usage": {
                "prompt_tokens": 2500,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 2313},
            },
            "timings": {
                "prompt_n": 187,
                "prompt_ms": 986.48,
                "prompt_per_second": 189.56,
                "predicted_n": 40,
                "predicted_ms": 1400.0,
                "predicted_per_token_ms": 35.0,
                "predicted_per_second": 28.57,
            },
        }
        metrics = extract_metrics(data, wall_ms=2500.0, n_ctx=100096)
        self.assertEqual(metrics["cache_tokens"], 2313)
        self.assertEqual(metrics["cache_hit_ratio"], 0.9252)
        self.assertEqual(metrics["tokens_in_context"], 2313 + 187 + 40)
        self.assertEqual(metrics["tool_calls"], 2)
        self.assertAlmostEqual(metrics["ttft_ms"], 986.48 + 35.0)

    def test_openai_usage_only(self) -> None:
        data = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        metrics = extract_metrics(data, wall_ms=123.4)
        self.assertEqual(metrics["prompt_tokens"], 10)
        self.assertEqual(metrics["completion_tokens"], 5)
        self.assertEqual(metrics["tokens_in_context"], 15)
        self.assertIsNone(metrics["ttft_ms"])
        self.assertIsNone(metrics["prefill_tok_s"])
        self.assertEqual(metrics["wall_ms"], 123.4)


class PropsAndLogTests(unittest.TestCase):
    def test_n_ctx_from_props(self) -> None:
        self.assertEqual(
            n_ctx_from_props({"default_generation_settings": {"n_ctx": 100096}}),
            100096,
        )
        self.assertEqual(n_ctx_from_props({"n_ctx": 8192}), 8192)
        self.assertIsNone(n_ctx_from_props(None))
        self.assertIsNone(n_ctx_from_props({}))

    def test_jsonl_write(self) -> None:
        data = {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm-perf.jsonl"
            log_completion(data, wall_ms=10.0, path=path, session_id="s")
            rows = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            parsed = json.loads(rows[0])
            self.assertEqual(parsed["session_id"], "s")
            self.assertEqual(parsed["prompt_tokens"], 1)


class PromptCacheHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_prompt_cache_stats()

    def tearDown(self) -> None:
        reset_prompt_cache_stats()

    def test_empty(self) -> None:
        stats = prompt_cache_health()
        self.assertIsNone(stats["cache_hit_ratio"])
        self.assertEqual(stats["completions"], 0)
        self.assertEqual(stats["source"], "completion_timings")

    def test_aggregates_completions(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "perf.jsonl"
            log_completion(
                {
                    "choices": [{"message": {"content": "a"}}],
                    "usage": {"prompt_tokens": 4852, "completion_tokens": 71},
                    "timings": {"cache_n": 4378, "prompt_n": 474},
                },
                wall_ms=5208.0,
                path=path,
                turn_id="24fc",
            )
            log_completion(
                {
                    "choices": [{"message": {"content": "b"}}],
                    "usage": {"prompt_tokens": 1413, "completion_tokens": 148},
                    "timings": {"cache_n": 0, "prompt_n": 1413},
                },
                wall_ms=9320.0,
                path=path,
            )
        stats = prompt_cache_health()
        self.assertEqual(stats["completions"], 2)
        self.assertEqual(stats["n_prompt_tokens"], 4852 + 1413)
        self.assertEqual(stats["n_prompt_tokens_cache"], 4378)
        self.assertEqual(stats["cache_hit_ratio"], round(4378 / (4852 + 1413), 4))
        self.assertEqual(stats["last"]["n_prompt_tokens_cache"], 0)
        self.assertEqual(stats["last"]["n_prompt_tokens"], 1413)


if __name__ == "__main__":
    unittest.main()
