"""Unit tests for the offline model replay and ablation evaluation harness."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from evals.ablations import (
    AblationConfig,
    build_chat_system_prompt,
    build_executor_system_prompt,
    select_history_messages,
    select_memory_context,
)
from evals.cases import EvalCase, all_cases, filter_cases
from evals.client import OpenAICompatClient, SamplingParams
from evals.cli import main, run_eval_suite
from evals.fixtures import EVAL_SESSION_ID
from evals.report import (
    build_report_payload,
    format_markdown_report,
    log_to_mlflow,
    summarize_results,
    write_report_files,
)
from evals.runner import run_case
from llm.base import LLMClient


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


class ScriptedEvalLLM(LLMClient):
    """Queue-driven fake LLMClient that also populates ``last_metrics``."""

    def __init__(
        self,
        steps: list[dict[str, Any]],
        *,
        prompt_tokens: int = 1200,
        completion_tokens: int = 60,
        cache_tokens: int = 900,
    ) -> None:
        self._steps = list(steps)
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cache_tokens = cache_tokens
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.seen_tools: list[list[dict[str, Any]] | None] = []
        self.last_metrics: dict[str, Any] | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del temperature, timeout
        self.seen_messages.append(messages)
        self.seen_tools.append(tools)
        self.last_metrics = {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "cache_tokens": self._cache_tokens,
            "finish_reason": "stop",
            "wall_ms": 12.5,
        }
        if not self._steps:
            return {"role": "assistant", "content": "Done."}
        return self._steps.pop(0)


class GoldenPolicyLLM(LLMClient):
    """Deterministic fake LLM that follows the ideal tool-calling policy across all cases."""

    model = "golden-fake-model"

    def __init__(self) -> None:
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.seen_tools: list[list[dict[str, Any]] | None] = []
        self.last_metrics: dict[str, Any] | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del temperature, timeout
        self.seen_messages.append(messages)
        self.seen_tools.append(tools)
        self.last_metrics = {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cache_tokens": 800,
            "finish_reason": "stop",
            "wall_ms": 10.0,
        }
        tool_names = {
            (spec.get("function") or {}).get("name")
            for spec in (tools or [])
            if isinstance(spec, dict)
        }
        last_msg = messages[-1]
        is_chat_agent = "spawn_agent" in tool_names

        if is_chat_agent:
            return self._respond_as_chat(messages, last_msg, tool_names)
        return self._respond_as_executor(messages, last_msg, tool_names)

    def _respond_as_chat(
        self,
        messages: list[dict[str, Any]],
        last_msg: dict[str, Any],
        tool_names: set[str | None],
    ) -> dict[str, Any]:
        if last_msg.get("role") == "tool":
            tool_content = str(last_msg.get("content") or "")
            if "connect.composio.dev" in tool_content:
                return {
                    "role": "assistant",
                    "content": "Open this link to connect your Google Calendar: https://connect.composio.dev/link/ca_mock_gcal_123",
                }
            if "app.snaptrade.com" in tool_content:
                return {
                    "role": "assistant",
                    "content": "Open this link to approve SnapTrade read access for your brokerage portfolio: https://app.snaptrade.com/oauth/device?user_code=MOCK-1234",
                }
            if '"status": "linked"' in tool_content:
                return {
                    "role": "assistant",
                    "content": "Yep, your Google Calendar is connected and linked!",
                }
            return {
                "role": "assistant",
                "content": tool_content,
            }

        raw_user = str(last_msg.get("content") or "")
        # Strip injected <context>...</context> when inspecting the user's actual utterance.
        user_only = re.sub(r"<context>.*?</context>\s*", "", raw_user, flags=re.S).strip()
        lower = user_only.lower()

        if "hope you're having a good wednesday" in lower:
            return {
                "role": "assistant",
                "content": "Thanks! Having a great Wednesday — let me know if you need anything.",
            }
        if "is my google calendar connected" in lower:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call("c_status", "composio_status", {"toolkit": "googlecalendar"})
                ],
            }
        if "link to connect my google calendar" in lower:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call("c_link_gcal", "link_composio", {"toolkit": "googlecalendar"})
                ],
            }
        if "snaptrade" in lower or "brokerage portfolio" in lower:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("c_link_st", "link_snaptrade", {})],
            }

        # Delegation cases: inspect available worker capability catalog in system prompt.
        sys_prompt = str(messages[0].get("content") or "")
        use_flat = "- calendar:" in sys_prompt and "- composio:" not in sys_prompt

        if "can’t see it" in lower or "can't see" in lower or "what events are actually" in lower:
            worker_tools = ["calendar"] if use_flat else ["composio"]
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "c_spawn",
                        "spawn_agent",
                        {
                            "acknowledgement": "Checking your Google Calendar now…",
                            "task": "List today's events on the user's Google Calendar for Wed Nov 11, 2026 and check whether 'Ideal Bedtime' is present.",
                            "tools": worker_tools,
                        },
                    ),
                ],
            }

        worker_tools = ["calendar"] if use_flat else ["composio"]
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "c_spawn",
                    "spawn_agent",
                    {
                        "acknowledgement": "Adding that Ideal Bedtime block to your calendar…",
                        "task": (
                            "Create a Google Calendar event titled 'Ideal Bedtime' "
                            "on Wed Nov 11, 2026 from 10:30 PM to 11:00 PM CST "
                            "(2026-11-11T22:30:00-06:00 to 2026-11-11T23:00:00-06:00)."
                        ),
                        "tools": worker_tools,
                    },
                ),
            ],
        }

    def _respond_as_executor(
        self,
        messages: list[dict[str, Any]],
        last_msg: dict[str, Any],
        tool_names: set[str | None],
    ) -> dict[str, Any]:
        task_text = str(messages[1].get("content") or "").lower()
        wants_list = "list " in task_text or "check whether" in task_text
        wants_create = "create " in task_text or "add " in task_text

        # Flat tool surface
        if "create_calendar_event" in tool_names or "list_calendar_events" in tool_names:
            if last_msg.get("role") == "user":
                calls = []
                if wants_create and "create_calendar_event" in tool_names:
                    calls.append(
                        _tool_call(
                            "e_flat_create",
                            "create_calendar_event",
                            {
                                "summary": "Ideal Bedtime",
                                "start_datetime": "2026-11-11T22:30:00-06:00",
                                "end_datetime": "2026-11-11T23:00:00-06:00",
                            },
                        )
                    )
                if wants_list and "list_calendar_events" in tool_names:
                    calls.append(
                        _tool_call(
                            "e_flat_list",
                            "list_calendar_events",
                            {"date": "2026-11-11"},
                        )
                    )
                return {"role": "assistant", "content": "", "tool_calls": calls}
            if wants_create and wants_list:
                return {
                    "role": "assistant",
                    "content": (
                        "Created 'Ideal Bedtime' for 10:30 PM CST on Google Calendar. "
                        "Today's calendar events are Morning Standup, Product Design Sync, and Ideal Bedtime."
                    ),
                }
            if wants_list:
                return {
                    "role": "assistant",
                    "content": (
                        "Checked your Google Calendar for Wed Nov 11: Morning Standup (9:30 AM) "
                        "and Product Design Sync (2:00 PM) are listed; Ideal Bedtime is not on the calendar yet."
                    ),
                }
            return {
                "role": "assistant",
                "content": "Created 'Ideal Bedtime' on your Google Calendar for Wed Nov 11 at 10:30 PM CST (22:30).",
            }

        # Composio Tool Router surface: step 1 search, step 2 multi_execute, step 3 reply
        if last_msg.get("role") == "user":
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "e_search",
                        "COMPOSIO_SEARCH_TOOLS",
                        {
                            "queries": [
                                {"use_case": "create and list Google Calendar events"}
                            ]
                        },
                    )
                ],
            }

        tool_results_so_far = [m for m in messages if m.get("role") == "tool"]
        if len(tool_results_so_far) == 1:
            if wants_list and not wants_create:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tool_call(
                            "e_exec_list",
                            "COMPOSIO_MULTI_EXECUTE_TOOL",
                            {
                                "tools": [
                                    {
                                        "tool_slug": "GOOGLECALENDAR_EVENTS_LIST",
                                        "arguments": {"calendarId": "primary"},
                                    }
                                ]
                            },
                        )
                    ],
                }
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "e_exec_create",
                        "COMPOSIO_MULTI_EXECUTE_TOOL",
                        {
                            "tools": [
                                {
                                    "tool_slug": "GOOGLECALENDAR_CREATE_EVENT",
                                    "arguments": {
                                        "calendar_id": "primary",
                                        "summary": "Ideal Bedtime",
                                        "start_datetime": "2026-11-11T22:30:00-06:00",
                                        "end_datetime": "2026-11-11T23:00:00-06:00",
                                    },
                                }
                            ]
                        },
                    )
                ],
            }

        if wants_list and not wants_create:
            return {
                "role": "assistant",
                "content": (
                    "Checked your Google Calendar for today: you have Morning Standup (9:30 AM) "
                    "and Product Design Sync (2:00 PM). The 'Ideal Bedtime' event is not on your calendar."
                ),
            }
        return {
            "role": "assistant",
            "content": "Added 'Ideal Bedtime' to your Google Calendar for Wed Nov 11 at 10:30 PM CST.",
        }


class EvalHarnessTests(unittest.TestCase):
    def test_benchmark_suite_is_complete_and_pii_free(self) -> None:
        cases = all_cases()
        self.assertGreaterEqual(len(cases), 10)
        self.assertLessEqual(len(cases), 13)
        categories = {c.category for c in cases}
        self.assertEqual(
            categories,
            {
                "polluted_replay",
                "clean_equivalent",
                "direct_routing",
                "executor_comparison",
            },
        )
        # Ensure no real phone numbers exist in any fixture (only the +15550001234 test dummy).
        serialized = json.dumps([c.__dict__ for c in cases], default=str)
        phones = set(re.findall(r"\+1\d{10}", serialized))
        self.assertLessEqual(phones, {EVAL_SESSION_ID})

        # Filter helper works by id or category
        only_polluted = filter_cases(cases, "polluted_replay")
        self.assertEqual(len(only_polluted), 3)
        only_t58 = filter_cases(cases, "t58")
        self.assertEqual(len(only_t58), 2)

    def test_golden_policy_passes_entire_suite(self) -> None:
        cases = all_cases()
        llm = GoldenPolicyLLM()
        results = [run_case(case, llm) for case in cases]
        for res in results:
            self.assertTrue(
                res.scores.all_passed,
                f"Case {res.case_id} failed with reasons: {res.scores.reasons}",
            )
            self.assertGreater(res.prompt_tokens, 0)
            self.assertEqual(res.cache_hit_ratio, 0.8)

    def test_turn_56_failure_mode_detected_by_scorers(self) -> None:
        """Turn 56 failure: no ack, snaptrade in task, missing COMPOSIO_SEARCH_TOOLS, executor loops to limit."""
        case = filter_cases(all_cases(), "replay_t56_polluted_bedtime_create")[0]
        # ChatAgent calls spawn_agent without send_acknowledgement, omits COMPOSIO_SEARCH_TOOLS,
        # and hallucinates snaptrade; Executor guesses unknown slugs 3 times then hits max_executor_iterations=3.
        steps = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "t56_spawn",
                        "spawn_agent",
                        {
                            "task": (
                                "Add 'Ideal Bedtime' at 10:30 PM. Keep the event concise "
                                "and in the calendar toolkits (googlecalendar, snaptrade)."
                            ),
                            "tools": [
                                "COMPOSIO_GET_TOOL_SCHEMAS",
                                "COMPOSIO_MANAGE_CONNECTIONS",
                                "COMPOSIO_MULTI_EXECUTE_TOOL",
                            ],
                        },
                    )
                ],
            },
            # Executor guesses 1
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "e1",
                        "COMPOSIO_GET_TOOL_SCHEMAS",
                        {"tool_slugs": ["GOOGLECALENDAR_ADD_EVENT"]},
                    )
                ],
            },
            # Executor guesses 2
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "e2",
                        "COMPOSIO_GET_TOOL_SCHEMAS",
                        {"tool_slugs": ["SNAPTRADE_CREATE_EVENT"]},
                    )
                ],
            },
            # ChatAgent final reply after executor hits limit
            {
                "role": "assistant",
                "content": "I hit a tool-call limit earlier, so I can't add the event right now.",
            },
        ]
        llm = ScriptedEvalLLM(steps)
        result = run_case(case, llm, max_executor_iterations=2)
        self.assertFalse(result.scores.routing_ok)
        self.assertIn("acknowledgement", result.scores.reasons["routing_ok"])
        self.assertFalse(result.scores.tool_args_ok)
        self.assertIn("snaptrade", result.scores.reasons["tool_args_ok"].lower())
        self.assertFalse(result.scores.request_fidelity)
        self.assertEqual(len(result.created_events), 0)

    def test_turn_56_missing_search_tools_grant_flagged(self) -> None:
        """Even without snaptrade, omitting COMPOSIO_SEARCH_TOOLS from spawn_agent.tools fails tool_args_ok."""
        case = filter_cases(all_cases(), "replay_t56_polluted_bedtime_create")[0]
        steps = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "sp",
                        "spawn_agent",
                        {
                            "acknowledgement": "On it!",
                            "task": "Create Ideal Bedtime event on Google Calendar at 10:30 PM.",
                            "tools": [
                                "COMPOSIO_GET_TOOL_SCHEMAS",
                                "COMPOSIO_MULTI_EXECUTE_TOOL",
                            ],
                        },
                    ),
                ],
            },
            {"role": "assistant", "content": "I could not find the schema."},
            {"role": "assistant", "content": "I wasn't able to add the event."},
        ]
        result = run_case(case, ScriptedEvalLLM(steps))
        self.assertTrue(result.scores.routing_ok)
        self.assertFalse(result.scores.tool_args_ok)
        self.assertIn("COMPOSIO_SEARCH_TOOLS", result.scores.reasons["tool_args_ok"])

    def test_turn_57_fabrication_and_contradiction_detected(self) -> None:
        """Turn 57 failure: Executor makes 0 create calls and fabricates creation while claiming SnapTrade is not allowed."""
        case = filter_cases(all_cases(), "replay_t57_polluted_retry_create")[0]
        steps = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "sp",
                        "spawn_agent",
                        {
                            "acknowledgement": "Retrying…",
                            "task": "Create Ideal Bedtime event on Google Calendar at 10:30 PM.",
                            "tools": ["composio"],
                        },
                    ),
                ],
            },
            # Executor returns fabricated + self-contradictory text with 0 tool calls
            {
                "role": "assistant",
                "content": (
                    "I cannot add the event to Google Calendar with the available "
                    "SnapTrade tools — SnapTrade is currently not allowed for this "
                    "session. I have successfully created the event on your Google Calendar: "
                    "Title: Ideal Bedtime, Wed Nov 11 at 10:30 PM."
                ),
            },
            # ChatAgent relays the fabrication
            {
                "role": "assistant",
                "content": 'Done! I added a "Ideal Bedtime" event to your Google Calendar for Wed Nov 11 at 10:30 PM.',
            },
        ]
        result = run_case(case, ScriptedEvalLLM(steps))
        self.assertFalse(result.scores.no_fabrication)
        self.assertFalse(result.scores.request_fidelity)
        self.assertEqual(len(result.created_events), 0)

    def test_turn_58_whoop_daily_digest_hijack_detected(self) -> None:
        """Turn 58 failure: ChatAgent hijacks calendar check with Whoop daily digest; Executor makes 0 tool calls."""
        case = filter_cases(all_cases(), "replay_t58_polluted_verify_calendar")[0]
        steps = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "sp",
                        "spawn_agent",
                        {
                            "acknowledgement": "Checking…",
                            "task": (
                                "Check Google Calendar again for today's events. The user wants "
                                "a short daily-digest iMessage summarizing what's on the calendar, "
                                "pulling out the key data (Recovery 53%, Sleep 5.4h, Weightlifting strain)."
                            ),
                            "tools": ["composio"],
                        },
                    ),
                ],
            },
            # Executor makes 0 tool calls and reports Whoop metrics as calendar events
            {
                "role": "assistant",
                "content": (
                    "I checked Google Calendar for today's events. Your recovery is at 53% "
                    "(yellow zone), sleep was 5.4 hours, and you have a weightlifting strain "
                    "of 9.3 on the calendar."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Here's a quick daily-digest for today: Your recovery is at 53%, "
                    "sleep was 5.4 hours, and you have a weightlifting strain of 9.3."
                ),
            },
        ]
        result = run_case(case, ScriptedEvalLLM(steps))
        self.assertFalse(result.scores.no_fabrication)
        self.assertFalse(result.scores.request_fidelity)
        self.assertIn("hijacked", result.scores.reasons["request_fidelity"])

    def test_ablation_modes_transform_history_context_prompt_and_surface(self) -> None:
        case = filter_cases(all_cases(), "replay_t58_polluted_verify_calendar")[0]

        # 1. History modes: full vs short vs clean
        full_hist = select_history_messages(case.prior_turns, "full")
        short_hist = select_history_messages(case.prior_turns, "short")
        clean_hist = select_history_messages(case.prior_turns, "clean")
        self.assertGreater(len(full_hist), len(short_hist))
        self.assertGreater(len(short_hist), 0)
        self.assertEqual(clean_hist, [])
        # Turn 53 daily digest is in full_hist[0], but dropped in short_hist
        self.assertIn("Daily digest", str(full_hist[0].get("content")))
        self.assertNotIn("Daily digest", json.dumps(short_hist))

        # 2. Context modes: full vs none
        self.assertIn("Skill: whoop-recovery", select_memory_context(case.memory_context, "full"))
        self.assertEqual(select_memory_context(case.memory_context, "none"), "")

        # Verify context-mode=none actually omits <context> from LLM messages during run_case
        llm_no_ctx = GoldenPolicyLLM()
        res_no_ctx = run_case(
            case,
            llm_no_ctx,
            config=AblationConfig(
                history_mode="clean",
                context_mode="none",
                prompt_mode="no_negative_priming",
                tool_surface="flat",
                override_case_surface=True,
            ),
        )
        self.assertTrue(res_no_ctx.scores.all_passed)
        first_chat_messages = llm_no_ctx.seen_messages[0]
        # System + current user message only (since history_mode=clean)
        self.assertEqual(len(first_chat_messages), 2)
        self.assertNotIn("<context>", str(first_chat_messages[-1]["content"]))
        self.assertIn("- calendar:", str(first_chat_messages[0]["content"]))

        # 3. Prompt modes: current vs no_negative_priming (both now free of negative SnapTrade priming)
        curr_chat = build_chat_system_prompt("current", "full_composio")
        clean_chat = build_chat_system_prompt("no_negative_priming", "full_composio")
        self.assertNotIn("Never call link_snaptrade for Calendar", curr_chat)
        self.assertNotIn("Never call link_snaptrade for Calendar", clean_chat)

        curr_exec = build_executor_system_prompt("current", "full_composio")
        clean_exec = build_executor_system_prompt("no_negative_priming", "full_composio")
        self.assertNotIn("Do not use SnapTrade tools for Calendar", curr_exec)
        self.assertNotIn("SnapTrade", clean_exec)

    def test_openai_compat_client_omits_unset_sampling_params(self) -> None:
        captured_bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={"data": [{"id": "LiquidAI/LFM3-Omni-2B-GGUF:BF16"}]},
                )
            body = json.loads(request.content.decode("utf-8"))
            captured_bodies.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Hello!"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 10,
                        "prompt_tokens_details": {"cached_tokens": 90},
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        # Default SamplingParams -> no temp/top_p/top_k/repeat_penalty keys sent
        client_default = OpenAICompatClient(
            "http://127.0.0.1:8080/v1",
            "lfm3",
            transport=transport,
        )
        self.assertEqual(
            client_default.served_model(), "LiquidAI/LFM3-Omni-2B-GGUF:BF16"
        )
        msg = client_default.chat([{"role": "user", "content": "hi"}], temperature=0.7)
        self.assertEqual(msg["content"], "Hello!")
        self.assertNotIn("temperature", captured_bodies[-1])
        self.assertNotIn("top_p", captured_bodies[-1])
        self.assertNotIn("top_k", captured_bodies[-1])
        self.assertNotIn("repeat_penalty", captured_bodies[-1])
        self.assertEqual(client_default.last_metrics["cache_hit_ratio"], 0.75)
        client_default.close()

        # Explicit sampling overrides -> included in request body
        client_custom = OpenAICompatClient(
            "http://127.0.0.1:8080/v1",
            "lfm3",
            sampling=SamplingParams(
                temperature=0.2,
                top_p=0.85,
                top_k=40,
                repeat_penalty=1.1,
            ),
            transport=transport,
        )
        client_custom.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(captured_bodies[-1]["temperature"], 0.2)
        self.assertEqual(captured_bodies[-1]["top_p"], 0.85)
        self.assertEqual(captured_bodies[-1]["top_k"], 40)
        self.assertEqual(captured_bodies[-1]["repeat_penalty"], 1.1)
        client_custom.close()

    def test_cli_report_generation_and_soft_mlflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            llm = GoldenPolicyLLM()
            payload, json_path, md_path = run_eval_suite(
                [
                    "--model",
                    "LiquidAI/LFM3-Omni-2B-GGUF:BF16",
                    "--history-mode",
                    "short",
                    "--context-mode",
                    "none",
                    "--prompt-mode",
                    "no_negative_priming",
                    "--cases",
                    "direct_composio_status,clean_t56",
                    "--repeats",
                    "2",
                    "--output-dir",
                    str(tmp_dir),
                ],
                llm=llm,
            )
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertEqual(payload["summary"]["total_runs"], 4)
            self.assertEqual(payload["summary"]["pass_all_rate"], 1.0)
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Model Eval Report", md_text)
            self.assertIn("direct_composio_status", md_text)
            self.assertIn("clean_t56_bedtime_create", md_text)

            # Verify --list-cases returns 0 without touching network
            rc = main(["--list-cases", "--cases", "polluted_replay"])
            self.assertEqual(rc, 0)

            # Verify soft-import graceful skip when mlflow is unavailable
            with patch.dict("sys.modules", {"mlflow": None}):
                run_id = log_to_mlflow(payload, json_path=json_path, md_path=md_path)
                self.assertIsNone(run_id)

            # Verify live per-case MLflow runs, step metrics, table, and traces against temp SQLite DB
            import mlflow  # type: ignore[import-untyped]
            from tracing import reset_tracing_for_tests

            mlflow_db = tmp_dir / "test_mlflow.db"
            mlflow_uri = f"sqlite:///{mlflow_db}"
            try:
                run_eval_suite(
                    [
                        "--model",
                        "LiquidAI/LFM3-Omni-2B-GGUF:BF16",
                        "--cases",
                        "direct_composio_status,clean_t56",
                        "--output-dir",
                        str(tmp_dir),
                        "--mlflow",
                        "--mlflow-uri",
                        mlflow_uri,
                    ],
                    llm=GoldenPolicyLLM(),
                )
                client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
                exp = client.get_experiment_by_name("baka-model-evals")
                self.assertIsNotNone(exp)
                runs = client.search_runs([exp.experiment_id])
                self.assertEqual(len(runs), 1)
                parent_run = runs[0]
                self.assertEqual(
                    parent_run.data.tags.get("eval.status"), "COMPLETED"
                )
                self.assertEqual(
                    parent_run.data.metrics.get("completed_cases"), 2.0
                )
                traces = client.search_traces(
                    locations=[exp.experiment_id],
                    run_id=parent_run.info.run_id,
                )
                self.assertEqual(len(traces), 2)
                self.assertEqual(len(traces[0].info.assessments), 5)
                datasets = mlflow.genai.search_datasets(
                    experiment_ids=[exp.experiment_id]
                )
                self.assertEqual(len(datasets), 1)
                self.assertEqual(datasets[0].name, "baka_eval_suite")
            finally:
                reset_tracing_for_tests()


if __name__ == "__main__":
    unittest.main()
