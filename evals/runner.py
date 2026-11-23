"""Case runner that wires ChatAgent / ExecutorAgent to mocked tools and scores traces."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agents.chat_agent import ChatAgent
from agents.executor_agent import ExecutorAgent
from evals.ablations import (
    AblationConfig,
    build_chat_system_prompt,
    build_executor_system_prompt,
    select_history_messages,
    select_memory_context,
    surface_worker_tool_names,
)
from evals.cases import EvalCase
from evals.client import RecordingLLM, usage_totals
from evals.fixtures import EVAL_SESSION_ID
from evals.mock_tools import (
    MockWorld,
    StaticContextBuilder,
    register_chat_tools,
    register_worker_tools,
)
from evals.scorers import ScoreBreakdown, score_case, surface_valid_worker_tools
from llm.base import LLMClient
from tools.registry import ToolRegistry
from tools.session_context import reset_session_id, set_session_id
from tools.spawn_agent import SpawnAgentTool


class RecordingSpawnAgentTool(SpawnAgentTool):
    """Wraps production ``SpawnAgentTool`` to log call & result on ``MockWorld``."""

    def __init__(
        self,
        executor: ExecutorAgent,
        world: MockWorld,
        context_builder: StaticContextBuilder | None = None,
    ) -> None:
        super().__init__(executor, context_builder)  # type: ignore[arg-type]
        self._world = world

    def execute(self, **kwargs: Any) -> str:
        result = super().execute(**kwargs)
        self._world.record_call("chat", self.name, kwargs, result)
        return result


@dataclass
class CaseRunResult:
    """Complete trace, usage metrics, and binary scores for one case execution."""

    case_id: str
    category: str
    mode: str
    repeat: int
    effective_surface: str
    ablation: AblationConfig
    replies: list[str]
    final_reply: str
    scores: ScoreBreakdown
    prompt_tokens: int
    completion_tokens: int
    cache_tokens: int
    cache_hit_ratio: float | None
    max_prompt_tokens: int
    llm_calls_count: int
    wall_ms: float
    tool_calls: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    created_events: list[dict[str, Any]]
    listed_events_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "mode": self.mode,
            "repeat": self.repeat,
            "effective_surface": self.effective_surface,
            "ablation": {
                "history_mode": self.ablation.history_mode,
                "context_mode": self.ablation.context_mode,
                "prompt_mode": self.ablation.prompt_mode,
                "tool_surface": self.ablation.tool_surface,
            },
            "replies": list(self.replies),
            "final_reply": self.final_reply,
            "scores": self.scores.to_dict(),
            "metrics": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cache_tokens": self.cache_tokens,
                "cache_hit_ratio": self.cache_hit_ratio,
                "max_prompt_tokens": self.max_prompt_tokens,
                "llm_calls": self.llm_calls_count,
                "wall_ms": self.wall_ms,
                "tool_calls": len(self.tool_calls),
                "created_events": len(self.created_events),
                "listed_events_count": self.listed_events_count,
            },
            "tool_calls": list(self.tool_calls),
            "llm_calls": list(self.llm_calls),
            "created_events": list(self.created_events),
        }


def resolve_tool_surface(case: EvalCase, config: AblationConfig) -> str:
    """Determine the effective worker tool surface for ``case``."""
    if config.override_case_surface or not case.tool_surface:
        return config.tool_surface
    return case.tool_surface


def run_case(
    case: EvalCase,
    llm: LLMClient,
    *,
    config: AblationConfig | None = None,
    repeat: int = 1,
    max_chat_iterations: int = 5,
    max_executor_iterations: int = 8,
) -> CaseRunResult:
    """Execute one ``EvalCase`` offline against ``llm`` and score the result."""
    ablation = config or AblationConfig()
    effective_surface = resolve_tool_surface(case, ablation)

    world = MockWorld(
        composio_linked=case.composio_linked,
        snaptrade_linked=case.snaptrade_linked,
    )
    registry = ToolRegistry()
    register_worker_tools(registry, world, tool_surface=effective_surface)
    register_chat_tools(registry, world)

    llm_calls: list[dict[str, Any]] = []
    chat_llm = RecordingLLM(llm, "chat", llm_calls)
    exec_llm = RecordingLLM(llm, "executor", llm_calls)

    exec_prompt = build_executor_system_prompt(ablation.prompt_mode, effective_surface)
    executor = ExecutorAgent(
        llm=exec_llm,
        tools=registry,
        system_prompt=exec_prompt,
        max_agent_iterations=max_executor_iterations,
    )

    rendered_context = select_memory_context(case.memory_context, ablation.context_mode)
    context_builder = (
        StaticContextBuilder(rendered_context) if rendered_context else None
    )
    registry.register(
        RecordingSpawnAgentTool(executor, world, context_builder=context_builder)
    )

    valid_chat_tools = set(registry.names(chat_agent=True))
    valid_worker_tools = surface_valid_worker_tools(effective_surface)

    started = time.perf_counter()
    replies: list[str] = []

    try:
        from tracing import flush_traces, trace_span
    except ImportError:
        from contextlib import nullcontext

        def trace_span(*_a: Any, **_kw: Any) -> Any:  # type: ignore[misc]
            return nullcontext(None)

        def flush_traces() -> None:
            return None

    with trace_span(
        f"eval_case:{case.id}",
        span_type="EVALUATOR",
        inputs={
            "case_id": case.id,
            "category": case.category,
            "mode": case.mode,
            "repeat": repeat,
            "user_text": case.user_text,
            "effective_surface": effective_surface,
            "ablation": {
                "history_mode": ablation.history_mode,
                "context_mode": ablation.context_mode,
                "prompt_mode": ablation.prompt_mode,
                "tool_surface": ablation.tool_surface,
            },
        },
        attributes={
            "case_id": case.id,
            "category": case.category,
            "mode": case.mode,
            "repeat": repeat,
            "effective_surface": effective_surface,
        },
    ) as case_span:
        if case.mode == "executor":
            granted = (
                list(case.executor_granted_tools)
                if case.executor_granted_tools is not None
                else surface_worker_tool_names(effective_surface)
            )
            token = set_session_id(EVAL_SESSION_ID)
            try:
                final_reply = executor.run_task(
                    EVAL_SESSION_ID,
                    case.user_text,
                    tool_names=granted,
                )
            finally:
                reset_session_id(token)
            replies = [final_reply]
        else:
            chat_prompt = build_chat_system_prompt(ablation.prompt_mode, effective_surface)
            chat_agent = ChatAgent(
                llm=chat_llm,
                tools=registry,
                system_prompt=chat_prompt,
                max_history_messages=40,
                max_agent_iterations=max_chat_iterations,
                context_builder=context_builder,  # type: ignore[arg-type]
            )
            prior_messages = select_history_messages(
                case.prior_turns, ablation.history_mode
            )
            if prior_messages:
                chat_agent._histories[EVAL_SESSION_ID] = [
                    {"role": "system", "content": chat_agent._render_system_prompt()},
                    *prior_messages,
                ]
            replies = list(chat_agent.run_turn(EVAL_SESSION_ID, case.user_text))
            final_reply = replies[-1] if replies else ""

        wall_ms = round((time.perf_counter() - started) * 1000.0, 2)
        totals = usage_totals(llm_calls)

        scores = score_case(
            case,
            final_reply=final_reply,
            tool_calls=world.tool_calls,
            llm_calls=llm_calls,
            created_events=world.created_events,
            listed_events_count=world.listed_events_count,
            effective_surface=effective_surface,
            valid_chat_tools=valid_chat_tools,
            valid_worker_tools=valid_worker_tools,
        )

        if case_span is not None:
            status_label = "PASS" if scores.all_passed else "FAIL"
            case_span.set_attributes(
                {
                    "eval.result": status_label,
                    "eval.routing_ok": scores.routing_ok,
                    "eval.tool_args_ok": scores.tool_args_ok,
                    "eval.no_fabrication": scores.no_fabrication,
                    "eval.request_fidelity": scores.request_fidelity,
                    "eval.prompt_tokens": totals["prompt_tokens"],
                    "eval.completion_tokens": totals["completion_tokens"],
                    "eval.cache_hit_ratio": totals["cache_hit_ratio"],
                    "eval.wall_ms": wall_ms,
                }
            )
            case_span.set_outputs(
                {
                    "result": status_label,
                    "final_reply": final_reply,
                    "scores": scores.to_dict(),
                    "tool_calls": [tc.get("name") for tc in world.tool_calls],
                    "created_events": len(world.created_events),
                    "listed_events_count": world.listed_events_count,
                }
            )
            case_span.update_trace(
                tags={
                    "case_id": case.id,
                    "category": case.category,
                    "surface": effective_surface,
                    "result": status_label,
                    "routing_ok": str(scores.routing_ok),
                    "tool_args_ok": str(scores.tool_args_ok),
                    "no_fabrication": str(scores.no_fabrication),
                    "request_fidelity": str(scores.request_fidelity),
                },
                request_preview=f"[{case.id}] {case.user_text}",
                response_preview=f"[{status_label}] {final_reply}",
            )
    flush_traces()

    return CaseRunResult(
        case_id=case.id,
        category=case.category,
        mode=case.mode,
        repeat=repeat,
        effective_surface=effective_surface,
        ablation=ablation,
        replies=replies,
        final_reply=final_reply,
        scores=scores,
        prompt_tokens=totals["prompt_tokens"],
        completion_tokens=totals["completion_tokens"],
        cache_tokens=totals["cache_tokens"],
        cache_hit_ratio=totals["cache_hit_ratio"],
        max_prompt_tokens=totals["max_prompt_tokens"],
        llm_calls_count=totals["llm_calls"],
        wall_ms=wall_ms,
        tool_calls=world.tool_calls,
        llm_calls=llm_calls,
        created_events=world.created_events,
        listed_events_count=world.listed_events_count,
    )
