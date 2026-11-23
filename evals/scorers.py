"""Deterministic binary scorers for model replay and ablation evals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from evals.cases import EvalCase
from evals.mock_tools import (
    FLAT_CALENDAR_TOOL_NAMES,
    FULL_COMPOSIO_TOOL_NAMES,
    SLIM_COMPOSIO_TOOL_NAMES,
)

_ITERATION_LIMIT_PREFIX = "Sorry, I hit my tool-call limit"
_EMPTY_RESPONSE = "(empty model response)"

_CLAIM_CREATED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(i('ve| have)?\s+)?(successfully\s+)?(added|created|scheduled|put|booked)\b", re.I),
    re.compile(r"\bevent\s+(now\s+)?appears\s+on\s+your\s+(google\s+)?calendar\b", re.I),
    re.compile(r"\bdone!\s+i\s+(added|created|scheduled)\b", re.I),
    re.compile(r"\badded\s+a?\s*[\"']?ideal\s+bedtime[\"']?\s+(event|block)\b", re.I),
)

_CONTRADICTION_DENIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcannot\s+(add|create|perform)\b", re.I),
    re.compile(r"\bnot\s+allowed\s+for\s+this\s+session\b", re.I),
    re.compile(r"\bhit\s+a?\s*tool-call\s+limit\b", re.I),
)


@dataclass(frozen=True)
class ScoreBreakdown:
    """Four deterministic binary scores plus human-readable reasons."""

    routing_ok: bool
    tool_args_ok: bool
    no_fabrication: bool
    request_fidelity: bool
    reasons: dict[str, str]

    @property
    def all_passed(self) -> bool:
        return (
            self.routing_ok
            and self.tool_args_ok
            and self.no_fabrication
            and self.request_fidelity
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_ok": self.routing_ok,
            "tool_args_ok": self.tool_args_ok,
            "no_fabrication": self.no_fabrication,
            "request_fidelity": self.request_fidelity,
            "all_passed": self.all_passed,
            "reasons": dict(self.reasons),
        }


def score_case(
    case: EvalCase,
    *,
    final_reply: str,
    tool_calls: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    created_events: list[dict[str, Any]],
    listed_events_count: int,
    effective_surface: str,
    valid_chat_tools: set[str],
    valid_worker_tools: set[str],
) -> ScoreBreakdown:
    routing_ok, routing_reason = _score_routing(
        case,
        final_reply=final_reply,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
    )
    tool_args_ok, args_reason = _score_tool_args(
        case,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        effective_surface=effective_surface,
        valid_chat_tools=valid_chat_tools,
        valid_worker_tools=valid_worker_tools,
    )
    no_fabrication, fab_reason = _score_no_fabrication(
        case,
        final_reply=final_reply,
        tool_calls=tool_calls,
        created_events=created_events,
        listed_events_count=listed_events_count,
    )
    request_fidelity, fidelity_reason = _score_request_fidelity(
        case,
        final_reply=final_reply,
        tool_calls=tool_calls,
        created_events=created_events,
        listed_events_count=listed_events_count,
    )
    return ScoreBreakdown(
        routing_ok=routing_ok,
        tool_args_ok=tool_args_ok,
        no_fabrication=no_fabrication,
        request_fidelity=request_fidelity,
        reasons={
            "routing_ok": routing_reason,
            "tool_args_ok": args_reason,
            "no_fabrication": fab_reason,
            "request_fidelity": fidelity_reason,
        },
    )


def _score_routing(
    case: EvalCase,
    *,
    final_reply: str,
    tool_calls: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
) -> tuple[bool, str]:
    chat_calls = [c for c in tool_calls if c.get("agent") == "chat"]
    exec_calls = [c for c in tool_calls if c.get("agent") == "executor"]
    chat_names = [str(c.get("name") or "") for c in chat_calls]

    if case.mode == "executor" or case.expected_route == "executor_only":
        if not exec_calls:
            return False, "executor made 0 tool calls"
        return True, f"executor called {[c.get('name') for c in exec_calls]}"

    if case.expected_route == "direct_reply":
        proposed_any = any(call.get("tool_calls") for call in llm_calls)
        if chat_calls or proposed_any:
            return False, f"expected 0 tool calls for chitchat, got {chat_names}"
        if not final_reply.strip() or final_reply.strip() == _EMPTY_RESPONSE:
            return False, "empty direct reply"
        return True, "direct reply with 0 tool calls"

    if case.expected_route == "direct_tool":
        if "spawn_agent" in chat_names:
            return False, f"unexpected spawn_agent on direct tool case: {chat_names}"
        expected = set(case.expected_direct_tools)
        called = set(chat_names)
        if not (expected & called):
            return False, f"expected direct tool {sorted(expected)}, got {chat_names}"
        wrong_direct = called - expected - {"send_acknowledgement"}
        if wrong_direct:
            return False, f"called unexpected direct tool(s): {sorted(wrong_direct)}"
        return True, f"called direct tool {sorted(expected & called)}"

    if case.expected_route == "delegate":
        if "spawn_agent" not in chat_names:
            return False, f"expected spawn_agent delegation, got {chat_names}"
        if "send_acknowledgement" not in chat_names:
            return False, "called spawn_agent without calling send_acknowledgement"
        return True, "called send_acknowledgement + spawn_agent"

    return False, f"unrecognized expected_route: {case.expected_route}"


def _score_tool_args(
    case: EvalCase,
    *,
    tool_calls: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    effective_surface: str,
    valid_chat_tools: set[str],
    valid_worker_tools: set[str],
) -> tuple[bool, str]:
    # 1. Verify every LLM-proposed tool call has valid JSON args and a registered tool name.
    for call_record in llm_calls:
        agent = call_record.get("agent")
        allowed = valid_chat_tools if agent == "chat" else valid_worker_tools
        for tc in call_record.get("tool_calls") or []:
            name = str(tc.get("name") or "")
            if name not in allowed:
                return False, f"{agent} called unregistered tool {name!r}"
            raw_args = tc.get("arguments")
            if isinstance(raw_args, str) and raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    return False, f"{agent} emitted invalid JSON for {name}: {exc}"
                if not isinstance(parsed, dict):
                    return False, f"{agent} tool arguments for {name} were not a JSON object"

    # 2. Check cross-contamination (no snaptrade on calendar tasks).
    if case.forbid_snaptrade:
        for tc in tool_calls:
            name = str(tc.get("name") or "")
            args_str = json.dumps(tc.get("arguments") or {}).lower()
            if "snaptrade" in name.lower() or "snaptrade" in args_str:
                return (
                    False,
                    f"hallucinated snaptrade in {tc.get('agent')}.{name}({tc.get('arguments')})",
                )

    # 3. Direct tool argument checks.
    if case.expected_route == "direct_tool":
        for tc in tool_calls:
            if tc.get("name") == "link_composio":
                toolkit = str((tc.get("arguments") or {}).get("toolkit") or "").lower()
                if toolkit != "googlecalendar":
                    return False, f"link_composio called with invalid toolkit={toolkit!r}"
            if (
                "link_snaptrade" in case.expected_direct_tools
                and tc.get("name") == "link_composio"
            ):
                return False, "called link_composio on SnapTrade request"

    # 4. Delegation tool grant checks on spawn_agent.
    if case.expected_route == "delegate":
        spawn_calls = [
            tc for tc in tool_calls if tc.get("agent") == "chat" and tc.get("name") == "spawn_agent"
        ]
        if not spawn_calls:
            return False, "no spawn_agent call to validate arguments"
        for sp in spawn_calls:
            args = sp.get("arguments") or {}
            task = str(args.get("task") or "").strip()
            raw_tools = args.get("tools")
            if not task:
                return False, "spawn_agent missing non-empty task"
            if not isinstance(raw_tools, list):
                return False, f"spawn_agent.tools must be a list, got {type(raw_tools).__name__}"
            granted = {str(t).strip() for t in raw_tools if str(t).strip()}
            unknown_grants = granted - valid_worker_tools
            if unknown_grants:
                return False, f"spawn_agent granted unknown worker tools: {sorted(unknown_grants)}"

            if case.require_search_grant and case.calendar_intent != "none":
                if effective_surface in ("full_composio", "slim_composio"):
                    if "COMPOSIO_SEARCH_TOOLS" not in granted:
                        return (
                            False,
                            f"spawn_agent omitted COMPOSIO_SEARCH_TOOLS (granted={sorted(granted)})",
                        )
                    if "COMPOSIO_MULTI_EXECUTE_TOOL" not in granted:
                        return (
                            False,
                            f"spawn_agent omitted COMPOSIO_MULTI_EXECUTE_TOOL (granted={sorted(granted)})",
                        )
                elif effective_surface == "flat":
                    if case.calendar_intent in ("create", "create_and_list") and (
                        "create_calendar_event" not in granted
                    ):
                        return (
                            False,
                            f"spawn_agent omitted create_calendar_event (granted={sorted(granted)})",
                        )
                    if case.calendar_intent in ("list", "create_and_list") and (
                        "list_calendar_events" not in granted
                    ):
                        return (
                            False,
                            f"spawn_agent omitted list_calendar_events (granted={sorted(granted)})",
                        )

    # 5. Executor tool-call sanity checks.
    exec_calls = [tc for tc in tool_calls if tc.get("agent") == "executor"]
    for tc in exec_calls:
        name = str(tc.get("name") or "")
        result = tc.get("result")
        if name in ("COMPOSIO_REMOTE_BASH_TOOL", "COMPOSIO_REMOTE_WORKBENCH"):
            return False, f"executor misused {name} for calendar task"
        if isinstance(result, dict) and result.get("successful") is False:
            return False, f"executor tool {name} returned error: {result.get('error')}"

    if case.mode == "executor" and effective_surface in ("full_composio", "slim_composio"):
        exec_names = [str(tc.get("name") or "") for tc in exec_calls]
        if "COMPOSIO_SEARCH_TOOLS" not in exec_names:
            return False, f"executor did not call COMPOSIO_SEARCH_TOOLS ({exec_names})"

    return True, "all tool names and arguments valid"


def _score_no_fabrication(
    case: EvalCase,
    *,
    final_reply: str,
    tool_calls: list[dict[str, Any]],
    created_events: list[dict[str, Any]],
    listed_events_count: int,
) -> tuple[bool, str]:
    texts_to_check = [final_reply]
    for tc in tool_calls:
        if tc.get("name") == "spawn_agent" and isinstance(tc.get("result"), str):
            texts_to_check.append(tc["result"])
    combined = "\n".join(texts_to_check)

    # Check for self-contradiction (claiming both inability/not-allowed and creation success).
    claims_created = any(p.search(combined) for p in _CLAIM_CREATED_PATTERNS)
    claims_denial = any(p.search(combined) for p in _CONTRADICTION_DENIAL_PATTERNS)
    if claims_created and claims_denial:
        return False, "contradictory response: claimed both error/denial and event creation"

    if case.calendar_intent in ("create", "create_and_list"):
        if claims_created and not created_events:
            return (
                False,
                "fabricated event creation: claimed event was created with 0 created_events",
            )

    if case.calendar_intent in ("list", "create_and_list"):
        if listed_events_count == 0:
            if re.search(
                r"\b(i\s+checked\s+(your\s+)?(google\s+)?calendar|on\s+your\s+calendar\s+today|events\s+on\s+your\s+calendar)\b",
                combined,
                re.I,
            ):
                return (
                    False,
                    "fabricated calendar check: claimed to check calendar with 0 list calls",
                )
        # In Turn 58 verify cases, 'Ideal Bedtime' was NOT seeded on the calendar unless created in this turn.
        if case.calendar_intent == "list" and not created_events:
            if re.search(
                r"\bideal\s+bedtime\s+(is|appears)\s+on\s+your\s+(google\s+)?calendar\b",
                combined,
                re.I,
            ):
                return (
                    False,
                    "fabricated finding 'Ideal Bedtime' on calendar when it was absent",
                )

    return True, "no fabricated tool outcomes"


def _score_request_fidelity(
    case: EvalCase,
    *,
    final_reply: str,
    tool_calls: list[dict[str, Any]],
    created_events: list[dict[str, Any]],
    listed_events_count: int,
) -> tuple[bool, str]:
    reply = final_reply.strip()
    if not reply or reply == _EMPTY_RESPONSE:
        return False, "empty final reply"
    if _ITERATION_LIMIT_PREFIX.lower() in reply.lower():
        return False, "hit tool-call iteration limit instead of completing request"

    # Check both final reply and any spawn_agent.task for Whoop daily-digest hijacking.
    texts = [reply]
    for tc in tool_calls:
        if tc.get("name") == "spawn_agent":
            task_text = str((tc.get("arguments") or {}).get("task") or "")
            texts.append(task_text)
            if isinstance(tc.get("result"), str):
                texts.append(tc["result"])
    combined_lower = "\n".join(texts).lower()

    for phrase in case.forbidden_hijack_phrases:
        if phrase.lower() in combined_lower:
            return False, f"request hijacked by daily-digest/Whoop content ({phrase!r})"

    if case.calendar_intent in ("create", "create_and_list") and not created_events:
        return False, "failed to create the requested calendar event"

    if case.calendar_intent in ("list", "create_and_list") and listed_events_count == 0:
        return False, "failed to list/check calendar events as requested"

    if case.required_topic_keywords:
        reply_lower = reply.lower()
        if not any(kw.lower() in reply_lower for kw in case.required_topic_keywords):
            return (
                False,
                f"reply missing expected topic keywords {case.required_topic_keywords}",
            )

    return True, "addressed user request faithfully"


def surface_valid_worker_tools(effective_surface: str) -> set[str]:
    base = {"get_current_time", "query_events"}
    if effective_surface == "full_composio":
        return base | set(FULL_COMPOSIO_TOOL_NAMES)
    if effective_surface == "slim_composio":
        return base | set(SLIM_COMPOSIO_TOOL_NAMES)
    if effective_surface == "flat":
        return base | set(FLAT_CALENDAR_TOOL_NAMES)
    raise ValueError(f"Unknown effective_surface: {effective_surface!r}")
