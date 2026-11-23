"""Curated PII-free benchmark suite (~12 cases) for model replay & ablation evals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evals.fixtures import (
    POLLUTED_CONTEXT_BLOCK,
    turns_before_56,
    turns_before_57,
    turns_before_58,
)


@dataclass(frozen=True)
class EvalCase:
    """One deterministic replay or ablation benchmark case."""

    id: str
    category: str  # polluted_replay | clean_equivalent | direct_routing | executor_comparison
    description: str
    mode: str  # "chat" | "executor"
    user_text: str
    prior_turns: list[list[dict[str, Any]]] = field(default_factory=list)
    memory_context: str = ""
    composio_linked: bool = True
    snaptrade_linked: bool = False
    tool_surface: str | None = None
    executor_granted_tools: list[str] | None = None
    # Expectations for deterministic binary scorers:
    expected_route: str = "delegate"  # delegate | direct_tool | direct_reply
    expected_direct_tools: tuple[str, ...] = ()
    calendar_intent: str = "none"  # create | list | create_and_list | none
    forbid_snaptrade: bool = True
    require_search_grant: bool = False
    required_topic_keywords: tuple[str, ...] = ()
    forbidden_hijack_phrases: tuple[str, ...] = (
        "weightlifting strain",
        "strain of 9.3",
        "daily-digest",
        "here's a quick daily-digest",
        "sleep was 5.4 hours",
    )


def all_cases() -> list[EvalCase]:
    """Return the full curated 12-case benchmark suite."""
    return [
        # --- 1. Reconstructed polluted-history replays (Turns 56, 57, 58) ---
        EvalCase(
            id="replay_t56_polluted_bedtime_create",
            category="polluted_replay",
            description="Turn 56 replay: polluted history (Whoop digest + prior status loops) + <context>; add Ideal Bedtime block.",
            mode="chat",
            user_text="Can you add a block on my calendar for ideal bedtime based on Whoop recovery?",
            prior_turns=turns_before_56(),
            memory_context=POLLUTED_CONTEXT_BLOCK,
            expected_route="delegate",
            calendar_intent="create",
            forbid_snaptrade=True,
            require_search_grant=True,
            required_topic_keywords=("bedtime", "calendar", "10:", "22:", "added", "scheduled", "created"),
        ),
        EvalCase(
            id="replay_t57_polluted_retry_create",
            category="polluted_replay",
            description="Turn 57 replay: 'Can you try again?' after Turn 56 tool-limit failure; must not fabricate event creation or cite SnapTrade.",
            mode="chat",
            user_text="Can you try again?",
            prior_turns=turns_before_57(),
            memory_context=POLLUTED_CONTEXT_BLOCK,
            expected_route="delegate",
            calendar_intent="create",
            forbid_snaptrade=True,
            require_search_grant=True,
            required_topic_keywords=("bedtime", "calendar", "10:", "22:", "added", "scheduled", "created"),
        ),
        EvalCase(
            id="replay_t58_polluted_verify_calendar",
            category="polluted_replay",
            description="Turn 58 replay: 'I can't see it. Can you check again? I think you hit an error with composio'; must list calendar events instead of hijacking with Whoop daily digest.",
            mode="chat",
            user_text="I can’t see it. Can you check again? I think you hit an error with composio",
            prior_turns=turns_before_58(),
            memory_context=POLLUTED_CONTEXT_BLOCK,
            expected_route="delegate",
            calendar_intent="list",
            forbid_snaptrade=True,
            require_search_grant=True,
            required_topic_keywords=("calendar", "standup", "design sync", "bedtime", "events"),
        ),
        # --- 2. Clean-history equivalents (no daily-digest turn, no <context>) ---
        EvalCase(
            id="clean_t56_bedtime_create",
            category="clean_equivalent",
            description="Clean-history equivalent of Turn 56: add Ideal Bedtime block at 10:30 PM on Wed Nov 11 based on 53% Whoop recovery.",
            mode="chat",
            user_text=(
                "Can you add a block on my calendar for tonight (Wed Nov 11) at "
                "10:30 PM titled 'Ideal Bedtime' based on my 53% Whoop recovery?"
            ),
            prior_turns=[],
            memory_context="",
            expected_route="delegate",
            calendar_intent="create",
            forbid_snaptrade=True,
            require_search_grant=True,
            required_topic_keywords=("bedtime", "calendar", "10:30", "22:30", "added", "scheduled", "created"),
        ),
        EvalCase(
            id="clean_t57_retry_bedtime_create",
            category="clean_equivalent",
            description="Clean-history equivalent of Turn 57: direct request to retry creating the 10:30 PM Ideal Bedtime event on Google Calendar.",
            mode="chat",
            user_text=(
                "Please create a Google Calendar event titled 'Ideal Bedtime' "
                "for Wed Nov 11, 2026 from 10:30 PM to 11:00 PM CST."
            ),
            prior_turns=[],
            memory_context="",
            expected_route="delegate",
            calendar_intent="create",
            forbid_snaptrade=True,
            require_search_grant=True,
            required_topic_keywords=("bedtime", "calendar", "10:30", "22:30", "added", "scheduled", "created"),
        ),
        EvalCase(
            id="clean_t58_verify_calendar",
            category="clean_equivalent",
            description="Clean-history equivalent of Turn 58: check Google Calendar for today's events to see if Ideal Bedtime is actually there.",
            mode="chat",
            user_text=(
                "I can't see the Ideal Bedtime event on my Google Calendar. "
                "Can you check my calendar for today (Wed Nov 11, 2026) and tell "
                "me what events are actually listed?"
            ),
            prior_turns=[],
            memory_context="",
            expected_route="delegate",
            calendar_intent="list",
            forbid_snaptrade=True,
            require_search_grant=True,
            required_topic_keywords=("standup", "design sync", "calendar", "bedtime"),
        ),
        # --- 3. Direct ChatAgent tool routing cases ---
        EvalCase(
            id="direct_composio_status",
            category="direct_routing",
            description="Direct account status check: 'Is my Google Calendar connected?' -> composio_status without spawning an agent or touching SnapTrade.",
            mode="chat",
            user_text="Is my Google Calendar connected right now?",
            composio_linked=True,
            expected_route="direct_tool",
            expected_direct_tools=("composio_status",),
            calendar_intent="none",
            forbid_snaptrade=True,
            require_search_grant=False,
            required_topic_keywords=("connected", "linked", "calendar"),
        ),
        EvalCase(
            id="direct_link_composio",
            category="direct_routing",
            description="Direct Connect Link request when Google Calendar is not linked -> link_composio(toolkit='googlecalendar').",
            mode="chat",
            user_text="Can you send me a link to connect my Google Calendar?",
            composio_linked=False,
            expected_route="direct_tool",
            expected_direct_tools=("link_composio",),
            calendar_intent="none",
            forbid_snaptrade=True,
            require_search_grant=False,
            required_topic_keywords=("connect.composio.dev", "link", "calendar", "connect"),
        ),
        EvalCase(
            id="direct_link_snaptrade",
            category="direct_routing",
            description="Direct brokerage link request -> link_snaptrade without cross-contaminating link_composio.",
            mode="chat",
            user_text="Can you link my brokerage portfolio with SnapTrade?",
            snaptrade_linked=False,
            expected_route="direct_tool",
            expected_direct_tools=("link_snaptrade",),
            calendar_intent="none",
            forbid_snaptrade=False,
            require_search_grant=False,
            required_topic_keywords=("snaptrade", "link", "approve", "oauth", "portfolio", "brokerage"),
        ),
        EvalCase(
            id="direct_chitchat_no_tools",
            category="direct_routing",
            description="Simple conversational chitchat -> direct reply with 0 tool calls.",
            mode="chat",
            user_text="Hey! Thanks for the help earlier, hope you're having a good Wednesday.",
            expected_route="direct_reply",
            expected_direct_tools=(),
            calendar_intent="none",
            forbid_snaptrade=True,
            require_search_grant=False,
            required_topic_keywords=(),
        ),
        # --- 4. Executor-only comparison cases (Composio full vs slim vs flat) ---
        EvalCase(
            id="executor_composio_full_create_event",
            category="executor_comparison",
            description="Executor-only with full 6-tool Composio surface: search tools then create 'Ideal Bedtime' on Wed Nov 11 at 10:30 PM CST.",
            mode="executor",
            tool_surface="full_composio",
            user_text=(
                "Create a Google Calendar event titled 'Ideal Bedtime' on "
                "Wednesday, Nov 11, 2026 from 22:30 to 23:00 America/Chicago "
                "(2026-11-11T22:30:00-06:00 to 2026-11-11T23:00:00-06:00)."
            ),
            expected_route="executor_only",
            calendar_intent="create",
            forbid_snaptrade=True,
            require_search_grant=False,
            required_topic_keywords=("ideal bedtime", "10:30", "22:30", "created", "added", "scheduled"),
        ),
        EvalCase(
            id="executor_composio_slim_create_event",
            category="executor_comparison",
            description="Executor-only with slim 3-tool Composio surface (no bash/workbench/get_schemas): search tools then create 'Ideal Bedtime'.",
            mode="executor",
            tool_surface="slim_composio",
            user_text=(
                "Create a Google Calendar event titled 'Ideal Bedtime' on "
                "Wednesday, Nov 11, 2026 from 22:30 to 23:00 America/Chicago "
                "(2026-11-11T22:30:00-06:00 to 2026-11-11T23:00:00-06:00)."
            ),
            expected_route="executor_only",
            calendar_intent="create",
            forbid_snaptrade=True,
            require_search_grant=False,
            required_topic_keywords=("ideal bedtime", "10:30", "22:30", "created", "added", "scheduled"),
        ),
        EvalCase(
            id="executor_flat_create_and_list",
            category="executor_comparison",
            description="Executor-only with flat direct calendar tools (create_calendar_event, list_calendar_events): create 'Ideal Bedtime' and list today's events.",
            mode="executor",
            tool_surface="flat",
            user_text=(
                "Create a Google Calendar event titled 'Ideal Bedtime' on "
                "Wednesday, Nov 11, 2026 from 22:30 to 23:00 America/Chicago "
                "(2026-11-11T22:30:00-06:00 to 2026-11-11T23:00:00-06:00), "
                "and list all events on the calendar for 2026-11-11."
            ),
            expected_route="executor_only",
            calendar_intent="create_and_list",
            forbid_snaptrade=True,
            require_search_grant=False,
            required_topic_keywords=("ideal bedtime", "standup", "design sync"),
        ),
    ]


def filter_cases(
    cases: list[EvalCase],
    pattern: str | None = None,
) -> list[EvalCase]:
    """Filter cases by comma-separated substring match on ``id`` or ``category``."""
    if not pattern or not pattern.strip():
        return list(cases)
    tokens = [tok.strip().lower() for tok in pattern.split(",") if tok.strip()]
    if not tokens:
        return list(cases)
    return [
        case
        for case in cases
        if any(tok in case.id.lower() or tok in case.category.lower() for tok in tokens)
    ]
