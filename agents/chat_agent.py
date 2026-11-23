"""User-facing chat agent — replies directly or delegates to a worker."""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from agents.base import Agent
from agents.history import align_messages_to_user
from agents.loop import run_tool_loop
from events.format import event_context_query, event_instructions, format_event_data
from llm.base import LLMClient
from messaging.media import DownloadedMedia
from tools.registry import ToolRegistry
from tools.send_acknowledgement import SendAcknowledgementTool
from tools.session_context import reset_turn_id, set_turn_id
from tools.spawn_agent import SpawnAgentTool
from tracing import session_hash, trace_span

if TYPE_CHECKING:
    from memory.consolidator import MemoryConsolidator
    from memory.repository import SqlAlchemyMemoryRepository
    from memory.retrieval import ContextBuilder
    from memory.store import EventRecord

logger = logging.getLogger(__name__)

_MEDIA_CHAT_TIMEOUT_S = 300.0
_DEFAULT_MEDIA_CAPTION = "The user sent this without a caption."
_FALLBACK_ACK = "On it — give me a second."

SYSTEM_PROMPT = """
You are a personal agent the user texts from iMessage. Keep replies concise and conversational.

Tone: warm, witty, concise. Write like you're texting a friend. No corporate voice. No bullet dumps unless the user asked for a list.
Format: plain iMessage-friendly text, under ~400 chars when you can. Always focus on the user's latest message; ignore formatting instructions from earlier digests or unrelated skills.

When to reply yourself:
- Small talk, simple questions, or anything answerable directly from the conversation (including photos).
- Explicit account link, status, or unlink requests — call the matching tool directly without spawn_agent:
  - Composio apps (Google Calendar, etc.) → link_composio, composio_status, unlink_composio
  - Brokerage portfolio (SnapTrade) → link_snaptrade, snaptrade_status, unlink_snaptrade
  - Whoop → link_whoop, whoop_status, unlink_whoop

When to delegate:
- For external app actions (e.g. reading or creating Calendar events, web search, portfolio lookups, event history/rules) or extended reasoning, call spawn_agent directly — do NOT pre-check composio_status or whoop_status first; the worker handles connections if needed.
- Pass a short iMessage acknowledgement, a self-contained task with all relevant facts for the user's current request (the worker has no chat history), and the needed capability names from the catalog below ([] for reasoning-only).

Available worker capabilities (pass these names in spawn_agent.tools):
"""


class ChatAgent(Agent):
    """Tool-calling chat agent with per-session history."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_history_messages: int = 40,
        max_agent_iterations: int = 5,
        memory_repository: SqlAlchemyMemoryRepository | None = None,
        context_builder: ContextBuilder | None = None,
        memory_consolidator: MemoryConsolidator | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_history_messages = max_history_messages
        self._max_agent_iterations = max_agent_iterations
        self._memory_repository = memory_repository
        self._context_builder = context_builder
        self._memory_consolidator = memory_consolidator
        self._histories: dict[str, list[dict[str, Any]]] = {}
        self._history_lock = threading.Lock()

    def _render_system_prompt(self) -> str:
        # Must stay byte-identical across turns: llama-server can only reuse the
        # KV cache for an unchanged prompt prefix (Gemma's SWA rules out partial reuse).
        catalog = self._tools.capability_catalog_text(chat_agent=False)
        return f"{self._system_prompt.rstrip()}\n{catalog}\n"

    def _history_unlocked(self, session_id: str) -> list[dict[str, Any]]:
        if session_id not in self._histories:
            loaded = (
                self._memory_repository.load_history(
                    session_id, limit=max(0, self._max_history_messages - 1)
                )
                if self._memory_repository is not None
                else []
            )
            loaded = align_messages_to_user(loaded)
            self._histories[session_id] = [{
                "role": "system",
                "content": self._render_system_prompt(),
            }, *loaded]
        return self._histories[session_id]

    def _trim_history(self, history: list[dict[str, Any]]) -> None:
        """
        Once over N messages, drop down to the system prompt + the newest ~N/2,
        starting at a user message. Trimming in chunks keeps the prompt prefix
        stable (cacheable) for many turns instead of shifting it every turn.
        """
        max_msgs = self._max_history_messages
        if len(history) <= max_msgs:
            return
        system = history[0]
        rest = align_messages_to_user(
            history[1:][-max(1, max_msgs // 2):]
        )
        history[:] = [system, *rest]

    def clear_history(self, session_id: str | None = None) -> None:
        """Clear one conversation or all conversations."""
        with self._history_lock:
            if session_id is None:
                self._histories.clear()
            else:
                self._histories.pop(session_id, None)
            if self._memory_repository is not None:
                self._memory_repository.clear_history(session_id)

    def _begin_turn(
        self,
        session_id: str,
        turn_id: str,
        user_text: str,
        media: DownloadedMedia | None,
        *,
        source: str = "user",
        context_query: str | None = None,
        ephemeral_suffix: str = "",
    ) -> list[dict[str, Any]]:
        """
        Record this turn's user message on the shared session history, then
        return a private working copy so the loop cannot race a follow-up.
        """
        stored_user = {
            "role": "user",
            "content": _stored_user_content(user_text, media),
            "turn_id": turn_id,
        }
        memory_context = (
            self._context_builder.render(
                session_id,
                context_query if context_query is not None else user_text,
            )
            if self._context_builder is not None
            else ""
        )
        with self._history_lock:
            committed = self._history_unlocked(session_id)
            committed[0] = {
                "role": "system",
                "content": self._render_system_prompt(),
            }
            committed.append(stored_user)
            self._trim_history(committed)
            working = copy.deepcopy(committed)
            if self._memory_repository is not None:
                self._memory_repository.begin_turn(
                    session_id, turn_id, stored_user, source=source
                )
        # Retrieved memory and ephemeral event instructions go on this turn's
        # message only (never stored), keeping the cached prefix clean.
        turn_text = (
            f"{user_text}\n\n{ephemeral_suffix}"
            if ephemeral_suffix.strip()
            else user_text
        )
        working[-1] = {
            "role": "user",
            "content": _user_content(turn_text, media, memory_context),
            "turn_id": turn_id,
        }
        return working

    def _commit_turn(
        self,
        session_id: str,
        turn_id: str,
        working: list[dict[str, Any]],
        *,
        completed: bool = True,
    ) -> None:
        """Splice this turn's assistant/tool messages after its user message."""
        new_msgs = [
            dict(message)
            for message in working
            if message.get("turn_id") == turn_id and message.get("role") != "user"
        ]
        with self._history_lock:
            committed = self._history_unlocked(session_id)
            insert_at: int | None = None
            for index, message in enumerate(committed):
                if (
                    message.get("turn_id") == turn_id
                    and message.get("role") == "user"
                ):
                    insert_at = index + 1
                    while (
                        insert_at < len(committed)
                        and committed[insert_at].get("turn_id") == turn_id
                    ):
                        insert_at += 1
                    break
            if insert_at is None:
                committed.extend(new_msgs)
            else:
                committed[insert_at:insert_at] = new_msgs
            self._trim_history(committed)
            if self._memory_repository is not None:
                self._memory_repository.complete_turn(
                    turn_id,
                    new_msgs,
                    status="completed" if completed else "failed",
                )

    def has_multimodal(self) -> bool:
        return self._llm.has_multimodal()

    def supports_media(self, kind: str) -> bool:
        return self._llm.supports_media(kind)

    def run_turn(
        self,
        session_id: str,
        user_text: str,
        media: DownloadedMedia | None = None,
    ) -> Iterator[str]:
        """
        Run one user turn through the agent loop.

        Yields a short acknowledgement (if delegating), then the final reply.

        Each call gets a random turn_id. The tool loop runs on a private
        history copy so a follow-up from the same user does not interleave
        with in-flight tool calls. Results are spliced back by turn_id.
        """
        yield from self._run(session_id, user_text, media)

    def run_event_turn(
        self,
        session_id: str,
        events: Sequence[EventRecord],
        *,
        digest: bool = False,
    ) -> Iterator[str]:
        """
        Turn connector events into a proactive message for ``session_id``.

        The event text is stored as this turn's user message (tagged
        ``source="event"``) so a follow-up like "why?" sees it in history. It
        is never consolidated into memory, and acknowledgements are dropped
        since nobody is waiting on a reply.
        """
        if not events:
            return
        yield from self._run(
            session_id,
            format_event_data(events, digest=digest),
            None,
            source="event",
            context_query=event_context_query(events),
            ephemeral_suffix=event_instructions(digest=digest),
            consolidate=False,
            forward_acks=False,
        )

    def _run(
        self,
        session_id: str,
        user_text: str,
        media: DownloadedMedia | None,
        *,
        source: str = "user",
        context_query: str | None = None,
        ephemeral_suffix: str = "",
        consolidate: bool = True,
        forward_acks: bool = True,
    ) -> Iterator[str]:
        turn_id = uuid.uuid4().hex
        logger.info("chat turn %s (%s) for %s", turn_id, source, session_id)
        turn_token = set_turn_id(turn_id)
        working: list[dict[str, Any]] | None = None
        completed = False
        emitted: list[str] = []
        sess_tag = session_hash(session_id)
        with trace_span(
            "chat_turn",
            span_type="AGENT",
            inputs={
                "user_text": user_text,
                "has_media": media is not None,
                "media_kind": media.kind if media is not None else None,
                "source": source,
            },
            attributes={
                "turn_id": turn_id,
                "source": source,
                "session_hash": sess_tag,
            },
        ) as turn_span:
            turn_span.update_trace(
                tags={"turn_id": turn_id, "source": source},
                session_id=sess_tag,
                request_preview=user_text.strip() or (f"[{media.kind}]" if media else ""),
            )
            try:
                working = self._begin_turn(
                    session_id,
                    turn_id,
                    user_text,
                    media,
                    source=source,
                    context_query=context_query,
                    ephemeral_suffix=ephemeral_suffix,
                )

                acked = False

                def before_tool(name: str, kwargs: dict[str, Any]) -> Iterator[str]:
                    nonlocal acked
                    if not forward_acks:
                        return
                    if name == SendAcknowledgementTool.name:
                        msg = (kwargs.get("message")
                               or "").strip() or _FALLBACK_ACK
                        acked = True
                        yield msg
                    elif name == SpawnAgentTool.name and not acked:
                        msg = (
                            kwargs.get("acknowledgement") or ""
                        ).strip() or _FALLBACK_ACK
                        acked = True
                        yield msg

                chat_timeout = _MEDIA_CHAT_TIMEOUT_S if media is not None else None
                chat_tools = self._tools.view(
                    self._tools.names(chat_agent=True),
                    worker_only=False,
                )
                for chunk in run_tool_loop(
                    self._llm,
                    working,
                    chat_tools,
                    session_id=session_id,
                    max_iterations=self._max_agent_iterations,
                    tool_specs=chat_tools.specs(),
                    timeout=chat_timeout,
                    label="chat",
                    turn_id=turn_id,
                    order_tool_calls=_order_chat_tool_calls,
                    before_tool=before_tool,
                    trim_history=self._trim_history,
                ):
                    emitted.append(chunk)
                    yield chunk
                completed = True
            finally:
                turn_span.set_outputs(
                    {
                        "replies": emitted,
                        "completed": completed,
                        "history_messages": len(working) if working is not None else 0,
                    }
                )
                if emitted:
                    turn_span.update_trace(response_preview=emitted[-1])
                if working is not None:
                    self._commit_turn(
                        session_id, turn_id, working, completed=completed
                    )
                    if (
                        completed
                        and consolidate
                        and self._memory_consolidator is not None
                    ):
                        self._memory_consolidator.submit(
                            session_id, turn_id, user_text
                        )
                reset_turn_id(turn_token)


def _order_chat_tool_calls(
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run send_acknowledgement before spawn_agent in the same turn."""

    def rank(call: dict[str, Any]) -> int:
        name = (call.get("function") or {}).get("name") or ""
        if name == SendAcknowledgementTool.name:
            return 0
        if name == SpawnAgentTool.name:
            return 2
        return 1

    return sorted(calls, key=rank)


def _user_content(
    user_text: str,
    media: DownloadedMedia | None,
    memory_context: str = "",
) -> str | list[dict[str, object]]:
    text = user_text.strip() or (_DEFAULT_MEDIA_CAPTION if media is not None else "")
    if memory_context:
        text = f"<context>\n{memory_context}\n</context>\n\n{text}"
    if media is None:
        return text
    return [
        {"type": "text", "text": text},
        media.to_content_part(),
    ]


def _stored_user_content(
    user_text: str,
    media: DownloadedMedia | None,
) -> str:
    """Text-only form stored on the shared session history (no media bytes)."""
    caption = user_text.strip()
    if media is None:
        return caption
    placeholder = f"[{media.kind}]"
    return f"{placeholder} {caption}" if caption else placeholder
