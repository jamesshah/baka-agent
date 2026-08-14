"""User-facing chat agent — replies directly or delegates to a worker."""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from collections.abc import Iterator
from typing import Any

from agents.base import Agent
from agents.loop import run_tool_loop
from llm.base import LLMClient
from messaging.media import DownloadedMedia
from tools.registry import ToolRegistry
from tools.send_acknowledgement import SendAcknowledgementTool
from tools.session_context import reset_turn_id, set_turn_id
from tools.spawn_agent import SpawnAgentTool

logger = logging.getLogger(__name__)

_MEDIA_CHAT_TIMEOUT_S = 300.0
_DEFAULT_MEDIA_CAPTION = "The user sent this without a caption."
_FALLBACK_ACK = "On it — give me a second."

SYSTEM_PROMPT = """
You are a personal agent the user texts from iMessage. Keep replies concise and conversational.

Tone: warm, witty, concise. Write like you're texting a friend. No corporate voice. No bullet dumps unless the user asked for a list.

Format: plain iMessage-friendly text. Keep replies under ~400 chars when you can.

When to reply yourself:
Answer small talk, simple questions, and anything you can handle from the conversation (including photos you can see) without tools.

When to delegate:
If the request needs tools, lookups, SnapTrade, or extended reasoning, do not answer it yourself.
1. Call send_acknowledgement with a short iMessage-style ping. Make it short and sweet but don't be too generic or robotic. 
2. Call spawn_agent with a complete task description and the worker tool names this task needs (from the catalog below). Use an empty tools list for reasoning-only work. The worker has no chat history — put every fact and instruction it needs in task. Grant only the tools required for this task.
Prefer calling both tools in the same turn. 
NEVER CALL spawn_agent WITHOUT ACKNOWLEDGING FIRST.

Available worker tools (pass these names in spawn_agent.tools):
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
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_history_messages = max_history_messages
        self._max_agent_iterations = max_agent_iterations
        self._histories: dict[str, list[dict[str, Any]]] = {}
        self._history_lock = threading.Lock()

    def _render_system_prompt(self) -> str:
        catalog = self._tools.catalog_text(chat_agent=False)
        return f"{self._system_prompt.rstrip()}\n{catalog}\n"

    def _history_unlocked(self, session_id: str) -> list[dict[str, Any]]:
        if session_id not in self._histories:
            self._histories[session_id] = [
                {"role": "system", "content": self._render_system_prompt()},
            ]
        return self._histories[session_id]

    def _trim_history(self, history: list[dict[str, Any]]) -> None:
        """Keep system prompt + the most recent N messages."""
        max_msgs = self._max_history_messages
        if len(history) <= max_msgs:
            return
        system = history[0]
        rest = history[1:]
        history[:] = [system, *rest[-(max_msgs - 1):]]

    def clear_history(self, session_id: str | None = None) -> None:
        """Clear one conversation or all conversations."""
        with self._history_lock:
            if session_id is None:
                self._histories.clear()
            else:
                self._histories.pop(session_id, None)

    def _begin_turn(
        self,
        session_id: str,
        turn_id: str,
        user_text: str,
        media: DownloadedMedia | None,
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
        with self._history_lock:
            committed = self._history_unlocked(session_id)
            committed[0] = {
                "role": "system",
                "content": self._render_system_prompt(),
            }
            committed.append(stored_user)
            self._trim_history(committed)
            working = copy.deepcopy(committed)
        working[-1] = {
            "role": "user",
            "content": _user_content(user_text, media),
            "turn_id": turn_id,
        }
        return working

    def _commit_turn(
        self,
        session_id: str,
        turn_id: str,
        working: list[dict[str, Any]],
    ) -> None:
        """Splice this turn's assistant/tool messages after its user message."""
        new_msgs = [
            dict(message)
            for message in working
            if message.get("turn_id") == turn_id and message.get("role") != "user"
        ]
        if not new_msgs:
            return
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
        turn_id = uuid.uuid4().hex
        logger.info("chat turn %s for %s", turn_id, session_id)
        turn_token = set_turn_id(turn_id)
        working: list[dict[str, Any]] | None = None
        try:
            working = self._begin_turn(session_id, turn_id, user_text, media)

            acked = False

            def before_tool(name: str, kwargs: dict[str, Any]) -> Iterator[str]:
                nonlocal acked
                if name == SendAcknowledgementTool.name:
                    msg = (kwargs.get("message")
                           or "").strip() or _FALLBACK_ACK
                    acked = True
                    yield msg
                elif name == SpawnAgentTool.name and not acked:
                    acked = True
                    yield _FALLBACK_ACK

            chat_timeout = _MEDIA_CHAT_TIMEOUT_S if media is not None else None
            yield from run_tool_loop(
                self._llm,
                working,
                self._tools,
                session_id=session_id,
                max_iterations=self._max_agent_iterations,
                tool_specs=self._tools.specs(chat_agent=True),
                timeout=chat_timeout,
                label="chat",
                turn_id=turn_id,
                order_tool_calls=_order_chat_tool_calls,
                before_tool=before_tool,
                trim_history=self._trim_history,
            )
        finally:
            if working is not None:
                self._commit_turn(session_id, turn_id, working)
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
) -> str | list[dict[str, object]]:
    text = user_text.strip() or (_DEFAULT_MEDIA_CAPTION if media is not None else "")
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
