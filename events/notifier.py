"""Run an event turn and text the result to the user."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from messaging.outbound import deliver_replies

if TYPE_CHECKING:
    from agents.chat_agent import ChatAgent
    from memory.store import EventRecord
    from messaging.base import MessagingClient

logger = logging.getLogger(__name__)


class EventNotifier:
    def __init__(self, agent: ChatAgent, messaging: MessagingClient) -> None:
        self._agent = agent
        self._messaging = messaging

    def __call__(
        self, phone: str, events: Sequence[EventRecord], digest: bool = False
    ) -> None:
        replies = self._agent.run_event_turn(phone, events, digest=digest)
        sent = deliver_replies(self._messaging, phone, replies)
        if sent == 0:
            raise RuntimeError("event turn produced no message to send")
