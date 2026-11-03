"""Format agent replies for iMessage and send them chunk by chunk."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from messaging.base import MessagingClient
from messaging.format import chunk_text, strip_markdown

logger = logging.getLogger(__name__)


def deliver_replies(
    messaging: MessagingClient,
    number: str,
    replies: Iterable[str],
    *,
    before_send: Callable[[], None] | None = None,
    after_send: Callable[[], None] | None = None,
) -> int:
    """
    Send each reply as plain-text chunks; return how many replies were sent.

    ``before_send``/``after_send`` wrap each reply's chunks (used to pause the
    typing indicator while text goes out). Replies are consumed lazily, so a
    generator's acknowledgement is sent before its final answer is computed.
    """
    sent = 0
    for reply in replies:
        chunks = chunk_text(strip_markdown(reply))
        if not chunks:
            logger.warning("empty reply after format for %s — not sending", number)
            continue
        logger.info(
            "replying to %s (%d chunk(s)): %s", number, len(chunks), chunks[0][:200]
        )
        if before_send is not None:
            before_send()
        for chunk in chunks:
            messaging.send_message(number, chunk)
        if after_send is not None:
            after_send()
        sent += 1
    return sent
