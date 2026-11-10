"""Send one daily summary of events triaged as ``digest``."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import date, datetime, time, tzinfo
from typing import TYPE_CHECKING

from events.triage import in_quiet_hours, local_now

if TYPE_CHECKING:
    from events.processor import Notifier
    from memory.repository import SqlAlchemyMemoryRepository

logger = logging.getLogger(__name__)


def parse_digest_time(value: str) -> time | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        hour_text, _, minute_text = text.partition(":")
        return time(int(hour_text), int(minute_text or 0))
    except ValueError:
        logger.warning("invalid EVENT_DIGEST_TIME %r; digest disabled", value)
        return None


class DigestScheduler:
    """
    Once per local day at ``digest_time``, send pending digest events.

    If the app was down at digest time it catches up on the next check that
    day (outside quiet hours) rather than dropping the digest.
    """

    def __init__(
        self,
        repository: SqlAlchemyMemoryRepository,
        notifier: Notifier,
        *,
        digest_time: time,
        quiet_hours: tuple[time, time] | None = None,
        tz: tzinfo | None = None,
        clock: Callable[[], datetime] | None = None,
        check_interval_s: float = 60.0,
    ) -> None:
        self._repository = repository
        self._notifier = notifier
        self._digest_time = digest_time
        self._quiet_hours = quiet_hours
        self._clock = clock or (lambda: local_now(tz))
        self._check_interval_s = check_interval_s
        self._last_run: date | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="event-digest", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._check_interval_s + 1.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:  # noqa: BLE001
                logger.exception("digest run failed")
            if self._stop.wait(self._check_interval_s):
                return

    def run_once(self, now: datetime | None = None) -> int:
        """Send due digests; return how many users were messaged."""
        now = now or self._clock()
        with self._lock:
            if self._last_run == now.date():
                return 0
            if now.timetz().replace(tzinfo=None) < self._digest_time:
                return 0
            if in_quiet_hours(now, self._quiet_hours):
                return 0
            self._last_run = now.date()

        sent = 0
        for phone in self._repository.principals_with_pending_digest():
            events = self._repository.pending_digest(phone)
            if not events:
                continue
            try:
                self._notifier(phone, events, True)
            except Exception:  # noqa: BLE001 — leave them pending for tomorrow
                logger.exception("failed to send digest to %s", phone)
                continue
            for event in events:
                self._repository.update_event(event.id, status="digested")
            sent += 1
            logger.info("sent digest of %d event(s) to %s", len(events), phone)
        return sent
