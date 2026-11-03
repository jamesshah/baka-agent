"""Background pipeline: enrich deliveries, triage events, and act on them."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from connectors.base import ExternalEvent, PendingEvent

if TYPE_CHECKING:
    from connectors.base import PollingConnector
    from connectors.registry import ConnectorRegistry
    from events.triage import EventTriage
    from memory.repository import SqlAlchemyMemoryRepository
    from memory.store import EventRecord

logger = logging.getLogger(__name__)

# (phone, events, digest) -> None; raises if the message could not be sent.
Notifier = Callable[[str, Sequence["EventRecord"], bool], None]

_STATUS_FOR_ACTION = {
    "digest": "digest_pending",
    "remember": "remembered",
    "ignore": "ignored",
}
_MAX_DELIVERY_ATTEMPTS = 3
_RETRY_DELAYS_S = (30.0, 300.0)


class EventProcessor:
    """Single worker so event turns are serialized like consolidation."""

    def __init__(
        self,
        repository: SqlAlchemyMemoryRepository,
        connectors: ConnectorRegistry,
        triage: EventTriage,
        notifier: Notifier,
    ) -> None:
        self._repository = repository
        self._connectors = connectors
        self._triage = triage
        self._notifier = notifier
        self._executor: ThreadPoolExecutor | None = None
        self._stop = threading.Event()
        self._timers: set[threading.Timer] = set()
        self._timers_lock = threading.Lock()
        self._poll_threads: list[threading.Thread] = []

    # --- lifecycle ------------------------------------------------------------

    def start(self, *, resume: bool = True, poll: bool = True) -> None:
        if self._executor is not None:
            return
        self._stop.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="event-processor"
        )
        if resume:
            self.resume()
        if poll:
            for connector in self._connectors.polling():
                thread = threading.Thread(
                    target=self._poll_loop,
                    args=(connector,),
                    name=f"event-poll-{connector.name}",
                    daemon=True,
                )
                thread.start()
                self._poll_threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        with self._timers_lock:
            for timer in self._timers:
                timer.cancel()
            self._timers.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        for thread in self._poll_threads:
            thread.join(timeout=2.0)
        self._poll_threads.clear()

    def resume(self) -> None:
        """Requeue work a crash or restart left unfinished."""
        deliveries = self._repository.delivery_ids_with_status("received")
        events = self._repository.event_ids_with_status("pending", "notifying")
        for delivery_id in deliveries:
            self.submit_delivery(delivery_id)
        for event_id in events:
            self.submit_event(event_id)
        if deliveries or events:
            logger.info(
                "resumed %d delivery(ies) and %d event(s)", len(deliveries), len(events)
            )

    # --- queueing -------------------------------------------------------------

    def submit_delivery(self, delivery_id: str) -> Future[None] | None:
        return self._submit(self.process_delivery, delivery_id)

    def submit_event(self, event_id: str) -> Future[None] | None:
        return self._submit(self.process_event, event_id)

    def _submit(self, fn: Callable[[str], None], item_id: str) -> Future[None] | None:
        if self._executor is None or self._stop.is_set():
            logger.warning("event processor not running; %s deferred", item_id)
            return None
        return self._executor.submit(self._guard, fn, item_id)

    @staticmethod
    def _guard(fn: Callable[[str], None], item_id: str) -> None:
        try:
            fn(item_id)
        except Exception:  # noqa: BLE001 — keep the worker alive
            logger.exception("event processing failed for %s", item_id)

    # --- work -----------------------------------------------------------------

    def process_delivery(self, delivery_id: str) -> None:
        delivery = self._repository.get_delivery(delivery_id)
        if delivery is None or delivery.status != "received":
            return
        connector = self._connectors.get(delivery.source)
        if connector is None:
            self._repository.mark_delivery(
                delivery_id, "failed", error=f"connector {delivery.source} not enabled"
            )
            return
        pending = PendingEvent(
            source=delivery.source,
            type=delivery.event_type,
            dedupe_key=delivery.dedupe_key,
            principal_phone=delivery.principal_external_id,
            payload=delivery.payload,
        )
        try:
            events = connector.enrich(pending)
        except Exception as exc:  # noqa: BLE001
            attempt = delivery.attempts + 1
            if attempt < _MAX_DELIVERY_ATTEMPTS:
                delay = _RETRY_DELAYS_S[min(attempt - 1, len(_RETRY_DELAYS_S) - 1)]
                logger.warning(
                    "enrich failed for %s (attempt %d), retrying in %.0fs: %s",
                    delivery_id, attempt, delay, exc,
                )
                self._repository.mark_delivery(
                    delivery_id, "received", error=str(exc), count_attempt=True
                )
                self._retry_later(delivery_id, delay)
            else:
                logger.exception("enrich failed for %s; giving up", delivery_id)
                self._repository.mark_delivery(
                    delivery_id, "failed", error=str(exc), count_attempt=True
                )
            return

        new_ids = self.store_events(events, delivery_id=delivery_id)
        self._repository.mark_delivery(delivery_id, "processed")
        for event_id in new_ids:
            self.process_event(event_id)

    def store_events(
        self, events: Sequence[ExternalEvent], *, delivery_id: str | None = None
    ) -> list[str]:
        """Persist normalized events; duplicates are dropped."""
        new_ids: list[str] = []
        for event in events:
            event_id = self._repository.insert_event(
                event.principal_phone,
                source=event.source,
                event_type=event.type,
                dedupe_key=event.dedupe_key,
                occurred_at=event.occurred_at,
                metrics=event.metrics,
                raw=event.raw,
                summary=event.summary,
                delivery_id=delivery_id,
            )
            if event_id is None:
                logger.info("duplicate event %s ignored", event.dedupe_key)
                continue
            new_ids.append(event_id)
        return new_ids

    def process_event(self, event_id: str) -> None:
        event = self._repository.get_event(event_id)
        if event is None or event.status not in {"pending", "notifying"}:
            return
        if event.status == "notifying" and event.decision == "notify":
            self._notify(event)
            return

        decision = self._triage.decide(event)
        logger.info(
            "event %s %s/%s -> %s (%s)%s",
            event.id, event.source, event.event_type, decision.action,
            decision.decided_by,
            f" downgraded from {decision.downgraded_from}"
            if decision.downgraded_from else "",
        )
        if decision.action == "notify":
            self._repository.update_event(
                event_id,
                status="notifying",
                decision="notify",
                decided_by=decision.decided_by,
                reason=decision.reason,
            )
            refreshed = self._repository.get_event(event_id)
            if refreshed is not None:
                self._notify(refreshed)
            return
        self._repository.update_event(
            event_id,
            status=_STATUS_FOR_ACTION[decision.action],
            decision=decision.action,
            decided_by=decision.decided_by,
            reason=decision.reason,
        )

    def _notify(self, event: EventRecord) -> None:
        try:
            self._notifier(event.principal_external_id, [event], False)
        except Exception:  # noqa: BLE001
            logger.exception("failed to notify for event %s", event.id)
            self._repository.update_event(event.id, status="failed")
            return
        self._repository.update_event(event.id, status="notified")

    def _retry_later(self, delivery_id: str, delay: float) -> None:
        def fire() -> None:
            with self._timers_lock:
                self._timers.discard(timer)
            self.submit_delivery(delivery_id)

        timer = threading.Timer(delay, fire)
        timer.daemon = True
        with self._timers_lock:
            if self._stop.is_set():
                return
            self._timers.add(timer)
        timer.start()

    def _poll_loop(self, connector: PollingConnector) -> None:
        while not self._stop.is_set():
            try:
                since = self._repository.last_event_at(connector.name)
                events = connector.poll(since)
                if events:
                    for event_id in self.store_events(events):
                        self.submit_event(event_id)
            except Exception:  # noqa: BLE001
                logger.exception("poll failed for %s", connector.name)
            if self._stop.wait(connector.poll_interval_seconds):
                return
