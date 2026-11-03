"""Shared test doubles for the connector event pipeline."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from connectors.base import Connector, ExternalEvent, PendingEvent

FAKE_SECRET = "fake-secret"


class FakeConnector(Connector):
    """Push connector signed with a shared header; enrich is scripted."""

    name = "fake"
    auth = "shared_secret"

    def __init__(self, events_per_delivery: int = 1) -> None:
        self.events_per_delivery = events_per_delivery
        self.enrich_calls = 0
        self.fail_enrich = False

    def verify(self, headers: Mapping[str, str], raw_body: bytes) -> bool:
        return hmac.compare_digest(headers.get("x-fake-secret") or "", FAKE_SECRET)

    def parse(self, body: Any) -> list[PendingEvent]:
        if not isinstance(body, dict) or not body.get("id"):
            return []
        return [
            PendingEvent(
                source=self.name,
                type=str(body.get("type") or "recovery.updated"),
                dedupe_key=f"fake:{body['id']}",
                principal_phone=str(body.get("phone") or "+15550001111"),
                payload=dict(body),
            )
        ]

    def enrich(self, pending: PendingEvent) -> list[ExternalEvent]:
        self.enrich_calls += 1
        if self.fail_enrich:
            raise RuntimeError("provider down")
        score = int(pending.payload.get("recovery_score", 50))
        return [
            ExternalEvent(
                source=self.name,
                type=pending.type,
                occurred_at=datetime.now(timezone.utc),
                dedupe_key=f"{pending.dedupe_key}:{index}",
                principal_phone=pending.principal_phone,
                metrics={"recovery_score": score},
                summary=f"Recovery {score}%",
            )
            for index in range(self.events_per_delivery)
        ]

    def metric_schema(self) -> dict[str, str]:
        return {"recovery_score": "number", "sleep_hours": "number", "nap": "bool"}

    def event_types(self) -> list[str]:
        return ["recovery.updated", "recovery.deleted", "sleep.updated"]
