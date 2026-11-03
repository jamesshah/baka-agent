"""Generic webhook route for every connector: verify, parse, store, queue."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from connectors.registry import ConnectorRegistry
    from events.processor import EventProcessor
    from memory.repository import SqlAlchemyMemoryRepository

logger = logging.getLogger(__name__)


class ConnectorWebhookHandler:
    """
    Store each delivery before doing any work and return 200 quickly, so
    providers don't retry. Enrichment and triage run on the EventProcessor.
    """

    def __init__(
        self,
        connectors: ConnectorRegistry,
        repository: SqlAlchemyMemoryRepository,
        processor: EventProcessor,
        *,
        allowed_numbers: set[str] | None = None,
    ) -> None:
        self._connectors = connectors
        self._repository = repository
        self._processor = processor
        self._allowed = allowed_numbers or set()

    async def handle(self, source: str, request: Request) -> JSONResponse:
        connector = self._connectors.get(source)
        if connector is None or connector.delivery != "push":
            return JSONResponse({"error": "unknown_connector"}, status_code=404)

        raw_body = await request.body()
        headers = {key.lower(): value for key, value in request.headers.items()}
        if not connector.verify(headers, raw_body):
            logger.warning(
                "%s webhook rejected: bad signature from %s",
                source,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        try:
            body = json.loads(raw_body or b"null")
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid_json"}, status_code=400)

        try:
            pending = connector.parse(body)
        except Exception:  # noqa: BLE001
            logger.exception("%s webhook parse failed", source)
            return JSONResponse({"error": "invalid_payload"}, status_code=400)

        queued = duplicates = skipped = 0
        for item in pending:
            if self._allowed and item.principal_phone not in self._allowed:
                skipped += 1
                continue
            delivery_id = self._repository.record_delivery(
                item.principal_phone,
                source=item.source,
                event_type=item.type,
                dedupe_key=item.dedupe_key,
                payload=item.payload,
            )
            if delivery_id is None:
                duplicates += 1
                continue
            self._processor.submit_delivery(delivery_id)
            queued += 1
        logger.info(
            "%s webhook: queued=%d duplicates=%d skipped=%d",
            source, queued, duplicates, skipped,
        )
        return JSONResponse({
            "received": True,
            "queued": queued,
            "duplicates": duplicates,
            "skipped": skipped,
        })
