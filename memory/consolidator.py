"""Post-turn extraction of durable, user-supplied semantic memories."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from llm.base import LLMClient
from memory.embeddings import EmbeddingClient
from memory.repository import SqlAlchemyMemoryRepository

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Extract only durable facts explicitly stated by the user.
Useful memories include identity, stable preferences, relationships, ongoing
projects, constraints, and commitments. Never retain passwords, API keys,
authentication codes, financial account numbers, or transient requests.
Return a JSON array. Each item must have: kind, key, value, confidence.
Use stable snake_case keys. Return [] when nothing should be remembered."""

_SECRET_PATTERN = re.compile(
    r"(password|passcode|api[_ -]?key|secret|token|authorization|"
    r"\b\d{12,19}\b)",
    re.IGNORECASE,
)


class MemoryConsolidator:
    def __init__(
        self,
        repository: SqlAlchemyMemoryRepository,
        llm: LLMClient,
        *,
        embeddings: EmbeddingClient | None = None,
        enabled: bool = True,
        summary_every_turns: int = 20,
    ) -> None:
        self._repository = repository
        self._llm = llm
        self._embeddings = embeddings
        self._enabled = enabled
        self._summary_every_turns = max(2, summary_every_turns)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="memory-consolidator"
        )

    def submit(
        self,
        principal_external_id: str,
        turn_id: str,
        user_text: str,
    ) -> Future[None] | None:
        if not self._enabled or not user_text.strip():
            return None
        return self._executor.submit(
            self.consolidate, principal_external_id, turn_id, user_text
        )

    def consolidate(
        self,
        principal_external_id: str,
        turn_id: str,
        user_text: str,
    ) -> None:
        if _SECRET_PATTERN.search(user_text):
            return
        source_id = self._repository.user_message_id(turn_id)
        try:
            response = self._llm.chat(
                [
                    {"role": "system", "content": _EXTRACTION_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                timeout=60.0,
            )
            candidates = _parse_candidates(response.get("content") or "")
        except Exception:
            logger.exception("Memory extraction failed for turn %s", turn_id)
            return
        for candidate in candidates:
            value = str(candidate.get("value") or "").strip()
            key = _safe_key(str(candidate.get("key") or ""))
            kind = _safe_key(str(candidate.get("kind") or "fact")) or "fact"
            if not key or not value or _SECRET_PATTERN.search(value):
                continue
            try:
                confidence = min(
                    1.0, max(0.0, float(candidate.get("confidence", 0.8)))
                )
                memory = self._repository.upsert_memory(
                    principal_external_id,
                    kind=kind,
                    key=key,
                    value=value,
                    confidence=confidence,
                    source_message_ids=[source_id] if source_id else [],
                )
                if self._embeddings is not None:
                    content = f"{kind} {key}: {value}"
                    vector = self._embeddings.embed_documents([content])[0]
                    self._repository.put_embedding(
                        entity_type="memory",
                        entity_id=memory.id,
                        model_id=self._embeddings.model_id,
                        content=content,
                        vector=vector,
                    )
            except Exception:
                logger.exception("Failed to persist extracted memory %s", key)
        self._maybe_summarize(principal_external_id)

    def _maybe_summarize(self, principal_external_id: str) -> None:
        batch = self._repository.summary_batch(
            principal_external_id, batch_size=self._summary_every_turns
        )
        if batch is None:
            return
        conversation_id, start_sequence, end_sequence, transcript = batch
        try:
            response = self._llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize this conversation segment for future "
                            "context. Preserve decisions, ongoing work, and "
                            "important outcomes. Omit secrets and small talk. "
                            "Return plain text under 800 words."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
                temperature=0.0,
                timeout=90.0,
            )
            body = str(response.get("content") or "").strip()
            if not body:
                return
            summary = self._repository.put_summary(
                conversation_id=conversation_id,
                start_sequence=start_sequence,
                end_sequence=end_sequence,
                body=body,
            )
            if self._embeddings is not None:
                vector = self._embeddings.embed_documents([body])[0]
                self._repository.put_embedding(
                    entity_type="summary",
                    entity_id=summary.id,
                    model_id=self._embeddings.model_id,
                    content=body,
                    vector=vector,
                )
        except Exception:
            logger.exception(
                "Conversation summary failed for %s", principal_external_id
            )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")[:255]


def _parse_candidates(content: str) -> list[dict[str, Any]]:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Memory extraction must return a JSON array")
    return [item for item in parsed if isinstance(item, dict)]
