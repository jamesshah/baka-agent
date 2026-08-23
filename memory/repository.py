"""SQLAlchemy repository for conversation, semantic memory, and skills."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select, text

from memory.database import Database
from memory.models import (
    Conversation,
    ConversationSummary,
    Embedding,
    Memory,
    Message,
    Principal,
    Skill,
    Turn,
)
from memory.store import MemoryHit, SkillHit


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fts_query(value: str) -> str:
    tokens = re.findall(r"[\w'-]+", value.lower())
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:16])


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(value: bytes, dimensions: int) -> tuple[float, ...]:
    return struct.unpack(f"<{dimensions}f", value)


def _cosine(left: list[float], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class SqlAlchemyMemoryRepository:
    """Thread-safe repository; SQLite remains the single source of truth."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._write_lock = threading.RLock()

    @staticmethod
    def _nearest_embedding_distances(
        session, query_vector: list[float], *, limit: int
    ) -> dict[int, float] | None:
        if len(query_vector) != 768:
            return None
        try:
            rows = session.execute(
                text(
                    "SELECT rowid, distance FROM memory_vectors "
                    "WHERE embedding MATCH :vector AND k = :limit "
                    "ORDER BY distance"
                ),
                {
                    "vector": _pack_vector(query_vector),
                    "limit": max(256, min(4096, limit * 32)),
                },
            ).all()
        except Exception:
            return None
        return {int(row.rowid): float(row.distance) for row in rows}

    def _identity(self, session, external_id: str) -> tuple[Principal, Conversation]:
        principal = session.scalar(
            select(Principal).where(Principal.external_id == external_id)
        )
        if principal is None:
            principal = Principal(external_id=external_id)
            session.add(principal)
            session.flush()
        conversation = session.scalar(
            select(Conversation).where(
                Conversation.principal_id == principal.id,
                Conversation.channel == "imessage",
                Conversation.external_id == external_id,
            )
        )
        if conversation is None:
            conversation = Conversation(
                principal_id=principal.id,
                channel="imessage",
                external_id=external_id,
            )
            session.add(conversation)
            session.flush()
        return principal, conversation

    def principal_id(self, external_id: str) -> str:
        with self._write_lock, self._database.session() as session:
            principal, _ = self._identity(session, external_id)
            return principal.id

    def load_history(
        self, principal_external_id: str, *, limit: int
    ) -> list[dict[str, Any]]:
        with self._database.session() as session:
            principal = session.scalar(
                select(Principal).where(
                    Principal.external_id == principal_external_id
                )
            )
            if principal is None:
                return []
            conversation = session.scalar(
                select(Conversation).where(
                    Conversation.principal_id == principal.id,
                    Conversation.channel == "imessage",
                    Conversation.external_id == principal_external_id,
                )
            )
            if conversation is None:
                return []
            rows = session.execute(
                select(Message, Turn.sequence)
                .join(Turn, Message.turn_id == Turn.id)
                .where(Turn.conversation_id == conversation.id)
                .order_by(Turn.sequence.desc(), Message.sequence.desc())
                .limit(limit)
            ).all()
            result: list[dict[str, Any]] = []
            for message, _turn_sequence in reversed(rows):
                item: dict[str, Any] = {
                    "role": message.role,
                    "content": message.content,
                    "turn_id": message.turn_id,
                }
                if message.tool_calls is not None:
                    item["tool_calls"] = message.tool_calls
                if message.tool_call_id is not None:
                    item["tool_call_id"] = message.tool_call_id
                result.append(item)
            return result

    def begin_turn(
        self,
        principal_external_id: str,
        turn_id: str,
        user_message: dict[str, Any],
    ) -> None:
        with self._write_lock, self._database.session() as session:
            if session.get(Turn, turn_id) is not None:
                return
            _, conversation = self._identity(session, principal_external_id)
            next_sequence = (
                session.scalar(
                    select(func.coalesce(func.max(Turn.sequence), 0)).where(
                        Turn.conversation_id == conversation.id
                    )
                )
                or 0
            ) + 1
            turn = Turn(
                id=turn_id,
                conversation_id=conversation.id,
                sequence=next_sequence,
                status="processing",
            )
            message = Message(
                turn_id=turn_id,
                sequence=0,
                role="user",
                content=user_message.get("content", ""),
            )
            session.add_all([turn, message])
            session.flush()
            session.execute(
                text(
                    "INSERT INTO messages_fts(entity_id, turn_id, role, content) "
                    "VALUES (:id, :turn_id, :role, :content)"
                ),
                {
                    "id": str(message.id),
                    "turn_id": turn_id,
                    "role": "user",
                    "content": str(message.content),
                },
            )

    def complete_turn(
        self,
        turn_id: str,
        messages: list[dict[str, Any]],
        *,
        status: str = "completed",
    ) -> list[int]:
        inserted_ids: list[int] = []
        with self._write_lock, self._database.session() as session:
            turn = session.get(Turn, turn_id)
            if turn is None:
                raise KeyError(f"Unknown turn: {turn_id}")
            existing = session.scalar(
                select(func.count(Message.id)).where(Message.turn_id == turn_id)
            ) or 0
            if turn.status in {"completed", "failed"}:
                return []
            sequence = int(existing)
            for item in messages:
                if item.get("role") == "user":
                    continue
                message = Message(
                    turn_id=turn_id,
                    sequence=sequence,
                    role=str(item.get("role") or "assistant"),
                    content=item.get("content", ""),
                    tool_calls=item.get("tool_calls"),
                    tool_call_id=item.get("tool_call_id"),
                )
                session.add(message)
                session.flush()
                inserted_ids.append(message.id)
                session.execute(
                    text(
                        "INSERT INTO messages_fts"
                        "(entity_id, turn_id, role, content) "
                        "VALUES (:id, :turn_id, :role, :content)"
                    ),
                    {
                        "id": str(message.id),
                        "turn_id": turn_id,
                        "role": message.role,
                        "content": str(message.content),
                    },
                )
                sequence += 1
            turn.status = status
            turn.completed_at = _utcnow()
        return inserted_ids

    def fail_turn(self, turn_id: str) -> None:
        with self._write_lock, self._database.session() as session:
            turn = session.get(Turn, turn_id)
            if turn is not None and turn.status != "completed":
                turn.status = "failed"
                turn.completed_at = _utcnow()

    def user_message_id(self, turn_id: str) -> int | None:
        with self._database.session() as session:
            return session.scalar(
                select(Message.id).where(
                    Message.turn_id == turn_id,
                    Message.role == "user",
                )
            )

    def clear_history(self, principal_external_id: str | None = None) -> None:
        with self._write_lock, self._database.session() as session:
            if principal_external_id is None:
                turn_ids = session.scalars(select(Turn.id)).all()
            else:
                turn_ids = session.scalars(
                    select(Turn.id)
                    .join(Conversation)
                    .join(Principal)
                    .where(Principal.external_id == principal_external_id)
                ).all()
            if not turn_ids:
                return
            session.execute(
                text(
                    "DELETE FROM messages_fts WHERE turn_id IN "
                    f"({','.join(':t' + str(i) for i in range(len(turn_ids)))})"
                ),
                {f"t{i}": value for i, value in enumerate(turn_ids)},
            )
            session.execute(delete(Turn).where(Turn.id.in_(turn_ids)))

    def upsert_memory(
        self,
        principal_external_id: str,
        *,
        kind: str,
        key: str,
        value: str,
        confidence: float = 0.8,
        source_message_ids: list[int] | None = None,
    ) -> Memory:
        with self._write_lock, self._database.session() as session:
            principal, _ = self._identity(session, principal_external_id)
            active = session.scalar(
                select(Memory).where(
                    Memory.principal_id == principal.id,
                    Memory.canonical_key == key,
                    Memory.state == "active",
                )
            )
            if active is not None and active.value == value:
                active.confidence = max(active.confidence, confidence)
                active.updated_at = _utcnow()
                return active
            if active is not None:
                active.state = "superseded"
                session.execute(
                    text("DELETE FROM memories_fts WHERE entity_id = :id"),
                    {"id": active.id},
                )
            memory = Memory(
                principal_id=principal.id,
                kind=kind,
                canonical_key=key,
                value=value,
                confidence=confidence,
                source_message_ids=source_message_ids or [],
            )
            session.add(memory)
            session.flush()
            session.execute(
                text(
                    "INSERT INTO memories_fts"
                    "(entity_id, principal_id, kind, canonical_key, value) "
                    "VALUES (:id, :principal, :kind, :key, :value)"
                ),
                {
                    "id": memory.id,
                    "principal": principal.id,
                    "kind": kind,
                    "key": key,
                    "value": value,
                },
            )
            return memory

    def forget_memory(self, principal_external_id: str, key: str) -> bool:
        with self._write_lock, self._database.session() as session:
            memory = session.scalar(
                select(Memory)
                .join(Principal)
                .where(
                    Principal.external_id == principal_external_id,
                    Memory.canonical_key == key,
                    Memory.state == "active",
                )
            )
            if memory is None:
                return False
            memory.state = "forgotten"
            memory.updated_at = _utcnow()
            session.execute(
                text("DELETE FROM memories_fts WHERE entity_id = :id"),
                {"id": memory.id},
            )
            return True

    def list_memories(
        self, principal_external_id: str, *, limit: int = 50
    ) -> list[MemoryHit]:
        with self._database.session() as session:
            rows = session.scalars(
                select(Memory)
                .join(Principal)
                .where(
                    Principal.external_id == principal_external_id,
                    Memory.state == "active",
                )
                .order_by(Memory.updated_at.desc())
                .limit(limit)
            ).all()
        return [
            MemoryHit(
                id=item.id,
                kind=item.kind,
                key=item.canonical_key,
                value=item.value,
                confidence=item.confidence,
                score=1.0,
            )
            for item in rows
        ]

    def search_memories_fts(
        self, principal_external_id: str, query: str, *, limit: int
    ) -> list[MemoryHit]:
        match = _fts_query(query)
        if not match:
            return []
        with self._database.session() as session:
            rows = session.execute(
                text(
                    "SELECT m.id, m.kind, m.canonical_key, m.value, "
                    "m.confidence, bm25(memories_fts) AS rank "
                    "FROM memories_fts "
                    "JOIN memories m ON m.id = memories_fts.entity_id "
                    "JOIN principals p ON p.id = m.principal_id "
                    "WHERE memories_fts MATCH :query "
                    "AND p.external_id = :external_id AND m.state = 'active' "
                    "ORDER BY rank LIMIT :limit"
                ),
                {
                    "query": match,
                    "external_id": principal_external_id,
                    "limit": limit,
                },
            ).all()
        return [
            MemoryHit(
                id=row.id,
                kind=row.kind,
                key=row.canonical_key,
                value=row.value,
                confidence=row.confidence,
                score=1.0 / (1.0 + abs(float(row.rank))),
            )
            for row in rows
        ]

    def vector_memory_candidates(
        self,
        principal_external_id: str,
        query_vector: list[float],
        *,
        model_id: str,
        limit: int,
    ) -> list[MemoryHit]:
        with self._database.session() as session:
            distances = self._nearest_embedding_distances(
                session, query_vector, limit=limit
            )
            rows = session.execute(
                select(Memory, Embedding)
                .join(
                    Embedding,
                    (Embedding.entity_type == "memory")
                    & (Embedding.entity_id == Memory.id),
                )
                .join(Principal, Principal.id == Memory.principal_id)
                .where(
                    Principal.external_id == principal_external_id,
                    Memory.state == "active",
                    Embedding.model_id == model_id,
                )
            ).all()
        hits = [
            MemoryHit(
                id=memory.id,
                kind=memory.kind,
                key=memory.canonical_key,
                value=memory.value,
                confidence=memory.confidence,
                score=max(0.0, (
                    1.0 - distances[embedding.id]
                    if distances is not None and embedding.id in distances
                    else _cosine(
                        query_vector,
                        _unpack_vector(embedding.vector, embedding.dimensions),
                    )
                )),
            )
            for memory, embedding in rows
            if distances is None or embedding.id in distances
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def summary_batch(
        self, principal_external_id: str, *, batch_size: int
    ) -> tuple[str, int, int, str] | None:
        with self._database.session() as session:
            conversation = session.scalar(
                select(Conversation)
                .join(Principal)
                .where(
                    Principal.external_id == principal_external_id,
                    Conversation.channel == "imessage",
                )
            )
            if conversation is None:
                return None
            last_end = session.scalar(
                select(func.coalesce(
                    func.max(ConversationSummary.end_turn_sequence), 0
                )).where(
                    ConversationSummary.conversation_id == conversation.id
                )
            ) or 0
            turns = session.scalars(
                select(Turn)
                .where(
                    Turn.conversation_id == conversation.id,
                    Turn.status == "completed",
                    Turn.sequence > last_end,
                )
                .order_by(Turn.sequence)
                .limit(batch_size)
            ).all()
            if len(turns) < batch_size:
                return None
            turn_ids = [turn.id for turn in turns]
            messages = session.scalars(
                select(Message)
                .where(Message.turn_id.in_(turn_ids))
                .order_by(Message.created_at, Message.sequence)
            ).all()
            transcript = "\n".join(
                f"{message.role}: {message.content}" for message in messages
            )
            return (
                conversation.id,
                turns[0].sequence,
                turns[-1].sequence,
                transcript,
            )

    def put_summary(
        self,
        *,
        conversation_id: str,
        start_sequence: int,
        end_sequence: int,
        body: str,
    ) -> ConversationSummary:
        with self._write_lock, self._database.session() as session:
            summary = session.scalar(
                select(ConversationSummary).where(
                    ConversationSummary.conversation_id == conversation_id,
                    ConversationSummary.start_turn_sequence == start_sequence,
                    ConversationSummary.end_turn_sequence == end_sequence,
                )
            )
            if summary is None:
                summary = ConversationSummary(
                    conversation_id=conversation_id,
                    start_turn_sequence=start_sequence,
                    end_turn_sequence=end_sequence,
                    body=body,
                )
                session.add(summary)
                session.flush()
            else:
                summary.body = body
                summary.updated_at = _utcnow()
            session.execute(
                text("DELETE FROM summaries_fts WHERE entity_id = :id"),
                {"id": summary.id},
            )
            session.execute(
                text(
                    "INSERT INTO summaries_fts"
                    "(entity_id, conversation_id, body) "
                    "VALUES (:id, :conversation, :body)"
                ),
                {
                    "id": summary.id,
                    "conversation": conversation_id,
                    "body": body,
                },
            )
            return summary

    def search_summaries_fts(
        self, principal_external_id: str, query: str, *, limit: int
    ) -> list[MemoryHit]:
        match = _fts_query(query)
        if not match:
            return []
        with self._database.session() as session:
            rows = session.execute(
                text(
                    "SELECT s.id, s.start_turn_sequence, "
                    "s.end_turn_sequence, s.body, "
                    "bm25(summaries_fts) AS rank "
                    "FROM summaries_fts "
                    "JOIN conversation_summaries s "
                    "ON s.id = summaries_fts.entity_id "
                    "JOIN conversations c ON c.id = s.conversation_id "
                    "JOIN principals p ON p.id = c.principal_id "
                    "WHERE summaries_fts MATCH :query "
                    "AND p.external_id = :external_id "
                    "ORDER BY rank LIMIT :limit"
                ),
                {
                    "query": match,
                    "external_id": principal_external_id,
                    "limit": limit,
                },
            ).all()
        return [
            MemoryHit(
                id=row.id,
                kind="summary",
                key=f"turns_{row.start_turn_sequence}_{row.end_turn_sequence}",
                value=row.body,
                confidence=0.85,
                score=1.0 / (1.0 + abs(float(row.rank))),
            )
            for row in rows
        ]

    def vector_summary_candidates(
        self,
        principal_external_id: str,
        query_vector: list[float],
        *,
        model_id: str,
        limit: int,
    ) -> list[MemoryHit]:
        with self._database.session() as session:
            distances = self._nearest_embedding_distances(
                session, query_vector, limit=limit
            )
            rows = session.execute(
                select(ConversationSummary, Embedding)
                .join(
                    Embedding,
                    (Embedding.entity_type == "summary")
                    & (Embedding.entity_id == ConversationSummary.id),
                )
                .join(
                    Conversation,
                    Conversation.id == ConversationSummary.conversation_id,
                )
                .join(Principal, Principal.id == Conversation.principal_id)
                .where(
                    Principal.external_id == principal_external_id,
                    Embedding.model_id == model_id,
                )
            ).all()
        hits = [
            MemoryHit(
                id=summary.id,
                kind="summary",
                key=(
                    f"turns_{summary.start_turn_sequence}_"
                    f"{summary.end_turn_sequence}"
                ),
                value=summary.body,
                confidence=0.85,
                score=max(0.0, (
                    1.0 - distances[embedding.id]
                    if distances is not None and embedding.id in distances
                    else _cosine(
                        query_vector,
                        _unpack_vector(embedding.vector, embedding.dimensions),
                    )
                )),
            )
            for summary, embedding in rows
            if distances is None or embedding.id in distances
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def put_embedding(
        self,
        *,
        entity_type: str,
        entity_id: str,
        model_id: str,
        content: str,
        vector: list[float],
    ) -> None:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._write_lock, self._database.session() as session:
            existing = session.scalar(
                select(Embedding).where(
                    Embedding.entity_type == entity_type,
                    Embedding.entity_id == entity_id,
                    Embedding.model_id == model_id,
                    Embedding.content_hash == content_hash,
                )
            )
            if existing is not None:
                return
            old = session.scalars(
                select(Embedding).where(
                    Embedding.entity_type == entity_type,
                    Embedding.entity_id == entity_id,
                    Embedding.model_id == model_id,
                )
            ).all()
            for item in old:
                try:
                    session.execute(
                        text("DELETE FROM memory_vectors WHERE rowid = :id"),
                        {"id": item.id},
                    )
                except Exception:
                    pass
                session.delete(item)
            embedding = Embedding(
                entity_type=entity_type,
                entity_id=entity_id,
                model_id=model_id,
                dimensions=len(vector),
                content_hash=content_hash,
                vector=_pack_vector(vector),
            )
            session.add(embedding)
            session.flush()
            if len(vector) == 768:
                try:
                    session.execute(
                        text(
                            "INSERT INTO memory_vectors(rowid, embedding) "
                            "VALUES (:id, :vector)"
                        ),
                        {"id": embedding.id, "vector": _pack_vector(vector)},
                    )
                except Exception:
                    # Portable BLOB storage + Python cosine remains available.
                    pass

    def upsert_skill(
        self,
        *,
        name: str,
        description: str,
        body: str,
        tools: list[str],
        triggers: list[str],
        path: str,
        content_hash: str,
    ) -> Skill:
        with self._write_lock, self._database.session() as session:
            skill = session.scalar(select(Skill).where(Skill.name == name))
            if skill is None:
                skill = Skill(
                    name=name,
                    description=description,
                    body=body,
                    tools=tools,
                    triggers=triggers,
                    path=path,
                    content_hash=content_hash,
                )
                session.add(skill)
                session.flush()
            elif skill.content_hash != content_hash:
                skill.description = description
                skill.body = body
                skill.tools = tools
                skill.triggers = triggers
                skill.path = path
                skill.content_hash = content_hash
                skill.updated_at = _utcnow()
            session.execute(
                text("DELETE FROM skills_fts WHERE entity_id = :id"),
                {"id": skill.id},
            )
            session.execute(
                text(
                    "INSERT INTO skills_fts"
                    "(entity_id, name, description, body, triggers) "
                    "VALUES (:id, :name, :description, :body, :triggers)"
                ),
                {
                    "id": skill.id,
                    "name": name,
                    "description": description,
                    "body": body,
                    "triggers": " ".join(triggers),
                },
            )
            return skill

    def remove_missing_skills(self, paths: set[str]) -> None:
        with self._write_lock, self._database.session() as session:
            skills = session.scalars(select(Skill)).all()
            for skill in skills:
                if skill.path not in paths:
                    session.execute(
                        text("DELETE FROM skills_fts WHERE entity_id = :id"),
                        {"id": skill.id},
                    )
                    session.delete(skill)

    def search_skills_fts(self, query: str, *, limit: int) -> list[SkillHit]:
        match = _fts_query(query)
        if not match:
            return []
        with self._database.session() as session:
            rows = session.execute(
                text(
                    "SELECT s.id, s.name, s.description, s.body, s.tools, "
                    "bm25(skills_fts) AS rank FROM skills_fts "
                    "JOIN skills s ON s.id = skills_fts.entity_id "
                    "WHERE skills_fts MATCH :query ORDER BY rank LIMIT :limit"
                ),
                {"query": match, "limit": limit},
            ).all()
        return [
            SkillHit(
                id=row.id,
                name=row.name,
                description=row.description,
                body=row.body,
                tools=(
                    json.loads(row.tools)
                    if isinstance(row.tools, str)
                    else list(row.tools or [])
                ),
                score=1.0 / (1.0 + abs(float(row.rank))),
            )
            for row in rows
        ]

    def vector_skill_candidates(
        self,
        query_vector: list[float],
        *,
        model_id: str,
        limit: int,
    ) -> list[SkillHit]:
        with self._database.session() as session:
            distances = self._nearest_embedding_distances(
                session, query_vector, limit=limit
            )
            rows = session.execute(
                select(Skill, Embedding).join(
                    Embedding,
                    (Embedding.entity_type == "skill")
                    & (Embedding.entity_id == Skill.id),
                ).where(Embedding.model_id == model_id)
            ).all()
        hits = [
            SkillHit(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                body=skill.body,
                tools=list(skill.tools or []),
                score=max(0.0, (
                    1.0 - distances[embedding.id]
                    if distances is not None and embedding.id in distances
                    else _cosine(
                        query_vector,
                        _unpack_vector(embedding.vector, embedding.dimensions),
                    )
                )),
            )
            for skill, embedding in rows
            if distances is None or embedding.id in distances
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
