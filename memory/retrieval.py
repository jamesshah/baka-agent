"""Hybrid lexical/vector retrieval and bounded context rendering."""

from __future__ import annotations

import logging

from memory.embeddings import EmbeddingClient
from memory.repository import SqlAlchemyMemoryRepository
from memory.store import MemoryHit, SkillHit

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        repository: SqlAlchemyMemoryRepository,
        *,
        embeddings: EmbeddingClient | None = None,
        lexical_weight: float = 0.45,
        vector_weight: float = 0.55,
        minimum_vector_score: float = 0.3,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._lexical_weight = lexical_weight
        self._vector_weight = vector_weight
        self._minimum_vector_score = minimum_vector_score

    def retrieve(
        self,
        principal_external_id: str,
        query: str,
        *,
        memory_limit: int = 6,
        skill_limit: int = 2,
    ) -> tuple[list[MemoryHit], list[SkillHit]]:
        memory_fts = self._repository.search_memories_fts(
            principal_external_id, query, limit=memory_limit * 2
        )
        memory_fts.extend(
            self._repository.search_summaries_fts(
                principal_external_id, query, limit=memory_limit
            )
        )
        skill_fts = self._repository.search_skills_fts(
            query, limit=skill_limit * 2
        )
        memory_vec: list[MemoryHit] = []
        skill_vec: list[SkillHit] = []
        if self._embeddings is not None and query.strip():
            try:
                vector = self._embeddings.embed_query(query)
                memory_vec = self._repository.vector_memory_candidates(
                    principal_external_id,
                    vector,
                    model_id=self._embeddings.model_id,
                    limit=memory_limit * 2,
                )
                memory_vec.extend(
                    self._repository.vector_summary_candidates(
                        principal_external_id,
                        vector,
                        model_id=self._embeddings.model_id,
                        limit=memory_limit,
                    )
                )
                skill_vec = self._repository.vector_skill_candidates(
                    vector,
                    model_id=self._embeddings.model_id,
                    limit=skill_limit * 2,
                )
            except Exception:
                logger.warning(
                    "Embedding retrieval unavailable; using FTS5 only",
                    exc_info=True,
                )
        memories = self._merge_memories(memory_fts, memory_vec)[:memory_limit]
        skills = self._merge_skills(skill_fts, skill_vec)[:skill_limit]
        return memories, skills

    def _merge_memories(
        self, lexical: list[MemoryHit], semantic: list[MemoryHit]
    ) -> list[MemoryHit]:
        combined: dict[str, MemoryHit] = {}
        scores: dict[str, float] = {}
        for hit in lexical:
            combined[hit.id] = hit
            scores[hit.id] = scores.get(hit.id, 0.0) + (
                self._lexical_weight * hit.score
            )
        for hit in semantic:
            if hit.score < self._minimum_vector_score:
                continue
            combined[hit.id] = hit
            scores[hit.id] = scores.get(hit.id, 0.0) + (
                self._vector_weight * hit.score
            )
        return [
            MemoryHit(
                id=hit.id,
                kind=hit.kind,
                key=hit.key,
                value=hit.value,
                confidence=hit.confidence,
                score=scores[hit.id] * hit.confidence,
            )
            for hit in sorted(
                combined.values(),
                key=lambda item: scores[item.id] * item.confidence,
                reverse=True,
            )
        ]

    def _merge_skills(
        self, lexical: list[SkillHit], semantic: list[SkillHit]
    ) -> list[SkillHit]:
        combined: dict[str, SkillHit] = {}
        scores: dict[str, float] = {}
        for hit in lexical:
            combined[hit.id] = hit
            scores[hit.id] = scores.get(hit.id, 0.0) + (
                self._lexical_weight * hit.score
            )
        for hit in semantic:
            if hit.score < self._minimum_vector_score:
                continue
            combined[hit.id] = hit
            scores[hit.id] = scores.get(hit.id, 0.0) + (
                self._vector_weight * hit.score
            )
        return [
            SkillHit(
                id=hit.id,
                name=hit.name,
                description=hit.description,
                body=hit.body,
                tools=hit.tools,
                score=scores[hit.id],
            )
            for hit in sorted(
                combined.values(),
                key=lambda item: scores[item.id],
                reverse=True,
            )
        ]


class ContextBuilder:
    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        max_memory_chars: int = 3000,
        max_skill_chars: int = 5000,
    ) -> None:
        self._retriever = retriever
        self._max_memory_chars = max_memory_chars
        self._max_skill_chars = max_skill_chars

    def render(self, principal_external_id: str, query: str) -> str:
        memories, skills = self._retriever.retrieve(
            principal_external_id, query
        )
        sections: list[str] = []
        if memories:
            lines = [
                f"- [{item.kind}] {item.key}: {item.value}"
                for item in memories
            ]
            rendered = "\n".join(lines)[: self._max_memory_chars]
            sections.append(
                "Relevant long-term memory (use only when helpful; "
                "do not mention this block):\n" + rendered
            )
        if skills:
            chunks: list[str] = []
            remaining = self._max_skill_chars
            for skill in skills:
                chunk = (
                    f"Skill: {skill.name}\n"
                    f"Allowed tools: {', '.join(skill.tools) or 'none'}\n"
                    f"{skill.body}"
                )
                if remaining <= 0:
                    break
                chunks.append(chunk[:remaining])
                remaining -= len(chunk)
            sections.append(
                "Relevant procedural skills (follow when applicable):\n"
                + "\n\n".join(chunks)
            )
        return "\n\n".join(sections)
