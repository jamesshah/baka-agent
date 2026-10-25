"""Git-editable Markdown skill loading and indexing."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from memory.embeddings import EmbeddingClient
from memory.repository import SqlAlchemyMemoryRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillDocument:
    name: str
    description: str
    body: str
    tools: list[str]
    triggers: list[str]
    path: str
    content_hash: str


def _list_value(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [
        item.strip().strip("'\"")
        for item in value.split(",")
        if item.strip().strip("'\"")
    ]


def parse_skill(path: Path) -> SkillDocument:
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    body = raw
    if raw.startswith("---\n"):
        parts = raw.split("---\n", 2)
        if len(parts) == 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip("'\"")
            body = parts[2].strip()
    name = metadata.get("name") or path.parent.name
    if not name:
        raise ValueError(f"Skill name missing in {path}")
    return SkillDocument(
        name=name,
        description=metadata.get("description", ""),
        body=body,
        tools=_list_value(metadata.get("tools", "")),
        triggers=_list_value(metadata.get("triggers", "")),
        path=str(path.resolve()),
        content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


class SkillIndexer:
    def __init__(
        self,
        repository: SqlAlchemyMemoryRepository,
        skills_dir: str,
        *,
        embeddings: EmbeddingClient | None = None,
    ) -> None:
        self._repository = repository
        self._skills_dir = Path(skills_dir)
        self._embeddings = embeddings

    def sync(self) -> int:
        paths = (
            sorted(self._skills_dir.glob("*/SKILL.md"))
            if self._skills_dir.exists()
            else []
        )
        seen: set[str] = set()
        count = 0
        for path in paths:
            try:
                document = parse_skill(path)
                skill = self._repository.upsert_skill(
                    name=document.name,
                    description=document.description,
                    body=document.body,
                    tools=document.tools,
                    triggers=document.triggers,
                    path=document.path,
                    content_hash=document.content_hash,
                )
                seen.add(document.path)
                count += 1
                if self._embeddings is not None:
                    text = (
                        f"{document.name}\n{document.description}\n"
                        f"{' '.join(document.triggers)}\n{document.body}"
                    )
                    vector = self._embeddings.embed_documents([text])[0]
                    self._repository.put_embedding(
                        entity_type="skill",
                        entity_id=skill.id,
                        model_id=self._embeddings.model_id,
                        content=text,
                        vector=vector,
                    )
            except Exception:
                logger.exception("Failed to index skill %s", path)
        self._repository.remove_missing_skills(seen)
        return count
