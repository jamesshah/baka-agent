"""Embedding clients for semantic-memory indexing and retrieval."""

from __future__ import annotations

from typing import Protocol

import httpx


class EmbeddingClient(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class LlamaEmbeddingClient:
    """OpenAI-compatible client for a llama.cpp embedding sidecar."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int = 768,
        timeout: float = 30.0,
        query_prefix: str = "search_query: ",
        document_prefix: str = "search_document: ",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._timeout = timeout
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"{self._query_prefix}{text}"])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([
            f"{self._document_prefix}{text}" for text in texts
        ])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": texts},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = sorted(payload.get("data") or [], key=lambda row: row["index"])
        vectors = [[float(value) for value in row["embedding"]] for row in rows]
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding server returned {len(vectors)} vectors "
                f"for {len(texts)} inputs"
            )
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise ValueError(
                    f"Expected {self._dimensions} embedding dimensions, "
                    f"received {len(vector)}"
                )
        return vectors

    def health_check(self) -> dict[str, object]:
        try:
            response = httpx.get(
                f"{self._base_url.removesuffix('/v1')}/health",
                timeout=2.0,
            )
            response.raise_for_status()
        except Exception as exc:
            return {"status": "degraded", "detail": str(exc)}
        return {
            "status": "ok",
            "model": self._model,
            "dimensions": self._dimensions,
        }
