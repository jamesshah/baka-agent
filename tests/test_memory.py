from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi import BackgroundTasks
from sqlalchemy import inspect
from starlette.requests import Request

from agents.chat_agent import ChatAgent
from config import Settings
from llm.base import LLMClient
from memory.database import Database
from memory.consolidator import MemoryConsolidator
from memory.migrate import upgrade
from memory.repository import SqlAlchemyMemoryRepository
from memory.retrieval import ContextBuilder, HybridRetriever
from memory.skills import SkillIndexer
from tools.registry import ToolRegistry
from webhooks.sendblue import SendblueWebhookHandler


class FakeEmbeddingClient:
    model_id = "fake-3d"
    dimensions = 3

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("coffee" in lowered or "espresso" in lowered),
                float("python" in lowered),
                float("travel" in lowered),
            ])
        return vectors


class FakeLLM(LLMClient):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return {"role": "assistant", "content": "stored reply"}


class ConsolidationLLM(LLMClient):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        system = str(messages[0]["content"])
        if "Summarize" in system:
            return {
                "role": "assistant",
                "content": "The user is building a local agent project.",
            }
        return {
            "role": "assistant",
            "content": (
                '[{"kind":"project","key":"current_project",'
                '"value":"local agent","confidence":0.9}]'
            ),
        }


class MemoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "memory.db")
        upgrade(self.db_path)
        self.database = Database(self.db_path)
        self.repository = SqlAlchemyMemoryRepository(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.temp.cleanup()

    def test_migration_is_repeatable_and_creates_search_schema(self) -> None:
        existing_logger = logging.getLogger("baka.test.existing")
        existing_logger.disabled = False
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        handler = logging.NullHandler()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        try:
            upgrade(self.db_path)
            self.assertFalse(existing_logger.disabled)
            self.assertEqual(logging.INFO, root_logger.level)
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(previous_level)
        names = set(inspect(self.database.engine).get_table_names())
        self.assertIn("principals", names)
        self.assertIn("messages", names)
        self.assertIn("seen_handles", names)
        self.assertIn("memories_fts", names)
        self.assertIn("skills_fts", names)
        self.assertIn("alembic_version", names)
        with self.database.engine.connect() as connection:
            checkpoint_pages = connection.exec_driver_sql(
                "PRAGMA wal_autocheckpoint"
            ).scalar_one()
        self.assertEqual(10, checkpoint_pages)

    def test_turns_persist_restart_and_stay_isolated(self) -> None:
        self.repository.begin_turn(
            "+15550000001",
            "turn-a",
            {"role": "user", "content": "hello", "turn_id": "turn-a"},
        )
        self.repository.complete_turn(
            "turn-a",
            [{
                "role": "assistant",
                "content": "hi",
                "turn_id": "turn-a",
            }],
        )
        self.repository.complete_turn("turn-a", [])
        self.repository.begin_turn(
            "+15550000002",
            "turn-b",
            {"role": "user", "content": "private", "turn_id": "turn-b"},
        )
        self.repository.complete_turn("turn-b", [])

        reopened = SqlAlchemyMemoryRepository(Database(self.db_path))
        try:
            history = reopened.load_history("+15550000001", limit=10)
            self.assertEqual(["user", "assistant"], [m["role"] for m in history])
            self.assertEqual("hello", history[0]["content"])
            other = reopened.load_history("+15550000002", limit=10)
            self.assertEqual(["private"], [m["content"] for m in other])
        finally:
            reopened._database.close()

    def test_concurrent_turns_get_unique_ordering(self) -> None:
        def write(number: int) -> None:
            turn_id = f"concurrent-{number}"
            self.repository.begin_turn(
                "+15550000001",
                turn_id,
                {
                    "role": "user",
                    "content": f"message {number}",
                    "turn_id": turn_id,
                },
            )
            self.repository.complete_turn(turn_id, [])

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(write, range(8)))
        history = self.repository.load_history("+15550000001", limit=20)
        self.assertEqual(8, len(history))
        self.assertEqual(
            {f"message {number}" for number in range(8)},
            {message["content"] for message in history},
        )

    def test_hybrid_memory_retrieval_and_fts_fallback(self) -> None:
        embeddings = FakeEmbeddingClient()
        memory = self.repository.upsert_memory(
            "+15550000001",
            kind="preference",
            key="favorite_drink",
            value="espresso",
            confidence=0.95,
        )
        vector = embeddings.embed_documents(
            ["preference favorite_drink: espresso"]
        )[0]
        self.repository.put_embedding(
            entity_type="memory",
            entity_id=memory.id,
            model_id=embeddings.model_id,
            content="preference favorite_drink: espresso",
            vector=vector,
        )
        retriever = HybridRetriever(self.repository, embeddings=embeddings)
        memories, _ = retriever.retrieve("+15550000001", "coffee order")
        self.assertEqual("favorite_drink", memories[0].key)

        fts_only = HybridRetriever(self.repository)
        memories, _ = fts_only.retrieve("+15550000001", "espresso")
        self.assertEqual("favorite_drink", memories[0].key)
        isolated, _ = fts_only.retrieve("+15550000002", "espresso")
        self.assertEqual([], isolated)

    def test_skill_index_and_context_rendering(self) -> None:
        skills = Path(self.temp.name) / "skills" / "python-helper"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text(
            "---\n"
            "name: python-helper\n"
            "description: Diagnose Python code\n"
            "tools: [get_current_time]\n"
            "triggers: [python, traceback]\n"
            "---\n"
            "Inspect the traceback before proposing a fix.\n",
            encoding="utf-8",
        )
        indexer = SkillIndexer(
            self.repository,
            str(Path(self.temp.name) / "skills"),
            embeddings=FakeEmbeddingClient(),
        )
        self.assertEqual(1, indexer.sync())
        context = ContextBuilder(
            HybridRetriever(
                self.repository, embeddings=FakeEmbeddingClient()
            )
        ).render("+15550000001", "help with python")
        self.assertIn("python-helper", context)
        self.assertIn("Inspect the traceback", context)

    def test_chat_agent_hydrates_and_commits_history(self) -> None:
        agent = ChatAgent(
            FakeLLM(),
            ToolRegistry(),
            memory_repository=self.repository,
            max_history_messages=10,
        )
        self.assertEqual(
            ["stored reply"],
            list(agent.run_turn("+15550000001", "persist me")),
        )
        restarted = ChatAgent(
            FakeLLM(),
            ToolRegistry(),
            memory_repository=self.repository,
            max_history_messages=10,
        )
        history = restarted._history_unlocked("+15550000001")
        self.assertEqual(
            ["system", "user", "assistant"],
            [message["role"] for message in history],
        )

    def test_consolidation_persists_memory_and_periodic_summary(self) -> None:
        consolidator = MemoryConsolidator(
            self.repository,
            ConsolidationLLM(),
            embeddings=FakeEmbeddingClient(),
            summary_every_turns=2,
        )
        try:
            for number in (1, 2):
                turn_id = f"turn-{number}"
                self.repository.begin_turn(
                    "+15550000001",
                    turn_id,
                    {
                        "role": "user",
                        "content": f"project update {number}",
                        "turn_id": turn_id,
                    },
                )
                self.repository.complete_turn(
                    turn_id,
                    [{
                        "role": "assistant",
                        "content": "noted",
                        "turn_id": turn_id,
                    }],
                )
                consolidator.consolidate(
                    "+15550000001", turn_id, f"project update {number}"
                )
            memories = self.repository.list_memories("+15550000001")
            self.assertEqual("local agent", memories[0].value)
            summaries = self.repository.search_summaries_fts(
                "+15550000001", "project", limit=5
            )
            self.assertEqual(1, len(summaries))
        finally:
            consolidator.close()

    def test_seen_handles_persist_and_dedupe_across_restarts(self) -> None:
        self.assertFalse(self.repository.is_duplicate_handle(""))
        self.assertFalse(self.repository.is_duplicate_handle("   "))
        self.assertFalse(self.repository.has_seen_handle("msg-1"))
        self.assertFalse(self.repository.is_duplicate_handle("msg-1"))
        self.assertTrue(self.repository.has_seen_handle("msg-1"))
        self.assertTrue(self.repository.is_duplicate_handle("msg-1"))

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    self.repository.is_duplicate_handle,
                    ["msg-concurrent"] * 8,
                )
            )
        self.assertEqual(1, results.count(False))
        self.assertEqual(7, results.count(True))

        reopened_db = Database(self.db_path)
        reopened_repo = SqlAlchemyMemoryRepository(reopened_db)
        try:
            agent = ChatAgent(
                FakeLLM(),
                ToolRegistry(),
                memory_repository=reopened_repo,
            )
            handler = SendblueWebhookHandler(
                agent=agent,
                messaging=object(),  # type: ignore[arg-type]
                memory_repository=reopened_repo,
            )
            self.assertTrue(handler._is_duplicate_handle("msg-1"))
            self.assertTrue(handler._is_duplicate_handle("msg-concurrent"))
            self.assertFalse(handler._is_duplicate_handle("msg-2"))
            self.assertTrue(handler._is_duplicate_handle("msg-2"))
        finally:
            reopened_db.close()

    def test_seen_handles_respect_max_entries(self) -> None:
        for idx in range(5):
            self.assertFalse(
                self.repository.is_duplicate_handle(
                    f"handle-{idx}", max_entries=3
                )
            )
        self.assertFalse(self.repository.has_seen_handle("handle-0"))
        self.assertFalse(self.repository.has_seen_handle("handle-1"))
        self.assertTrue(self.repository.has_seen_handle("handle-2"))
        self.assertTrue(self.repository.has_seen_handle("handle-3"))
        self.assertTrue(self.repository.has_seen_handle("handle-4"))

    def test_webhook_handle_receive_dedupes_via_sqlite_across_restarts(
        self,
    ) -> None:
        secret = "test-webhook-secret"
        payload = {
            "from_number": "+15550000001",
            "content": "hello there",
            "message_handle": "sb-handle-123",
            "service": "iMessage",
        }
        raw_body = json.dumps(payload).encode("utf-8")

        def make_request() -> Request:
            async def receive() -> dict[str, Any]:
                return {
                    "type": "http.request",
                    "body": raw_body,
                    "more_body": False,
                }

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/webhooks/receive",
                "headers": [(b"sb-signing-secret", secret.encode("utf-8"))],
                "client": ("127.0.0.1", 12345),
            }
            return Request(scope, receive)

        settings = Settings(
            sendblue_global_webhook_secret=secret,
            allowed_numbers="",
        )
        with patch("webhooks.sendblue.get_settings", return_value=settings):
            handler1 = SendblueWebhookHandler(
                agent=ChatAgent(
                    FakeLLM(),
                    ToolRegistry(),
                    memory_repository=self.repository,
                ),
                messaging=object(),  # type: ignore[arg-type]
                memory_repository=self.repository,
            )
            tasks1 = BackgroundTasks()
            resp1 = asyncio.run(handler1.handle_receive(make_request(), tasks1))
            self.assertEqual(200, resp1.status_code)
            self.assertEqual(
                {"received": True},
                json.loads(bytes(resp1.body).decode("utf-8")),
            )
            self.assertEqual(1, len(tasks1.tasks))

            tasks_dup = BackgroundTasks()
            resp_dup = asyncio.run(
                handler1.handle_receive(make_request(), tasks_dup)
            )
            self.assertEqual(200, resp_dup.status_code)
            self.assertEqual(
                {"received": True, "ignored": "duplicate"},
                json.loads(bytes(resp_dup.body).decode("utf-8")),
            )
            self.assertEqual(0, len(tasks_dup.tasks))

            reopened_db = Database(self.db_path)
            reopened_repo = SqlAlchemyMemoryRepository(reopened_db)
            try:
                handler2 = SendblueWebhookHandler(
                    agent=ChatAgent(
                        FakeLLM(),
                        ToolRegistry(),
                        memory_repository=reopened_repo,
                    ),
                    messaging=object(),  # type: ignore[arg-type]
                    memory_repository=reopened_repo,
                )
                tasks2 = BackgroundTasks()
                resp2 = asyncio.run(
                    handler2.handle_receive(make_request(), tasks2)
                )
                self.assertEqual(200, resp2.status_code)
                self.assertEqual(
                    {"received": True, "ignored": "duplicate"},
                    json.loads(bytes(resp2.body).decode("utf-8")),
                )
                self.assertEqual(0, len(tasks2.tasks))
            finally:
                reopened_db.close()


if __name__ == "__main__":
    unittest.main()
