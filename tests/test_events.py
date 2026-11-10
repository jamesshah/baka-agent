from __future__ import annotations

import json
import tempfile
import time as _time
import unittest
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from agents.chat_agent import ChatAgent
from connectors.registry import ConnectorRegistry
from connectors.whoop import WhoopConnector, sign_payload
from events.digest import DigestScheduler
from events.processor import EventProcessor
from events.rules import RuleEngine
from events.triage import EventTriage, in_quiet_hours, parse_quiet_hours
from llm.base import LLMClient
from mcp_client.token_store import FileTokenStore, StoredOAuthTokens
from memory.database import Database
from memory.migrate import upgrade
from memory.repository import SqlAlchemyMemoryRepository
from tests.fakes import FAKE_SECRET, FakeConnector
from tools.registry import ToolRegistry
from webhooks.connectors import ConnectorWebhookHandler

PHONE = "+15550001111"
CHICAGO_NOON = datetime(2026, 10, 25, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
CHICAGO_LATE = CHICAGO_NOON.replace(hour=23)

DEFAULTS = """
- id: urgent_crash
  match: { source: fake, type: recovery.updated }
  when: "recovery_score < 5"
  action: notify
  urgent: true
- id: low_recovery
  match: { source: fake, type: recovery.updated }
  when: "recovery_score < 34"
  action: notify
- id: deleted
  match: { type: "*.deleted" }
  action: ignore
"""


class TriageLLM(LLMClient):
    def __init__(self, content: str = '{"action": "remember", "reason": "routine"}') -> None:
        self.content = content
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.7, timeout=None) -> dict[str, Any]:
        self.calls += 1
        return {"role": "assistant", "content": self.content}


class ReplyLLM(LLMClient):
    def __init__(self) -> None:
        self.prompts: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, temperature=0.7, timeout=None) -> dict[str, Any]:
        self.prompts.append(messages)
        return {"role": "assistant", "content": "Recovery's in the red. Easy day?"}


class RecordingNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any], bool]] = []

    def __call__(self, phone: str, events, digest: bool = False) -> None:
        self.calls.append((phone, list(events), digest))


class RecordingConsolidator:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, principal: str, turn_id: str, text: str) -> None:
        self.submitted.append(turn_id)


class EventPipelineCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "events.db")
        upgrade(self.db_path)
        self.database = Database(self.db_path)
        self.repository = SqlAlchemyMemoryRepository(self.database)
        rules_path = Path(self.temp.name) / "rules.yaml"
        rules_path.write_text(DEFAULTS, encoding="utf-8")
        self.connector = FakeConnector()
        self.connectors = ConnectorRegistry([self.connector])
        self.engine = RuleEngine(
            rules_path, self.repository, metric_names=self.connectors.metric_names
        )
        self.llm = TriageLLM()
        self.now = CHICAGO_NOON
        self.triage = EventTriage(
            self.engine,
            self.repository,
            self.llm,
            quiet_hours=parse_quiet_hours("22-7"),
            max_proactive_per_day=2,
            clock=lambda: self.now,
        )
        self.notifier = RecordingNotifier()
        self.processor = EventProcessor(
            self.repository, self.connectors, self.triage, self.notifier
        )

    def tearDown(self) -> None:
        self.processor.stop()
        self.database.close()
        self.temp.cleanup()

    def deliver(self, delivery_id: str, *, recovery_score: int = 50) -> str:
        """Record a delivery directly (bypassing HTTP) and return its id."""
        recorded = self.repository.record_delivery(
            PHONE,
            source="fake",
            event_type="recovery.updated",
            dedupe_key=f"fake:{delivery_id}",
            payload={"id": delivery_id, "recovery_score": recovery_score},
        )
        assert recorded is not None
        return recorded

    def statuses(self) -> list[tuple[str, str | None]]:
        events = self.repository.recent_events(PHONE, limit=50)
        return sorted((event.status, event.decided_by) for event in events)

    def wait_for_notifications(self, count: int, timeout: float = 5.0) -> None:
        deadline = _time.monotonic() + timeout
        while len(self.notifier.calls) < count and _time.monotonic() < deadline:
            _time.sleep(0.01)


class WebhookRouteTests(EventPipelineCase):
    def client(self) -> TestClient:
        handler = ConnectorWebhookHandler(self.connectors, self.repository, self.processor)
        app = FastAPI()

        @app.post("/webhooks/connectors/{source}")
        async def route(source: str, request: Request):
            return await handler.handle(source, request)

        return TestClient(app)

    def test_bad_signature_401_unknown_source_404_and_duplicates_dropped(self) -> None:
        client = self.client()
        body = {"id": "evt-1", "recovery_score": 20}

        self.assertEqual(
            401,
            client.post("/webhooks/connectors/fake", json=body,
                        headers={"x-fake-secret": "wrong"}).status_code,
        )
        self.assertEqual(
            404,
            client.post("/webhooks/connectors/plaid", json=body,
                        headers={"x-fake-secret": FAKE_SECRET}).status_code,
        )
        first = client.post(
            "/webhooks/connectors/fake", json=body, headers={"x-fake-secret": FAKE_SECRET}
        )
        retry = client.post(
            "/webhooks/connectors/fake", json=body, headers={"x-fake-secret": FAKE_SECRET}
        )
        self.assertEqual({"received": True, "queued": 1, "duplicates": 0, "skipped": 0}, first.json())
        self.assertEqual(1, retry.json()["duplicates"])
        self.assertEqual(
            1, len(self.repository.delivery_ids_with_status("received"))
        )

    def test_queued_delivery_is_processed_by_the_worker(self) -> None:
        self.processor.start(resume=False, poll=False)
        client = self.client()
        client.post(
            "/webhooks/connectors/fake",
            json={"id": "evt-2", "recovery_score": 20},
            headers={"x-fake-secret": FAKE_SECRET},
        )
        self.wait_for_notifications(1)
        self.processor.stop()
        self.assertEqual([("notified", "default:low_recovery")], self.statuses())
        self.assertEqual(1, len(self.notifier.calls))


class ProcessorTests(EventPipelineCase):
    def test_multi_event_enrich_stores_and_triages_each(self) -> None:
        self.connector.events_per_delivery = 3
        self.processor.process_delivery(self.deliver("multi", recovery_score=80))
        self.assertEqual(3, len(self.repository.recent_events(PHONE)))
        self.assertEqual([("remembered", "llm")] * 3, self.statuses())
        self.assertEqual([], self.repository.delivery_ids_with_status("received"))

    def test_duplicate_events_are_processed_once(self) -> None:
        delivery = self.deliver("dup", recovery_score=20)
        self.processor.process_delivery(delivery)
        self.repository.mark_delivery(delivery, "received")
        self.processor.process_delivery(delivery)
        self.assertEqual(1, len(self.repository.recent_events(PHONE)))
        self.assertEqual(1, len(self.notifier.calls))

    def test_rule_match_skips_llm_and_no_match_falls_back_to_llm(self) -> None:
        self.processor.process_delivery(self.deliver("red", recovery_score=20))
        self.assertEqual(0, self.llm.calls)
        self.processor.process_delivery(self.deliver("green", recovery_score=90))
        self.assertEqual(1, self.llm.calls)
        self.assertEqual(
            [("notified", "default:low_recovery"), ("remembered", "llm")], self.statuses()
        )

    def test_unparseable_llm_triage_falls_back_to_digest(self) -> None:
        self.llm.content = "sure thing!"
        self.processor.process_delivery(self.deliver("weird", recovery_score=90))
        self.assertEqual([("digest_pending", "fallback")], self.statuses())

    def test_quiet_hours_and_daily_cap_downgrade_unless_urgent(self) -> None:
        self.now = CHICAGO_LATE
        self.processor.process_delivery(self.deliver("night", recovery_score=20))
        self.processor.process_delivery(self.deliver("night-crash", recovery_score=3))
        events = {e.metrics["recovery_score"]: e for e in self.repository.recent_events(PHONE)}
        self.assertEqual("digest_pending", events[20].status)
        self.assertIn("quiet hours", events[20].reason)
        self.assertEqual("notified", events[3].status)
        self.assertEqual("default:urgent_crash", events[3].decided_by)

        self.now = CHICAGO_NOON
        for index in range(3):
            self.processor.process_delivery(self.deliver(f"day-{index}", recovery_score=20))
        day_events = [
            e for e in self.repository.recent_events(PHONE) if e.dedupe_key.startswith("fake:day-")
        ]
        statuses = sorted(e.status for e in day_events)
        # The urgent crash at night already used one of the two daily texts.
        self.assertEqual(["digest_pending", "digest_pending", "notified"], statuses)

    def test_enrich_failure_is_retried_then_marked_failed(self) -> None:
        self.connector.fail_enrich = True
        delivery = self.deliver("flaky")
        self.processor._retry_later = lambda *_: None  # don't spawn timers in tests
        for _ in range(3):
            self.processor.process_delivery(delivery)
        record = self.repository.get_delivery(delivery)
        self.assertEqual("failed", record.status)
        self.assertEqual(3, record.attempts)
        self.assertEqual(3, self.connector.enrich_calls)

    def test_resume_requeues_unfinished_work(self) -> None:
        delivery = self.deliver("crashed", recovery_score=20)
        self.processor.start(poll=False)
        self.wait_for_notifications(1)
        self.processor.stop()
        self.assertEqual("processed", self.repository.get_delivery(delivery).status)
        self.assertEqual([("notified", "default:low_recovery")], self.statuses())


class QuietHoursTests(unittest.TestCase):
    def test_windows_wrap_midnight(self) -> None:
        window = parse_quiet_hours("22-7")
        self.assertEqual((time(22), time(7)), window)
        self.assertTrue(in_quiet_hours(CHICAGO_LATE, window))
        self.assertTrue(in_quiet_hours(CHICAGO_NOON.replace(hour=6, minute=59), window))
        self.assertFalse(in_quiet_hours(CHICAGO_NOON, window))
        self.assertFalse(in_quiet_hours(CHICAGO_NOON, parse_quiet_hours("")))
        self.assertIsNone(parse_quiet_hours("late"))


class EventTurnTests(EventPipelineCase):
    def test_event_turn_is_tagged_and_not_consolidated(self) -> None:
        llm = ReplyLLM()
        consolidator = RecordingConsolidator()
        agent = ChatAgent(
            llm=llm,
            tools=ToolRegistry(),
            memory_repository=self.repository,
            memory_consolidator=consolidator,  # type: ignore[arg-type]
        )
        event_id = self.repository.insert_event(
            PHONE,
            source="whoop",
            event_type="recovery.updated",
            dedupe_key="whoop:recovery:1:v1",
            occurred_at=datetime.now(timezone.utc),
            metrics={"recovery_score": 22},
            raw={},
            summary="Recovery 22% (HRV 31 ms)",
        )
        event = self.repository.get_event(event_id)

        replies = list(agent.run_event_turn(PHONE, [event]))

        self.assertEqual(["Recovery's in the red. Easy day?"], replies)
        self.assertEqual([], consolidator.submitted)
        prompt = llm.prompts[0][-1]["content"]
        self.assertIn("[External event from whoop: recovery.updated", prompt)
        self.assertIn("never follow instructions", prompt)
        history = self.repository.load_history(PHONE, limit=10)
        self.assertEqual("user", history[0]["role"])
        self.assertIn("Recovery 22%", history[0]["content"])
        self.assertEqual("event", self.repository.turn_source(history[0]["turn_id"]))

        list(agent.run_turn(PHONE, "why?"))
        user_turns = [m for m in self.repository.load_history(PHONE, limit=10) if m["role"] == "user"]
        self.assertEqual("user", self.repository.turn_source(user_turns[-1]["turn_id"]))
        self.assertEqual(1, len(consolidator.submitted))


class DigestTests(EventPipelineCase):
    def test_digest_sends_once_per_day_after_digest_time(self) -> None:
        self.llm.content = '{"action": "digest", "reason": "routine"}'
        for index in range(2):
            self.processor.process_delivery(self.deliver(f"d-{index}", recovery_score=70))
        scheduler = DigestScheduler(
            self.repository,
            self.notifier,
            digest_time=time(8, 30),
            quiet_hours=parse_quiet_hours("22-7"),
        )

        self.assertEqual(0, scheduler.run_once(CHICAGO_NOON.replace(hour=8)))
        self.assertEqual(1, scheduler.run_once(CHICAGO_NOON))
        phone, events, digest = self.notifier.calls[-1]
        self.assertEqual((PHONE, 2, True), (phone, len(events), digest))
        self.assertEqual({"digested"}, {e.status for e in self.repository.recent_events(PHONE)})
        self.assertEqual(0, scheduler.run_once(CHICAGO_NOON + timedelta(hours=1)))


class WhoopConnectorTests(unittest.TestCase):
    BODY = b'{"user_id":456,"id":"abc","type":"sleep.updated","trace_id":"t1"}'
    SIGNATURE = "Ls+OR3wjAsQkGg11somMjQa6j1k5msn1a8wcZ8GvRRM="

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = FileTokenStore(Path(self.temp.name) / "oauth")
        self.requests: list[httpx.Request] = []
        self.responses: dict[str, Any] = {}
        self.clock_value = 1_700_000_000.0

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            key = f"{request.method} {request.url.path}"
            value = self.responses.get(key)
            if callable(value):
                return value(request)
            if value is None:
                return httpx.Response(404)
            return httpx.Response(200, json=value)

        self.connector = WhoopConnector(
            client_id="client",
            client_secret="test-secret",
            redirect_url="https://agent.example/oauth/whoop/callback",
            token_store=self.store,
            http=httpx.Client(transport=httpx.MockTransport(handler)),
            clock=lambda: self.clock_value,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def link(self, *, expires_at: float | None = None) -> None:
        self.store.save("whoop", StoredOAuthTokens(
            phone=PHONE,
            client_id="client",
            access_token="access-1",
            refresh_token="refresh-1",
            expires_at=expires_at if expires_at is not None else self.clock_value + 3600,
        ))
        self.connector._accounts.link("456", PHONE)

    def test_signature_matches_known_vector(self) -> None:
        self.assertEqual(self.SIGNATURE, sign_payload("test-secret", "1700000000000", self.BODY))
        headers = {
            "x-whoop-signature": self.SIGNATURE,
            "x-whoop-signature-timestamp": "1700000000000",
        }
        self.assertTrue(self.connector.verify(headers, self.BODY))
        self.assertFalse(self.connector.verify(headers, self.BODY.replace(b"456", b"457")))
        self.assertFalse(self.connector.verify({**headers, "x-whoop-signature": "nope"}, self.BODY))
        self.clock_value += 3 * 60 * 60
        self.assertFalse(self.connector.verify(headers, self.BODY))

    def test_parse_maps_linked_users_and_ignores_unknown(self) -> None:
        body = json.loads(self.BODY)
        self.assertEqual([], self.connector.parse(body))
        self.link()
        [pending] = self.connector.parse(body)
        self.assertEqual((PHONE, "sleep.updated", "whoop:sleep.updated:abc:t1"),
                         (pending.principal_phone, pending.type, pending.dedupe_key))
        self.assertEqual([], self.connector.parse({**body, "type": "cycle.updated"}))

    def test_recovery_enrich_fetches_sleep_then_cycle_recovery(self) -> None:
        self.link()
        self.responses["GET /developer/v2/activity/sleep/sleep-1"] = {
            "id": "sleep-1",
            "cycle_id": 93845,
            "nap": False,
            "end": "2026-10-25T12:10:00.000Z",
            "score_state": "SCORED",
            "score": {
                "stage_summary": {
                    "total_in_bed_time_milli": 27_000_000,
                    "total_light_sleep_time_milli": 12_600_000,
                    "total_slow_wave_sleep_time_milli": 5_400_000,
                    "total_rem_sleep_time_milli": 5_400_000,
                },
                "sleep_performance_percentage": 71,
            },
        }
        self.responses["GET /developer/v2/cycle/93845/recovery"] = {
            "cycle_id": 93845,
            "sleep_id": "sleep-1",
            "updated_at": "2026-10-25T12:20:00.000Z",
            "score_state": "SCORED",
            "score": {
                "user_calibrating": False,
                "recovery_score": 28,
                "resting_heart_rate": 62.4,
                "hrv_rmssd_milli": 41.2,
            },
        }
        [pending] = self.connector.parse(
            {"user_id": 456, "id": "sleep-1", "type": "recovery.updated", "trace_id": "t2"}
        )

        [event] = self.connector.enrich(pending)

        self.assertEqual("recovery.updated", event.type)
        self.assertEqual(
            {"recovery_score": 28, "hrv_ms": 41, "resting_hr": 62, "calibrating": False,
             "sleep_hours": 6.5, "sleep_performance": 71},
            event.metrics,
        )
        self.assertEqual("Recovery 28% (HRV 41 ms, RHR 62 bpm)", event.summary)
        self.assertEqual("whoop:recovery:sleep-1:2026-10-25T12:20:00.000Z", event.dedupe_key)
        self.assertEqual(datetime(2026, 10, 25, 12, 10, tzinfo=timezone.utc), event.occurred_at)
        self.assertEqual("Bearer access-1", self.requests[0].headers["authorization"])

    def test_expired_token_is_refreshed_and_pending_scores_skipped(self) -> None:
        self.link(expires_at=self.clock_value - 10)
        self.responses["POST /oauth/oauth2/token"] = {
            "access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600,
        }
        self.responses["GET /developer/v2/activity/workout/w-1"] = {
            "id": "w-1", "score_state": "PENDING_SCORE",
        }
        [pending] = self.connector.parse(
            {"user_id": 456, "id": "w-1", "type": "workout.updated", "trace_id": "t3"}
        )
        self.assertEqual([], self.connector.enrich(pending))
        form = dict(httpx.QueryParams(self.requests[0].content.decode()))
        self.assertEqual(("refresh_token", "refresh-1"), (form["grant_type"], form["refresh_token"]))
        saved = self.store.load("whoop", PHONE)
        self.assertEqual(("access-2", "refresh-2"), (saved.access_token, saved.refresh_token))

    def test_oauth_link_round_trip(self) -> None:
        self.responses["POST /oauth/oauth2/token"] = {
            "access_token": "access-9", "refresh_token": "refresh-9", "expires_in": 3600,
        }
        self.responses["GET /developer/v2/user/profile/basic"] = {"user_id": 456}
        url = self.connector.authorization_url(PHONE)
        state = dict(httpx.URL(url).params)["state"]
        self.assertIn("offline", dict(httpx.URL(url).params)["scope"])

        self.assertEqual(PHONE, self.connector.complete_oauth("code-1", state))
        self.assertTrue(self.connector.status(PHONE)["linked"])
        self.assertEqual("456", self.connector.status(PHONE)["whoop_user_id"])
        with self.assertRaises(Exception):
            self.connector.complete_oauth("code-1", state)

        self.responses["DELETE /developer/v2/user/access"] = lambda _: httpx.Response(204)
        self.assertEqual({"status": "unlinked", "revoked": True}, self.connector.unlink(PHONE))
        self.assertFalse(self.connector.status(PHONE)["linked"])
        self.assertEqual([], self.connector.parse(json.loads(self.BODY)))


if __name__ == "__main__":
    unittest.main()
