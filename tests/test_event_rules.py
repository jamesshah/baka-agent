from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from connectors.registry import ConnectorRegistry
from events.rules import (
    ConditionError,
    RuleEngine,
    RuleError,
    build_rule,
    compile_condition,
    evaluate_condition,
)
from memory.database import Database
from memory.migrate import upgrade
from memory.repository import SqlAlchemyMemoryRepository
from tests.fakes import FakeConnector
from tools.event_rules import ManageEventRulesTool
from tools.session_context import reset_session_id, set_session_id

PHONE = "+15550001111"

DEFAULTS = """
- id: ignore_deletes
  match: { type: "*.deleted" }
  action: ignore
- id: low_recovery
  match: { source: fake, type: recovery.updated }
  when: "recovery_score < 34"
  action: notify
- id: fake_routine
  match: { source: fake }
  action: digest
"""


def evaluate(expression: str, **metrics) -> bool:
    return evaluate_condition(compile_condition(expression), metrics)


class ConditionTests(unittest.TestCase):
    def test_comparisons_and_boolean_logic(self) -> None:
        self.assertTrue(evaluate("recovery_score < 34", recovery_score=20))
        self.assertFalse(evaluate("recovery_score < 34", recovery_score=50))
        self.assertTrue(evaluate("not nap and sleep_hours < 5.5", nap=False, sleep_hours=4.9))
        self.assertFalse(evaluate("not nap and sleep_hours < 5.5", nap=True, sleep_hours=4.9))
        self.assertTrue(evaluate("20 <= strain < 30", strain=21))
        self.assertTrue(evaluate("sport in ['running', 'cycling']", sport="running"))
        self.assertTrue(evaluate("sleep_hours * 60 < 300", sleep_hours=4))
        self.assertTrue(evaluate("calibrating == true", calibrating=True))

    def test_missing_metrics_and_type_errors_are_false(self) -> None:
        self.assertFalse(evaluate("recovery_score < 34"))
        self.assertFalse(evaluate("recovery_score < 34", recovery_score=None))
        self.assertFalse(evaluate("sport > 3", sport="running"))
        self.assertTrue(evaluate("missing > 1 or strain > 10", strain=12))

    def test_unsafe_expressions_are_rejected(self) -> None:
        for expression in (
            "__import__('os').system('id')",
            "open('x')",
            "metrics.items",
            "values[0]",
            "(lambda: 1)()",
            "[x for x in range(3)]",
            "__class__",
            "a if b else c",
            "x ** 999999",
            "",
            "x <" * 200,
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ConditionError):
                    compile_condition(expression)

    def test_unknown_metric_names_are_rejected_when_schema_known(self) -> None:
        compile_condition("recovery_score < 34", {"recovery_score"})
        with self.assertRaisesRegex(ConditionError, "unknown metric 'recovry'"):
            compile_condition("recovry < 34", {"recovery_score"})

    def test_build_rule_validates_fields(self) -> None:
        with self.assertRaises(RuleError):
            build_rule(rule_id="Bad Id", action="notify")
        with self.assertRaises(RuleError):
            build_rule(rule_id="ok", action="shout")
        with self.assertRaises(RuleError):
            build_rule(rule_id="ok", action="notify", match={"merchant": "x"})
        with self.assertRaises(RuleError):
            build_rule(rule_id="ok", action="notify", when="open('x')")


class _RulesFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "rules.db")
        upgrade(self.db_path)
        self.database = Database(self.db_path)
        self.repository = SqlAlchemyMemoryRepository(self.database)
        self.rules_path = Path(self.temp.name) / "rules.yaml"
        self.rules_path.write_text(DEFAULTS, encoding="utf-8")
        self.connectors = ConnectorRegistry([FakeConnector()])
        self.engine = RuleEngine(
            self.rules_path,
            self.repository,
            metric_names=self.connectors.metric_names,
        )

    def tearDown(self) -> None:
        self.database.close()
        self.temp.cleanup()

    def match_id(self, event_type: str, **metrics) -> str | None:
        rule = self.engine.match(PHONE, "fake", event_type, metrics)
        return rule.id if rule is not None else None


class RuleEngineTests(_RulesFixture):
    def test_defaults_match_in_file_order_with_globs(self) -> None:
        self.assertEqual("ignore_deletes", self.match_id("recovery.deleted"))
        self.assertEqual("low_recovery", self.match_id("recovery.updated", recovery_score=20))
        self.assertEqual("fake_routine", self.match_id("recovery.updated", recovery_score=80))
        self.assertIsNone(self.engine.match(PHONE, "other", "thing.updated", {}))

    def test_user_rules_take_priority_and_disabled_rules_are_skipped(self) -> None:
        self.repository.put_event_rule(
            PHONE,
            rule_id="mute_recovery",
            match={"source": "fake", "type": "recovery.updated"},
            when=None,
            action="ignore",
        )
        self.assertEqual("mute_recovery", self.match_id("recovery.updated", recovery_score=20))
        other_user = self.engine.match(
            "+15559999999", "fake", "recovery.updated", {"recovery_score": 20}
        )
        self.assertEqual("low_recovery", other_user.id)
        self.repository.set_event_rule_enabled(PHONE, "mute_recovery", False)
        self.assertEqual("low_recovery", self.match_id("recovery.updated", recovery_score=20))

    def test_newest_user_rule_wins(self) -> None:
        for rule_id, action in (("first", "digest"), ("second", "remember")):
            self.repository.put_event_rule(
                PHONE, rule_id=rule_id, match={"source": "fake"}, when=None, action=action,
            )
            time.sleep(0.01)
        self.assertEqual("second", self.match_id("sleep.updated"))

    def test_yaml_reloads_when_file_changes(self) -> None:
        self.assertEqual("fake_routine", self.match_id("sleep.updated"))
        self.rules_path.write_text(
            "- id: everything_remembered\n  action: remember\n", encoding="utf-8"
        )
        future = time.time() + 5
        os.utime(self.rules_path, (future, future))
        self.assertEqual("everything_remembered", self.match_id("sleep.updated"))

    def test_invalid_default_rules_are_skipped(self) -> None:
        self.rules_path.write_text(
            "- id: bad\n  match: { source: fake }\n  when: \"nope < 1\"\n  action: notify\n"
            "- id: good\n  action: digest\n",
            encoding="utf-8",
        )
        future = time.time() + 10
        os.utime(self.rules_path, (future, future))
        self.assertEqual(["good"], [rule.id for rule in self.engine.defaults()])


class ManageEventRulesToolTests(_RulesFixture):
    def run_tool(self, **kwargs):
        tool = ManageEventRulesTool(self.repository, self.engine, self.connectors)
        token = set_session_id(PHONE)
        try:
            return tool.execute(**kwargs)
        finally:
            reset_session_id(token)

    def test_create_list_disable_delete(self) -> None:
        created = self.run_tool(
            operation="create",
            rule_id="recovery_under_40",
            source="fake",
            type="recovery.updated",
            when="recovery_score < 40",
            action="notify",
            urgent=True,
            note="I want to know before training",
        )
        self.assertEqual("ok", created["status"])
        self.assertEqual("recovery_under_40", self.match_id("recovery.updated", recovery_score=38))

        listed = self.run_tool(operation="list")
        self.assertEqual(["recovery_under_40"], [r["id"] for r in listed["user_rules"]])
        self.assertTrue(listed["user_rules"][0]["urgent"])
        self.assertIn("low_recovery", [r["id"] for r in listed["default_rules"]])
        self.assertIn("recovery_score", listed["sources"]["fake"]["metrics"])

        self.assertEqual("ok", self.run_tool(operation="disable", rule_id="recovery_under_40")["status"])
        self.assertEqual("fake_routine", self.match_id("recovery.updated", recovery_score=38))
        self.assertEqual("ok", self.run_tool(operation="delete", rule_id="recovery_under_40")["status"])
        self.assertEqual([], self.run_tool(operation="list")["user_rules"])
        self.assertEqual("not_found", self.run_tool(operation="delete", rule_id="recovery_under_40")["status"])

    def test_create_rejects_unknown_source_metric_and_bad_action(self) -> None:
        unknown_source = self.run_tool(
            operation="create", rule_id="x", source="plaid", action="notify"
        )
        self.assertEqual("error", unknown_source["status"])
        self.assertIn("unknown source", unknown_source["detail"])

        unknown_metric = self.run_tool(
            operation="create", rule_id="x", source="fake", when="recovry < 1", action="notify"
        )
        self.assertEqual("error", unknown_metric["status"])
        self.assertIn("unknown metric", unknown_metric["detail"])

        bad_action = self.run_tool(operation="create", rule_id="x", action="yell")
        self.assertEqual("error", bad_action["status"])
        self.assertEqual([], self.repository.list_event_rules(PHONE))


if __name__ == "__main__":
    unittest.main()
