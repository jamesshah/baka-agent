"""Rule matching: user rules (newest first), then YAML defaults (file order)."""

from __future__ import annotations

import ast
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from events.rules.condition import (
    ConditionError,
    compile_condition,
    evaluate_condition,
)

if TYPE_CHECKING:
    from memory.repository import SqlAlchemyMemoryRepository
    from memory.store import EventRuleRecord

logger = logging.getLogger(__name__)

ACTIONS = ("notify", "digest", "remember", "ignore")
_RULE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Metric names for a source, or None when the source is unknown.
MetricLookup = Callable[[str], "set[str] | None"]


class RuleError(ValueError):
    """A rule definition is invalid."""


@dataclass(frozen=True)
class Rule:
    id: str
    action: str
    source: str = "*"
    type: str = "*"
    when: str | None = None
    urgent: bool = False
    note: str = ""
    origin: str = "default"
    condition: ast.Expression | None = field(
        default=None, compare=False, repr=False
    )

    def matches(self, source: str, event_type: str, metrics: dict[str, Any]) -> bool:
        if not fnmatchcase(source, self.source):
            return False
        if not fnmatchcase(event_type, self.type):
            return False
        if self.condition is None:
            return True
        return evaluate_condition(self.condition, metrics)

    def describe(self) -> dict[str, Any]:
        described: dict[str, Any] = {
            "id": self.id,
            "origin": self.origin,
            "match": {"source": self.source, "type": self.type},
            "action": self.action,
        }
        if self.when:
            described["when"] = self.when
        if self.urgent:
            described["urgent"] = True
        if self.note:
            described["note"] = self.note
        return described


def build_rule(
    *,
    rule_id: str,
    action: str,
    match: dict[str, Any] | None = None,
    when: str | None = None,
    urgent: bool = False,
    note: str = "",
    origin: str = "default",
    metric_names: MetricLookup | None = None,
) -> Rule:
    """Validate and compile one rule; raise RuleError when it is invalid."""
    rule_id = (rule_id or "").strip()
    if not _RULE_ID_RE.match(rule_id):
        raise RuleError(
            "rule id must be 1-64 lowercase letters, digits, '_' or '-'"
        )
    action = (action or "").strip().lower()
    if action not in ACTIONS:
        raise RuleError(f"action must be one of {', '.join(ACTIONS)}")
    match = dict(match or {})
    unknown_keys = set(match) - {"source", "type"}
    if unknown_keys:
        raise RuleError(f"unknown match keys: {', '.join(sorted(unknown_keys))}")
    source = str(match.get("source") or "*").strip() or "*"
    event_type = str(match.get("type") or "*").strip() or "*"

    condition = None
    when = (when or "").strip() or None
    if when is not None:
        allowed = None
        if metric_names is not None and not _is_glob(source):
            allowed = metric_names(source)
        try:
            condition = compile_condition(when, allowed)
        except ConditionError as exc:
            raise RuleError(f"invalid when: {exc}") from None

    return Rule(
        id=rule_id,
        action=action,
        source=source,
        type=event_type,
        when=when,
        urgent=bool(urgent),
        note=str(note or ""),
        origin=origin,
        condition=condition,
    )


def _is_glob(value: str) -> bool:
    return any(char in value for char in "*?[")


class RuleEngine:
    """Resolve the first matching rule for an event."""

    def __init__(
        self,
        defaults_path: str | Path,
        repository: SqlAlchemyMemoryRepository | None = None,
        *,
        metric_names: MetricLookup | None = None,
    ) -> None:
        self._path = Path(defaults_path)
        self._repository = repository
        self._metric_names = metric_names
        self._lock = threading.Lock()
        self._defaults: list[Rule] = []
        self._mtime: float | None = None

    def defaults(self) -> list[Rule]:
        """YAML default rules, reloaded when the file's mtime changes."""
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            mtime = None
        with self._lock:
            if mtime != self._mtime:
                self._defaults = self._load_defaults() if mtime is not None else []
                self._mtime = mtime
            return list(self._defaults)

    def user_rules(self, phone: str) -> list[Rule]:
        if self._repository is None or not phone:
            return []
        rules: list[Rule] = []
        for record in self._repository.list_event_rules(phone, enabled_only=True):
            try:
                rules.append(self.from_record(record))
            except RuleError as exc:
                logger.warning("skipping invalid user rule %s: %s", record.rule_id, exc)
        return rules

    def from_record(self, record: EventRuleRecord) -> Rule:
        return build_rule(
            rule_id=record.rule_id,
            action=record.action,
            match=record.match,
            when=record.when,
            urgent=record.urgent,
            note=record.note,
            origin="user",
        )

    def match(
        self,
        phone: str,
        source: str,
        event_type: str,
        metrics: dict[str, Any],
    ) -> Rule | None:
        for rule in [*self.user_rules(phone), *self.defaults()]:
            if rule.matches(source, event_type, metrics):
                return rule
        return None

    def validate(
        self,
        *,
        rule_id: str,
        action: str,
        match: dict[str, Any] | None,
        when: str | None,
        urgent: bool = False,
        note: str = "",
    ) -> Rule:
        """Validate a user rule against known connectors and metrics."""
        return build_rule(
            rule_id=rule_id,
            action=action,
            match=match,
            when=when,
            urgent=urgent,
            note=note,
            origin="user",
            metric_names=self._metric_names,
        )

    def _load_defaults(self) -> list[Rule]:
        try:
            data = yaml.safe_load(self._path.read_text(encoding="utf-8")) or []
        except (OSError, yaml.YAMLError):
            logger.exception("failed to read event rules from %s", self._path)
            return []
        if not isinstance(data, list):
            logger.error("event rules file %s must be a YAML list", self._path)
            return []
        rules: list[Rule] = []
        seen: set[str] = set()
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                logger.error("event rule #%d in %s is not a mapping", index, self._path)
                continue
            try:
                rule = build_rule(
                    rule_id=str(item.get("id") or ""),
                    action=str(item.get("action") or ""),
                    match=item.get("match") or {},
                    when=item.get("when"),
                    urgent=bool(item.get("urgent", False)),
                    note=str(item.get("note") or ""),
                    origin="default",
                    metric_names=self._metric_names,
                )
            except RuleError as exc:
                logger.error(
                    "skipping invalid default rule #%d (%s): %s",
                    index, item.get("id"), exc,
                )
                continue
            if rule.id in seen:
                logger.error("duplicate default rule id %s; keeping the first", rule.id)
                continue
            seen.add(rule.id)
            rules.append(rule)
        logger.info("loaded %d default event rule(s) from %s", len(rules), self._path)
        return rules
