"""Let the user shape which connector events text them, from chat."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from events.rules import ACTIONS, RuleError
from tools.base import Tool
from tools.session_context import require_session_id

if TYPE_CHECKING:
    from connectors.registry import ConnectorRegistry
    from events.rules import RuleEngine
    from memory.repository import SqlAlchemyMemoryRepository


class ManageEventRulesTool(Tool):
    name = "manage_event_rules"
    description = (
        "List, create, enable, disable, or delete the user's rules for "
        "connected-service events (e.g. Whoop). Use when the user says things "
        "like 'tell me if my recovery is under 40', 'stop texting me about "
        "workouts', or 'what alerts do I have'. Actions: notify (text now), "
        "digest (daily summary), remember (store quietly), ignore. User rules "
        "override the defaults. Call list first to see sources, event types, "
        "and metric names."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "create", "enable", "disable", "delete"],
            },
            "rule_id": {
                "type": "string",
                "description": (
                    "Short snake_case id, e.g. low_recovery_alert. Creating "
                    "with an existing id replaces that rule. To silence a "
                    "default rule, create a user rule with the same match "
                    "and action ignore."
                ),
            },
            "source": {
                "type": "string",
                "description": "Connector name, e.g. whoop. Omit for any source.",
            },
            "type": {
                "type": "string",
                "description": (
                    "Event type, e.g. recovery.updated; globs like "
                    "'*.updated' allowed. Omit for any type."
                ),
            },
            "when": {
                "type": "string",
                "description": (
                    "Optional condition over metrics, e.g. "
                    "'recovery_score < 40' or 'not nap and sleep_hours < 6'."
                ),
            },
            "action": {"type": "string", "enum": list(ACTIONS)},
            "urgent": {
                "type": "boolean",
                "description": "Text even during quiet hours or past the daily cap.",
            },
            "note": {
                "type": "string",
                "description": "Why the user wants this, in their words.",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        repository: SqlAlchemyMemoryRepository,
        engine: RuleEngine,
        connectors: ConnectorRegistry,
    ) -> None:
        self._repository = repository
        self._engine = engine
        self._connectors = connectors

    def is_chat_agent_tool(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> Any:
        phone = require_session_id()
        operation = str(kwargs.get("operation") or "").strip()
        rule_id = str(kwargs.get("rule_id") or "").strip()

        if operation == "list":
            return self._list(phone)
        if operation == "create":
            return self._create(phone, rule_id, kwargs)
        if operation in {"enable", "disable"}:
            if not rule_id:
                return {"status": "error", "detail": "rule_id is required"}
            changed = self._repository.set_event_rule_enabled(
                phone, rule_id, operation == "enable"
            )
            return {
                "status": "ok" if changed else "not_found",
                "rule_id": rule_id,
                "enabled": operation == "enable",
            }
        if operation == "delete":
            if not rule_id:
                return {"status": "error", "detail": "rule_id is required"}
            deleted = self._repository.delete_event_rule(phone, rule_id)
            return {"status": "ok" if deleted else "not_found", "rule_id": rule_id}
        return {"status": "error", "detail": f"unsupported operation: {operation}"}

    def _list(self, phone: str) -> dict[str, Any]:
        user_rules = []
        for record in self._repository.list_event_rules(phone):
            entry = {
                "id": record.rule_id,
                "origin": "user",
                "enabled": record.enabled,
                "match": record.match or {"source": "*", "type": "*"},
                "action": record.action,
            }
            if record.when:
                entry["when"] = record.when
            if record.urgent:
                entry["urgent"] = True
            if record.note:
                entry["note"] = record.note
            user_rules.append(entry)
        return {
            "evaluation": (
                "User rules (newest first) are checked before defaults; "
                "the first match wins. Unmatched events are triaged by the model."
            ),
            "user_rules": user_rules,
            "default_rules": [rule.describe() for rule in self._engine.defaults()],
            "sources": {
                connector.name: {
                    "event_types": connector.event_types(),
                    "metrics": connector.metric_schema(),
                }
                for connector in self._connectors.all()
            },
        }

    def _create(
        self, phone: str, rule_id: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        source = str(kwargs.get("source") or "").strip()
        event_type = str(kwargs.get("type") or "").strip()
        if source and not any(c in source for c in "*?[") and (
            self._connectors.get(source) is None
        ):
            known = ", ".join(self._connectors.names()) or "none connected"
            return {
                "status": "error",
                "detail": f"unknown source '{source}' (available: {known})",
            }
        match = {}
        if source:
            match["source"] = source
        if event_type:
            match["type"] = event_type
        try:
            rule = self._engine.validate(
                rule_id=rule_id,
                action=str(kwargs.get("action") or ""),
                match=match,
                when=kwargs.get("when"),
                urgent=bool(kwargs.get("urgent", False)),
                note=str(kwargs.get("note") or ""),
            )
        except RuleError as exc:
            return {"status": "error", "detail": str(exc)}
        self._repository.put_event_rule(
            phone,
            rule_id=rule.id,
            match=match,
            when=rule.when,
            action=rule.action,
            urgent=rule.urgent,
            note=rule.note,
        )
        return {"status": "ok", "rule": rule.describe()}
