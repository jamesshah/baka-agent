"""Per-turn session identity for tool execution (phone number + turn)."""

from __future__ import annotations

from contextvars import ContextVar, Token

_session_id: ContextVar[str | None] = ContextVar("tool_session_id", default=None)
_turn_id: ContextVar[str | None] = ContextVar("tool_turn_id", default=None)


def set_session_id(session_id: str) -> Token[str | None]:
    return _session_id.set(session_id)


def reset_session_id(token: Token[str | None]) -> None:
    _session_id.reset(token)


def get_session_id() -> str | None:
    return _session_id.get()


def require_session_id() -> str:
    value = _session_id.get()
    if not value:
        raise RuntimeError("No session_id in tool context")
    return value


def set_turn_id(turn_id: str) -> Token[str | None]:
    return _turn_id.set(turn_id)


def reset_turn_id(token: Token[str | None]) -> None:
    _turn_id.reset(token)


def get_turn_id() -> str | None:
    return _turn_id.get()
