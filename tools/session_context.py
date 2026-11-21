"""Per-turn session identity for tool execution (phone number + turn)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_session_id: ContextVar[str | None] = ContextVar("tool_session_id", default=None)
_turn_id: ContextVar[str | None] = ContextVar("tool_turn_id", default=None)
_llm_label: ContextVar[str | None] = ContextVar("llm_call_label", default=None)
_llm_iteration: ContextVar[int | None] = ContextVar("llm_call_iteration", default=None)


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


def get_llm_label() -> str | None:
    return _llm_label.get()


def get_llm_iteration() -> int | None:
    return _llm_iteration.get()


@contextmanager
def llm_call_context(
    label: str | None,
    iteration: int | None = None,
) -> Iterator[None]:
    """Attach caller label and loop iteration for LLM perf logs and traces."""
    label_token = _llm_label.set(label)
    iter_token = _llm_iteration.set(iteration)
    try:
        yield
    finally:
        _llm_iteration.reset(iter_token)
        _llm_label.reset(label_token)
