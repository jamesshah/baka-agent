"""Local-first MLflow tracing with PII redaction for baka-agent."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from log_redact import redact_data, redact_text

logger = logging.getLogger(__name__)

_SECRET_KEYS = frozenset(
    {
        "password",
        "passcode",
        "api_key",
        "apikey",
        "api_secret",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "credentials",
        "authorization",
    }
)
_CONTEXT_BLOCK_RE = re.compile(r"<context>\s*(.*?)\s*</context>", re.DOTALL)

_lock = threading.Lock()
_enabled = False
_tracking_uri: str | None = None
_experiment: str | None = None
_last_error: str | None = None
_mlflow: Any = None


def _resolve_tracking_uri(uri: str) -> str:
    """Normalize sqlite:/// relative paths and ensure parent directories exist."""
    cleaned = (uri or "").strip()
    if not cleaned:
        cleaned = "sqlite:///.data/mlflow.db"
    prefix = "sqlite:///"
    if cleaned.startswith(prefix) and cleaned != "sqlite:///:memory:":
        raw_path = cleaned[len(prefix) :]
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"{prefix}{path}"
    return cleaned


def _artifact_location_for_uri(resolved_uri: str) -> str | None:
    """Keep MLflow artifact directories next to a local SQLite DB (never ./mlruns)."""
    prefix = "sqlite:///"
    if resolved_uri.startswith(prefix) and resolved_uri != "sqlite:///:memory:":
        db_path = Path(resolved_uri[len(prefix) :])
        artifacts = db_path.parent / "mlflow-artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        return artifacts.as_uri()
    return None


def configure_tracing(
    *,
    enabled: bool = True,
    tracking_uri: str = "sqlite:///.data/mlflow.db",
    experiment: str = "baka-agent",
) -> bool:
    """Initialize MLflow tracing once. Never raises; disables cleanly on error."""
    global _enabled, _tracking_uri, _experiment, _last_error, _mlflow

    with _lock:
        _last_error = None
        if not enabled:
            _enabled = False
            _tracking_uri = tracking_uri
            _experiment = experiment
            if _mlflow is not None:
                try:
                    _mlflow.tracing.disable()
                except Exception:  # noqa: BLE001
                    logger.debug("mlflow.tracing.disable() failed", exc_info=True)
            return False

        try:
            import mlflow  # type: ignore[import-untyped]
        except ImportError as exc:
            _enabled = False
            _last_error = f"mlflow not installed: {exc}"
            logger.warning("MLflow tracing unavailable: %s", exc)
            return False

        try:
            resolved_uri = _resolve_tracking_uri(tracking_uri)
            mlflow.set_tracking_uri(resolved_uri)
            artifact_loc = _artifact_location_for_uri(resolved_uri)
            if artifact_loc and mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=artifact_loc)
            mlflow.set_experiment(experiment_name=experiment)
            mlflow.tracing.configure(span_processors=[_redact_live_span])
            mlflow.tracing.enable()
            _mlflow = mlflow
            _tracking_uri = resolved_uri
            _experiment = experiment
            _enabled = True
            logger.info(
                "MLflow tracing ready: uri=%s experiment=%s",
                resolved_uri,
                experiment,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            _enabled = False
            _last_error = str(exc)
            logger.exception("Failed to configure MLflow tracing; continuing disabled")
            return False


def is_tracing_enabled() -> bool:
    return _enabled and _mlflow is not None


def tracing_health() -> dict[str, Any]:
    """Status block for GET /health."""
    with _lock:
        enabled = _enabled
        uri = _tracking_uri
        experiment = _experiment
        last_error = _last_error

    if last_error:
        return {
            "status": "degraded",
            "backend": "mlflow",
            "tracking_uri": uri,
            "experiment": experiment,
            "detail": last_error,
        }
    if not enabled:
        return {
            "status": "disabled",
            "backend": "mlflow",
            "tracking_uri": uri,
            "experiment": experiment,
        }
    return {
        "status": "ok",
        "backend": "mlflow",
        "tracking_uri": uri,
        "experiment": experiment,
        "ui_command": "./run.sh mlflow",
    }


def flush_traces() -> None:
    """Flush any buffered MLflow traces on shutdown or test readback."""
    mlflow = _mlflow
    if mlflow is None:
        return
    try:
        mlflow.flush_trace_async_logging()
    except Exception:  # noqa: BLE001
        logger.debug("mlflow.flush_trace_async_logging() failed", exc_info=True)


def reset_tracing_for_tests() -> None:
    """Test helper: disable tracing and clear module state."""
    global _enabled, _tracking_uri, _experiment, _last_error
    with _lock:
        _enabled = False
        _tracking_uri = None
        _experiment = None
        _last_error = None
        if _mlflow is not None:
            try:
                _mlflow.tracing.disable()
            except Exception:  # noqa: BLE001
                pass


def session_hash(session_id: str | None) -> str | None:
    """Stable non-PII session identifier for trace grouping."""
    if not session_id or not session_id.strip():
        return None
    digest = hashlib.sha256(session_id.strip().encode("utf-8")).hexdigest()
    return f"sess_{digest[:12]}"


def sanitize_for_trace(value: Any) -> Any:
    """Redact PII, mask secret keys, and strip large inline base64 media."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:80]:
            header, _, payload = value.partition(";base64,")
            mime = header[len("data:") :] or "application/octet-stream"
            return f"<media-base64 mime={mime} chars={len(payload)}>"
        return redact_text(value)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str.casefold() in _SECRET_KEYS:
                cleaned[key_str] = "<REDACTED>"
                continue
            if (
                key_str == "data"
                and isinstance(item, str)
                and len(item) > 512
                and value.get("format") in {"wav", "mp3", "m4a", "ogg", "flac", "pcm"}
            ):
                cleaned[key_str] = f"<audio-base64 format={value.get('format')} chars={len(item)}>"
                continue
            cleaned[key_str] = sanitize_for_trace(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize_for_trace(item) for item in value]
    return redact_data(str(value))


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a PII-redacted, media-stripped copy of chat messages."""
    return [sanitize_for_trace(message) for message in messages]


def prompt_breakdown(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate where prompt characters/tokens come from (system, tools, history, context, tool outputs)."""
    system_chars = 0
    history_chars = 0
    history_messages = 0
    context_block_chars = 0
    tool_result_chars = 0
    tool_result_messages = 0
    current_user_chars = 0

    last_user_index = -1
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_index = idx

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        text = _message_text_size(msg)
        if role == "system":
            system_chars += text
        elif role == "tool":
            tool_result_chars += text
            tool_result_messages += 1
        elif idx == last_user_index:
            ctx_chars = _context_block_size(msg.get("content"))
            context_block_chars += ctx_chars
            current_user_chars += max(0, text - ctx_chars)
        else:
            history_chars += text
            history_messages += 1

    tool_names: list[str] = []
    tools_chars = 0
    if tools:
        try:
            tools_chars = len(json.dumps(tools, separators=(",", ":"), default=str))
        except Exception:  # noqa: BLE001
            tools_chars = 0
        for spec in tools:
            fn = spec.get("function") if isinstance(spec, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                tool_names.append(str(fn["name"]))

    total_chars = (
        system_chars
        + tools_chars
        + history_chars
        + context_block_chars
        + current_user_chars
        + tool_result_chars
    )
    return {
        "system_chars": system_chars,
        "tools_chars": tools_chars,
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "history_messages": history_messages,
        "history_chars": history_chars,
        "context_block_chars": context_block_chars,
        "current_user_chars": current_user_chars,
        "tool_result_messages": tool_result_messages,
        "tool_result_chars": tool_result_chars,
        "total_chars": total_chars,
        "estimated_tokens": round(total_chars / 4.0),
    }


def _message_text_size(msg: dict[str, Any]) -> int:
    total = 0
    content = msg.get("content")
    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                total += len(str(part.get("text") or ""))
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        try:
            total += len(json.dumps(tool_calls, separators=(",", ":"), default=str))
        except Exception:  # noqa: BLE001
            pass
    return total


def _context_block_size(content: Any) -> int:
    if isinstance(content, str):
        match = _CONTEXT_BLOCK_RE.search(content)
        return len(match.group(0)) if match else 0
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                match = _CONTEXT_BLOCK_RE.search(str(part.get("text") or ""))
                if match:
                    return len(match.group(0))
    return 0


def _redact_live_span(span: Any) -> None:
    """Defense-in-depth MLflow span processor that scrubs PII before export."""
    try:
        if span.inputs is not None:
            span.set_inputs(sanitize_for_trace(span.inputs))
        if span.outputs is not None:
            span.set_outputs(sanitize_for_trace(span.outputs))
    except Exception:  # noqa: BLE001
        logger.debug("span redaction processor failed", exc_info=True)


class SpanHandle:
    """Thin wrapper over an active MLflow LiveSpan (or a no-op when disabled)."""

    def __init__(self, span: Any = None) -> None:
        self._span = span

    @property
    def active(self) -> bool:
        return self._span is not None

    def set_inputs(self, data: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.set_inputs(sanitize_for_trace(data))
        except Exception:  # noqa: BLE001
            logger.debug("span.set_inputs failed", exc_info=True)

    def set_outputs(self, data: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.set_outputs(sanitize_for_trace(data))
        except Exception:  # noqa: BLE001
            logger.debug("span.set_outputs failed", exc_info=True)

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is None or value is None:
            return
        try:
            self._span.set_attribute(key, sanitize_for_trace(value))
        except Exception:  # noqa: BLE001
            logger.debug("span.set_attribute(%s) failed", key, exc_info=True)

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        if self._span is None or not attrs:
            return
        cleaned = {
            key: sanitize_for_trace(val)
            for key, val in attrs.items()
            if val is not None
        }
        if not cleaned:
            return
        try:
            self._span.set_attributes(cleaned)
        except Exception:  # noqa: BLE001
            logger.debug("span.set_attributes failed", exc_info=True)

    def set_status_error(self, message: str) -> None:
        if self._span is None:
            return
        try:
            from mlflow.entities import SpanStatus, SpanStatusCode  # type: ignore[import-untyped]

            self._span.set_status(
                SpanStatus(SpanStatusCode.ERROR, redact_text(message)[:500])
            )
        except Exception:  # noqa: BLE001
            logger.debug("span.set_status failed", exc_info=True)

    def update_trace(
        self,
        *,
        tags: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        session_id: str | None = None,
        request_preview: str | None = None,
        response_preview: str | None = None,
    ) -> None:
        if self._span is None or _mlflow is None:
            return
        kwargs: dict[str, Any] = {}
        if tags:
            kwargs["tags"] = {
                k: redact_text(str(v)) for k, v in tags.items() if v is not None
            }
        if metadata:
            kwargs["metadata"] = {
                k: redact_text(str(v)) for k, v in metadata.items() if v is not None
            }
        if session_id:
            kwargs["session_id"] = session_id
        if request_preview is not None:
            kwargs["request_preview"] = redact_text(request_preview)[:250]
        if response_preview is not None:
            kwargs["response_preview"] = redact_text(response_preview)[:250]
        if not kwargs:
            return
        try:
            _mlflow.update_current_trace(**kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("mlflow.update_current_trace failed", exc_info=True)


_NOOP_SPAN = SpanHandle(None)


@contextmanager
def trace_span(
    name: str,
    *,
    span_type: str = "UNKNOWN",
    inputs: Any = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[SpanHandle]:
    """Open an MLflow span when tracing is enabled, else yield a no-op handle."""
    if not is_tracing_enabled():
        yield _NOOP_SPAN
        return

    try:
        cm = _mlflow.start_span(name=name, span_type=span_type)
        live_span = cm.__enter__()
    except Exception:  # noqa: BLE001
        logger.debug("mlflow.start_span(%s) failed", name, exc_info=True)
        yield _NOOP_SPAN
        return

    handle = SpanHandle(live_span)
    if inputs is not None:
        handle.set_inputs(inputs)
    if attributes:
        handle.set_attributes(attributes)

    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield handle
    except BaseException as exc:
        exc_info = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception:  # noqa: BLE001
            logger.debug("mlflow span exit(%s) failed", name, exc_info=True)


def set_llm_span_metrics(
    span: SpanHandle,
    metrics: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> None:
    """Attach standard MLflow LLM token usage + llama.cpp timing attributes."""
    if not span.active:
        return

    prompt_tokens = metrics.get("prompt_tokens")
    completion_tokens = metrics.get("completion_tokens")
    cache_tokens = metrics.get("cache_tokens")

    token_usage: dict[str, int] = {}
    if isinstance(prompt_tokens, int):
        token_usage["input_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        token_usage["output_tokens"] = completion_tokens
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        token_usage["total_tokens"] = prompt_tokens + completion_tokens
    if isinstance(cache_tokens, int):
        token_usage["cache_read_input_tokens"] = cache_tokens

    attrs: dict[str, Any] = {
        "mlflow.message.format": "openai",
        "mlflow.llm.provider": "llama.cpp",
        "mlflow.llm.model": metrics.get("model"),
        "llm.finish_reason": metrics.get("finish_reason"),
        "llm.tool_call_count": metrics.get("tool_calls"),
        "llm.tokens_in_context": metrics.get("tokens_in_context"),
        "llm.n_ctx": metrics.get("n_ctx"),
        "llm.context_used_pct": metrics.get("context_used_pct"),
        "llm.cache_tokens": cache_tokens,
        "llm.cache_hit_ratio": metrics.get("cache_hit_ratio"),
        "llm.prefill_tokens": metrics.get("prefill_tokens"),
        "llm.prefill_ms": metrics.get("prefill_ms"),
        "llm.prefill_tok_s": metrics.get("prefill_tok_s"),
        "llm.decode_ms": metrics.get("decode_ms"),
        "llm.decode_tok_s": metrics.get("decode_tok_s"),
        "llm.ttft_ms": metrics.get("ttft_ms"),
        "llm.wall_ms": metrics.get("wall_ms"),
    }
    if token_usage:
        attrs["mlflow.chat.tokenUsage"] = token_usage
    if tools:
        attrs["mlflow.chat.tools"] = tools

    span.set_attributes(attrs)
