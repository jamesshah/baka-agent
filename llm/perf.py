"""Per-completion llama.cpp performance metrics (tokens, prefill, decode, TTFT)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PERF_LOG_PATH = Path(__file__).resolve().parent.parent / ".run" / "logs" / "llm-perf.jsonl"

_lock = threading.Lock()
_path_logged = False
_cache_completions = 0
_cache_prompt_tokens = 0
_cache_cached_tokens = 0
_cache_prefill_tokens = 0
_cache_last: dict[str, Any] | None = None


def extract_metrics(
    data: dict[str, Any],
    *,
    wall_ms: float,
    n_ctx: int | None = None,
    model: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Pull usage/timings from a llama-server (or OpenAI-shaped) completion."""
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage.get("prompt_tokens_details"), dict)
        else {}
    )
    choice = _first_choice(data)

    cache_n = _as_int(timings.get("cache_n"))
    if cache_n is None:
        cache_n = _as_int(details.get("cached_tokens"))

    prompt_n = _as_int(timings.get("prompt_n"))  # tokens actually prefilled
    predicted_n = _as_int(timings.get("predicted_n"))

    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    if prompt_tokens is None and prompt_n is not None:
        prompt_tokens = prompt_n + (cache_n or 0)

    completion_tokens = _as_int(usage.get("completion_tokens"))
    if completion_tokens is None:
        completion_tokens = predicted_n

    tokens_in_context: int | None = None
    if prompt_n is not None or cache_n is not None or predicted_n is not None:
        tokens_in_context = (cache_n or 0) + (prompt_n or 0) + (predicted_n or 0)
    elif prompt_tokens is not None and completion_tokens is not None:
        tokens_in_context = prompt_tokens + completion_tokens

    prefill_ms = _as_float(timings.get("prompt_ms"))
    decode_ms = _as_float(timings.get("predicted_ms"))
    prefill_tok_s = _as_float(timings.get("prompt_per_second"))
    decode_tok_s = _as_float(timings.get("predicted_per_second"))
    first_token_ms = _as_float(timings.get("predicted_per_token_ms"))

    # TTFT ≈ prefill + first decode token (server-side; non-streaming).
    ttft_ms: float | None = None
    if prefill_ms is not None:
        ttft_ms = prefill_ms + (first_token_ms or 0.0)

    context_used_pct: float | None = None
    if n_ctx and n_ctx > 0 and tokens_in_context is not None:
        context_used_pct = round(100.0 * tokens_in_context / n_ctx, 2)

    cache_hit_ratio: float | None = None
    if prompt_tokens and cache_n is not None and prompt_tokens > 0:
        cache_hit_ratio = round(cache_n / prompt_tokens, 4)

    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    tool_calls = message.get("tool_calls") or []

    metrics: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "turn_id": turn_id,
        "label": label,
        "model": data.get("model") or model,
        "id": data.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_in_context": tokens_in_context,
        "n_ctx": n_ctx,
        "context_used_pct": context_used_pct,
        "cache_tokens": cache_n,
        "cache_hit_ratio": cache_hit_ratio,
        "prefill_tokens": prompt_n,
        "prefill_ms": _round(prefill_ms, 2),
        "prefill_tok_s": _round(prefill_tok_s, 2),
        "decode_ms": _round(decode_ms, 2),
        "decode_tok_s": _round(decode_tok_s, 2),
        "ttft_ms": _round(ttft_ms, 2),
        "wall_ms": _round(wall_ms, 2),
    }
    return metrics


def format_perf_message(metrics: dict[str, Any]) -> str:
    """One-line summary for the agent log."""
    ctx = metrics.get("tokens_in_context")
    n_ctx = metrics.get("n_ctx")
    pct = metrics.get("context_used_pct")
    if ctx is not None and n_ctx:
        ctx_part = f"{ctx}/{n_ctx}"
        if pct is not None:
            ctx_part += f" ({pct}%)"
    elif ctx is not None:
        ctx_part = str(ctx)
    else:
        ctx_part = "?"

    prefill = metrics.get("prefill_tok_s")
    prefill_ms = metrics.get("prefill_ms")
    decode = metrics.get("decode_tok_s")
    decode_ms = metrics.get("decode_ms")
    cache = metrics.get("cache_tokens")
    cache_ratio = metrics.get("cache_hit_ratio")
    cache_part = str(cache) if cache is not None else "?"
    if cache_ratio is not None:
        cache_part += f" ({cache_ratio:.0%})"

    return (
        "llm perf"
        f" session={metrics.get('session_id') or '-'}"
        f" turn={metrics.get('turn_id') or '-'}"
        f" prompt={metrics.get('prompt_tokens') if metrics.get('prompt_tokens') is not None else '?'}"
        f" gen={metrics.get('completion_tokens') if metrics.get('completion_tokens') is not None else '?'}"
        f" cache={cache_part}"
        f" ctx={ctx_part}"
        f" prefill={_speed(prefill, prefill_ms)}"
        f" decode={_speed(decode, decode_ms)}"
        f" ttft={_ms(metrics.get('ttft_ms'))}"
        f" wall={_ms(metrics.get('wall_ms'))}"
    )


def log_completion(
    data: dict[str, Any],
    *,
    wall_ms: float,
    n_ctx: int | None = None,
    model: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    label: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Write JSONL + an INFO line. Never raises."""
    metrics = extract_metrics(
        data,
        wall_ms=wall_ms,
        n_ctx=n_ctx,
        model=model,
        session_id=session_id,
        turn_id=turn_id,
        label=label,
    )
    record_prompt_cache(metrics)
    try:
        logger.info(format_perf_message(metrics))
        _append_jsonl(metrics, path or PERF_LOG_PATH)
    except Exception:
        logger.exception("failed to write llm perf log")
    return metrics


def record_prompt_cache(metrics: dict[str, Any]) -> None:
    """Accumulate cache hits from a completion for /health."""
    global _cache_completions, _cache_prompt_tokens, _cache_cached_tokens
    global _cache_prefill_tokens, _cache_last
    prompt_tokens = metrics.get("prompt_tokens")
    cache_tokens = metrics.get("cache_tokens")
    if prompt_tokens is None and cache_tokens is None:
        return
    prompt_n = int(prompt_tokens or 0)
    cache_n = int(cache_tokens or 0)
    prefill_n = metrics.get("prefill_tokens")
    last = {
        "cache_hit_ratio": metrics.get("cache_hit_ratio"),
        "n_prompt_tokens_cache": cache_n,
        "n_prompt_tokens": prompt_n,
        "n_prompt_tokens_processed": prefill_n,
        "turn_id": metrics.get("turn_id"),
    }
    with _lock:
        _cache_completions += 1
        _cache_prompt_tokens += prompt_n
        _cache_cached_tokens += cache_n
        if prefill_n is not None:
            _cache_prefill_tokens += int(prefill_n)
        _cache_last = last


def prompt_cache_health() -> dict[str, Any]:
    """Lifetime cache hit ratio from completions (idle /slots zeros cache_n)."""
    with _lock:
        completions = _cache_completions
        prompt_tokens = _cache_prompt_tokens
        cache_tokens = _cache_cached_tokens
        prefill_tokens = _cache_prefill_tokens
        last = dict(_cache_last) if _cache_last else None
    if completions <= 0 or prompt_tokens <= 0:
        return {
            "status": "ok",
            "cache_hit_ratio": None,
            "n_prompt_tokens_cache": 0,
            "n_prompt_tokens": 0,
            "completions": completions,
            "detail": "no completions recorded yet",
            "source": "completion_timings",
        }
    return {
        "status": "ok",
        "cache_hit_ratio": round(cache_tokens / prompt_tokens, 4),
        "n_prompt_tokens_cache": cache_tokens,
        "n_prompt_tokens": prompt_tokens,
        "n_prompt_tokens_processed": prefill_tokens,
        "completions": completions,
        "last": last,
        "source": "completion_timings",
    }


def reset_prompt_cache_stats() -> None:
    """Test helper: clear lifetime cache totals."""
    global _cache_completions, _cache_prompt_tokens, _cache_cached_tokens
    global _cache_prefill_tokens, _cache_last
    with _lock:
        _cache_completions = 0
        _cache_prompt_tokens = 0
        _cache_cached_tokens = 0
        _cache_prefill_tokens = 0
        _cache_last = None


def n_ctx_from_props(props: dict[str, Any] | None) -> int | None:
    """Read slot context size from llama-server GET /props."""
    if not isinstance(props, dict):
        return None
    settings = props.get("default_generation_settings")
    if isinstance(settings, dict):
        n_ctx = _as_int(settings.get("n_ctx"))
        if n_ctx:
            return n_ctx
    return _as_int(props.get("n_ctx"))


def _append_jsonl(metrics: dict[str, Any], path: Path) -> None:
    global _path_logged
    line = json.dumps(metrics, default=str, separators=(",", ":"))
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if not _path_logged:
            _path_logged = True
            logger.info("llm perf jsonl: %s", path)


def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _round(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _ms(value: Any) -> str:
    if value is None:
        return "?"
    return f"{value:.0f}ms"


def _speed(tok_s: Any, duration_ms: Any) -> str:
    if tok_s is None and duration_ms is None:
        return "?"
    parts: list[str] = []
    if tok_s is not None:
        parts.append(f"{tok_s:.1f} tok/s")
    if duration_ms is not None:
        parts.append(f"{duration_ms:.0f}ms")
    return " ".join(parts) if len(parts) == 1 else f"{parts[0]} ({parts[1]})"
