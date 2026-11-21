"""Summary aggregation, Markdown table + JSON report writing, and optional MLflow logging."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.ablations import AblationConfig
from evals.client import SamplingParams
from evals.runner import CaseRunResult

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(".run/evals")
DEFAULT_MLFLOW_URI = "sqlite:///.data/mlflow.db"


def summarize_results(results: list[CaseRunResult]) -> dict[str, Any]:
    """Compute overall and per-category pass rates and token/latency totals."""
    total = len(results)
    if total == 0:
        return {
            "total_runs": 0,
            "pass_all_rate": 0.0,
            "routing_ok_rate": 0.0,
            "tool_args_ok_rate": 0.0,
            "no_fabrication_rate": 0.0,
            "request_fidelity_rate": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "mean_cache_hit_ratio": None,
            "total_wall_ms": 0.0,
            "by_category": {},
        }

    def _rate(pred: list[bool]) -> float:
        return round(sum(1 for ok in pred if ok) / len(pred), 4) if pred else 0.0

    cache_ratios = [r.cache_hit_ratio for r in results if r.cache_hit_ratio is not None]
    by_cat: dict[str, list[CaseRunResult]] = {}
    for res in results:
        by_cat.setdefault(res.category, []).append(res)

    cat_summary: dict[str, Any] = {}
    for cat, items in by_cat.items():
        cat_summary[cat] = {
            "count": len(items),
            "pass_all_rate": _rate([i.scores.all_passed for i in items]),
            "routing_ok_rate": _rate([i.scores.routing_ok for i in items]),
            "tool_args_ok_rate": _rate([i.scores.tool_args_ok for i in items]),
            "no_fabrication_rate": _rate([i.scores.no_fabrication for i in items]),
            "request_fidelity_rate": _rate([i.scores.request_fidelity for i in items]),
        }

    return {
        "total_runs": total,
        "passed_all": sum(1 for r in results if r.scores.all_passed),
        "pass_all_rate": _rate([r.scores.all_passed for r in results]),
        "routing_ok_rate": _rate([r.scores.routing_ok for r in results]),
        "tool_args_ok_rate": _rate([r.scores.tool_args_ok for r in results]),
        "no_fabrication_rate": _rate([r.scores.no_fabrication for r in results]),
        "request_fidelity_rate": _rate([r.scores.request_fidelity for r in results]),
        "total_prompt_tokens": sum(r.prompt_tokens for r in results),
        "total_completion_tokens": sum(r.completion_tokens for r in results),
        "mean_cache_hit_ratio": (
            round(sum(cache_ratios) / len(cache_ratios), 4) if cache_ratios else None
        ),
        "total_wall_ms": round(sum(r.wall_ms for r in results), 2),
        "by_category": cat_summary,
    }


def build_report_payload(
    results: list[CaseRunResult],
    *,
    model: str,
    base_url: str,
    ablation: AblationConfig,
    sampling: SamplingParams,
    repeats: int = 1,
) -> dict[str, Any]:
    """Build the full JSON-serializable report dictionary."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "repeats": repeats,
        "ablation": {
            "history_mode": ablation.history_mode,
            "context_mode": ablation.context_mode,
            "prompt_mode": ablation.prompt_mode,
            "tool_surface": ablation.tool_surface,
            "override_case_surface": ablation.override_case_surface,
        },
        "sampling": sampling.payload(),
        "summary": summarize_results(results),
        "cases": [r.to_dict() for r in results],
    }


def format_markdown_report(payload: dict[str, Any]) -> str:
    """Render a human-readable Markdown table + failure breakdown."""
    summary = payload.get("summary") or {}
    ablation = payload.get("ablation") or {}
    sampling = payload.get("sampling") or {}
    cases = payload.get("cases") or []

    def _b(flag: Any) -> str:
        return "PASS" if flag else "FAIL"

    def _pct(val: Any) -> str:
        if val is None:
            return "-"
        return f"{float(val) * 100:.1f}%"

    lines: list[str] = [
        f"# Model Eval Report — `{payload.get('model')}`",
        "",
        f"- **Timestamp**: `{payload.get('timestamp')}`",
        f"- **Base URL**: `{payload.get('base_url')}`",
        (
            f"- **Ablation**: `history={ablation.get('history_mode')}` "
            f"`context={ablation.get('context_mode')}` "
            f"`prompt={ablation.get('prompt_mode')}` "
            f"`surface={ablation.get('tool_surface')}`"
        ),
        f"- **Sampling overrides**: `{json.dumps(sampling, sort_keys=True) if sampling else 'server-defaults'}`",
        (
            f"- **Overall**: **{summary.get('passed_all', 0)}/{summary.get('total_runs', 0)}** "
            f"({_pct(summary.get('pass_all_rate'))}) | "
            f"routing `{_pct(summary.get('routing_ok_rate'))}` | "
            f"tool_args `{_pct(summary.get('tool_args_ok_rate'))}` | "
            f"no_fabrication `{_pct(summary.get('no_fabrication_rate'))}` | "
            f"request_fidelity `{_pct(summary.get('request_fidelity_rate'))}`"
        ),
        "",
        "## Case Breakdown",
        "",
        "| Case | Surface | Run | Route | Args | NoFab | Fidelity | All | Prompt Tok | Comp Tok | Cache Hit | Wall (ms) | Tools Called |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for item in cases:
        scores = item.get("scores") or {}
        metrics = item.get("metrics") or {}
        tool_names = [str(tc.get("name") or "") for tc in (item.get("tool_calls") or [])]
        tools_str = ", ".join(tool_names) if tool_names else "(none)"
        if len(tools_str) > 55:
            tools_str = tools_str[:52] + "..."
        cache_str = _pct(metrics.get("cache_hit_ratio"))
        lines.append(
            f"| `{item.get('case_id')}` | `{item.get('effective_surface')}` | "
            f"{item.get('repeat')} | "
            f"{_b(scores.get('routing_ok'))} | "
            f"{_b(scores.get('tool_args_ok'))} | "
            f"{_b(scores.get('no_fabrication'))} | "
            f"{_b(scores.get('request_fidelity'))} | "
            f"**{_b(scores.get('all_passed'))}** | "
            f"{metrics.get('prompt_tokens', 0)} | "
            f"{metrics.get('completion_tokens', 0)} | "
            f"{cache_str} | "
            f"{metrics.get('wall_ms', 0.0):.1f} | "
            f"`{tools_str}` |"
        )

    failed = [c for c in cases if not (c.get("scores") or {}).get("all_passed")]
    if failed:
        lines.extend(["", "## Failure Reasons", ""])
        for item in failed:
            scores = item.get("scores") or {}
            reasons = scores.get("reasons") or {}
            bad = [
                f"{key}: {reasons.get(key)}"
                for key in ("routing_ok", "tool_args_ok", "no_fabrication", "request_fidelity")
                if not scores.get(key)
            ]
            lines.append(
                f"- **`{item.get('case_id')}` (run {item.get('repeat')})**: "
                + "; ".join(bad)
            )

    lines.append("")
    return "\n".join(lines)


def write_report_files(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not stem:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_model = str(payload.get("model") or "model")
        safe_model = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw_model
        ).strip("_")
        ablation = payload.get("ablation") or {}
        stem = (
            f"eval_{safe_model}_{ablation.get('history_mode', 'full')}"
            f"_{ablation.get('context_mode', 'full')}"
            f"_{ablation.get('prompt_mode', 'current')}"
            f"_{ablation.get('tool_surface', 'full_composio')}_{ts}"
        )
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown_report(payload), encoding="utf-8")
    return json_path, md_path


def log_to_mlflow(
    payload: dict[str, Any],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
    experiment_name: str = "baka-model-evals",
) -> str | None:
    """Soft-import ``mlflow`` and log params, summary metrics, and artifacts."""
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("mlflow is not installed; skipping --mlflow logging")
        return None

    try:
        if tracking_uri.startswith("sqlite:///"):
            db_rel = tracking_uri[len("sqlite:///") :]
            if db_rel:
                Path(db_rel).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        ablation = payload.get("ablation") or {}
        sampling = payload.get("sampling") or {}
        summary = payload.get("summary") or {}
        run_name = (
            f"{payload.get('model', 'model')}_"
            f"h-{ablation.get('history_mode')}_"
            f"c-{ablation.get('context_mode')}_"
            f"p-{ablation.get('prompt_mode')}_"
            f"s-{ablation.get('tool_surface')}"
        )
        with mlflow.start_run(run_name=run_name) as run:
            params: dict[str, Any] = {
                "model": payload.get("model"),
                "base_url": payload.get("base_url"),
                "repeats": payload.get("repeats"),
                **{f"ablation_{k}": v for k, v in ablation.items()},
                **{f"sampling_{k}": v for k, v in sampling.items()},
            }
            mlflow.log_params(params)

            metrics: dict[str, float] = {
                "pass_all_rate": float(summary.get("pass_all_rate") or 0.0),
                "routing_ok_rate": float(summary.get("routing_ok_rate") or 0.0),
                "tool_args_ok_rate": float(summary.get("tool_args_ok_rate") or 0.0),
                "no_fabrication_rate": float(summary.get("no_fabrication_rate") or 0.0),
                "request_fidelity_rate": float(summary.get("request_fidelity_rate") or 0.0),
                "total_prompt_tokens": float(summary.get("total_prompt_tokens") or 0),
                "total_completion_tokens": float(summary.get("total_completion_tokens") or 0),
                "total_wall_ms": float(summary.get("total_wall_ms") or 0.0),
            }
            if summary.get("mean_cache_hit_ratio") is not None:
                metrics["mean_cache_hit_ratio"] = float(summary["mean_cache_hit_ratio"])
            mlflow.log_metrics(metrics)

            if hasattr(mlflow, "log_table"):
                rows = []
                for item in payload.get("cases") or []:
                    sc = item.get("scores") or {}
                    mt = item.get("metrics") or {}
                    rows.append(
                        {
                            "case_id": item.get("case_id"),
                            "category": item.get("category"),
                            "surface": item.get("effective_surface"),
                            "repeat": item.get("repeat"),
                            "routing_ok": sc.get("routing_ok"),
                            "tool_args_ok": sc.get("tool_args_ok"),
                            "no_fabrication": sc.get("no_fabrication"),
                            "request_fidelity": sc.get("request_fidelity"),
                            "all_passed": sc.get("all_passed"),
                            "prompt_tokens": mt.get("prompt_tokens"),
                            "completion_tokens": mt.get("completion_tokens"),
                            "wall_ms": mt.get("wall_ms"),
                            "final_reply": item.get("final_reply"),
                        }
                    )
                if rows:
                    cols = list(rows[0].keys())
                    table_data = {col: [r[col] for r in rows] for col in cols}
                    mlflow.log_table(data=table_data, artifact_file="eval_cases.json")

            if json_path is not None and json_path.exists():
                mlflow.log_artifact(str(json_path))
            if md_path is not None and md_path.exists():
                mlflow.log_artifact(str(md_path))
            return str(run.info.run_id)
    except Exception:  # noqa: BLE001
        logger.warning("MLflow logging failed; continuing cleanly", exc_info=True)
        return None
