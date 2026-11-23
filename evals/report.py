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


def _prepare_mlflow_experiment(
    mlflow: Any,
    tracking_uri: str,
    experiment_name: str,
) -> str:
    """Resolve SQLite tracking URI and keep artifacts next to the DB under .data/mlflow-artifacts."""
    resolved = tracking_uri.strip() or DEFAULT_MLFLOW_URI
    artifact_loc: str | None = None
    if resolved.startswith("sqlite:///") and resolved != "sqlite:///:memory:":
        raw_path = resolved[len("sqlite:///") :]
        db_path = Path(raw_path)
        if not db_path.is_absolute():
            db_path = (Path.cwd() / db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = f"sqlite:///{db_path}"
        artifacts_dir = db_path.parent / "mlflow-artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_loc = artifacts_dir.as_uri()
    mlflow.set_tracking_uri(resolved)
    if artifact_loc and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_loc)
    mlflow.set_experiment(experiment_name)
    return resolved


def _cases_table_data(cases: list[dict[str, Any]]) -> dict[str, list[Any]]:
    rows: list[dict[str, Any]] = []
    for item in cases:
        sc = item.get("scores") or {}
        mt = item.get("metrics") or {}
        reasons = sc.get("reasons") or {}
        tool_names = [str(tc.get("name") or "") for tc in (item.get("tool_calls") or [])]
        rows.append(
            {
                "case_id": item.get("case_id"),
                "category": item.get("category"),
                "surface": item.get("effective_surface"),
                "repeat": item.get("repeat"),
                "status": "PASS" if sc.get("all_passed") else "FAIL",
                "all_passed": sc.get("all_passed"),
                "routing_ok": sc.get("routing_ok"),
                "tool_args_ok": sc.get("tool_args_ok"),
                "no_fabrication": sc.get("no_fabrication"),
                "request_fidelity": sc.get("request_fidelity"),
                "prompt_tokens": mt.get("prompt_tokens"),
                "completion_tokens": mt.get("completion_tokens"),
                "cache_hit_ratio": mt.get("cache_hit_ratio"),
                "wall_ms": mt.get("wall_ms"),
                "tools_called": ", ".join(tool_names) if tool_names else "(none)",
                "failure_reasons": "; ".join(
                    f"{k}: {reasons.get(k)}"
                    for k in ("routing_ok", "tool_args_ok", "no_fabrication", "request_fidelity")
                    if not sc.get(k)
                ),
                "final_reply": item.get("final_reply"),
            }
        )
    if not rows:
        return {}
    cols = list(rows[0].keys())
    return {col: [r[col] for r in rows] for col in cols}


def _sync_mlflow_dataset(
    mlflow: Any,
    client: Any,
    *,
    experiment_id: str,
    run_id: str,
    dataset_name: str = "baka_eval_suite",
) -> None:
    """Register/update the 13-case benchmark dataset in MLflow GenAI Datasets and link it to run_id."""
    try:
        from mlflow.entities import DatasetInput  # type: ignore[import-untyped]
        from evals.cases import all_cases

        cases = all_cases()
        existing = [
            d
            for d in mlflow.genai.search_datasets(experiment_ids=[experiment_id])
            if getattr(d, "name", None) == dataset_name
        ]
        if existing:
            ds = mlflow.genai.get_dataset(dataset_id=existing[0].dataset_id)
        else:
            ds = mlflow.genai.create_dataset(
                name=dataset_name,
                experiment_id=experiment_id,
                tags={"suite": "lfm3_vs_gemma", "cases": str(len(cases))},
            )
        records = [
            {
                "inputs": {
                    "case_id": c.id,
                    "category": c.category,
                    "mode": c.mode,
                    "user_text": c.user_text,
                    "prior_turns_count": len(c.prior_turns),
                    "has_polluted_context": bool(c.memory_context),
                    "tool_surface": c.tool_surface or "full_composio",
                },
                "expectations": {
                    "expected_route": c.expected_route,
                    "expected_direct_tools": list(c.expected_direct_tools),
                    "calendar_intent": c.calendar_intent,
                    "forbid_snaptrade": c.forbid_snaptrade,
                    "require_search_grant": c.require_search_grant,
                },
            }
            for c in cases
        ]
        ds.merge_records(records)
        client.log_inputs(
            run_id=run_id,
            datasets=[DatasetInput(dataset=ds._to_mlflow_entity())],
        )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to sync MLflow GenAI EvaluationDataset", exc_info=True)


class LiveMlflowTracker:
    """Streams eval suite progress, linked per-case traces, assessments, and dataset live to MLflow."""

    def __init__(
        self,
        *,
        tracking_uri: str = DEFAULT_MLFLOW_URI,
        experiment_name: str = "baka-model-evals",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._mlflow: Any = None
        self._client: Any = None
        self._parent_run: Any = None
        self._run_id: str | None = None
        self._total_runs: int = 0

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def start_suite(
        self,
        *,
        model: str,
        base_url: str,
        ablation: AblationConfig,
        sampling: SamplingParams,
        repeats: int,
        total_runs: int,
    ) -> str | None:
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("mlflow is not installed; skipping --mlflow logging")
            return None

        try:
            resolved = _prepare_mlflow_experiment(
                mlflow, self._tracking_uri, self._experiment_name
            )
            try:
                from tracing import configure_tracing

                configure_tracing(
                    enabled=True,
                    tracking_uri=resolved,
                    experiment=self._experiment_name,
                )
            except ImportError:
                pass

            run_name = (
                f"{model}_"
                f"h-{ablation.history_mode}_"
                f"c-{ablation.context_mode}_"
                f"p-{ablation.prompt_mode}_"
                f"s-{ablation.tool_surface}"
            )
            self._mlflow = mlflow
            self._client = mlflow.tracking.MlflowClient(tracking_uri=resolved)
            self._total_runs = total_runs
            self._parent_run = mlflow.start_run(run_name=run_name)
            self._run_id = str(self._parent_run.info.run_id)
            exp_id = str(self._parent_run.info.experiment_id)

            try:
                from mlflow.utils.mlflow_tags import (  # type: ignore[import-untyped]
                    MLFLOW_RUN_TYPE,
                    MLFLOW_RUN_TYPE_GENAI_EVALUATE,
                )

                self._client.set_tag(
                    self._run_id, MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_GENAI_EVALUATE
                )
            except Exception:  # noqa: BLE001
                pass

            _sync_mlflow_dataset(
                mlflow,
                self._client,
                experiment_id=exp_id,
                run_id=self._run_id,
            )

            ablation_dict = {
                "history_mode": ablation.history_mode,
                "context_mode": ablation.context_mode,
                "prompt_mode": ablation.prompt_mode,
                "tool_surface": ablation.tool_surface,
                "override_case_surface": ablation.override_case_surface,
            }
            params: dict[str, Any] = {
                "model": model,
                "base_url": base_url,
                "repeats": repeats,
                "total_cases": total_runs,
                **{f"ablation_{k}": v for k, v in ablation_dict.items()},
                **{f"sampling_{k}": v for k, v in sampling.payload().items()},
            }
            mlflow.log_params(params)
            mlflow.set_tags(
                {
                    "eval.status": "RUNNING",
                    "eval.progress": f"0/{total_runs}",
                }
            )
            mlflow.log_metrics(
                {
                    "completed_cases": 0.0,
                    "total_cases": float(total_runs),
                    "progress_pct": 0.0,
                },
                step=0,
            )
            return self._run_id
        except Exception:  # noqa: BLE001
            logger.warning("MLflow start_suite failed; continuing without live MLflow", exc_info=True)
            self._mlflow = None
            self._client = None
            self._parent_run = None
            self._run_id = None
            return None

    def log_case_step(
        self,
        result: CaseRunResult,
        *,
        step: int,
        results_so_far: list[CaseRunResult],
    ) -> None:
        if self._mlflow is None or self._parent_run is None or self._run_id is None:
            return
        mlflow = self._mlflow
        try:
            status_str = "PASS" if result.scores.all_passed else "FAIL"
            # 1. Link this case's trace to the evaluation run and attach the 5 binary scorer assessments
            if result.trace_id:
                try:
                    mlflow.flush_trace_async_logging()
                    from mlflow.entities import (  # type: ignore[import-untyped]
                        AssessmentSource,
                        AssessmentSourceType,
                    )

                    source = AssessmentSource(
                        source_type=AssessmentSourceType.CODE,
                        source_id="eval_models",
                    )
                    score_dict = result.scores.to_dict()
                    reasons = result.scores.reasons
                    for key in (
                        "all_passed",
                        "routing_ok",
                        "tool_args_ok",
                        "no_fabrication",
                        "request_fidelity",
                    ):
                        mlflow.log_feedback(
                            trace_id=result.trace_id,
                            name=key,
                            value=bool(score_dict.get(key)),
                            source=source,
                            rationale=reasons.get(key)
                            or (
                                "All 4 checks passed"
                                if result.scores.all_passed
                                else "One or more checks failed"
                            ),
                        )
                    if self._client is not None:
                        self._client.link_traces_to_run(
                            run_id=self._run_id,
                            trace_ids=[result.trace_id],
                        )
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to attach feedback/link trace %s", result.trace_id, exc_info=True)

            # 2. Update the evaluation run's live summary metrics & table at this step.
            running_summary = summarize_results(results_so_far)
            total = max(1, self._total_runs)
            parent_metrics: dict[str, float] = {
                "completed_cases": float(step),
                "progress_pct": round(100.0 * step / total, 2),
                "passed_all": float(running_summary.get("passed_all") or 0),
                "pass_all_rate": float(running_summary.get("pass_all_rate") or 0.0),
                "routing_ok_rate": float(running_summary.get("routing_ok_rate") or 0.0),
                "tool_args_ok_rate": float(running_summary.get("tool_args_ok_rate") or 0.0),
                "no_fabrication_rate": float(running_summary.get("no_fabrication_rate") or 0.0),
                "request_fidelity_rate": float(running_summary.get("request_fidelity_rate") or 0.0),
                "total_prompt_tokens": float(running_summary.get("total_prompt_tokens") or 0),
                "total_completion_tokens": float(running_summary.get("total_completion_tokens") or 0),
                "total_wall_ms": float(running_summary.get("total_wall_ms") or 0.0),
            }
            if running_summary.get("mean_cache_hit_ratio") is not None:
                parent_metrics["mean_cache_hit_ratio"] = float(
                    running_summary["mean_cache_hit_ratio"]
                )
            mlflow.log_metrics(parent_metrics, step=step)
            mlflow.set_tags(
                {
                    "eval.status": "RUNNING",
                    "eval.progress": f"{step}/{self._total_runs}",
                    "eval.last_case": f"{result.case_id} ({status_str})",
                }
            )
            if hasattr(mlflow, "log_table"):
                table_data = _cases_table_data([r.to_dict() for r in results_so_far])
                if table_data:
                    mlflow.log_table(data=table_data, artifact_file="eval_cases.json")
        except Exception:  # noqa: BLE001
            logger.warning("MLflow log_case_step failed", exc_info=True)

    def finish_suite(
        self,
        payload: dict[str, Any],
        *,
        json_path: Path | None = None,
        md_path: Path | None = None,
    ) -> str | None:
        if self._mlflow is None or self._parent_run is None:
            return None
        mlflow = self._mlflow
        try:
            summary = payload.get("summary") or {}
            total = int(summary.get("total_runs") or 0)
            passed = int(summary.get("passed_all") or 0)
            metrics: dict[str, float] = {
                "completed_cases": float(total),
                "progress_pct": 100.0 if total else 0.0,
                "passed_all": float(passed),
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
            mlflow.log_metrics(metrics, step=total)
            mlflow.set_tags(
                {
                    "eval.status": "COMPLETED",
                    "eval.progress": f"{total}/{total}",
                    "eval.pass_summary": f"{passed}/{total} ({float(summary.get('pass_all_rate') or 0.0) * 100:.1f}%)",
                }
            )
            if hasattr(mlflow, "log_table"):
                table_data = _cases_table_data(payload.get("cases") or [])
                if table_data:
                    mlflow.log_table(data=table_data, artifact_file="eval_cases.json")
            if json_path is not None and json_path.exists():
                mlflow.log_artifact(str(json_path))
            if md_path is not None and md_path.exists():
                mlflow.log_artifact(str(md_path))
            return self._run_id
        except Exception:  # noqa: BLE001
            logger.warning("MLflow finish_suite failed; continuing cleanly", exc_info=True)
            return None
        finally:
            try:
                mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass
            self._parent_run = None


def log_to_mlflow(
    payload: dict[str, Any],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
    experiment_name: str = "baka-model-evals",
) -> str | None:
    """One-shot helper to log a completed report payload to MLflow."""
    ablation_raw = payload.get("ablation") or {}
    sampling_raw = payload.get("sampling") or {}
    ablation = AblationConfig(
        history_mode=ablation_raw.get("history_mode", "full"),
        context_mode=ablation_raw.get("context_mode", "full"),
        prompt_mode=ablation_raw.get("prompt_mode", "current"),
        tool_surface=ablation_raw.get("tool_surface", "full_composio"),
        override_case_surface=bool(ablation_raw.get("override_case_surface", False)),
    )
    sampling = SamplingParams(
        temperature=sampling_raw.get("temperature"),
        top_p=sampling_raw.get("top_p"),
        top_k=sampling_raw.get("top_k"),
        repeat_penalty=sampling_raw.get("repeat_penalty"),
        seed=sampling_raw.get("seed"),
    )
    cases = payload.get("cases") or []
    tracker = LiveMlflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    run_id = tracker.start_suite(
        model=str(payload.get("model") or "model"),
        base_url=str(payload.get("base_url") or ""),
        ablation=ablation,
        sampling=sampling,
        repeats=int(payload.get("repeats") or 1),
        total_runs=len(cases),
    )
    if run_id is None:
        return None
    return tracker.finish_suite(payload, json_path=json_path, md_path=md_path)
