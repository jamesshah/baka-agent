"""Command-line interface for offline model replay and ablation evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from evals.ablations import (
    CONTEXT_MODES,
    HISTORY_MODES,
    PROMPT_MODES,
    TOOL_SURFACES,
    AblationConfig,
)
from evals.cases import all_cases, filter_cases
from evals.client import OpenAICompatClient, SamplingParams
from evals.report import (
    DEFAULT_MLFLOW_URI,
    DEFAULT_OUTPUT_DIR,
    LiveMlflowTracker,
    build_report_payload,
    format_markdown_report,
    write_report_files,
)
from evals.runner import CaseRunResult, run_case
from llm.base import LLMClient


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_models.py",
        description=(
            "Offline replay and ablation eval harness for local llama-server models "
            "(LFM3 vs Gemma debugging suite)."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible base URL (default: http://127.0.0.1:8080/v1).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (auto-detected from /v1/models when omitted).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Sampling temperature (omitted when unset so server default applies).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p (omitted when unset).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter (omitted when unset).",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help="Repetition penalty (omitted when unset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional sampling seed (omitted when unset).",
    )
    parser.add_argument(
        "--history-mode",
        choices=list(HISTORY_MODES),
        default="full",
        help="Conversation history ablation mode: full, short, or clean.",
    )
    parser.add_argument(
        "--context-mode",
        choices=list(CONTEXT_MODES),
        default="full",
        help="Injected <context> (memories/skills/events) ablation mode: full or none.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=list(PROMPT_MODES),
        default="current",
        help="System prompt ablation mode: current or no_negative_priming.",
    )
    parser.add_argument(
        "--tool-surface",
        choices=list(TOOL_SURFACES),
        default=None,
        help=(
            "Worker calendar tool surface: full_composio (6 meta-tools), "
            "slim_composio (3 meta-tools), or flat (direct calendar tools)."
        ),
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated substring filter on case id or category.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to run each selected case (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Markdown and JSON reports (default: .run/evals).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-completion HTTP timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log run parameters, metrics, and report artifacts to MLflow SQLite.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=DEFAULT_MLFLOW_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_MLFLOW_URI}).",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List matching benchmark cases and exit without calling the server.",
    )
    return parser


def run_eval_suite(
    argv: Sequence[str] | None = None,
    *,
    llm: LLMClient | None = None,
) -> tuple[dict, Path, Path]:
    """Parse CLI args, run matching cases, write `.run/evals/` outputs, and return payload."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    selected = filter_cases(all_cases(), args.cases)
    if not selected:
        raise ValueError(f"No benchmark cases matched --cases={args.cases!r}")

    override_surface = args.tool_surface is not None
    effective_default_surface = args.tool_surface or "full_composio"
    ablation = AblationConfig(
        history_mode=args.history_mode,
        context_mode=args.context_mode,
        prompt_mode=args.prompt_mode,
        tool_surface=effective_default_surface,
        override_case_surface=override_surface,
    )
    sampling = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
    )

    owned_client: OpenAICompatClient | None = None
    model_name = args.model or "local-model"
    if llm is None:
        probe = OpenAICompatClient(
            args.base_url,
            args.model or "default",
            sampling=sampling,
            timeout=args.timeout,
        )
        detected = probe.served_model() if not args.model else args.model
        model_name = detected or args.model or "local-model"
        probe.close()
        owned_client = OpenAICompatClient(
            args.base_url,
            model_name,
            sampling=sampling,
            timeout=args.timeout,
        )
        active_llm: LLMClient = owned_client
    else:
        active_llm = llm
        model_name = args.model or getattr(llm, "model", None) or "fake-llm"

    repeats = max(1, int(args.repeats))
    total_runs = len(selected) * repeats
    tracker: LiveMlflowTracker | None = None
    if args.mlflow:
        tracker = LiveMlflowTracker(tracking_uri=args.mlflow_uri)
        tracker.start_suite(
            model=model_name,
            base_url=args.base_url,
            ablation=ablation,
            sampling=sampling,
            repeats=repeats,
            total_runs=total_runs,
        )

    results: list[CaseRunResult] = []
    step = 0
    try:
        for case in selected:
            for rep in range(1, repeats + 1):
                step += 1
                print(
                    f"[{step}/{total_runs}] Running {case.id} (repeat {rep})…",
                    flush=True,
                )
                result = run_case(
                    case,
                    active_llm,
                    config=ablation,
                    repeat=rep,
                )
                results.append(result)
                status = "PASS" if result.scores.all_passed else "FAIL"
                print(
                    f"[{step}/{total_runs}] {status} {case.id} "
                    f"({result.wall_ms:.0f}ms, prompt={result.prompt_tokens}, "
                    f"comp={result.completion_tokens})",
                    flush=True,
                )
                if tracker is not None:
                    tracker.log_case_step(
                        result,
                        step=step,
                        results_so_far=results,
                    )
    finally:
        if owned_client is not None:
            owned_client.close()

    payload = build_report_payload(
        results,
        model=model_name,
        base_url=args.base_url,
        ablation=ablation,
        sampling=sampling,
        repeats=repeats,
    )
    json_path, md_path = write_report_files(payload, output_dir=args.output_dir)
    if tracker is not None:
        tracker.finish_suite(
            payload,
            json_path=json_path,
            md_path=md_path,
        )
    return payload, json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.list_cases:
        for case in filter_cases(all_cases(), args.cases):
            print(f"{case.id:40s} [{case.category:19s}] ({case.mode}) - {case.description}")
        return 0

    payload, json_path, md_path = run_eval_suite(argv)
    sys.stdout.write(format_markdown_report(payload))
    sys.stdout.write(f"\nSaved JSON report: {json_path}\nSaved Markdown report: {md_path}\n")
    return 0
