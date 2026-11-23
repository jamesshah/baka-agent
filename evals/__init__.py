"""Offline replay + ablation evals for local models behind llama-server.

The harness drives the production prompts and tool loop
(``agents.loop.run_tool_loop``, ``ChatAgent``/``ExecutorAgent`` prompts,
``SpawnAgentTool``) against mocked, deterministic tools, so the only live
dependency is an OpenAI-compatible ``/v1/chat/completions`` endpoint.

Modules:
    client     OpenAI-compatible client with optional sampling knobs + usage recording
    fixtures   Fixed clock, PII-free Whoop data, memories, skill, replayed history turns
    mock_tools Mock world, Composio Tool Router / flat calendar / link tools
    ablations  history / context / prompt / tool-surface transforms
    cases      The curated benchmark suite
    runner     Runs one case (chat or executor-only) and captures a trace
    scorers    Deterministic binary scorers
    report     Aggregation, Markdown/JSON output, optional MLflow logging
    cli        ``scripts/eval_models.py`` entry point
"""
