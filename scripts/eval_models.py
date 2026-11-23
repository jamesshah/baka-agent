#!/usr/bin/env python3
"""CLI entrypoint for offline model replay and ablation evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
