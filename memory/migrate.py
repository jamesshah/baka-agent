"""Command-line wrapper around Alembic migrations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from alembic import command
from alembic.config import Config


def alembic_config(db_path: str | None = None) -> Config:
    root = Path(__file__).resolve().parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "memory" / "alembic"))
    if db_path:
        os.environ["BAKA_DB_PATH"] = db_path
    return config


def upgrade(db_path: str | None = None) -> None:
    command.upgrade(alembic_config(db_path), "head")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage baka-agent database schema")
    parser.add_argument(
        "command", choices=("upgrade", "current", "history"),
        help="Migration operation",
    )
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()
    config = alembic_config(args.db_path)
    if args.command == "upgrade":
        command.upgrade(config, "head")
    elif args.command == "current":
        command.current(config, verbose=True)
    else:
        command.history(config, verbose=True)


if __name__ == "__main__":
    main()
