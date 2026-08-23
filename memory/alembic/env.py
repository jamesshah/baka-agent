"""Alembic environment for baka-agent's local SQLite database."""

from __future__ import annotations

import os
import logging
from logging.config import fileConfig

from alembic import context

from memory.database import create_database_engine
from memory.models import Base

config = context.config
if (
    config.config_file_name is not None
    and not logging.getLogger().handlers
):
    # Migrations run inside the already-configured uvicorn process at startup.
    # Only install Alembic's standalone CLI logging when no host application
    # has configured logging already.
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    if type_ == "table" and reflected and (
        "_fts" in name
        or name.startswith("memory_vectors")
    ):
        return False
    return True


def database_path() -> str:
    return os.environ.get("BAKA_DB_PATH", ".data/baka.db")


def run_migrations_offline() -> None:
    context.configure(
        url=f"sqlite+pysqlite:///{database_path()}",
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_database_engine(database_path())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
