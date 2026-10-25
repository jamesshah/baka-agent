"""SQLAlchemy engine and session configuration for local SQLite storage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def sqlite_url(path: str) -> str:
    if path == ":memory:":
        return "sqlite+pysqlite:///:memory:"
    return f"sqlite+pysqlite:///{Path(path).expanduser().resolve()}"


def create_database_engine(path: str, *, echo: bool = False) -> Engine:
    if path != ":memory:":
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        sqlite_url(path),
        echo=echo,
        future=True,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
        dbapi_connection.execute("PRAGMA busy_timeout=5000")
        if path != ":memory:":
            dbapi_connection.execute("PRAGMA journal_mode=WAL")
            dbapi_connection.execute("PRAGMA synchronous=NORMAL")
            # Keep local DB viewers reasonably fresh while retaining WAL's
            # concurrency benefits. SQLite's default (1000 pages) can leave a
            # small agent database looking stale for a long time.
            dbapi_connection.execute("PRAGMA wal_autocheckpoint=10")
        # Apple's system SQLite can be compiled without extension loading.
        # Keep vector persistence/search functional through the portable BLOB
        # fallback in the repository when sqlite-vec cannot be loaded.
        if hasattr(dbapi_connection, "enable_load_extension"):
            try:
                dbapi_connection.enable_load_extension(True)
                sqlite_vec.load(dbapi_connection)
            except Exception:
                pass
            finally:
                dbapi_connection.enable_load_extension(False)

    return engine


class Database:
    def __init__(self, path: str, *, echo: bool = False) -> None:
        self.engine = create_database_engine(path, echo=echo)
        self._factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self.engine.dispose()
