"""SQLite connection management.

The original notebook opened a brand-new `sqlite3.connect(...)` inside every
single tool function and never closed it on the error path. Here connections
are always opened via a context manager (`get_connection`), guaranteeing
`close()` runs even on exceptions, and the DB path is read from Settings
instead of being hardcoded as a literal string in five different places.

SQLite only supports one writer at a time; a process-wide lock serializes
writes so concurrent FastAPI requests (which run tool calls in a thread pool)
don't collide. For real multi-worker/high-throughput production deployments,
swap this module for a PostgreSQL connection pool (SQLAlchemy) -- the
`repository.py` layer above it is written against plain parameterized SQL and
would need only its connection acquisition changed.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from app.config import get_settings

_write_lock = threading.Lock()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access by column name."""
    settings = get_settings()
    _ensure_parent_dir(settings.database_path)
    conn = sqlite3.connect(settings.database_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_lock() -> Iterator[None]:
    """Serialize write operations against the single SQLite file."""
    with _write_lock:
        yield


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]
