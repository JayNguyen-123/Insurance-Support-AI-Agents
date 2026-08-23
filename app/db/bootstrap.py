"""One-time (or re-runnable) database setup: create schema + load sample data."""

from __future__ import annotations

import sqlite3

from app.db.schema import CREATE_TABLES_SQL, DROP_TABLES_SQL
from app.db.seed_data import generate_sample_data
from app.db.session import get_connection, write_lock
from app.logging_config import get_logger

logger = get_logger(__name__)


def drop_and_create_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.executescript(DROP_TABLES_SQL)
    cursor.executescript(CREATE_TABLES_SQL)
    conn.commit()


def insert_data(conn: sqlite3.Connection, data: dict) -> None:
    for table, df in data.items():
        df.to_sql(table, conn, if_exists="append", index=False)
    conn.commit()


def setup_insurance_database(random_state: int = 42) -> None:
    """Create schema and populate it with synthetic sample data."""
    data = generate_sample_data(random_state=random_state)
    with write_lock(), get_connection() as conn:
        drop_and_create_tables(conn)
        insert_data(conn, data)
    logger.info("Database created and populated with synthetic sample data.")


if __name__ == "__main__":
    setup_insurance_database()
