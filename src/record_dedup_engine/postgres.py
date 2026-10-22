"""Optional PostgreSQL persistence through psycopg 3."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from typing import TYPE_CHECKING

import polars as pl

from record_dedup_engine.exceptions import InputValidationError, PersistenceError

if TYPE_CHECKING:
    from psycopg import Connection


def _batches(frame: pl.DataFrame, size: int) -> Iterator[list[tuple[object, ...]]]:
    rows = frame.iter_rows()
    while batch := list(islice(rows, size)):
        yield batch


def persist_postgres(
    frame: pl.DataFrame,
    *,
    connection: Connection,
    table: str,
    batch_size: int = 1_000,
) -> int:
    """Insert rows using safely quoted identifiers; caller owns transaction/commit."""
    if not isinstance(frame, pl.DataFrame):
        raise InputValidationError("frame must be a Polars DataFrame")
    if not isinstance(table, str) or not table.strip():
        raise InputValidationError("table must be a non-empty identifier")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise InputValidationError("batch_size must be greater than zero")
    if frame.width == 0:
        raise InputValidationError("cannot persist a frame with no columns")
    try:
        from psycopg import Error, sql
    except ImportError as exc:
        raise PersistenceError("PostgreSQL support requires the 'postgres' extra") from exc

    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(column) for column in frame.columns),
        sql.SQL(", ").join(sql.Placeholder() for _ in frame.columns),
    )
    inserted = 0
    try:
        with connection.cursor() as cursor:
            for batch in _batches(frame, batch_size):
                cursor.executemany(statement, batch)
                inserted += len(batch)
    except Error as exc:
        raise PersistenceError(f"PostgreSQL insert into {table!r} failed") from exc
    return inserted
