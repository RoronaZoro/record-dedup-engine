"""Validation tests for the optional PostgreSQL adapter."""

from unittest.mock import MagicMock

import polars as pl
import psycopg
import pytest

from record_dedup_engine.exceptions import InputValidationError, PersistenceError
from record_dedup_engine.postgres import persist_postgres


@pytest.mark.parametrize(
    ("table", "batch_size"),
    [("", 1), ("users", 0), ("users", -1)],
)
def test_invalid_persistence_options_fail_before_connecting(table, batch_size):
    with pytest.raises(InputValidationError):
        persist_postgres(
            pl.DataFrame({"id": [1]}), connection=object(), table=table, batch_size=batch_size
        )


def test_persistence_requires_columns():
    with pytest.raises(InputValidationError, match="no columns"):
        persist_postgres(pl.DataFrame(), connection=object(), table="users")


def test_persistence_inserts_in_bounded_batches():
    connection = MagicMock()
    frame = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

    inserted = persist_postgres(frame, connection=connection, table="users", batch_size=2)

    cursor = connection.cursor.return_value.__enter__.return_value
    assert inserted == 3
    assert cursor.executemany.call_count == 2
    assert cursor.executemany.call_args_list[0].args[1] == [(1, "a"), (2, "b")]
    assert cursor.executemany.call_args_list[1].args[1] == [(3, "c")]


def test_database_errors_are_wrapped_with_original_cause():
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.executemany.side_effect = psycopg.Error("synthetic database failure")

    with pytest.raises(PersistenceError, match="insert into") as error:
        persist_postgres(pl.DataFrame({"id": [1]}), connection=connection, table="users")

    assert isinstance(error.value.__cause__, psycopg.Error)
