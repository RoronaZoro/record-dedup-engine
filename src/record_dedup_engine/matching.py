"""Lazy deterministic matching and deduplication operations."""

import logging

import polars as pl

from record_dedup_engine.exceptions import InputValidationError, SchemaValidationError
from record_dedup_engine.models import DedupConfig
from record_dedup_engine.normalization import normalized_key_expr

LOGGER = logging.getLogger(__name__)
_ROW_ORDER = "__rde_row_order"
_KEY_PREFIX = "__rde_key_"


def validate_keys(frame: pl.LazyFrame, columns: tuple[str, ...]) -> None:
    """Validate key column names against the lazy schema without collecting rows."""
    if not isinstance(frame, pl.LazyFrame):
        raise InputValidationError("frame must be a Polars LazyFrame")
    schema = frame.collect_schema()
    missing = [column for column in columns if column not in schema]
    if missing:
        raise SchemaValidationError(f"key columns are missing from input: {missing}")


def deduplicate_exact(frame: pl.LazyFrame, subset: tuple[str, ...] | None = None) -> pl.LazyFrame:
    """Remove identical rows (or identical selected columns), keeping first-seen rows."""
    if not isinstance(frame, pl.LazyFrame):
        raise InputValidationError("frame must be a Polars LazyFrame")
    schema = frame.collect_schema()
    if subset is not None:
        if isinstance(subset, str) or not isinstance(subset, (tuple, list)):
            raise InputValidationError("subset must be a tuple or list of column names")
        subset = tuple(subset)
    columns = subset if subset is not None else tuple(schema.names())
    if not columns:
        raise SchemaValidationError("exact deduplication requires at least one column")
    validate_keys(frame, columns)
    if _ROW_ORDER in schema:
        raise SchemaValidationError(f"input uses reserved internal column name: {_ROW_ORDER}")
    LOGGER.info("Applying exact deduplication", extra={"key_columns": columns})
    return (
        frame.with_columns(pl.int_range(pl.len(), dtype=pl.UInt64).alias(_ROW_ORDER))
        .unique(subset=list(columns), keep="first", maintain_order=True)
        .sort(_ROW_ORDER)
        .drop(_ROW_ORDER)
    )


def deduplicate_composite(frame: pl.LazyFrame, config: DedupConfig) -> pl.LazyFrame:
    """Deduplicate by normalized columns, retaining the first original record."""
    if not isinstance(config, DedupConfig):
        raise InputValidationError("config must be a DedupConfig")
    validate_keys(frame, config.key_columns)
    schema = frame.collect_schema()
    reserved = {_ROW_ORDER, *[f"{_KEY_PREFIX}{index}" for index in range(len(config.key_columns))]}
    collisions = sorted(reserved.intersection(schema.names()))
    if collisions:
        raise SchemaValidationError(f"input uses reserved internal column names: {collisions}")

    LOGGER.info(
        "Applying composite-key deduplication",
        extra={"key_columns": config.key_columns},
    )
    keyed = frame.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt64).alias(_ROW_ORDER)
    ).with_columns(
        [
            normalized_key_expr(column, config.normalization, schema[column]).alias(
                f"{_KEY_PREFIX}{index}"
            )
            for index, column in enumerate(config.key_columns)
        ]
    )
    key_columns = [f"{_KEY_PREFIX}{index}" for index in range(len(config.key_columns))]
    return (
        keyed.unique(subset=key_columns, keep="first", maintain_order=True)
        .sort(_ROW_ORDER)
        .drop([_ROW_ORDER, *key_columns])
    )
