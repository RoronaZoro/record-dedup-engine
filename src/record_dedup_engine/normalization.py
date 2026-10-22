"""Polars expression helpers for composite key normalization."""

import polars as pl

from record_dedup_engine.models import NormalizationConfig


def normalized_key_expr(column: str, config: NormalizationConfig, dtype: pl.DataType) -> pl.Expr:
    """Build a lazy expression that normalizes string values without changing source data."""
    expression = pl.col(column)
    if dtype != pl.String:
        return expression
    if config.strip_whitespace:
        expression = expression.str.strip_chars()
    if config.lowercase:
        expression = expression.str.to_lowercase()
    return expression
