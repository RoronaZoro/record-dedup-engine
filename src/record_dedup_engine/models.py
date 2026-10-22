"""Small immutable configuration and result models."""

from dataclasses import dataclass, field

import polars as pl

from record_dedup_engine.exceptions import InputValidationError


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """String transforms applied only while constructing composite keys."""

    strip_whitespace: bool = True
    lowercase: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.strip_whitespace, bool) or not isinstance(self.lowercase, bool):
            raise InputValidationError("normalization options must be booleans")


@dataclass(frozen=True, slots=True)
class DedupConfig:
    """Configuration for composite-key deduplication."""

    key_columns: tuple[str, ...]
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    def __post_init__(self) -> None:
        if isinstance(self.key_columns, str) or not isinstance(self.key_columns, (tuple, list)):
            raise InputValidationError("key_columns must be a tuple or list of column names")
        object.__setattr__(self, "key_columns", tuple(self.key_columns))
        if not self.key_columns:
            raise InputValidationError("key_columns must contain at least one column")
        if any(not isinstance(column, str) or not column.strip() for column in self.key_columns):
            raise InputValidationError("key_columns must contain non-empty strings")
        if len(set(self.key_columns)) != len(self.key_columns):
            raise InputValidationError("key_columns must not contain duplicates")
        if not isinstance(self.normalization, NormalizationConfig):
            raise InputValidationError("normalization must be a NormalizationConfig")


@dataclass(frozen=True, slots=True)
class DedupResult:
    """Materialized output and basic accounting from a pipeline run."""

    frame: pl.DataFrame
    input_rows: int
    output_rows: int

    @property
    def removed_rows(self) -> int:
        """Number of rows removed by the configured deduplication steps."""
        return self.input_rows - self.output_rows
