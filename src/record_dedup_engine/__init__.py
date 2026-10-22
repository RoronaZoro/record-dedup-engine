"""Deterministic deduplication tools for structured datasets."""

from record_dedup_engine.exceptions import (
    DeduplicationError,
    InputValidationError,
    PersistenceError,
    SchemaValidationError,
)
from record_dedup_engine.matching import deduplicate_composite, deduplicate_exact
from record_dedup_engine.models import DedupConfig, DedupResult, NormalizationConfig
from record_dedup_engine.pipeline import DedupPipeline

__all__ = [
    "DedupConfig",
    "DedupPipeline",
    "DedupResult",
    "DeduplicationError",
    "deduplicate_composite",
    "deduplicate_exact",
    "InputValidationError",
    "NormalizationConfig",
    "PersistenceError",
    "SchemaValidationError",
]
