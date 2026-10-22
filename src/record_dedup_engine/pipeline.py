"""Orchestration for lazy record deduplication."""

import logging

import polars as pl

from record_dedup_engine.exceptions import InputValidationError
from record_dedup_engine.matching import deduplicate_composite, deduplicate_exact
from record_dedup_engine.models import DedupConfig, DedupResult

LOGGER = logging.getLogger(__name__)


class DedupPipeline:
    """Compose optional exact and composite-key deduplication steps."""

    def __init__(
        self,
        *,
        exact: bool = True,
        composite: DedupConfig | None = None,
    ) -> None:
        if not isinstance(exact, bool):
            raise InputValidationError("exact must be a boolean")
        if composite is not None and not isinstance(composite, DedupConfig):
            raise InputValidationError("composite must be a DedupConfig or None")
        self.exact = exact
        self.composite = composite

    def lazy(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Return a lazy plan; source data is not collected by this method."""
        if not isinstance(frame, pl.LazyFrame):
            raise InputValidationError("frame must be a Polars LazyFrame")
        result = frame
        if self.exact:
            result = deduplicate_exact(result)
        if self.composite is not None:
            result = deduplicate_composite(result, self.composite)
        return result

    def run(self, frame: pl.LazyFrame) -> DedupResult:
        """Execute the plan and return output row counts."""
        if not isinstance(frame, pl.LazyFrame):
            raise InputValidationError("frame must be a Polars LazyFrame")
        LOGGER.info("Collecting deduplication pipeline")
        original_count = frame.select(pl.len()).collect().item()
        output = self.lazy(frame).collect()
        LOGGER.info(
            "Deduplication pipeline complete",
            extra={"input_rows": original_count, "output_rows": output.height},
        )
        return DedupResult(frame=output, input_rows=original_count, output_rows=output.height)
