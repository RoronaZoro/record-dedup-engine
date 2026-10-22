# record-dedup-engine

A Python 3.12 library for deterministic, memory-conscious deduplication of structured datasets. It uses Polars `LazyFrame` plans so callers can compose deduplication with filters, projections, and scans before materializing results.

## Features

- Exact row deduplication and composite-key deduplication.
- Stable first-seen winner selection and output ordering.
- Configurable whitespace trimming and lowercase normalization for string keys.
- Explicit schema and option validation with domain-specific exceptions.
- Structured logging through Python's standard `logging` module.
- Optional batched PostgreSQL inserts using safely quoted identifiers.
- Synthetic, seeded benchmark harness; no benchmark results are asserted or pre-reported.

## Install

Requires Python 3.12 or later.

```shell
python -m pip install -e ".[dev]"
```

For PostgreSQL persistence in an application, install `record-dedup-engine[postgres]`. The adapter uses psycopg 3. The library does not create tables or commit transactions.

## Quick start

```python
import polars as pl

from record_dedup_engine import DedupConfig, DedupPipeline, NormalizationConfig

source = pl.scan_parquet("incoming/*.parquet")
pipeline = DedupPipeline(
    composite=DedupConfig(
        key_columns=("email", "country"),
        normalization=NormalizationConfig(strip_whitespace=True, lowercase=True),
    )
)

# Keep the plan lazy to compose more query operations.
deduplicated_plan = pipeline.lazy(source.filter(pl.col("active")))
deduplicated = deduplicated_plan.collect()

# Or execute and receive row accounting in one result object.
result = pipeline.run(source)
print(result.input_rows, result.output_rows, result.removed_rows)
```

By default the pipeline first removes exact duplicate rows, then removes records sharing a normalized composite key. Set `exact=False` when only key-based deduplication is desired. With `exact=True` and no composite config, exact full-row deduplication is performed.

## Architecture

`models` contains immutable configuration and result types. `normalization` builds Polars expressions for normalized keys. `matching` validates the lazy schema and constructs exact or composite lazy plans. `pipeline` composes those operations and owns materialization/accounting. `postgres` is an optional sink adapter and is not imported by the package's normal API.

The input must be a Polars `LazyFrame`. `lazy()` only builds a query plan. `run()` collects it and returns a Polars `DataFrame`. The latter currently evaluates the source once for the input row count and again for the deduplicated result; for remote scans or expensive sources, use `lazy()` and manage metrics at the source layer if the extra pass is undesirable.

## Semantics and trade-offs

- Both modes retain the first record in source order. An internal row index is used to make the winner and returned ordering explicit. This requires maintaining order and may prevent some streaming optimizations in Polars.
- Exact deduplication compares every column. It can take a subset through `deduplicate_exact(frame, subset=(...))`; the pipeline's exact stage uses all columns.
- Composite keys are compared across all configured columns. String key values are trimmed and lowercased by default; normalization does not alter the returned source columns. Case conversion follows Polars' Unicode lowercase behavior, not locale-specific rules or full Unicode case folding. Non-string columns remain unchanged.
- Nulls compare as equal key values under Polars `unique` semantics. Duplicate null-containing composite keys retain their first row. Nulls are not rejected or treated as unique records.
- Input row order defines the first record; results can therefore depend on source ordering. Sort the source explicitly when winner selection must follow a business priority.
- Exact and composite uniqueness require state proportional to the number of distinct keys. Lazy execution can reduce intermediate materialization, but it does not make global deduplication constant-memory.
- Schema validation happens when a plan is built. Value-level data errors are raised by Polars during collection and are not broadly caught or rewritten.
- `run()` counts source and output records, but repeated evaluation assumes the source is stable across both executions. Avoid it for non-repeatable sources.

## PostgreSQL persistence

The optional helper inserts a collected Polars `DataFrame` into an existing table:

```python
from record_dedup_engine.postgres import persist_postgres

inserted = persist_postgres(
    result.frame,
    connection=connection,
    table="deduplicated_records",
    batch_size=2_000,
)
```

Identifiers are quoted with psycopg's SQL composition API. Batches bound the adapter's extra row-buffer memory. The caller supplies a psycopg connection and owns the surrounding transaction, including commit or rollback. Table schema creation, type coercion, retries, and upsert behavior are deliberately outside this library.

## Development

```shell
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

GitHub Actions runs lint, formatting, and tests on Python 3.12. Tests use only synthetic in-memory data and do not require a database.

## Benchmarks

The harness generates deterministic synthetic records and prints runtime and row counts for the current environment. It does not ship or claim measured results:

```shell
python benchmarks/benchmark_dedup.py --rows 1000000 --seed 42
```

For useful comparisons, record Python and Polars versions, hardware, operating system, row count, key cardinality, and whether data is already in memory or read from storage. Run multiple trials and compare medians; benchmark values are workload- and machine-dependent.

## Limitations

This is a single-process library, not a distributed deduplication service. It does not infer fuzzy matches, apply entity resolution, persist checkpoints, or guarantee order for an unordered upstream source. Full Unicode case folding, locale-aware rules, null policy selection, streaming persistence from lazy sources, and PostgreSQL schema management are not included.

