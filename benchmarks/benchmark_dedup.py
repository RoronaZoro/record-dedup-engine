"""Run reproducible synthetic deduplication benchmarks (no results are checked in)."""

import argparse
import platform
import random
import time

import polars as pl

from record_dedup_engine import DedupConfig, DedupPipeline, NormalizationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be greater than zero")

    rng = random.Random(args.seed)
    unique_records = max(1, args.rows // 2)
    frame = pl.DataFrame(
        {
            "record_id": [rng.randrange(unique_records) for _ in range(args.rows)],
            "email": [
                f" User{rng.randrange(unique_records)}@Example.test " for _ in range(args.rows)
            ],
            "payload": [rng.randrange(10_000) for _ in range(args.rows)],
        }
    ).lazy()
    pipeline = DedupPipeline(
        composite=DedupConfig(
            ("email",), NormalizationConfig(strip_whitespace=True, lowercase=True)
        )
    )
    started = time.perf_counter()
    result = pipeline.run(frame)
    elapsed = time.perf_counter() - started
    print(f"python={platform.python_version()} polars={pl.__version__}")
    print(f"rows={args.rows} seed={args.seed} output_rows={result.output_rows}")
    print(f"elapsed_seconds={elapsed:.6f}")


if __name__ == "__main__":
    main()
