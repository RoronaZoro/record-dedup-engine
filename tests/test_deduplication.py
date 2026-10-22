"""Behavioral tests for deterministic lazy deduplication."""

import polars as pl
import pytest

from record_dedup_engine import (
    DedupConfig,
    DedupPipeline,
    InputValidationError,
    NormalizationConfig,
    SchemaValidationError,
)
from record_dedup_engine.matching import deduplicate_composite, deduplicate_exact


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([], []),
        ([{"id": 1}, {"id": 1}], [{"id": 1}]),
        ([{"id": 1}, {"id": 2}, {"id": 1}], [{"id": 1}, {"id": 2}]),
    ],
)
def test_exact_deduplication_keeps_first_occurrence(records, expected):
    frame = pl.DataFrame(schema={"id": pl.Int64}) if not records else pl.DataFrame(records)
    result = deduplicate_exact(frame.lazy()).collect()
    assert result.to_dicts() == expected


def test_exact_subset_preserves_first_original_row_and_order():
    frame = pl.DataFrame({"email": ["a", "b", "a"], "value": [1, 2, 3]}).lazy()
    result = deduplicate_exact(frame, subset=("email",)).collect()
    assert result.to_dicts() == [{"email": "a", "value": 1}, {"email": "b", "value": 2}]


def test_both_deduplication_modes_use_uint64_row_ordinals(monkeypatch):
    observed_dtypes = []
    original_unique = pl.LazyFrame.unique

    def inspect_ordinal_dtype(frame, *args, **kwargs):
        schema = frame.collect_schema()
        if "__rde_row_order" in schema:
            observed_dtypes.append(schema["__rde_row_order"])
        return original_unique(frame, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "unique", inspect_ordinal_dtype)
    source = pl.DataFrame({"id": [1, 1], "name": ["a", "b"]}).lazy()

    deduplicate_exact(source).collect()
    deduplicate_composite(source, DedupConfig(("id",))).collect()

    assert observed_dtypes == [pl.UInt64, pl.UInt64]


@pytest.mark.parametrize(
    ("strip", "lowercase", "expected"),
    [
        (True, True, ([" Alice ", " Bob "], [1, 3])),
        (True, False, ([" Alice ", "alice", " Bob ", "bob"], [1, 2, 3, 4])),
        (False, True, ([" Alice ", "alice", " Bob ", "bob"], [1, 2, 3, 4])),
    ],
)
def test_composite_deduplication_normalizes_keys_but_preserves_data(strip, lowercase, expected):
    frame = pl.DataFrame(
        {"email": [" Alice ", "alice", " Bob ", "bob"], "source": [1, 2, 3, 4]}
    ).lazy()
    config = DedupConfig(
        ("email",), NormalizationConfig(strip_whitespace=strip, lowercase=lowercase)
    )
    result = deduplicate_composite(frame, config).collect()
    expected_emails, expected_sources = expected
    assert result.get_column("email").to_list() == expected_emails
    assert result.get_column("source").to_list() == expected_sources


def test_composite_key_uses_all_components_and_nulls_match():
    frame = pl.DataFrame(
        {"first": ["a", "a", "a", None, None], "last": ["x", "y", "x", "z", "z"]}
    ).lazy()
    result = deduplicate_composite(frame, DedupConfig(("first", "last"))).collect()
    assert result.to_dicts() == [
        {"first": "a", "last": "x"},
        {"first": "a", "last": "y"},
        {"first": None, "last": "z"},
    ]


def test_non_string_key_is_left_unchanged():
    frame = pl.DataFrame({"id": [1, 1, 2], "value": ["a", "b", "c"]}).lazy()
    result = deduplicate_composite(frame, DedupConfig(("id",))).collect()
    assert result.to_dicts() == [{"id": 1, "value": "a"}, {"id": 2, "value": "c"}]


@pytest.mark.parametrize("columns", [(), ("",), ("id", "id")])
def test_invalid_composite_columns_raise(columns):
    with pytest.raises(InputValidationError):
        DedupConfig(columns)


def test_column_names_must_be_a_sequence_not_a_string():
    with pytest.raises(InputValidationError, match="tuple or list"):
        DedupConfig("email")


def test_wrong_frame_type_raises_domain_input_error():
    with pytest.raises(InputValidationError, match="LazyFrame"):
        deduplicate_exact(pl.DataFrame({"id": [1]}))


def test_missing_column_raises_domain_schema_error():
    with pytest.raises(SchemaValidationError, match="missing"):
        deduplicate_composite(pl.DataFrame({"id": [1]}).lazy(), DedupConfig(("email",)))


def test_reserved_internal_names_raise_clear_error():
    frame = pl.DataFrame({"id": [1], "__rde_row_order": [8]}).lazy()
    with pytest.raises(SchemaValidationError, match="reserved"):
        deduplicate_exact(frame)


def test_pipeline_collects_and_reports_accounting():
    frame = pl.DataFrame({"id": [1, 1, 2], "name": ["A", "A", "B"]}).lazy()
    result = DedupPipeline(composite=DedupConfig(("id",))).run(frame)
    assert result.input_rows == 3
    assert result.output_rows == 2
    assert result.removed_rows == 1
    assert result.frame.get_column("id").to_list() == [1, 2]


def test_lazy_method_returns_a_lazy_frame():
    plan = DedupPipeline(composite=DedupConfig(("id",))).lazy(pl.DataFrame({"id": [1, 1]}).lazy())
    assert isinstance(plan, pl.LazyFrame)
    assert plan.collect().height == 1
