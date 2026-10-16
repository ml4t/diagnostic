"""Prevent explicit qcut selection from silently running uniform signal bins."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.diagnostic import analyze_signal
from ml4t.diagnostic.signal import prepare_data


def _priced_factor(values: list[float]) -> tuple[pl.DataFrame, pl.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=2)
    assets = [f"A{i:02}" for i in range(len(values))]
    factor = pl.DataFrame({"date": [dates[0]] * len(values), "asset": assets, "factor": values})
    prices = pl.DataFrame(
        {
            "date": [dates[0]] * len(values) + [dates[1]] * len(values),
            "asset": assets * 2,
            "price": [100.0] * len(values) + [100.0 + i for i in range(len(values))],
        }
    )
    return factor, prices


@pytest.mark.parametrize("scale", [1.0, 1e-14])
def test_explicit_qcut_uses_interpolated_edges_not_uniform_or_rank_membership(scale):
    factor, prices = _priced_factor([float(2**i) * scale for i in range(12)])
    before_factor, before_prices = factor.clone(), prices.clone()

    result = prepare_data(
        factor, prices, periods=(1,), quantiles=5, filter_zscore=None, quantile_method="qcut"
    ).sort("asset")

    assert result["quantile"].to_list() == [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
    np.testing.assert_allclose(result["1D_fwd_return"], np.arange(12) / 100, atol=1e-14)
    assert_frame_equal(factor, before_factor)
    assert_frame_equal(prices, before_prices)


def test_high_level_qcut_analysis_uses_exact_bucket_return_membership():
    factor, prices = _priced_factor([float(2**i) for i in range(12)])

    result = analyze_signal(
        factor,
        prices,
        periods=(1,),
        quantiles=5,
        filter_zscore=None,
        quantile_method="qcut",
        compute_turnover_flag=False,
    )

    assert result.count_by_quantile == {1: 3, 2: 2, 3: 2, 4: 2, 5: 3}
    for quantile, expected in {1: 0.01, 2: 0.035, 3: 0.055, 4: 0.075, 5: 0.1}.items():
        assert result.quantile_returns["1D"][quantile] == pytest.approx(expected)
    assert result.preparation_metadata["bucket_method"] == "pandas.qcut"
    assert result.preparation_metadata["pandas_version"] == pd.__version__
    assert result.preparation_metadata["numpy_version"] == np.__version__
    assert json.loads(result.to_json())["preparation_metadata"] == result.preparation_metadata


def test_qcut_keeps_equal_values_together_at_interpolated_boundaries():
    factor, prices = _priced_factor([float(i // 2) for i in range(12)])

    result = prepare_data(
        factor, prices, periods=(1,), quantiles=5, filter_zscore=None, quantile_method="qcut"
    )

    assert result["quantile"].to_list() == [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


@pytest.mark.parametrize("count", [2, 3, 5])
def test_qcut_rejects_duplicate_edges_instead_of_inventing_uniform_buckets(count):
    factor, prices = _priced_factor([1.0] * 12)

    with pytest.raises((pl.exceptions.DuplicateError, ValueError)):
        prepare_data(
            factor,
            prices,
            periods=(1,),
            quantiles=count,
            filter_zscore=None,
            quantile_method="qcut",
        )


def test_qcut_rejects_a_repeated_endpoint_even_with_one_interior_edge():
    factor, prices = _priced_factor([0.0, 0.0, 1.0])

    with pytest.raises((pl.exceptions.DuplicateError, ValueError)):
        prepare_data(
            factor, prices, periods=(1,), quantiles=2, filter_zscore=None, quantile_method="qcut"
        )


@pytest.mark.parametrize("scale", [1.0, 1e-14])
def test_qcut_matches_runtime_reference_edges_at_fractional_quantile_boundaries(scale):
    values = np.arange(4, dtype=float) * scale
    factor, prices = _priced_factor(values.tolist())
    expected = (pd.qcut(values, 3, labels=False) + 1).tolist()

    result = prepare_data(
        factor, prices, periods=(1,), quantiles=3, filter_zscore=None, quantile_method="qcut"
    )

    assert result["quantile"].to_list() == expected


def test_qcut_one_bucket_keeps_a_constant_cross_section():
    factor, prices = _priced_factor([1.0] * 12)

    result = prepare_data(
        factor, prices, periods=(1,), quantiles=1, filter_zscore=None, quantile_method="qcut"
    )

    assert result["quantile"].to_list() == [1] * 12


def test_qcut_groups_by_date_preserves_input_order_and_unrelated_columns():
    first_factor, first_prices = _priced_factor([float(2**i) for i in range(12)])
    second_factor = first_factor.head(7).with_columns(pl.col("date").dt.offset_by("7d"))
    second_prices = first_prices.filter(pl.col("asset").is_in(second_factor["asset"].implode()))
    second_prices = second_prices.with_columns(pl.col("date").dt.offset_by("7d"))
    factor = pl.concat([second_factor.reverse(), first_factor.reverse()]).with_row_index("source")
    factor = factor.with_columns(pl.lit("original").alias("factor_quantile"))

    result = prepare_data(
        factor,
        pl.concat([first_prices, second_prices]),
        periods=(1,),
        quantiles=5,
        filter_zscore=None,
        quantile_method="qcut",
    )

    expected = [5, 5, 4, 3, 2, 1, 1] + [5, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1]
    assert result["quantile"].to_list() == expected
    assert result["source"].to_list() == factor["source"].to_list()
    assert result["factor_quantile"].to_list() == ["original"] * 19


@pytest.mark.parametrize("method", ["qct", "uniformm", "", None])
def test_misspelled_method_does_not_silently_substitute_uniform_bins(method):
    factor, prices = _priced_factor([float(2**i) for i in range(12)])

    with pytest.raises(ValueError, match="QuantileMethod"):
        prepare_data(factor, prices, periods=(1,), filter_zscore=None, quantile_method=method)


@pytest.mark.parametrize("count", [0, -1, 2.5, True, "5"])
def test_qcut_rejects_invalid_bucket_counts_before_assignment(count):
    factor, prices = _priced_factor([float(2**i) for i in range(12)])

    with pytest.raises(ValueError, match="positive integer"):
        prepare_data(
            factor,
            prices,
            periods=(1,),
            quantiles=count,
            filter_zscore=None,
            quantile_method="qcut",
        )


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_qcut_does_not_hide_missing_factor_input_without_a_cleaning_policy(missing):
    values = [float(2**i) for i in range(12)]
    values[5] = missing
    factor, prices = _priced_factor(values)

    with pytest.raises(ValueError, match="finite"):
        prepare_data(factor, prices, periods=(1,), filter_zscore=None, quantile_method="qcut")


def test_legacy_rank_and_uniform_defaults_retain_their_distinct_membership():
    factor, prices = _priced_factor([float(2**i) for i in range(12)])

    rank = prepare_data(factor, prices, periods=(1,), quantiles=5, filter_zscore=None)
    uniform = prepare_data(
        factor, prices, periods=(1,), quantiles=5, filter_zscore=None, quantile_method="uniform"
    )

    assert rank["quantile"].to_list() == [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    assert uniform["quantile"].to_list() == [1] * 9 + [2, 3, 5]
