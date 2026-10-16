"""Equivalent datetime units must join without rounding signal observation keys."""

from datetime import UTC, datetime, timezone
from itertools import product

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import analyze_feature_outcome, compute_ic_by_horizon
from ml4t.diagnostic.signal import prepare_data


def _inputs(pred_unit, price_unit, pandas_input):
    dates = pd.bdate_range("2025-01-02", periods=4, tz="UTC")
    prices = (
        pl.DataFrame(
            {
                "timestamp": [day for day in dates for _ in range(2)],
                "instrument": ["A", "B"] * 4,
                "mark": [256.0, 512.0, 288.0, 640.0, 270.0, 720.0, 337.5, 675.0],
            }
        )
        .with_columns(pl.col("timestamp").cast(pl.Datetime(price_unit, "UTC")))
        .reverse()
    )
    predictions = pl.DataFrame(
        {
            "timestamp": [day for day in dates[:3] for _ in range(2)],
            "instrument": ["A", "B"] * 3,
            "alpha": [0.125, 0.25, -0.0625, 0.125, 0.25, -0.0625],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime(pred_unit, "UTC")))
    if pandas_input:
        return predictions.to_pandas(), prices.to_pandas()
    return predictions, prices


def _run(route, predictions, prices):
    if route == "prepare":
        result = prepare_data(
            predictions,
            prices,
            periods=(1,),
            quantiles=2,
            filter_zscore=None,
            factor_col="alpha",
            date_col="timestamp",
            asset_col="instrument",
            price_col="mark",
        )
        np.testing.assert_allclose(
            result["1D_fwd_return"], [0.125, 0.25, -0.0625, 0.125, 0.25, -0.0625]
        )
        assert result.height == 6
    elif route == "horizon":
        result = compute_ic_by_horizon(
            predictions,
            prices,
            horizons=[1],
            pred_col="alpha",
            price_col="mark",
            date_col="timestamp",
            group_col="instrument",
        )
        assert result[1] == pytest.approx(1.0)
    else:
        result = analyze_feature_outcome(
            predictions,
            prices,
            pred_col="alpha",
            price_col="mark",
            date_col="timestamp",
            group_col="instrument",
            include_decay=False,
            include_monotonicity=False,
            include_hac=False,
        )
        assert result["ic_summary"]["mean_ic"] == pytest.approx(1.0)
        assert result["ic_series"]["n_obs"].to_list() == [2, 2, 2]


@pytest.mark.parametrize("pred_unit,price_unit", list(product(["ms", "us", "ns"], repeat=2)))
@pytest.mark.parametrize("pandas_input", [False, True])
@pytest.mark.parametrize("route", ["prepare", "horizon", "feature"])
def test_equivalent_instants_with_different_storage_units_preserve_public_workflow(
    pred_unit, price_unit, pandas_input, route
):
    predictions, prices = _inputs(pred_unit, price_unit, pandas_input)
    before_pred = predictions.copy(deep=True) if pandas_input else predictions.clone()
    before_prices = prices.copy(deep=True) if pandas_input else prices.clone()
    _run(route, predictions, prices)
    if pandas_input:
        pd.testing.assert_frame_equal(predictions, before_pred)
        pd.testing.assert_frame_equal(prices, before_prices)
    else:
        assert predictions.equals(before_pred)
        assert prices.equals(before_prices)


def test_coarse_price_keys_never_round_distinct_nanosecond_factor_queries_together():
    predictions, prices = _inputs("ns", "us", False)
    first = predictions.head(1)
    next_query = first.with_columns(
        (pl.col("timestamp").cast(pl.Int64) + 100).cast(pl.Datetime("ns", "UTC"))
    )
    queries = pl.concat([first, next_query])
    result = prepare_data(
        queries,
        prices,
        periods=(1,),
        quantiles=2,
        filter_zscore=None,
        factor_col="alpha",
        date_col="timestamp",
        asset_col="instrument",
        price_col="mark",
    )
    assert result.height == 2
    assert (
        result["timestamp"].cast(pl.Int64).to_list()
        == queries["timestamp"].cast(pl.Int64).to_list()
    )
    assert result["1D_fwd_return"].to_list() == [0.125, None]


@pytest.mark.parametrize("route", ["prepare", "horizon", "feature"])
def test_unrepresentable_finer_key_unit_raises_instead_of_wrapping_observation_identity(route):
    predictions, prices = _inputs("us", "ns", False)
    predictions = predictions.with_columns(
        pl.lit(datetime(2500, 1, 2, tzinfo=UTC)).alias("timestamp")
    )
    predictions = predictions.head(2)
    before_pred, before_prices = predictions.clone(), prices.clone()
    with pytest.raises(ValueError, match="lossless|represent"):
        _run(route, predictions, prices)
    assert predictions.equals(before_pred)
    assert prices.equals(before_prices)


@pytest.mark.parametrize("zone", [None, "America/New_York"])
@pytest.mark.parametrize("route", ["prepare", "horizon", "feature"])
def test_unit_alignment_does_not_infer_a_timezone_policy(zone, route):
    predictions, prices = _inputs("us", "ns", False)
    expression = pl.col("timestamp")
    expression = (
        expression.dt.replace_time_zone(None)
        if zone is None
        else expression.dt.convert_time_zone(zone)
    )
    prices = prices.with_columns(expression)
    with pytest.raises(pl.exceptions.SchemaError):
        _run(route, predictions, prices)
