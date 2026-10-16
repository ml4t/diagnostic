"""Reject overlapping batches before they duplicate signal observations or horizon steps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.diagnostic import analyze_signal
from ml4t.diagnostic.metrics import (
    analyze_feature_outcome,
    compute_forward_returns,
    compute_ic_by_horizon,
    compute_ic_decay,
)
from ml4t.diagnostic.signal import prepare_data


def _panel() -> tuple[pl.DataFrame, pl.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=3)
    assets = [f"A{i:02}" for i in range(12)]
    factor = pl.DataFrame(
        {
            "date": [day for day in dates[:2] for _ in assets],
            "asset": assets * 2,
            "factor": [float(i) for i in range(12)] * 2,
        }
    )
    prices = pl.DataFrame(
        {
            "date": [day for day in dates for _ in assets],
            "asset": assets * 3,
            "price": [100.0 + 5 * day + asset for day in range(3) for asset in range(12)],
        }
    )
    return factor, prices


def _metric_route(route, predictions, prices, *, date_col="date", group_col="asset"):
    options = {
        "pred_col": "factor",
        "price_col": "price",
        "date_col": date_col,
        "group_col": group_col,
    }
    if route == "horizon":
        return compute_ic_by_horizon(predictions, prices, horizons=[1], **options)
    if route == "decay":
        return compute_ic_decay(predictions, prices, horizons=[1], **options)
    return analyze_feature_outcome(
        predictions,
        prices,
        include_decay=False,
        include_monotonicity=False,
        include_hac=False,
        **options,
    )


@pytest.mark.parametrize("side", ["factor", "price"])
@pytest.mark.parametrize("conflicting", [False, True])
@pytest.mark.parametrize("pandas_input", [False, True])
@pytest.mark.parametrize("route", ["prepare", "analyze", "horizon", "decay", "feature"])
def test_overlapping_batches_cannot_duplicate_a_factor_sample(
    side, conflicting, pandas_input, route
):
    factor, prices = _panel()
    original_factor, original_prices = factor.clone(), prices.clone()
    if side == "factor":
        overlap = factor.head(1)
        if conflicting:
            overlap = overlap.with_columns((pl.col("factor") + 20).alias("factor"))
        factor = pl.concat([factor, overlap])
    else:
        overlap = prices.head(1)
        if conflicting:
            overlap = overlap.with_columns((pl.col("price") + 20).alias("price"))
        prices = pl.concat([prices, overlap])
    before_factor, before_prices = factor.clone(), prices.clone()
    factor_input = factor.to_pandas() if pandas_input else factor
    price_input = prices.to_pandas() if pandas_input else prices

    with pytest.raises(ValueError, match="(?i)duplicate"):
        if route == "prepare":
            prepare_data(factor_input, price_input, periods=(1,), filter_zscore=None)
        elif route == "analyze":
            analyze_signal(
                factor_input,
                price_input,
                periods=(1,),
                filter_zscore=None,
                compute_turnover_flag=False,
            )
        else:
            _metric_route(route, factor_input, price_input)

    assert_frame_equal(factor, before_factor)
    assert_frame_equal(prices, before_prices)
    if pandas_input:
        pd.testing.assert_frame_equal(factor_input, before_factor.to_pandas())
        pd.testing.assert_frame_equal(price_input, before_prices.to_pandas())
    assert original_factor.height == 24 and original_prices.height == 36


@pytest.mark.parametrize("route", ["horizon", "decay", "feature"])
@pytest.mark.parametrize("pandas_input", [False, True])
def test_custom_time_axis_sorts_each_asset_before_ic(route, pandas_input):
    dates = pd.bdate_range("2025-01-02", periods=4)
    prices = pl.DataFrame(
        {
            "timestamp": [day for day in dates for _ in range(2)],
            "asset": ["A", "B"] * 4,
            "price": [256.0, 512.0, 288.0, 640.0, 270.0, 720.0, 337.5, 675.0],
        }
    ).reverse()
    predictions = pl.DataFrame(
        {
            "timestamp": [day for day in dates[:3] for _ in range(2)],
            "asset": ["A", "B"] * 3,
            "factor": [0.125, 0.25, -0.0625, 0.125, 0.25, -0.0625],
        }
    )
    result = _metric_route(
        route,
        predictions.to_pandas() if pandas_input else predictions,
        prices.to_pandas() if pandas_input else prices,
        date_col="timestamp",
    )

    if route == "horizon":
        assert result[1] == pytest.approx(1.0)
    elif route == "decay":
        assert result["ic_by_horizon"][1] == pytest.approx(1.0)
    else:
        assert result["ic_summary"]["mean_ic"] == pytest.approx(1.0)
        assert len(result["ic_series"]) == 3
        np.testing.assert_array_equal(result["ic_series"]["n_obs"], [2, 2, 2])
        np.testing.assert_allclose(result["ic_series"]["ic"], [1.0, 1.0, 1.0])


@pytest.mark.parametrize("route", ["horizon", "decay", "feature"])
@pytest.mark.parametrize("pandas_input", [False, True])
def test_multi_asset_ic_requires_its_declared_entity_key(route, pandas_input):
    predictions, prices = _panel()

    with pytest.raises(ValueError, match="(?i)duplicate"):
        _metric_route(
            route,
            predictions.to_pandas() if pandas_input else predictions,
            prices.to_pandas() if pandas_input else prices,
            group_col=None,
        )


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("conflicting", [False, True])
@pytest.mark.parametrize("pandas_input", [False, True])
def test_public_forward_producer_rejects_duplicate_dated_price_observations(
    grouped, conflicting, pandas_input
):
    _, prices = _panel()
    if not grouped:
        prices = prices.filter(pl.col("asset") == "A00")
    overlap = prices.head(1)
    if conflicting:
        overlap = overlap.with_columns((pl.col("price") + 20).alias("price"))
    overlapping = pl.concat([prices, overlap])
    before = overlapping.clone()
    supplied = overlapping.to_pandas() if pandas_input else overlapping

    with pytest.raises(ValueError, match="duplicate"):
        compute_forward_returns(
            supplied, price_col="price", group_col="asset" if grouped else None, periods=1
        )

    assert_frame_equal(overlapping, before)
    if pandas_input:
        pd.testing.assert_frame_equal(supplied, before.to_pandas())


@pytest.mark.parametrize("side", ["factor", "price"])
def test_coerced_missing_timestamp_is_not_a_forward_return_observation(side):
    factor, prices = (frame.to_pandas() for frame in _panel())
    target = factor if side == "factor" else prices
    raw_dates = target["date"].astype(str).tolist()
    raw_dates[0] = "not a timestamp"
    target["date"] = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    assert pd.isna(target.loc[0, "date"])

    with pytest.raises(ValueError, match="nonmissing"):
        prepare_data(factor, prices, periods=(1,), filter_zscore=None)


@pytest.mark.parametrize("side", ["factor", "price"])
def test_missing_asset_identity_does_not_form_an_unknown_shared_group(side):
    factor, prices = (frame.to_pandas() for frame in _panel())
    target = factor if side == "factor" else prices
    target["asset"] = target["asset"].where(target["asset"] != "A00")
    assert target["asset"].isna().any()

    with pytest.raises(ValueError, match="nonmissing"):
        prepare_data(factor, prices, periods=(1,), filter_zscore=None)


@pytest.mark.parametrize("pandas_input", [False, True])
def test_explicit_row_order_mode_does_not_invent_uniqueness_of_timestamps(pandas_input):
    dates = pd.bdate_range("2025-01-02", periods=3)
    prices = pl.DataFrame(
        {
            "date": [dates[0]] * 2 + [dates[1]] * 2 + [dates[2]] * 2 + [dates[0]] * 2,
            "asset": ["A", "B"] * 4,
            "price": [100.0, 200.0, 105.0, 180.0, 110.0, 220.0, 100.0, 200.0],
        }
    )
    before = prices.clone()
    supplied = prices.to_pandas() if pandas_input else prices

    result = compute_forward_returns(
        supplied,
        periods=1,
        price_col="price",
        group_col="asset",
        date_col=None,
    )
    if isinstance(result, pd.DataFrame):
        result = pl.from_pandas(result)

    assert result["price"].to_list() == prices["price"].to_list()
    expected = [0.05, -0.1, 5 / 105, 40 / 180, -10 / 110, -20 / 220, np.nan, np.nan]
    np.testing.assert_allclose(result["fwd_ret_1"].to_numpy(), expected, atol=1e-14, equal_nan=True)
    assert_frame_equal(prices, before)
    if pandas_input:
        pd.testing.assert_frame_equal(supplied, before.to_pandas())


def test_distinct_assets_at_one_time_are_not_duplicate_observations():
    factor, prices = _panel()

    result = prepare_data(factor, prices, periods=(1,), filter_zscore=None)

    assert result.height == result.select("date", "asset").n_unique() == 24
    first_asset = result.filter(pl.col("asset") == "A00").sort("date")
    np.testing.assert_allclose(first_asset["1D_fwd_return"], [0.05, 5 / 105], atol=1e-14)


def test_zero_horizon_can_share_the_group_and_observation_key_column():
    _, prices = _panel()
    daily_marks = prices.group_by("date").agg(pl.col("price").mean()).sort("date")

    result = compute_forward_returns(
        daily_marks, periods=0, price_col="price", group_col="date", date_col="date"
    )

    np.testing.assert_array_equal(result["fwd_ret_0"], [0.0, 0.0, 0.0])


def test_nanosecond_distinct_price_keys_are_not_collapsed_into_one_observation():
    factor, prices = (frame.to_pandas() for frame in _panel())
    dates = pd.bdate_range("2025-01-02", periods=3)
    base = pd.Timestamp("2025-01-02", tz="UTC")
    precise = [base + pd.Timedelta(value, unit="ns") for value in [0, 100, 200]]
    replacement = dict(zip(dates, precise, strict=True))
    factor["date"] = factor["date"].map(replacement)
    prices["date"] = prices["date"].map(replacement)

    result = prepare_data(factor, prices, periods=(1,), filter_zscore=None)

    assert result.height == result.select("date", "asset").n_unique() == 24
    observed_keys = result["date"].cast(pl.Int64).unique().sort().to_list()
    assert observed_keys == [base.value, base.value + 100]
