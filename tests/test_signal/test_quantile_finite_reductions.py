"""Finite quote/PnL outcomes must not overflow into unavailable sample counts."""

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic import analyze_prepared_signal, analyze_signal
from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import compute_quantile_returns, prepare_signal_panel


def _inputs(profile):
    dates = pd.bdate_range("2025-01-02", periods=4, tz="UTC")
    assets = (
        ["A", "B", "C", "D"]
        if profile == "near_constant"
        else ["A"]
        if profile == "singleton"
        else ["A", "B"]
    )
    initial = (
        0.5e-308 if profile in {"constant_large", "signed_large", "unrepresentable_std"} else 0.5
    )
    middle = initial * 2
    if profile == "constant_large":
        closing, expected = [1.0, 1.0], (1e308, 0.0, 0.0)
    elif profile == "signed_large":
        closing, expected = [1.0, -1.0], (0.0, math.sqrt(2) * 1e308, 1e308)
    elif profile == "unrepresentable_std":
        closing, expected = [1.7, -1.7], (0.0, math.inf, 1.7e308)
    elif profile == "near_constant":
        closing = [2.0, 2.0, 2.0, np.nextafter(2.0, math.inf)]
        expected = (1.0, np.spacing(2.0) / 2, np.spacing(2.0) / 4)
    elif profile == "tiny_nav":
        closing, expected = [1.25, 0.75], (0.0, math.sqrt(2) * 1e-308, 1e-308)
    elif profile == "zero":
        closing, expected = [1.0, 1.0], (0.0, 0.0, 0.0)
    elif profile == "singleton":
        closing, expected = [1.25], (0.25, math.nan, math.nan)
    else:
        closing, expected = [1.25, 0.75], (0.0, math.sqrt(2) * 0.25, 0.25)
    prices = pd.DataFrame(
        {
            asset: [initial, middle, close, close]
            for asset, close in zip(assets, closing, strict=True)
        },
        index=dates,
    )
    factor = (
        prices.pct_change(fill_method=None)
        .loc[[dates[1]]]
        .stack(future_stack=True)
        .rename("factor")
    )
    factor.index = factor.index.set_names(["date", "asset"])
    marks = (
        prices.stack(future_stack=True).rename("price").rename_axis(["date", "asset"]).reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=1, price_col="price", group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1"]]
        .rename(columns={"fwd_ret_1": "1D"})
    )
    if profile == "tiny_nav":
        pnl = pd.Series([1.0, -1.0], index=factor.index)
        forward.loc[factor.index, "1D"] = pnl / 1e308
    return factor, marks, forward, expected


CASES = [
    (profile, route, representation)
    for profile in [
        "constant_large",
        "signed_large",
        "unrepresentable_std",
        "near_constant",
        "tiny_nav",
        "ordinary",
        "zero",
        "singleton",
    ]
    for route in ["raw", "prepared", "owned_daily", "owned_equal_date"]
    for representation in ["pandas", "polars"]
    if not (profile == "tiny_nav" and route == "raw")
]


@pytest.mark.parametrize("profile,route,representation", CASES)
def test_finite_sample_reduction_preserves_mathematical_mean_uncertainty_and_counts(
    profile, route, representation
):
    factor, prices, forward, expected = _inputs(profile)
    input_count = len(factor)
    panel = prepare_signal_panel(factor, forward, quantiles=1, max_loss=0)
    assert panel.metadata["retained_rows"] == input_count
    if route.startswith("owned"):
        # Both representations reach the public input boundary, not just a stored frame.
        f, r = factor.reset_index(), forward.reset_index()
        if representation == "polars":
            f, r = pl.from_pandas(f), pl.from_pandas(r)
        panel = prepare_signal_panel(f, r, quantiles=1, max_loss=0)
        result = panel.quantile_statistics(by_date=route == "owned_daily", demeaned=False)
        row = result.row(0, named=True)
        assert row["n_obs"] == (input_count if route == "owned_daily" else 1)
        np.testing.assert_allclose(row["mean"], expected[0], atol=0, rtol=1e-14)
        if route == "owned_equal_date":
            assert row["std"] is None and row["standard_error"] is None
        elif input_count == 1:
            assert row["std"] is None and row["standard_error"] is None
        else:
            np.testing.assert_allclose(
                [row["std"], row["standard_error"]], expected[1:], atol=0, rtol=1e-14
            )
    else:
        if route == "raw":
            f = factor.reset_index()
            if representation == "polars":
                f, prices = pl.from_pandas(f), pl.from_pandas(prices)
            result = analyze_signal(
                f,
                prices,
                periods=(1,),
                quantiles=1,
                min_assets=2,
                filter_zscore=None,
                compute_turnover_flag=False,
            )
        else:
            data = panel.data
            if representation == "pandas":
                data = data.to_pandas().set_index(["date", "asset"])
            result = analyze_prepared_signal(data, periods=(1,), quantiles=1, min_assets=2)
        assert result.n_assets == input_count and result.n_dates == 1
        assert result.count_by_quantile[1] == input_count
        np.testing.assert_allclose(
            result.quantile_returns["1D"][1], expected[0], atol=0, rtol=1e-14
        )
        np.testing.assert_allclose(
            result.quantile_returns_std["1D"][1], expected[1], atol=0, rtol=1e-14, equal_nan=True
        )


def test_direct_pooled_helper_preserves_nonfinite_quote_producer_policy():
    dates = pd.bdate_range("2025-01-02", periods=3)
    prices = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "asset": ["A"] * 3 + ["B"] * 3,
            "close": [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    produced = compute_forward_returns(prices, periods=1, group_col="asset")
    selected = pl.from_pandas(produced[produced["date"] == dates[1]]).with_columns(
        pl.lit(1).alias("quantile"), pl.col("fwd_ret_1").alias("1D_fwd_return")
    )
    assert math.isinf(compute_quantile_returns(selected, period=1, n_quantiles=1)[1])
