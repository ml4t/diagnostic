"""Persistence summaries must not merge distinct sub-microsecond observations."""

import itertools

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic import analyze_prepared_signal, analyze_signal
from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import (
    QuantileMethod,
    compute_autocorrelation,
    compute_turnover,
    prepare_signal_panel,
)

SCENARIOS = [
    ("UTC", "ns", 100, "2025-01-02"),
    ("America/New_York", "ns", 100, "2025-01-02"),
    (None, "ns", 100, "2025-01-02"),
    ("UTC", "ns", 100, "1969-12-31"),
    ("UTC", "us", 1, "2025-01-02"),
    (None, "ms", 1, "2025-01-02"),
    ("UTC", "D", 1, "2025-01-02"),
]


def _inputs(timezone, unit, step, origin, representation):
    start = pd.Timestamp(origin, tz=timezone)
    dates = pd.DatetimeIndex([start + pd.Timedelta(i * step, unit=unit) for i in range(8)])
    profiles = [
        [128.0, 144.0, 128.0, 112.0, 128.0, 144.0, 128.0, 128.0],
        [128.0, 160.0, 128.0, 96.0, 128.0, 160.0, 128.0, 128.0],
        [128.0, 192.0, 128.0, 64.0, 128.0, 192.0, 128.0, 128.0],
        [128.0, 256.0, 128.0, 32.0, 128.0, 256.0, 128.0, 128.0],
    ]
    prices = pd.DataFrame(
        {
            f"{name}{replica}": values
            for name, values in zip("ABCD", profiles, strict=True)
            for replica in range(3)
        },
        index=dates,
    )
    factor = (
        prices.pct_change(fill_method=None)
        .loc[dates[[1, 3, 5]]]
        .stack(future_stack=True)
        .rename("factor")
    )
    factor.index = factor.index.set_names(["date", "asset"])
    marks = (
        prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=[1, 2], group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]]
        .rename(columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"})
    )
    factor, forward = factor.reset_index(), forward.reset_index()
    if representation == "polars":
        factor, forward, marks = (pl.from_pandas(frame) for frame in [factor, forward, marks])
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0)
    assert panel.data.height == 36 and panel.data["date"].n_unique() == 3
    expected = {
        (dates[day].value, f"{asset}{replica}"): 1 if (asset in "AB") == (day != 3) else 2
        for day, asset, replica in itertools.product([1, 3, 5], "ABCD", range(3))
    }
    assert {
        (pd.Timestamp(row["date"]).value, row["asset"]): row["quantile"]
        for row in panel.data.to_pandas().to_dict(orient="records")
    } == expected
    return factor, marks, panel


@pytest.mark.parametrize(
    "scenario,route,representation",
    list(
        itertools.product(
            SCENARIOS,
            ["turnover", "autocorrelation", "raw", "prepared", "panel"],
            ["pandas", "polars"],
        )
    ),
)
def test_exact_quote_observations_retain_reversing_membership_and_ranks(
    scenario, route, representation
):
    factor, prices, panel = _inputs(*scenario, representation)
    before = panel.data
    if route == "turnover":
        for lag, expected in [(1, 1.0), (2, 0.0)]:
            assert compute_turnover(panel.data, n_quantiles=2, lag=lag) == expected
    elif route == "autocorrelation":
        np.testing.assert_allclose(
            compute_autocorrelation(panel.data, lags=[1, 2], min_obs=2),
            [-1.0, 1.0],
            atol=1e-14,
            rtol=0,
        )
    else:
        options = {
            "periods": (1, 2),
            "quantiles": 2,
            "min_assets": 2,
            "compute_turnover_flag": True,
            "autocorrelation_lags": 2,
        }
        if route == "raw":
            result = analyze_signal(
                factor,
                prices,
                price_col="close",
                filter_zscore=None,
                quantile_method=QuantileMethod.QCUT,
                **options,
            )
        elif route == "prepared":
            data = panel.data
            if representation == "pandas":
                data = data.to_pandas().set_index(["date", "asset"])
            result = analyze_prepared_signal(data, **options)
        else:
            result = panel.analyze(min_assets=2, compute_turnover_flag=True, autocorrelation_lags=2)
        assert result.turnover == {"1D": 1.0, "2D": 0.0}
        np.testing.assert_allclose(result.autocorrelation, [-1.0, 1.0], atol=1e-14, rtol=0)
        assert result.n_dates == 3 and result.n_assets == 12
    assert panel.data.equals(before)
