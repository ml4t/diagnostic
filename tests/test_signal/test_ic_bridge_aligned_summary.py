"""Report statistics must use the quote-derived IC dates their bridge retains."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp

from ml4t.diagnostic import analyze_prepared_signal, analyze_signal
from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import SignalResult, prepare_signal_panel
from ml4t.diagnostic.visualization.signal import plot_ic_ts


def _inputs(profile, periods=(1, 2)):
    count = 6 if profile == "empty" else 7 if profile == "missing_tail" else 8
    dates = pd.bdate_range("2025-01-02", periods=count, tz="UTC")
    prices = {}
    for asset, past, rising, falling in zip(
        ["A", "B", "C", "D"],
        [1.125, 1.25, 1.5, 2.0],
        [0.125, 0.25, 0.5, 1.0],
        [1.0, 0.5, 0.25, 0.125],
        strict=True,
    ):
        if profile == "empty":
            gross = {"A": 2.0, "B": 4.0, "C": 8.0, "D": 16.0}[asset]
            multipliers = [past, gross, 16 / gross, 1.25, past]
        else:
            middle = 1 + rising if profile == "steady" else 1.25
            multipliers = [past, 1 + rising, past, middle, past, 1 + falling, past]
        quotes = [128.0]
        for multiplier in multipliers[: count - 1]:
            quotes.append(quotes[-1] * multiplier)
        prices[asset] = quotes
    prices = pd.DataFrame(prices, index=dates)
    scored = [1, 3] if profile == "empty" else [1, 3, 5]
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[scored]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    marks = (
        prices.stack(future_stack=True).rename("price").rename_axis(["date", "asset"]).reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=[1, 2], group_col="asset", price_col="price")
        .set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]]
        .rename(columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"})
    )
    panel = prepare_signal_panel(
        factor, forward[[f"{period}D" for period in periods]], quantiles=2, max_loss=0.5
    )
    return factor.reset_index(), marks, panel, dates


def _analysis(profile, route, periods=(1, 2)):
    factor, marks, panel, dates = _inputs(profile, periods)
    if route == "raw":
        result = analyze_signal(
            factor,
            marks,
            periods=periods,
            quantiles=2,
            min_assets=2,
            quantile_method="qcut",
            filter_zscore=None,
            compute_turnover_flag=False,
        )
    elif route == "prepared":
        result = analyze_prepared_signal(panel.data, periods=periods, quantiles=2, min_assets=2)
    else:
        result = panel.analyze(min_assets=2)
    return result, dates


def _assert_statistics(model, period, values):
    mean = float(np.mean(values)) if values else float("nan")
    std = float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
    positive = sum(value > 0 for value in values) / len(values) * 100 if values else float("nan")
    t_stat, p_value = (
        ttest_1samp(values, 0) if len(values) >= 2 and std > 0 else (math.nan, math.nan)
    )
    ir = mean / std if std > 0 else float("nan")
    for actual, expected in [
        (model.ic_mean[period], mean),
        (model.ic_std[period], std),
        (model.ic_positive_pct[period], positive),
        (model.ic_ir[period], ir),
        (model.ic_t_stat[period], t_stat),
        (model.ic_p_value[period], p_value),
    ]:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14, equal_nan=True)


@pytest.mark.parametrize("profile", ["full", "steady", "empty"])
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
@pytest.mark.parametrize("periods", [(1, 2), (2, 1)])
def test_aligned_quote_series_and_report_statistics_describe_the_same_sample(
    profile, route, periods
):
    result, dates = _analysis(profile, route, periods)
    before = result.to_json()
    model = result.to_ic_result()
    indices = [] if profile == "empty" else [1, 3, 5] if profile == "steady" else [1, 5]
    assert [pd.Timestamp(day).value for day in model.dates] == [
        dates[index].value for index in indices
    ]
    expected = {
        "1D": []
        if profile == "empty"
        else [1.0, 1.0, -1.0]
        if profile == "steady"
        else [1.0, -1.0],
        "2D": [] if profile == "empty" else [1.0, 1.0, 0.0] if profile == "steady" else [1.0, 0.0],
    }
    assert list(model.ic_by_date) == [f"{period}D" for period in periods]
    for period, values in expected.items():
        np.testing.assert_allclose(model.ic_by_date[period], values, rtol=0, atol=1e-14)
        _assert_statistics(model, period, values)
        if values:
            figure = plot_ic_ts(model, period=period, rolling_window=2)
            np.testing.assert_allclose(figure.data[0].y, values, rtol=0, atol=1e-14)
            assert any(
                f"Mean IC: {np.mean(values):.4f}" == annotation.text
                for annotation in figure.layout.annotations
            )
    assert result.to_json() == before


@pytest.mark.parametrize("profile", ["full", "steady", "empty"])
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
@pytest.mark.parametrize("period", [1, 2])
def test_explicit_period_keeps_its_original_quote_sample_and_statistics(profile, route, period):
    result, _ = _analysis(profile, route)
    model = result.to_ic_result(period=period)
    key = f"{period}D"
    assert model.dates == result.ic_dates[key]
    np.testing.assert_allclose(model.ic_by_date[key], result.ic_series[key], rtol=0, atol=1e-14)
    _assert_statistics(model, key, result.ic_series[key])


@pytest.mark.parametrize("profile", ["full", "empty"])
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_actual_result_json_and_tearsheet_preserve_aligned_summary_consistency(
    profile, route, tmp_path
):
    result, _ = _analysis(profile, route)
    path = tmp_path / "signal.json"
    result.to_json(str(path))
    restored = SignalResult.from_json(str(path))
    model = restored.to_tear_sheet().ic_analysis
    assert model is not None
    decoded = type(model).model_validate_json(model.model_dump_json())
    expected = {"1D": [1.0, -1.0], "2D": [1.0, 0.0]} if profile == "full" else {"1D": [], "2D": []}
    for period, values in expected.items():
        _assert_statistics(decoded, period, values)
    assert restored.to_json() == result.to_json()


def test_missing_long_horizon_quote_tail_does_not_leave_discarded_short_horizon_mean():
    result, dates = _analysis("missing_tail", "raw")
    np.testing.assert_allclose(result.ic_series["1D"], [1.0, -1.0], rtol=0, atol=1e-14)
    assert result.ic["1D"] == 0.0
    model = result.to_ic_result()
    assert [pd.Timestamp(day).value for day in model.dates] == [dates[1].value]
    assert model.ic_by_date == {"1D": [1.0], "2D": [1.0]}
    for period in ["1D", "2D"]:
        _assert_statistics(model, period, [1.0])
