"""Return-report panels must retain raw observations, splits and training-only forecasts."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import plotting, timeseries


def _inputs():
    dates = pd.date_range("2025-02-03", periods=4, tz="UTC")
    prices = pd.Series([100.0, 110.0, 99.0, 103.95], index=dates)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    benchmark = pd.Series([0.03, -0.02, 0.01], index=returns.index, name="Market")
    return returns, benchmark


def test_price_produced_return_panel_keeps_original_wealth_and_raw_missingness():
    pytest.importorskip("plotly")
    returns, benchmark = _inputs()
    before = returns.copy(deep=True)
    figure = plotting.plot_rolling_returns(returns, factor_returns=benchmark, cone_std=None)
    np.testing.assert_allclose(figure.data[1].y, [1.1, 0.99, 1.0395])
    pd.testing.assert_series_equal(returns, before)
    returns.iloc[1] = np.nan
    missing_before = returns.copy(deep=True)
    gaps = plotting.plot_returns(returns)
    assert pd.isna(gaps.data[0].y[1])
    flat_wealth = plotting.plot_rolling_returns(returns, cone_std=None)
    np.testing.assert_allclose(flat_wealth.data[0].y, [1.1, 1.1, 1.155])
    pd.testing.assert_series_equal(returns, missing_before)


def test_live_forecast_only_receives_pre_live_rows_and_uses_that_final_wealth():
    pytest.importorskip("plotly")
    returns, _ = _inputs()
    received = []

    def forecast(sample, horizon, cone_std, starting_value):
        received.append((sample.copy(deep=True), starting_value))
        return timeseries.forecast_cone_bootstrap(
            sample,
            horizon,
            cone_std=cone_std,
            starting_value=starting_value,
            num_samples=16,
            random_seed=8,
        )

    original = returns.copy(deep=True)
    first = plotting.plot_rolling_returns(
        returns, live_start_date=returns.index[2], cone_function=forecast, cone_std=(1, 2)
    )
    changed = returns.copy()
    changed.iloc[-1] = 5.0
    second = plotting.plot_rolling_returns(
        changed, live_start_date=returns.index[2], cone_function=forecast, cone_std=(1, 2)
    )
    assert all(
        sample.index.equals(returns.index[:2]) and start == pytest.approx(0.99)
        for sample, start in received
    )
    assert first.layout.meta["cone_bounds"] == second.layout.meta["cone_bounds"]
    assert first.data[1].y[-1] != second.data[1].y[-1]
    pd.testing.assert_series_equal(returns, original)


@pytest.mark.parametrize("live,logy,matched", list(product([False, True], repeat=3)))
def test_actual_reference_complete_wealth_panel_preserves_benchmark_split_scale_and_values(
    live, logy, matched
):
    reference = pytest.importorskip("pyfolio.plotting")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    returns, benchmark = _inputs()
    options = {
        "factor_returns": benchmark,
        "live_start_date": returns.index[2] if live else None,
        "cone_std": None,
        "logy": logy,
        "volatility_match": matched,
    }
    try:
        _, axis = pyplot.subplots()
        reference.plot_rolling_returns(returns, ax=axis, **options)
        axis.figure.canvas.draw()
        actual = plotting.plot_rolling_returns(returns, **options)
        assert len(actual.data) == len(axis.lines) - 1
        for trace, line in zip(actual.data, axis.lines[:-1], strict=True):
            np.testing.assert_allclose(
                trace.y, line.get_ydata(), rtol=1e-12, atol=1e-12, equal_nan=True
            )
        assert actual.layout.yaxis.type == ("log" if logy else "linear")
    finally:
        pyplot.close("all")


@pytest.mark.parametrize("live", [False, True])
def test_actual_reference_performance_display_preserves_percent_rows_and_split_tables(live):
    reference = pytest.importorskip("pyfolio.plotting")
    returns, benchmark = _inputs()
    options = {
        "factor_returns": benchmark,
        "live_start_date": returns.index[2] if live else None,
        "return_df": True,
    }
    expected = reference.show_perf_stats(returns, **options)
    actual = plotting.show_perf_stats(returns, **options)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    assert "<table" in plotting.show_perf_stats(returns, factor_returns=benchmark)


def test_invalid_volatility_matching_and_callback_cannot_mutate_the_callers_returns():
    pytest.importorskip("plotly")
    returns, _ = _inputs()
    before = returns.copy(deep=True)
    with pytest.raises(ValueError, match="factor_returns"):
        plotting.plot_rolling_returns(returns, volatility_match=True)

    def invalid(sample, horizon, cone_std, starting_value):
        sample.iloc[0] = 99
        return pd.DataFrame()

    with pytest.raises(ValueError, match="cone"):
        plotting.plot_rolling_returns(
            returns, live_start_date=returns.index[2], cone_function=invalid, cone_std=1
        )
    pd.testing.assert_series_equal(returns, before)


def test_default_live_panel_does_not_invent_a_forecast_and_named_legend_is_placed():
    returns, _ = _inputs()
    actual = plotting.plot_rolling_returns(
        returns, live_start_date=returns.index[-1], legend_loc="lower left"
    )
    assert actual.layout.meta["cone_bounds"] is None
    assert len(actual.data) == 2
    assert actual.layout.legend.x == 0
    assert actual.layout.legend.y == 0
    with pytest.raises(TypeError, match="unknown_option"):
        plotting.plot_rolling_returns(returns, unknown_option=True)


def test_actual_reference_bootstrap_display_drops_real_statistics_not_copied_by_adapter():
    reference = pytest.importorskip("pyfolio.plotting")
    dates = pd.date_range("2025-01-01", periods=12)
    returns = pd.Series(np.full(12, 0.01), index=dates)
    original = returns.copy(deep=True)
    expected = reference.show_perf_stats(returns, bootstrap=True, return_df=True)
    assert expected.loc["Annual return", "Backtest"] == "nan%"
    actual = plotting.show_perf_stats(
        returns, bootstrap=True, return_df=True, n_samples=16, random_seed=3
    )
    assert actual.columns.tolist() == ["mean", "median", "5%", "95%"]
    independent = str(np.round(((1.01**252) - 1) * 100, 3)) + "%"
    assert actual.loc["Annual return"].tolist() == [independent] * 4
    pd.testing.assert_series_equal(returns, original)


def test_live_bootstrap_display_keeps_each_split_and_summary_column():
    returns, benchmark = _inputs()
    actual = plotting.show_perf_stats(
        returns,
        factor_returns=benchmark,
        live_start_date=returns.index[-1],
        bootstrap=True,
        return_df=True,
        n_samples=8,
        random_seed=3,
    )
    assert actual.columns.tolist() == list(
        product(["In-sample", "Out-of-sample", "All"], ["mean", "median", "5%", "95%"])
    )
    assert actual.loc["Cumulative returns", ("All", "mean")].endswith("%")
