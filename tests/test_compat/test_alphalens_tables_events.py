"""Standalone table/event plots must use supplied samples and named units."""

from importlib.metadata import version
from itertools import product

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.alphalens import performance, plotting, utils


def _inputs():
    dates = pd.date_range("2025-02-03", periods=2, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    factors = pd.Series([1.0, 3.0, 2.0, 6.0], index=index)
    data = utils.get_clean_factor(
        factors,
        pd.DataFrame({"1D": [0.01, 0.03, -0.02, 0.04]}, index=index),
        quantiles=2,
        max_loss=0,
    )
    return data


def test_familiar_tables_preserve_explicit_units_samples_and_supplied_undefined_errors():
    data = _inputs()
    original = data.copy(deep=True)
    quantiles = plotting.plot_quantile_statistics_table(data, return_df=True)
    np.testing.assert_allclose(quantiles.loc[1], [1, 2, 1.5, np.sqrt(0.5), 2, 50])
    np.testing.assert_allclose(quantiles.loc[2], [3, 6, 4.5, np.sqrt(4.5), 2, 50])
    means, errors = performance.mean_return_by_quantile(data, demeaned=False)
    daily, daily_errors = performance.mean_return_by_quantile(data, by_date=True, demeaned=False)
    spread, _ = performance.compute_mean_returns_spread(daily, 2, 1, daily_errors)
    regression = pd.DataFrame({"1D": [0.12, 0.6]}, index=["Ann. alpha", "beta"])
    table = plotting.plot_returns_table(regression, means, spread, return_df=True)
    np.testing.assert_allclose(table["1D"], [0.12, 0.6, 350, -50, 200])
    assert table.index[-1] == "Mean Period Wise Spread (bps, 100% gross)"
    ic = pd.DataFrame({"1D": [0.1, 0.3, np.nan, -0.2], "5D": [np.nan] * 4})
    information = plotting.plot_information_table(ic, return_df=True)
    assert information.loc["1D", "IC Mean"] == pytest.approx(1 / 15)
    assert information.loc["1D", "IC Std."] == pytest.approx(np.sqrt(19 / 300))
    assert information.loc["5D"].isna().all()
    membership = {
        1: pd.concat(
            [performance.quantile_turnover(data.factor_quantile, q) for q in [1, 2]], axis=1
        )
    }
    ranks = {1: performance.factor_rank_autocorrelation(data)}
    turnover, autocorrelation = plotting.plot_turnover_table(ranks, membership, return_df=True)
    np.testing.assert_array_equal(turnover["1D"], [0.0, 0.0])
    assert autocorrelation.loc["Mean Factor Rank Autocorrelation", "1D"] == 1
    assert "<table" in plotting.plot_returns_table(regression, means, spread)
    assert "<table" in plotting.plot_information_table(ic)
    assert "<table" in plotting.plot_quantile_statistics_table(data)
    assert all("<table" in html for html in plotting.plot_turnover_table(ranks, membership))
    pd.testing.assert_frame_equal(data, original)
    assert errors.shape == means.shape


@pytest.mark.parametrize("route", ["returns", "turnover", "information", "quantile_statistics"])
def test_actual_reference_standalone_tables_match_each_named_statistic_and_label(route):
    reference = pytest.importorskip("alphalens.plotting")
    data = _inputs()
    if route == "returns":
        means, _ = performance.mean_return_by_quantile(data, demeaned=False)
        daily, _ = performance.mean_return_by_quantile(data, by_date=True, demeaned=False)
        spread, _ = performance.compute_mean_returns_spread(
            daily, 2, 1, reference_version=version("alphalens-reloaded")
        )
        args = (pd.DataFrame({"1D": [0.12, 0.6]}, index=["Ann. alpha", "beta"]), means, spread)
    elif route == "turnover":
        args = (
            {1: performance.factor_rank_autocorrelation(data)},
            {
                1: pd.concat(
                    [performance.quantile_turnover(data.factor_quantile, q) for q in [1, 2]], axis=1
                )
            },
        )
    elif route == "information":
        args = (pd.DataFrame({"1D": [0.1, 0.3, -0.2], "5D": [0.2, -0.1, 0.1]}),)
    else:
        args = (data,)
    expected = getattr(reference, "plot_" + route + "_table")(*args, return_df=True)
    options = {"reference_version": version("alphalens-reloaded")} if route == "returns" else {}
    actual = getattr(plotting, "plot_" + route + "_table")(*args, return_df=True, **options)
    for left, right in zip(
        actual if isinstance(actual, tuple) else (actual,),
        expected if isinstance(expected, tuple) else (expected,),
        strict=True,
    ):
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("by_quantile,std_bar", list(product([False, True], repeat=2)))
def test_event_chart_keeps_supplied_mean_std_gaps_without_imputing_sample_sizes(
    by_quantile, std_bar
):
    pytest.importorskip("plotly")
    index = pd.MultiIndex.from_product([[1, 3], ["mean", "std"]], names=["factor_quantile", None])
    supplied = pd.DataFrame(
        [[0.01, 0.02, 0.03], [np.nan, 0, 0.01], [-0.02, np.nan, 0.04], [0.02, np.nan, 0.03]],
        index=index,
        columns=[-1, 0, 1],
    )
    before = supplied.copy(deep=True)
    actual = plotting.plot_quantile_average_cumulative_return(
        supplied, by_quantile=by_quantile, std_bar=std_bar
    )
    figures = actual if isinstance(actual, list) else [actual]
    assert len(figures) == (2 if by_quantile else 1)
    means = [trace for fig in figures for trace in fig.data if trace.meta["role"] == "mean"]
    assert len(means) == 2
    np.testing.assert_allclose(means[0].y, [100, 200, 300])
    np.testing.assert_allclose(means[1].y, [-200, np.nan, 400], equal_nan=True)
    for fig in figures:
        assert fig.layout.meta["counts"] == "unavailable"
    if std_bar:
        # Unknown deviations must not become zero caps; known zero remains visible.
        errors = [
            trace for fig in figures for trace in fig.data if trace.meta["role"] == "sample_std"
        ]
        assert len(errors) == 2
        assert list(errors[0].x) == [0, 1]
        np.testing.assert_allclose(errors[0].error_y.array, [0, 100])
    pd.testing.assert_frame_equal(supplied, before)


def test_event_distribution_keeps_nan_counts_singleton_and_submillisecond_bin_labels():
    pytest.importorskip("plotly")
    dates = pd.date_range("2025-02-03", periods=3, freq="ns", tz="UTC")
    events = pd.Series(
        [1.0, np.nan, 3.0],
        index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "asset"]),
    )
    figure = plotting.plot_events_distribution(events, num_bars=2)
    assert list(figure.data[0].y) == [1, 0, 1]
    assert list(figure.data[0].x) == [date.isoformat() for date in dates]
    assert figure.layout.xaxis.type == "category"
    singleton = plotting.plot_events_distribution(events.iloc[:1])
    assert list(singleton.data[0].y) == [1]
    with pytest.raises(ValueError, match="resolution"):
        plotting.plot_events_distribution(events, num_bars=10)


@pytest.mark.parametrize("by_quantile,std_bar", list(product([False, True], repeat=2)))
def test_actual_reference_event_plot_renders_each_supplied_path_and_sample_deviation(
    by_quantile, std_bar
):
    reference = pytest.importorskip("alphalens.plotting")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    index = pd.MultiIndex.from_product([[1, 3], ["mean", "std"]], names=["factor_quantile", None])
    supplied = pd.DataFrame(
        [[0.01, 0.02, 0.03], [0.02, 0, 0.01], [-0.02, -0.01, 0.04], [0.02, 0.01, 0.03]],
        index=index,
        columns=[-1, 0, 1],
    )
    try:
        axes = np.asarray(
            reference.plot_quantile_average_cumulative_return(
                supplied, by_quantile=by_quantile, std_bar=std_bar
            )
        ).reshape(-1)
        actual = plotting.plot_quantile_average_cumulative_return(
            supplied, by_quantile=by_quantile, std_bar=std_bar
        )
        figures = actual if isinstance(actual, list) else [actual]
        for axis in axes:
            axis.figure.canvas.draw()
        means = [trace for fig in figures for trace in fig.data if trace.meta["role"] == "mean"]
        for position, trace in enumerate(means):
            axis = axes[position] if by_quantile else axes[0]
            expected_line = axis.lines[0 if by_quantile else position]
            np.testing.assert_allclose(trace.y, expected_line.get_ydata(), rtol=1e-12, atol=1e-12)
            if std_bar:
                errors = [
                    trace
                    for fig in figures
                    for trace in fig.data
                    if trace.meta["role"] == "sample_std"
                ][position]
                collection = axis.collections[0 if by_quantile else position]
                rendered_deviations = [
                    (segment[1, 1] - segment[0, 1]) / 2 for segment in collection.get_segments()
                ]
                np.testing.assert_allclose(
                    errors.error_y.array, rendered_deviations, rtol=1e-12, atol=1e-12
                )
    finally:
        pyplot.close("all")


def test_actual_reference_event_distribution_keeps_nonmissing_counts_and_empty_elapsed_bins():
    reference = pytest.importorskip("alphalens.plotting")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    events = pd.Series(
        [1.0, np.nan, 3.0],
        index=pd.MultiIndex.from_product(
            [pd.date_range("2025-02-03", periods=3, freq="2D", tz="UTC"), ["A"]],
            names=["date", "asset"],
        ),
    )
    try:
        axis = reference.plot_events_distribution(events, num_bars=4)
        axis.figure.canvas.draw()
        actual = plotting.plot_events_distribution(events, num_bars=4)
        np.testing.assert_array_equal(
            actual.data[0].y, [rectangle.get_height() for rectangle in axis.patches]
        )
        assert list(actual.data[0].y) == [1, 0, 0, 0, 1]
    finally:
        pyplot.close("all")


def test_unavailable_bucket_scores_and_empty_ic_observations_remain_unknown_not_zero():
    prices = pd.Series([10.0, 11.0, 13.0])
    data = pd.DataFrame(
        {"factor": prices.pct_change(fill_method=None), "factor_quantile": [3, 1, 1]}
    )
    table = plotting.plot_quantile_statistics_table(data, return_df=True)
    assert table.loc[3, "count"] == 0 and table.loc[3, "count %"] == 0
    assert table.loc[3, ["min", "max", "mean", "std"]].isna().all()
    empty = plotting.plot_information_table(
        pd.DataFrame(columns=["1D", "5D"], dtype=float), return_df=True
    )
    assert empty.shape == (2, 7) and empty.isna().all().all()
