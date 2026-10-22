"""Overview chart adapters must retain supplied samples, units and unknown values."""

from fractions import Fraction
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest


def _panel():
    from ml4t.diagnostic.compat import alphalens as al

    dates = pd.bdate_range("2025-01-27", periods=6, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["date", "asset"])
    factors = pd.Series(np.tile([1.0, 2.0, 4.0, 3.0], len(dates)), index=index)
    returns = np.array(
        [
            [-0.02, 0.01, 0.03, 0.04],
            [0.01, -0.01, 0.02, 0.03],
            [0.0, 0.04, -0.02, 0.01],
            [0.03, 0.02, 0.01, -0.01],
            [-0.01, 0.01, 0.03, 0.02],
            [0.02, -0.03, 0.01, 0.04],
        ]
    )
    data = al.utils.get_clean_factor(
        factors,
        pd.DataFrame({"1D": returns.ravel(), "2D": 2 * returns.ravel()}, index=index),
        groupby={"A": "red", "B": "red", "C": "blue", "D": "blue"},
        quantiles=2,
        max_loss=0,
    )
    return data


def test_cleaned_public_statistics_flow_to_every_original_overview_chart_without_invented_counts():
    pytest.importorskip("plotly")
    from ml4t.diagnostic.compat import alphalens as al

    data = _panel()
    before = data.copy(deep=True)
    means, errors = al.performance.mean_return_by_quantile(data, by_date=True, demeaned=False)
    reduced, _ = al.performance.mean_return_by_quantile(data, demeaned=False)
    spread, spread_error = al.performance.compute_mean_returns_spread(means, 2, 1, errors)
    chart = al.plotting.plot_quantile_returns_bar(reduced)
    np.testing.assert_allclose(
        chart.data[0].y, [10000 * float(Fraction(7, 1200)), 10000 * float(Fraction(21, 1200))]
    )
    assert chart.data[0].error_y.array is None
    assert chart.layout.meta["counts"] == "unavailable"
    violin = al.plotting.plot_quantile_returns_violin(means)
    assert len(violin.data) == 2 and len(violin.data[0].y) == 12
    curves = al.plotting.plot_mean_quantile_returns_spread_time_series(
        spread, spread_error, bandwidth=0.5
    )
    assert len(curves) == 2 and curves[0].data[0].y[0] == pytest.approx(200)
    assert curves[0].data[3].y[0] > curves[0].data[0].y[0]
    factor_returns = al.performance.factor_returns(data)
    cumulative = al.plotting.plot_cumulative_returns(factor_returns["1D"], "1D")
    assert cumulative.data[0].y[0] == pytest.approx(1.0225)
    quantile_paths = al.plotting.plot_cumulative_returns_by_quantile(means["1D"], "1D")
    assert [trace.name for trace in quantile_paths.data] == ["2", "1"]
    assert quantile_paths.data[0].y[0] == pytest.approx(1.035)
    ic = al.performance.factor_information_coefficient(data)
    for route in (al.plotting.plot_ic_ts, al.plotting.plot_ic_hist, al.plotting.plot_ic_qq):
        assert len(route(ic)) == 2
    monthly = al.performance.mean_information_coefficient(data, by_time="M")
    assert len(al.plotting.plot_monthly_ic_heatmap(monthly)) == 2
    groups = al.performance.mean_information_coefficient(data, by_group=True)
    assert len(al.plotting.plot_ic_by_group(groups).data) == 2
    group_means, _ = al.performance.mean_return_by_quantile(data, by_group=True)
    assert len(al.plotting.plot_quantile_returns_bar(group_means, by_group=True)) == 2
    membership = pd.concat(
        [al.performance.quantile_turnover(data["factor_quantile"], q) for q in (1, 2)], axis=1
    )
    assert len(al.plotting.plot_top_bottom_quantile_turnover(membership).data) == 2
    rank = al.performance.factor_rank_autocorrelation(data)
    assert len(al.plotting.plot_factor_rank_auto_correlation(rank).data) == 1
    pd.testing.assert_frame_equal(data, before)


def test_plotting_preserves_supplied_missingness_and_rejects_artist_requests_before_outputs():
    pytest.importorskip("plotly")
    from ml4t.diagnostic.compat import alphalens as al

    data = _panel()
    ic = al.performance.factor_information_coefficient(data)
    ic.iloc[1, 0] = np.nan
    before = ic.copy(deep=True)
    figure = al.plotting.plot_ic_ts(ic)[0]
    assert pd.isna(figure.data[0].y[1])
    histogram = al.plotting.plot_ic_hist(ic)[0]
    assert len(histogram.data[0].x) == 5
    with pytest.raises(TypeError, match="Matplotlib ax"):
        al.plotting.plot_ic_ts(ic, ax=object())
    pd.testing.assert_frame_equal(ic, before)


@pytest.mark.parametrize("by_group", [False, True])
@pytest.mark.parametrize("limits", [None, (5, 95)])
def test_actual_reference_quantile_bars_preserve_supplied_group_means_and_bps(by_group, limits):
    reference = pytest.importorskip("alphalens.plotting")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    if version("alphalens-reloaded") not in {"0.5.0", "0.4.6"}:
        pytest.skip("Corrected and frozen references are qualified independently")
    from ml4t.diagnostic.compat import alphalens as al

    data = _panel()
    means, _ = al.performance.mean_return_by_quantile(data, by_group=by_group)
    try:
        axes = np.asarray(
            reference.plot_quantile_returns_bar(means, by_group=by_group, ylim_percentiles=limits)
        ).reshape(-1)
        actual = al.plotting.plot_quantile_returns_bar(
            means, by_group=by_group, ylim_percentiles=limits
        )
        figures = actual if isinstance(actual, list) else [actual]
        samples = (
            [sample.droplevel("group") for _, sample in means.groupby(level="group", observed=True)]
            if by_group
            else [means]
        )
        for axis, figure, sample in zip(axes, figures, samples, strict=True):
            expected = [patch.get_height() for patch in axis.patches]
            supplied = np.concatenate([sample[column].to_numpy() for column in sample])
            actual_values = np.array([value for trace in figure.data for value in trace.y])
            unavailable = np.isnan(supplied)
            np.testing.assert_array_equal(np.isnan(actual_values), unavailable)
            np.testing.assert_allclose(
                actual_values[~unavailable], np.array(expected)[~unavailable], atol=1e-12
            )
            # Pandas/Matplotlib paints an absent group/bucket mean as a zero bar.
            # Its original supplied cell and the Diagnostic chart remain unknown.
            np.testing.assert_array_equal(np.array(expected)[unavailable], 0)
            if limits is not None:
                np.testing.assert_allclose(figure.layout.yaxis.range, axis.get_ylim(), atol=1e-12)
        assert len(figures) == (2 if by_group else 1)
    finally:
        pyplot.close("all")
