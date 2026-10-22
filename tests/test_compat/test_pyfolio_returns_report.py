"""An original returns report must expose every financial panel, not only wealth."""

from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import plotting, tears, timeseries


def _inputs():
    dates = pd.bdate_range("2024-01-02", periods=280, tz="UTC")
    benchmark = pd.Series(np.sin(np.arange(280) / 8) * 0.008 + 0.0001, index=dates)
    returns = benchmark * 0.7 + np.cos(np.arange(280) / 5) * 0.004 + 0.0002
    return returns, benchmark


def test_returns_report_from_price_changes_contains_the_complete_financial_manifest(tmp_path):
    returns, benchmark = _inputs()
    prices = (1 + returns).cumprod()
    prices = pd.concat([pd.Series([1.0], index=[returns.index[0] - pd.Timedelta(days=1)]), prices])
    produced = prices.pct_change(fill_method=None).iloc[1:]
    before = produced.copy(deep=True)
    report = tears.create_returns_tear_sheet(
        produced,
        benchmark_rets=benchmark,
        live_start_date=returns.index[260],
        cone_std=(1, 2),
        return_fig=True,
        random_seed=7,
        num_samples=16,
        periods_per_year=12,
    )
    expected = {
        "Performance statistics",
        "Top drawdowns",
        "Cumulative wealth",
        "Volatility-matched wealth",
        "Log wealth",
        "Observed returns",
        "Rolling beta",
        "Rolling volatility",
        "Rolling Sharpe",
        "Drawdown periods",
        "Underwater",
        "Monthly returns",
        "Annual returns",
        "Monthly distribution",
        "Return quantiles",
    }
    assert set(report.figures) == expected
    assert report.availability["returns"]["status"] == "available"
    assert report.figures["Performance statistics"].layout.meta["periods_per_year"] == 12
    assert report.figures["Rolling volatility"].layout.meta["periods_per_year"] == 12
    assert report.figures["Rolling Sharpe"].layout.meta["periods_per_year"] == 12
    wealth = np.asarray(report.figures["Cumulative wealth"].data[1].y)
    np.testing.assert_allclose(wealth, (1 + produced.iloc[:260]).cumprod(), rtol=1e-12)
    output = tmp_path / "returns.html"
    report.save_html(output, include_plotlyjs=True)
    html = output.read_text(encoding="utf-8")
    assert "Rolling beta" in html and "Return quantiles" in html

    class ScriptSources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "script":
                for key, value in attrs:
                    if key == "src":
                        assert not value.startswith("http"), value

    ScriptSources().feed(html)
    pd.testing.assert_series_equal(produced, before)


def test_first_return_loss_is_measured_from_actual_initial_capital_in_all_report_panels():
    returns = pd.Series([-0.2, 0.25, -0.1], index=pd.date_range("2025-01-01", periods=3))
    report = tears.create_returns_tear_sheet(returns, return_fig=True)
    np.testing.assert_allclose(report.figures["Underwater"].data[0].y, [-0.2, 0, -0.1])
    table = timeseries.gen_drawdown_table(returns, top=2)
    assert table["Net drawdown in %"].tolist() == pytest.approx([20, 10])
    assert pd.isna(table.loc[0, "Peak date"])
    assert table.loc[0, "Valley date"] == returns.index[0]
    assert table.loc[0, "Recovery date"] == returns.index[1]
    assert table.attrs["diagnostic"]["initial_capital_peak_timestamp"] == "unavailable"


@pytest.mark.parametrize("name", ["rolling_volatility", "rolling_sharpe", "rolling_beta"])
def test_actual_reference_rolling_samples_preserve_warmup_nan_windows_and_inclusive_beta(name):
    reference = pytest.importorskip("pyfolio.timeseries")
    returns, benchmark = _inputs()
    returns.iloc[30] = np.nan
    if name == "rolling_beta":
        expected = reference.rolling_beta(returns, benchmark, rolling_window=17)
        actual = timeseries.rolling_beta(returns, benchmark, rolling_window=17)
    else:
        expected = getattr(reference, name)(returns, 17)
        actual = getattr(timeseries, name)(returns, 17)
    pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-12)


def test_actual_reference_drawdown_initial_loss_is_not_a_valid_zero_drawdown():
    reference = pytest.importorskip("pyfolio.timeseries")
    returns = pd.Series([-0.2, 0.25, -0.1], index=pd.date_range("2025-01-01", periods=3))
    original = reference.gen_drawdown_table(returns, top=2)
    assert float(original.loc[0, "Net drawdown in %"]) == pytest.approx(10)
    assert reference.perf_stats(returns)["Max drawdown"] == pytest.approx(-0.2)
    repaired = timeseries.gen_drawdown_table(returns, top=2)
    assert repaired.loc[0, "Net drawdown in %"] == pytest.approx(20)


def test_report_bootstrap_requires_benchmark_before_rendering_or_mutating_inputs():
    returns, _ = _inputs()
    original = returns.copy(deep=True)
    with pytest.raises(ValueError, match="benchmark"):
        tears.create_returns_tear_sheet(returns, bootstrap=True, return_fig=True)
    pd.testing.assert_series_equal(returns, original)


def test_calendar_return_panel_samples_use_shared_products_not_rebalance_cadence():
    returns = pd.Series(
        [0.1, -0.1, 0.05], index=pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02"])
    )
    annual = plotting.plot_annual_returns(returns)
    monthly = plotting.plot_monthly_returns_dist(returns)
    quantiles = plotting.plot_return_quantiles(returns, live_start_date=returns.index[2])
    np.testing.assert_allclose(annual.data[0].x, [5, -1])
    assert list(annual.data[0].y) == ["2025", "2024"]
    np.testing.assert_allclose(monthly.data[0].x, [-1, 5])
    np.testing.assert_allclose(quantiles.data[0].y, [10, -10])
    np.testing.assert_allclose(quantiles.data[3].y, [5])


def test_drawdown_table_preserves_actual_nanosecond_peak_valley_and_recovery_keys():
    dates = pd.date_range("2025-01-01", periods=3, freq="ns", tz="UTC")
    returns = pd.Series([0.1, -0.1, 0.25], index=dates)
    actual = timeseries.gen_drawdown_table(returns, top=1)
    assert actual.loc[0, "Peak date"] == dates[0]
    assert actual.loc[0, "Valley date"] == dates[1]
    assert actual.loc[0, "Recovery date"] == dates[2]


def test_absent_benchmark_does_not_present_an_unmatched_copy_as_volatility_matched():
    returns, _ = _inputs()
    report = tears.create_returns_tear_sheet(returns, return_fig=True)
    assert report.availability["volatility_match"]["status"] == "not_requested"
    assert len(report.figures["Volatility-matched wealth"].data) == 0


def test_one_observation_report_retains_actual_loss_without_inventing_dispersion():
    returns = pd.Series([-0.2], index=pd.date_range("2025-01-01", periods=1))
    benchmark = returns * 0.5
    report = tears.create_returns_tear_sheet(returns, benchmark_rets=benchmark, return_fig=True)
    assert report.availability["volatility_match"]["status"] == "unavailable"
    assert report.figures["Underwater"].data[0].y == pytest.approx([-0.2])


def test_short_price_history_does_not_present_unavailable_rolling_risk_as_an_estimate():
    prices = pd.Series([100.0, 110, 99, 103.95], index=pd.bdate_range("2025-01-02", periods=4))
    returns = prices.pct_change(fill_method=None).iloc[1:]
    report = tears.create_returns_tear_sheet(returns, benchmark_rets=returns * 0.5, return_fig=True)
    for name in ("Rolling volatility", "Rolling Sharpe", "Rolling beta"):
        figure = report.figures[name]
        assert all(not np.isfinite(np.asarray(trace.y, dtype=float)).any() for trace in figure.data)
        assert figure.layout.meta["status"] == "unavailable"
        assert figure.layout.yaxis.visible is False
        assert any("no finite" in annotation.text for annotation in figure.layout.annotations)


def test_observed_flat_prices_keep_known_zero_rolling_volatility_instead_of_an_unavailable_panel():
    prices = pd.Series(100.0, index=pd.bdate_range("2025-01-02", periods=130))
    report = tears.create_returns_tear_sheet(
        prices.pct_change(fill_method=None).iloc[1:], return_fig=True
    )
    figure = report.figures["Rolling volatility"]
    finite = np.asarray(figure.data[0].y, dtype=float)
    np.testing.assert_array_equal(finite[np.isfinite(finite)], [0.0] * 4)
    assert figure.layout.meta["status"] == "available"
    assert figure.layout.meta["finite_estimates"]["Volatility"] == 4
    assert figure.layout.yaxis.visible is not False
    assert not figure.layout.annotations


def test_available_benchmark_risk_is_not_hidden_when_only_strategy_windows_are_incomplete():
    dates = pd.bdate_range("2025-01-02", periods=130)
    short = pd.Series([0.01, -0.02, 0.03], index=dates[:3])
    benchmark = pd.Series(np.sin(np.arange(130)) * 0.01, index=dates)
    figure = plotting.plot_rolling_volatility(short, benchmark)
    assert figure.layout.meta["status"] == "available"
    assert figure.layout.meta["finite_estimates"] == {"Volatility": 0, "Benchmark": 5}
    assert figure.layout.yaxis.visible is not False
    assert not figure.layout.annotations


def test_benchmark_range_clipping_retains_actual_keys_and_discloses_excluded_observations():
    returns, benchmark = _inputs()
    selected = benchmark.iloc[10:-10]
    report = tears.create_returns_tear_sheet(returns, benchmark_rets=selected, return_fig=True)
    assert report.availability["returns"]["benchmark_range_exclusions"] == 20
    assert list(report.figures["Observed returns"].data[0].x) == [
        value.isoformat() for value in selected.index
    ]


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_actual_reference_whole_returns_sheet_executes_each_live_and_bootstrap_manifest(
    live, bootstrap
):
    reference = pytest.importorskip("pyfolio.tears")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    returns, benchmark = _inputs()
    options = {
        "benchmark_rets": benchmark,
        "live_start_date": returns.index[260] if live else None,
        "cone_std": None,
        "bootstrap": bootstrap,
        "return_fig": True,
    }
    try:
        original = reference.create_returns_tear_sheet(returns, set_context=False, **options)
        original.set_dpi(30)
        original.canvas.draw()
        actual = tears.create_returns_tear_sheet(returns, n_samples=16, random_seed=3, **options)
        assert len(original.axes) == 13 + int(bootstrap)
        for title, axis_position, trace_positions in [
            ("Cumulative wealth", 0, [0, 1, 2] if live else [0, 1]),
            ("Volatility-matched wealth", 1, [0, 1, 2] if live else [0, 1]),
            ("Log wealth", 2, [0, 1, 2] if live else [0, 1]),
            ("Observed returns", 3, [0, 1] if live else [0]),
            ("Rolling beta", 4, [0, 1]),
            ("Rolling volatility", 5, [0, 1]),
            ("Rolling Sharpe", 6, [0]),
        ]:
            for position in trace_positions:
                np.testing.assert_allclose(
                    actual.figures[title].data[position].y,
                    original.axes[axis_position].lines[position].get_ydata(),
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=True,
                )
        np.testing.assert_allclose(
            actual.figures["Underwater"].data[0].y,
            original.axes[8].lines[0].get_ydata() / 100,
            rtol=1e-12,
            atol=1e-12,
        )
        assert ("Bootstrap distributions" in actual.figures) is bootstrap
        if bootstrap:
            assert len(actual.figures["Bootstrap distributions"].data) == 14
            assert all(
                len(trace.x) == 16 for trace in actual.figures["Bootstrap distributions"].data
            )
    finally:
        pyplot.close("all")
