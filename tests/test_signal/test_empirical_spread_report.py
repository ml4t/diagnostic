"""Report spreads must use actual dated means/errors, not cumulative increments."""

import json
import math
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import analyze_prepared_signal, analyze_signal, prepare_signal_panel
from ml4t.diagnostic.visualization.signal import SignalDashboard, plot_spread_timeseries


def _inputs(nanoseconds=False, one_bucket=False):
    dates = pd.bdate_range("2025-01-02", periods=7, tz="UTC")
    if nanoseconds:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, unit="ns") for i in range(7)])
    prices = pd.DataFrame(
        {
            "A": [128.0, 144.0, 180.0, 202.5, 253.125, 284.765625, 355.95703125],
            "B": [128.0, 160.0, 140.0, 175.0, 153.125, 191.40625, 167.48046875],
            "C": [128.0, 192.0, 288.0, 432.0, 648.0, 972.0, 1458.0],
            "D": [128.0, 256.0, 288.0, 576.0, 648.0, 1296.0, 1458.0],
        },
        index=dates,
    )
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[[1, 3]]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
        .drop((dates[3], "C"))
    )
    marks = (
        prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=[1, 2], group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]]
        .rename(columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"})
    )
    panel = prepare_signal_panel(
        factor,
        forward,
        groups={"A": "growth", "C": "growth", "B": "value", "D": "value"},
        quantiles=1 if one_bucket else 2,
        max_loss=0,
    )
    return factor.reset_index(), marks, panel, dates


def _analysis(route, factor, marks, panel):
    if route == "owned":
        return panel.analyze(min_assets=2)
    if route == "prepared":
        return analyze_prepared_signal(
            panel.data, periods=(1, 2), quantiles=panel.data["quantile"].max(), min_assets=2
        )
    return analyze_signal(
        factor,
        marks,
        periods=(1, 2),
        quantiles=2,
        price_col="close",
        filter_zscore=None,
        min_assets=2,
        compute_turnover_flag=False,
    )


def _long_inputs():
    dates = pd.bdate_range("2025-01-02", periods=30, tz="UTC")
    returns = pd.DataFrame(
        {
            "A": [0.01 + 0.001 * (i % 3) for i in range(30)],
            "B": [-0.015 + 0.0005 * (i % 5) for i in range(30)],
            "C": [0.02 + 0.001 * (i % 4) for i in range(30)],
            "D": [-0.005 + 0.0002 * (i % 7) for i in range(30)],
        },
        index=dates,
    )
    prices = 100 * (1 + returns).cumprod()
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[1:27]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    marks = (
        prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=[1, 2], group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]]
        .rename(columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"})
    )
    panel = prepare_signal_panel(
        factor,
        forward,
        groups={"A": "growth", "C": "growth", "B": "value", "D": "value"},
        quantiles=2,
        max_loss=0,
    )
    assert panel.data.height == 104
    return factor.reset_index(), marks, panel, dates


def _trace(figure, name):
    return next(trace for trace in figure.data if trace.name == name)


@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
@pytest.mark.parametrize("nanoseconds", [False, True])
@pytest.mark.parametrize("period", ["1D", "2D"])
def test_quote_to_spread_plot_uses_actual_pair_means_and_counts_without_cumulative_paths(
    route, nanoseconds, period
):
    factor, marks, panel, dates = _inputs(nanoseconds)
    result = _analysis(route, factor, marks, panel)
    model = result.to_quantile_result()
    assert model.cumulative_returns is None
    figure = plot_spread_timeseries(model, period=period, rolling_window=2)
    spread = _trace(figure, "Observed Spread")
    expected = [0.25, 0.0625] if period == "1D" else [1.0, 1.0]
    np.testing.assert_allclose(spread.y, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        _trace(figure, "2-Observation Rolling Mean").y,
        [np.nan, sum(expected) / 2],
        rtol=0,
        atol=1e-14,
        equal_nan=True,
    )
    assert [row[1] for row in spread.customdata] == [2, 1]
    assert [row[2] for row in spread.customdata] == [2, 2]
    assert len({row[0] for row in spread.customdata}) == 2
    if nanoseconds:
        assert spread.customdata[0][0].endswith("000000100+00:00")
        assert figure.layout.xaxis.type == "category"
    assert figure.layout.meta["return_unit"] == "decimal"
    assert figure.layout.meta["exposure_convention"] == "long_upper_short_lower_unit_each"
    assert "95%" not in json.dumps(figure.to_plotly_json(), default=str)


@pytest.mark.parametrize("bandwidth", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("period", ["1D", "2D"])
def test_propagated_sample_error_band_is_not_a_rolling_deviation_confidence_claim(
    bandwidth, period
):
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    figure = plot_spread_timeseries(model, period=period, rolling_window=2, bandwidth=bandwidth)
    band = _trace(figure, "Propagated SE band")
    spread = 0.25 if period == "1D" else 1.0
    error = math.sqrt(2) * 0.1875 if period == "1D" else 0.15625
    np.testing.assert_allclose(
        band.y,
        [spread + bandwidth * error, np.nan, np.nan, spread - bandwidth * error],
        rtol=0,
        atol=1e-14,
        equal_nan=True,
    )
    assert any("unavailable" in annotation.text.lower() for annotation in figure.layout.annotations)
    assert figure.layout.meta["error_method"] == "independent_propagation"


@pytest.mark.parametrize(
    "demeaned,group_adjust", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("representation", ["pandas", "polars"])
def test_explicit_shared_adjustment_changes_actual_sample_not_pooled_result_summary(
    demeaned, group_adjust, representation
):
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    observations = panel.quantile_statistics(
        by_date=True, demeaned=demeaned, group_adjust=group_adjust
    )
    before = observations.clone()
    data = (
        observations.to_pandas().set_index(["date", "quantile"])
        if representation == "pandas"
        else observations
    )
    figure = plot_spread_timeseries(model, period="1D", observations=data, rolling_window=2)
    expected = [0.25, 0.1875] if group_adjust else [0.25, 0.0625]
    np.testing.assert_allclose(_trace(figure, "Observed Spread").y, expected, rtol=0, atol=1e-14)
    assert observations.equals(before)


def test_grouped_observations_require_selection_instead_of_cross_group_pooling():
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    observations = panel.quantile_statistics(by_date=True, by_group=True, demeaned=False)
    with pytest.raises(ValueError, match="select one group"):
        plot_spread_timeseries(model, observations=observations)
    selected = observations.filter(pl.col("group") == "growth")
    figure = plot_spread_timeseries(
        model, observations=selected, period="1D", show_confidence=False
    )
    np.testing.assert_allclose(
        _trace(figure, "Observed Spread").y, [0.25, np.nan], rtol=0, atol=1e-14, equal_nan=True
    )


def test_summary_only_historical_result_cannot_invent_actual_dates_or_errors():
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    historical = type(model).model_validate(
        {
            key: value
            for key, value in model.model_dump().items()
            if key != "observations_ipc_base64"
        }
    )
    with pytest.raises(ValueError, match="empirical|observations"):
        plot_spread_timeseries(historical)


def test_single_bucket_report_does_not_invent_an_automatic_top_bottom_spread():
    factor, marks, panel, _ = _inputs(one_bucket=True)
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    with pytest.raises(ValueError, match="two|2"):
        plot_spread_timeseries(model)


@pytest.mark.parametrize("bandwidth", [np.nan, np.inf, -1, True, "1", None])
def test_invalid_error_multiplier_is_rejected_instead_of_publishing_misleading_bands(bandwidth):
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    with pytest.raises(ValueError, match="bandwidth"):
        plot_spread_timeseries(model, bandwidth=bandwidth)


@pytest.mark.parametrize("window", [0, -1, 1.5, True, "2"])
def test_rolling_smoothing_requires_a_positive_observation_count(window):
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    with pytest.raises(ValueError, match="rolling_window"):
        plot_spread_timeseries(model, rolling_window=window)


@pytest.mark.parametrize("show", [1, None])
def test_error_band_switch_requires_boolean_not_truthy_coercion(show):
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    with pytest.raises(ValueError, match="show_confidence"):
        plot_spread_timeseries(model, show_confidence=show)


def test_unknown_result_period_is_not_silently_substituted_with_the_first_horizon():
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    with pytest.raises(ValueError, match="Period"):
        plot_spread_timeseries(model, period="7D")


def test_selected_statistics_must_actually_include_the_requested_return_column():
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    selected = panel.quantile_statistics(by_date=True).filter(pl.col("return_column") == "1D")
    with pytest.raises(ValueError, match="selected period"):
        plot_spread_timeseries(model, period="2D", observations=selected)


def test_empty_actual_date_query_has_no_invented_spreads_or_uncertainty():
    factor, marks, panel, dates = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    selected = panel.quantile_statistics(by_date=True).filter(pl.col("date") < dates[0])
    figure = plot_spread_timeseries(model, observations=selected)
    assert all(len(trace.y) == 0 for trace in figure.data)
    assert any("No defined observed spread" in item.text for item in figure.layout.annotations)


@pytest.mark.parametrize("upper,lower,expected", [(1, 2, [-0.25, -0.0625]), (1, 1, [0.0, 0.0])])
def test_arbitrary_shared_bucket_pairs_keep_their_sign_and_same_bucket_convention(
    upper, lower, expected
):
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    figure = plot_spread_timeseries(model, upper_quantile=upper, lower_quantile=lower)
    np.testing.assert_allclose(_trace(figure, "Observed Spread").y, expected, rtol=0, atol=1e-14)


def test_absent_bucket_is_rejected_not_filled_with_a_zero_mean():
    factor, marks, panel, _ = _inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    with pytest.raises(ValueError, match="absent"):
        plot_spread_timeseries(model, upper_quantile=3)


def test_actual_long_quote_sample_exercises_complete_22_observation_means_not_only_warmup():
    factor, marks, panel, _ = _long_inputs()
    model = _analysis("owned", factor, marks, panel).to_quantile_result()
    figure = plot_spread_timeseries(model, rolling_window=22)
    expected = [
        0.025
        + (
            0.001 * ((i + 1) % 3)
            + 0.001 * ((i + 1) % 4)
            - 0.0005 * ((i + 1) % 5)
            - 0.0002 * ((i + 1) % 7)
        )
        / 2
        for i in range(1, 27)
    ]
    np.testing.assert_allclose(_trace(figure, "Observed Spread").y, expected, rtol=0, atol=1e-14)
    expected_smoothing = [np.nan] * 21 + [sum(expected[i - 21 : i + 1]) / 22 for i in range(21, 26)]
    np.testing.assert_allclose(
        _trace(figure, "22-Observation Rolling Mean").y,
        expected_smoothing,
        rtol=0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.isfinite(np.asarray(_trace(figure, "Propagated SE band").y)).all()


@pytest.mark.parametrize("width", [1280, 375])
def test_actual_offline_report_browser_renders_spread_values_and_exact_nanosecond_hover(
    width, tmp_path
):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium is required for actual report browser checks")
    factor, marks, panel, _ = _inputs(True)
    html = SignalDashboard().generate(_analysis("owned", factor, marks, panel).to_tear_sheet())
    probe = """<script>document.addEventListener('DOMContentLoaded',()=>{
    Array.from(document.querySelectorAll('button')).find(node=>node.textContent==='Quantile Analysis').click();
    const panel=document.getElementById('quantile'), marker=document.createElement('output');marker.id='spread-browser-check';
    const plot=Array.from(panel.querySelectorAll('.js-plotly-plot')).find(plot=>Array.from(plot._fullData||[]).some(trace=>trace.name==='Observed Spread'));
    marker.textContent=JSON.stringify({width:innerWidth,documentWidth:document.documentElement.scrollWidth,pointCount:plot?plot.querySelectorAll('.scatterlayer .point').length:null,legends:plot?Array.from(plot.querySelectorAll('.legendtext')).map(node=>node.textContent):[],traces:Array.from(panel.querySelectorAll('.js-plotly-plot')).flatMap(plot=>Array.from(plot._fullData||[]).filter(trace=>trace.name==='Observed Spread').map(trace=>({y:Array.from(trace.y||[]),customdata:Array.from(trace.customdata||[])})))});
    document.body.appendChild(marker);});</script>"""
    path = tmp_path / "report.html"
    path.write_text(html.replace("</body>", probe + "</body>"))
    completed = subprocess.run(
        [
            chrome,
            "--headless",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--host-resolver-rules=MAP * ~NOTFOUND",
            f"--user-data-dir={tmp_path / 'browser-profile'}",
            f"--window-size={width},1000",
            "--virtual-time-budget=1500",
            "--dump-dom",
            path.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    match = re.search(r'<output id="spread-browser-check">([^<]+)</output>', completed.stdout)
    assert match, completed.stderr
    measured = json.loads(match[1])
    assert len(measured["traces"]) == 1
    np.testing.assert_allclose(measured["traces"][0]["y"], [0.25, 0.0625], rtol=0, atol=1e-14)
    assert measured["traces"][0]["customdata"][0][0].endswith("000000100+00:00")
    assert measured["documentWidth"] <= measured["width"]
    assert measured["pointCount"] == 2
    assert measured["legends"] == [
        "Observed Spread",
        "21-Observation Rolling Mean",
        "Propagated SE band",
    ]
