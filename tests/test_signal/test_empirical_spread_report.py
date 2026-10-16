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
    marker.textContent=JSON.stringify({width:innerWidth,documentWidth:document.documentElement.scrollWidth,traces:Array.from(panel.querySelectorAll('.js-plotly-plot')).flatMap(plot=>Array.from(plot._fullData||[]).filter(trace=>trace.name==='Observed Spread').map(trace=>({y:Array.from(trace.y||[]),customdata:Array.from(trace.customdata||[])})))});
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
