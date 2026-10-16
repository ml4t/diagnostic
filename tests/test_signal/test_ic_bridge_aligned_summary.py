"""Report statistics must use the quote-derived IC dates their bridge retains."""

import json
import math
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp

from ml4t.diagnostic import analyze_prepared_signal, analyze_signal
from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import SignalResult, prepare_signal_panel
from ml4t.diagnostic.visualization.signal import SignalDashboard, plot_ic_ts


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
    assert decoded.ic_t_stat_hac is None and decoded.ic_p_value_hac is None
    assert decoded.ras_adjusted_ic is None and decoded.ras_significant is None
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


@pytest.mark.parametrize(
    "profile,periods,mean", [("full", (2, 1), 0.5), ("missing_tail", (1, 2), 1.0)]
)
@pytest.mark.parametrize("width", [1280, 375])
def test_actual_report_browser_mean_line_uses_retained_quote_ic_observations(
    profile, periods, mean, width, tmp_path
):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium required for actual report checks")
    result, _ = _analysis(profile, "raw", periods)
    html = SignalDashboard().generate(result.to_tear_sheet())
    probe = """<script>document.addEventListener('DOMContentLoaded',()=>{
    Array.from(document.querySelectorAll('button')).find(node=>node.textContent==='IC Analysis').click();
    const plot=Array.from(document.querySelectorAll('#ic .js-plotly-plot')).find(plot=>Array.from(plot.data||[]).some(trace=>trace.name==='Daily IC'));
    const marker=document.createElement('output');marker.id='ic-bridge-browser-check';
    const rect=plot.getBoundingClientRect();
    const box=node=>{const rect=node.getBoundingClientRect();return {left:rect.left,right:rect.right,top:rect.top,bottom:rect.bottom}};
    const summary=Array.from(plot.querySelectorAll('.annotation-text')).find(node=>node.textContent.includes('Summary:'));
    marker.textContent=JSON.stringify({values:Array.from(plot._fullData[0].y||[]),summary:box(summary),legend:box(plot.querySelector('.legend')),annotations:Array.from(plot.querySelectorAll('.annotation-text')).map(node=>{const box=node.getBoundingClientRect();return {text:node.textContent,left:box.left-rect.left,right:box.right-rect.left}}),plot_width:rect.width});document.body.appendChild(marker);});</script>"""
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
        check=True,
        timeout=30,
    )
    match = re.search(r'<output id="ic-bridge-browser-check">([^<]+)</output>', completed.stdout)
    assert match, completed.stderr
    measured = json.loads(match[1])
    expected = [1.0, 0.0] if profile == "full" else [1.0]
    np.testing.assert_allclose(measured["values"], expected, rtol=0, atol=1e-14)
    label = next(
        annotation
        for annotation in measured["annotations"]
        if annotation["text"] == f"Mean IC: {mean:.4f}"
    )
    assert label["left"] >= -1 and label["right"] <= measured["plot_width"] + 1
    legend, summary = measured["legend"], measured["summary"]
    overlap_width = min(legend["right"], summary["right"]) - max(legend["left"], summary["left"])
    overlap_height = min(legend["bottom"], summary["bottom"]) - max(legend["top"], summary["top"])
    assert overlap_width <= 0 or overlap_height <= 0, measured
    if profile == "missing_tail":
        assert any(annotation["text"] == "CI unavailable" for annotation in measured["annotations"])


@pytest.mark.parametrize("show_significance", [True, False])
def test_singleton_aligned_ic_does_not_render_confidence_from_undefined_variance(show_significance):
    result, _ = _analysis("missing_tail", "raw")
    figure = plot_ic_ts(result.to_ic_result(), period="1D", show_significance=show_significance)
    assert not any(shape.type == "rect" for shape in figure.layout.shapes)
    annotations = [annotation.text for annotation in figure.layout.annotations]
    assert ("CI unavailable" in annotations) is show_significance
    assert not any("% CI" in annotation for annotation in annotations)
