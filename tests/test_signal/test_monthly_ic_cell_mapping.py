"""Quote-derived monthly IC values must occupy their actual visible month/year cells."""

import math
import shutil

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic import analyze_prepared_signal, analyze_signal
from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel
from ml4t.diagnostic.visualization.signal import SignalDashboard, plot_monthly_ic_heatmap

PROFILES = ["jan", "mar", "sparse", "cross_year", "offset_change", "zero", "singleton", "empty"]


def _inputs(profile):
    count = 6 if profile == "empty" else 7 if profile == "singleton" else 8
    if profile == "sparse":
        dates = pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
                "2025-02-03",
                "2025-02-04",
                "2025-03-03",
                "2025-03-04",
                "2025-03-05",
            ],
            utc=True,
        )
    elif profile == "cross_year":
        dates = pd.to_datetime(
            [
                "2024-12-23",
                "2024-12-24",
                "2024-12-26",
                "2024-12-27",
                "2024-12-30",
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
            ],
            utc=True,
        )
    elif profile == "offset_change":
        dates = pd.DatetimeIndex(
            [
                "2025-03-03 16:00",
                "2025-03-04 16:00",
                "2025-03-05 16:00",
                "2025-03-06 16:00",
                "2025-03-07 16:00",
                "2025-03-10 16:00",
                "2025-03-11 16:00",
                "2025-03-12 16:00",
            ],
            tz="America/New_York",
        )
    else:
        dates = pd.bdate_range(
            "2025-03-03" if profile == "mar" else "2025-01-02", periods=count, tz="UTC"
        )
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
            multipliers = [past, 1 + rising, past, 1.25, past, 1 + falling, past]
        quotes = [128.0]
        for multiplier in multipliers[: count - 1]:
            quotes.append(quotes[-1] * multiplier)
        prices[asset] = quotes
    prices = pd.DataFrame(prices, index=dates)
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[[1, 3] if profile == "empty" else [1, 3, 5]]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    marks = (
        prices.stack(future_stack=True).rename("price").rename_axis(["date", "asset"]).reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=[2, 1], group_col="asset", price_col="price")
        .set_index(["date", "asset"])[["fwd_ret_2", "fwd_ret_1"]]
        .rename(columns={"fwd_ret_2": "2D", "fwd_ret_1": "1D"})
    )
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.5)
    return factor.reset_index(), marks, panel


def _analysis(profile, route="raw"):
    factor, marks, panel = _inputs(profile)
    if route == "raw":
        result = analyze_signal(
            factor,
            marks,
            periods=(2, 1),
            quantiles=2,
            quantile_method="qcut",
            min_assets=2,
            filter_zscore=None,
            compute_turnover_flag=False,
        )
    elif route == "prepared":
        result = analyze_prepared_signal(panel.data, periods=(2, 1), quantiles=2, min_assets=2)
    else:
        result = panel.analyze(min_assets=2)
    return result


def _expected(profile):
    if profile == "empty":
        return [], [], []
    if profile == "cross_year":
        return ["Jan", "Dec"], ["2025", "2024"], [[0.0, math.nan], [math.nan, 1.0]]
    if profile == "sparse":
        return ["Jan", "Mar"], ["2025"], [[1.0, 0.0]]
    return (
        ["Mar" if profile in {"mar", "offset_change"} else "Jan"],
        ["2025"],
        [[0.0 if profile == "zero" else 1.0 if profile == "singleton" else 0.5]],
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
@pytest.mark.parametrize("json_roundtrip", [False, True])
def test_monthly_cells_keep_actual_quote_month_year_mean_and_missingness(
    profile, route, json_roundtrip
):
    result = _analysis(profile, route)
    original = result.to_json()
    model = result.to_ic_result(period=1 if profile == "zero" else None)
    before = model.model_dump_json()
    if json_roundtrip:
        model = type(model).model_validate_json(before)
    expected_ic = (
        []
        if profile == "empty"
        else [1.0, -1.0]
        if profile == "zero"
        else [1.0]
        if profile == "singleton"
        else [1.0, 0.0]
    )
    np.testing.assert_allclose(
        model.ic_by_date["1D" if profile == "zero" else "2D"], expected_ic, rtol=0, atol=1e-14
    )
    if profile == "offset_change":
        assert {day[-6:] for day in model.dates} == {"-05:00", "-04:00"}
    figure = plot_monthly_ic_heatmap(model)
    months, years, means = _expected(profile)
    if not months:
        assert not figure.data
        assert any("Monthly IC unavailable" == note.text for note in figure.layout.annotations)
    else:
        assert list(figure.data[0].x) == months
        assert list(figure.data[0].y) == years
        np.testing.assert_allclose(figure.data[0].z, means, rtol=0, atol=1e-14, equal_nan=True)
    assert model.model_dump_json() == before
    assert result.to_json() == original


@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_explicit_single_period_component_retains_its_other_observed_month(route):
    model = _analysis("sparse", route).to_ic_result(period=2)
    np.testing.assert_allclose(model.ic_by_date["2D"], [1.0, 1.0, 0.0], rtol=0, atol=1e-14)
    figure = plot_monthly_ic_heatmap(model)
    assert list(figure.data[0].x) == ["Jan", "Feb", "Mar"]
    np.testing.assert_allclose(figure.data[0].z, [[1.0, 1.0, 0.0]], rtol=0, atol=1e-14)


def test_unknown_horizon_does_not_silently_plot_a_different_monthly_sample():
    with pytest.raises(ValueError, match="not found"):
        plot_monthly_ic_heatmap(_analysis("mar").to_ic_result(), period="7D")


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("width", [1280, 375])
def test_actual_monthly_report_browser_renders_defined_cells_at_their_real_coordinates(
    profile, width, tmp_path
):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium required for actual report checks")
    playwright = pytest.importorskip("playwright.sync_api")
    result = _analysis(profile)
    tear_sheet = result.to_tear_sheet()
    if profile == "zero":
        tear_sheet.ic_analysis = result.to_ic_result(period=1)
    path = tmp_path / "report.html"
    path.write_text(SignalDashboard().generate(tear_sheet))
    errors = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000}, offline=True)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(path.as_uri(), wait_until="networkidle")
            page.get_by_role("button", name="IC Analysis", exact=True).click()
            page.wait_for_function(
                """()=>Array.from(document.querySelectorAll('#ic .js-plotly-plot')).every(plot=>Math.abs(plot._fullLayout.width-plot.getBoundingClientRect().width)<=1)"""
            )
            measured = page.locator("#ic").evaluate("""element=>{
                const plot=Array.from(element.querySelectorAll('.js-plotly-plot')).find(plot=>plot._fullLayout.title.text.startsWith('Monthly IC Heatmap'));
                if(!plot)return {width:innerWidth,plot:null};
                const image=plot.querySelector('.heatmaplayer image');
                return {width:innerWidth,plot:{x:plot._fullData[0]?.x||[],y:plot._fullData[0]?.y||[],z:plot.calcdata[0]?.[0]?.z||[],image_height:image?.getBoundingClientRect().height||0,image_width:image?.getBoundingClientRect().width||0,notes:Array.from(plot.querySelectorAll('.annotation-text')).map(node=>node.textContent)}};
            }""")
        finally:
            browser.close()
    assert not errors
    assert measured["width"] == width
    assert measured["plot"] is not None
    actual = measured["plot"]
    months, years, means = _expected(profile)
    if not months:
        assert actual["image_height"] == 0
        assert "Monthly IC unavailable" in actual["notes"]
    else:
        assert actual["x"] == months and actual["y"] == years
        assert actual["image_height"] > 100 and actual["image_width"] > 50
        restored = [[math.nan if value is None else value for value in row] for row in actual["z"]]
        np.testing.assert_allclose(restored, means, rtol=0, atol=1e-14, equal_nan=True)
