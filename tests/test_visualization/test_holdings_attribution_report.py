"""Supplied-weight attribution must reach reports without refitting or relinking."""

import shutil

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor import compute_holdings_attribution
from ml4t.diagnostic.visualization.backtest import (
    BacktestTearsheet,
    export_workspaces,
    generate_backtest_tearsheet,
)
from ml4t.diagnostic.visualization.factor import (
    plot_holdings_attribution,
    plot_holdings_exposures,
)


def supplied_result(**kwargs):
    dates = pd.DatetimeIndex(["2025-07-07T20:00Z", "2025-07-08T20:00Z"])
    starts = pd.DatetimeIndex(["2025-07-03T20:00Z", "2025-07-07T20:00Z"])
    returns = pl.DataFrame({"timestamp": dates, "period_start": starts, "return": [0.03, -0.01]})
    keys = {
        "timestamp": dates.repeat(2),
        "asset": ["A", "B", "A", "B"],
        "known_at": starts.repeat(2),
    }
    weights = pl.DataFrame({**keys, "weight": [0.5, 0.5, 0.25, 0.75]})
    loadings = pl.DataFrame({**keys, "market": [2.0, 1.0, 1.0, 1.0], "style": [0.0, 0.5, 2.0, 0.0]})
    factors = pl.DataFrame({"timestamp": dates, "market": [0.01, -0.02], "style": [0.02, 0.01]})
    result = compute_holdings_attribution(
        returns, weights, factors, loadings, complete_snapshots=True, **kwargs
    )
    return returns["return"].to_numpy(), result


def report(route, returns, metrics, output=None):
    if route == "function":
        return generate_backtest_tearsheet(
            returns=returns,
            metrics=metrics,
            periods_per_year=12,
            include_statistical=False,
            output_path=output,
        )
    if route == "builder":
        return (
            BacktestTearsheet(periods_per_year=12)
            .add_returns(returns)
            .add_metrics(metrics)
            .generate(output_path=output)
        )
    return export_workspaces(
        ("factors",), returns=returns, metrics=metrics, periods_per_year=12, output_path=output
    )


@pytest.mark.parametrize("route", ["function", "builder", "export"])
def test_supplied_results_render_separate_exposure_attribution_and_coverage_sections(route):
    returns, result = supplied_result()
    html = report(route, returns, {"holdings_attribution": result})
    for name in (
        "holdings_factor_exposures",
        "holdings_factor_attribution",
        "holdings_factor_coverage",
    ):
        assert html.count(f'data-section="{name}"') == 1
    assert "Wealth-linked" in html and "not estimated regression" in html
    assert "0.4550%" in html and "1.5150%" in html and "1.9700%" in html
    assert "not annualized" in html


@pytest.mark.parametrize("route", ["function", "builder", "export"])
def test_serialized_result_is_rejected_before_metrics_or_previous_output_change(route, tmp_path):
    returns, result = supplied_result()
    metrics = {"holdings_attribution": result.to_dict()}
    before = {"holdings_attribution": result.to_dict()}
    output = tmp_path / "report.html"
    output.write_text("Previous report")
    with pytest.raises(TypeError, match="HoldingsAttributionResult"):
        report(route, returns, metrics, output)
    assert metrics == before and output.read_text() == "Previous report"


@pytest.mark.parametrize("route", ["function", "builder", "export"])
def test_other_account_returns_cannot_be_displayed_with_wrong_attribution(route, tmp_path):
    _, result = supplied_result()
    other = pd.Series([100.0, 104.0, 103.0]).pct_change(fill_method=None).iloc[1:].to_numpy()
    metrics = {"holdings_attribution": result}
    output = tmp_path / "report.html"
    output.write_text("Previous report")
    with pytest.raises(ValueError, match="same returns|match.*return|attribution.*return"):
        report(route, other, metrics, output)
    assert list(metrics) == ["holdings_attribution"] and output.read_text() == "Previous report"


def test_performance_only_export_does_not_leak_auto_enabled_factor_sections():
    returns, result = supplied_result()
    html = export_workspaces(
        ("performance",), returns=returns, metrics={"holdings_attribution": result}
    )
    assert 'data-section="holdings_factor_' not in html


def test_builder_explicit_section_disable_is_preserved_during_generation():
    returns, result = supplied_result()
    builder = BacktestTearsheet().add_returns(returns).add_metrics({"holdings_attribution": result})
    builder.disable_section("holdings_factor_coverage")
    html = builder.generate()
    assert 'data-section="holdings_factor_exposures"' in html
    assert 'data-section="holdings_factor_coverage"' not in html


@pytest.mark.parametrize(
    "method, common, specific, total",
    [
        ("wealth_linked", 0.00455, 0.01515, 0.0197),
        ("arithmetic", 0.005, 0.015, 0.02),
    ],
)
def test_figures_use_actual_supplied_exposures_and_selected_linking_without_recompounding(
    method, common, specific, total
):
    _, result = supplied_result(cumulative_method=method)
    exposures = plot_holdings_exposures(result)
    assert {t.name: list(t.y) for t in exposures.data} == {
        "market": [1.5, 1.0],
        "style": [0.25, 0.5],
    }
    assert exposures.data[0].line.color != exposures.data[1].line.color
    contributions = {t.name: list(t.y) for t in plot_holdings_attribution(result).data}
    np.testing.assert_allclose(contributions["Common"], [0.02, common], atol=1e-15)
    np.testing.assert_allclose(contributions["Specific"], [0.01, specific], atol=1e-15)
    np.testing.assert_allclose(contributions["Total"], [0.03, total], atol=1e-15)


@pytest.mark.parametrize("width", [375, 1280])
def test_public_report_retains_values_and_readable_nonoverlapping_labels_in_browser(
    width, tmp_path
):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium required for actual report checks")
    playwright = pytest.importorskip("playwright.sync_api")
    returns, result = supplied_result()
    path = tmp_path / "report.html"
    report("function", returns, {"holdings_attribution": result}, path)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.goto(path.as_uri())
            page.locator('button.workspace-tab[data-target="factors"]').click()
            page.wait_for_function("""() => {
                const charts=Array.from(document.querySelectorAll('[data-section^="holdings_factor_"] .js-plotly-plot'));
                return charts.length===2 && charts.every(p => p._fullLayout &&
                    Math.abs(p._fullLayout.width-p.getBoundingClientRect().width)<1);
            }""")
            actual = page.evaluate("""() => {
                const overlaps=(a,b)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>1 &&
                    Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1;
                return {
                    scroll:document.documentElement.scrollWidth, width:innerWidth,
                    prose:parseFloat(getComputedStyle(document.querySelector('.holdings-attribution-summary p')).fontSize),
                    charts:Array.from(document.querySelectorAll('[data-section^="holdings_factor_"] .js-plotly-plot')).map(p=>{
                        const bounds=p.getBoundingClientRect();
                        const labels=Array.from(p.querySelectorAll('.xtick text,.ytick text,.legendtext,.xtitle,.ytitle'));
                        const ticks=Array.from(p.querySelectorAll('.xtick text,.xtitle'));
                        return {
                            text:labels.map(t=>({size:parseFloat(getComputedStyle(t).fontSize),family:getComputedStyle(t).fontFamily})),
                            clipped:labels.some(t=>{const b=t.getBoundingClientRect(); return b.left<bounds.left-2 || b.right>bounds.right+2 || b.top<bounds.top-2 || b.bottom>bounds.bottom+2}),
                            overlap:Array.from(p.querySelectorAll('.legendtext')).some(t=>ticks.some(a=>overlaps(t.getBoundingClientRect(),a.getBoundingClientRect()))),
                            traces:p.data.map(t=>({name:t.name,y:Array.from(t.y)}))
                        };
                    })
                };
            }""")
        finally:
            browser.close()
    assert actual["width"] == width and actual["scroll"] <= width
    assert actual["prose"] == 13
    assert len(actual["charts"]) == 2
    for chart in actual["charts"]:
        assert not chart["clipped"] and not chart["overlap"]
        assert all(label["size"] == 14 and "DM Sans" in label["family"] for label in chart["text"])
    traces = {t["name"]: t["y"] for chart in actual["charts"] for t in chart["traces"]}
    np.testing.assert_allclose(traces["Common"], [0.02, 0.00455], atol=1e-15)
    np.testing.assert_allclose(traces["Total"], [0.03, 0.0197], atol=1e-15)
