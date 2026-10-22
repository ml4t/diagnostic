"""Supplied GPD fits remain inspectable in real portfolio reports, including failures."""

import re
from xml.etree import ElementTree

import numpy as np
import pytest

from ml4t.diagnostic.metrics import estimate_gpd_tail_risk
from ml4t.diagnostic.visualization.backtest import (
    BacktestTearsheet,
    export_workspaces,
    generate_backtest_tearsheet,
)


def model_table(html):
    matches = re.findall(r'<section class="gpd-tail-fit".*?</section>', html, re.DOTALL)
    assert len(matches) == 1, "The supplied fitted model was silently omitted or duplicated"
    root = ElementTree.fromstring(matches[0])
    return {row.find("th").text: row.find("td").text for row in root.findall(".//tbody/tr")}


@pytest.mark.parametrize("route", ["function", "builder", "export"])
def test_actual_portfolio_report_renders_supplied_fit_counts_units_and_boundary_without_refitting(
    route,
    monkeypatch,
):
    from ml4t.diagnostic.metrics import tail_risk

    returns = np.r_[
        np.repeat(0.002, 40), -np.array([0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2])
    ]
    fitted = estimate_gpd_tail_risk(returns, threshold=0.02, tail_probability=0.02)
    original_fit = tail_risk._fit
    refits = []

    def observe_fit(*args, **kwargs):
        refits.append(True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(tail_risk, "_fit", observe_fit)
    supplied = {"gpd_tail_risk": fitted}
    if route == "function":
        html = generate_backtest_tearsheet(
            returns=returns, metrics=supplied, periods_per_year=12, include_statistical=False
        )
    elif route == "builder":
        html = (
            BacktestTearsheet(periods_per_year=12)
            .add_returns(returns)
            .add_metrics(supplied)
            .generate()
        )
    else:
        html = export_workspaces(
            ("performance",), returns=returns, metrics=supplied, periods_per_year=12
        )
    rows = model_table(html)
    assert rows["Method"] == "Peaks over threshold"
    assert rows["Fit status"] == "ok"
    assert rows["Finite / supplied observations"] == "48 / 48"
    assert rows["Exceedances / probability denominator"] == "8 / 48"
    assert rows["Tail probability"] == "2.00%"
    assert rows["VaR (positive periodic loss)"] == "17.84%"
    assert rows["ES (positive periodic loss)"] == "18.92%"
    assert rows["At shape lower bound"] == "Yes"
    assert supplied["gpd_tail_risk"] is fitted
    assert not refits


def test_report_keeps_a_genuinely_infinite_fitted_mean_instead_of_null_or_zero():
    losses = np.array([0.021, 0.022, 0.023, 0.024, 0.026, 0.03, 0.04, 0.08, 0.2, 0.8])
    returns = np.r_[np.repeat(0.002, 40), -losses]
    fitted = estimate_gpd_tail_risk(returns, threshold=0.02)
    assert fitted.status == "ok" and fitted.shape > 1 and np.isposinf(fitted.expected_shortfall)
    html = generate_backtest_tearsheet(
        returns=returns, metrics={"gpd_tail_risk": fitted}, include_statistical=False
    )
    assert model_table(html)["ES (positive periodic loss)"] == "Positive infinity"


def test_report_discloses_unavailable_fitted_risk_instead_of_the_reference_zero_sentinel():
    returns = np.repeat(0.01, 20)
    fitted = estimate_gpd_tail_risk(returns)
    assert fitted.status == "insufficient_exceedances"
    html = generate_backtest_tearsheet(
        returns=returns, metrics={"gpd_tail_risk": fitted}, include_statistical=False
    )
    rows = model_table(html)
    assert rows["Fit status"] == "insufficient_exceedances"
    assert (
        rows["VaR (positive periodic loss)"] == rows["ES (positive periodic loss)"] == "Unavailable"
    )


@pytest.mark.parametrize("route", ["function", "builder", "export"])
def test_serialized_model_dictionary_is_rejected_before_report_metrics_or_output_are_changed(
    route, tmp_path
):
    from dataclasses import asdict

    returns = np.r_[
        np.repeat(0.002, 40), -np.array([0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2])
    ]
    supplied = {"gpd_tail_risk": asdict(estimate_gpd_tail_risk(returns, threshold=0.02))}
    before = dict(supplied)
    output = tmp_path / "report.html"
    output.write_text("Previous report")
    with pytest.raises(TypeError, match="GPDTailRiskResult"):
        if route == "function":
            generate_backtest_tearsheet(returns=returns, metrics=supplied, output_path=output)
        elif route == "builder":
            builder = BacktestTearsheet().add_returns(returns).add_metrics(supplied)
            builder.generate(output_path=output)
        else:
            export_workspaces(
                ("performance",), returns=returns, metrics=supplied, output_path=output
            )
    assert supplied == before
    if route == "builder":
        assert builder.metrics == before
    assert output.read_text() == "Previous report"


@pytest.mark.parametrize("width", [375, 1280])
def test_rendered_fitted_tail_table_matches_prose_and_remains_readable_at_phone_width(
    width, tmp_path
):
    import shutil

    playwright_api = pytest.importorskip("playwright.sync_api")
    executable = shutil.which("chromium") or shutil.which("google-chrome")
    if executable is None:
        pytest.skip("Actual Chrome/Chromium is required for browser typography checks")
    returns = np.r_[
        np.repeat(0.002, 40), -np.array([0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2])
    ]
    fitted = estimate_gpd_tail_risk(returns, threshold=0.02)
    path = tmp_path / "report.html"
    generate_backtest_tearsheet(
        returns=returns,
        metrics={"gpd_tail_risk": fitted},
        output_path=path,
        include_statistical=False,
    )
    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.goto(path.as_uri())
            page.locator('button.workspace-tab[data-target="performance"]').click()
            details = page.locator('details[data-section="distribution"]')
            if details.count() and not details.evaluate("e=>e.open"):
                details.locator("summary").click()
            component = page.locator(".gpd-tail-fit")
            component.wait_for(state="visible")
            measured = component.evaluate("""e=>({
                proseSize:parseFloat(getComputedStyle(e.querySelector('p')).fontSize),
                proseFamily:getComputedStyle(e.querySelector('p')).fontFamily,
                cells:[...e.querySelectorAll('th,td,caption')].map(x=>({size:parseFloat(getComputedStyle(x).fontSize),family:getComputedStyle(x).fontFamily})),
                tableWidth:e.querySelector('table').getBoundingClientRect().width,
                wrapperWidth:e.querySelector('.metrics-table-wrap').getBoundingClientRect().width,
                width:e.getBoundingClientRect().width,
                scroll:e.scrollWidth,client:e.clientWidth,
                noteSize:parseFloat(getComputedStyle(e.querySelector('.gpd-tail-fit-note')).fontSize)
            })""")
            assert measured["width"] > 0 and measured["scroll"] <= measured["client"] + 1
            assert measured["tableWidth"] <= measured["wrapperWidth"] + 1
            assert all(cell["family"] == measured["proseFamily"] for cell in measured["cells"])
            assert all(
                0.8 * measured["proseSize"] <= cell["size"] <= measured["proseSize"]
                for cell in measured["cells"]
            )
            if width == 375:
                assert all(cell["size"] >= 14 for cell in measured["cells"])
                assert measured["noteSize"] >= 13
            component.screenshot(path=str(tmp_path / f"gpd-tail-fit-{width}.png"))
        finally:
            browser.close()
