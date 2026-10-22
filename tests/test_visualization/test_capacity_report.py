"""Prepared liquidity reaches public reports with honest coverage and scale."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

pytest.importorskip("plotly")

from ml4t.diagnostic.evaluation.portfolio_analysis import CapacityAnalysisResult, PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import create_portfolio_dashboard


def _analysis():
    dates = [datetime(2025, 1, 13, 21, tzinfo=UTC) + timedelta(days=i) for i in range(8)]
    returns = pl.DataFrame(
        {"date": dates, "return": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.01, 0.02]}
    )
    positions = returns.select("date").with_columns(
        pl.lit("A").alias("asset"), pl.lit(10000.0).alias("value")
    )
    market = returns.select("date").with_columns(
        pl.lit("A").alias("asset"), pl.lit(10.0).alias("price"), pl.lit(10000.0).alias("volume")
    )
    fills = pl.DataFrame(
        {"timestamp": [dates[1]], "asset": ["A"], "quantity": [100], "price": [10.0]}
    )
    analysis = PortfolioAnalysis.from_frames(
        returns,
        positions=positions,
        transactions=fills,
        complete_snapshots=True,
        calendar=None,
        periods_per_year=12,
    )
    return analysis, market


def test_native_capacity_tables_and_figures_export_supplied_results_without_refitting(tmp_path):
    analysis, market = _analysis()
    result = analysis.compute_capacity(
        market,
        currency="USD",
        price_basis="raw",
        volume_basis="raw",
        lookback=1,
        base_capital=100000.0,
        scenario_capitals=[100000.0, 200000.0],
    )
    restored = CapacityAnalysisResult.from_json(result.to_json())
    assert restored.liquidation.equals(result.liquidation)
    sheet = analysis.create_tear_sheet(capacity=restored)
    assert sheet.capacity is restored
    assert sheet.figures["Hypothetical Capital Sweep"].layout.meta["periods_per_year"] == 12
    for capital, ratio in zip(
        sheet.figures["Hypothetical Capital Sweep"].data[0].x,
        sheet.figures["Hypothetical Capital Sweep"].data[0].y,
        strict=True,
    ):
        sample = result.scenarios.filter(pl.col("capital") == capital)["adjusted_return"].to_numpy()
        assert ratio == pytest.approx(sample.mean() / sample.std(ddof=1) * np.sqrt(12))
    path = tmp_path / "capacity.html"
    sheet.save_html(path, include_plotlyjs=True)
    html = path.read_text()
    assert "Worst liquidation: full/recent observations" in html
    assert "Capacity assumptions and coverage" in html
    assert "hypothetical" in html
    assert CapacityAnalysisResult.from_json(
        __import__("json").dumps(sheet.to_dict()["capacity"])
    ).timeline.equals(result.timeline)


def test_capacity_from_other_return_sample_is_rejected_before_output_changes(tmp_path):
    analysis, market = _analysis()
    result = analysis.compute_capacity(
        market, currency="USD", price_basis="raw", volume_basis="raw", lookback=1
    )
    other = PortfolioAnalysis(analysis.returns + 0.001, dates=analysis.dates)
    path = tmp_path / "previous.html"
    path.write_text("previous verified report")
    with pytest.raises(ValueError, match="return observations"):
        create_portfolio_dashboard(other, capacity=result).save_html(path)
    assert path.read_text() == "previous verified report"
    with pytest.raises(TypeError, match="CapacityAnalysisResult"):
        analysis.create_tear_sheet(capacity={})


def test_unbounded_liquidation_is_named_in_report_not_plotted_as_zero(tmp_path):
    analysis, market = _analysis()
    result = analysis.compute_capacity(
        market.with_columns(pl.lit(0.0).alias("volume")),
        currency="USD",
        price_basis="raw",
        volume_basis="raw",
        lookback=1,
    )
    sheet = analysis.create_tear_sheet(capacity=result)
    assert all(value is None for value in sheet.figures["Lagged Market Liquidation"].data[0].y)
    path = tmp_path / "unbounded.html"
    sheet.save_html(path)
    assert "Unbounded" in path.read_text()
    assert result.metadata["unbounded_liquidation_rows"] == 7


def test_recent_participation_uses_the_full_account_clock_not_last_active_trade():
    analysis, market = _analysis()
    result = analysis.compute_capacity(
        market, currency="USD", price_basis="raw", volume_basis="raw", lookback=1, recent_window=1
    )
    assert result.worst_transactions["sample"].to_list() == ["full"]


@pytest.mark.parametrize("width", [375, 1280])
def test_rendered_capacity_tables_match_prose_and_do_not_clip_at_phone_width(width, tmp_path):
    import os
    import shutil
    from pathlib import Path

    playwright_api = pytest.importorskip("playwright.sync_api")
    executable = shutil.which("chromium") or shutil.which("google-chrome")
    if executable is None:
        pytest.skip("Actual Chrome/Chromium required for capacity layout checks")
    analysis, market = _analysis()
    result = analysis.compute_capacity(
        market, currency="USD", price_basis="raw", volume_basis="raw", lookback=1
    )
    path = tmp_path / "capacity.html"
    analysis.create_tear_sheet(capacity=result).save_html(path, include_plotlyjs=True)
    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.route("https://**", lambda route: route.abort())
            page.goto(path.as_uri())
            page.locator(".js-plotly-plot .main-svg").first.wait_for()
            measured = page.locator("table.capacity-table").first.evaluate("""e=>({
                prose:getComputedStyle(document.querySelector('.metrics-summary p')),
                family:getComputedStyle(document.querySelector('.metrics-summary p')).fontFamily,
                proseSize:parseFloat(getComputedStyle(document.querySelector('.metrics-summary p')).fontSize),
                cells:[...e.querySelectorAll('th,td')].map(c=>({family:getComputedStyle(c).fontFamily,size:parseFloat(getComputedStyle(c).fontSize)})),
                wrapper:e.parentElement.getBoundingClientRect().width,
                body:document.body.getBoundingClientRect().width,
                viewport:innerWidth,
                pageWidth:document.documentElement.scrollWidth
            })""")
            assert measured["cells"] and measured["wrapper"] <= measured["body"]
            assert measured["pageWidth"] <= width + 1
            assert all(
                cell["family"] == measured["family"] and 14 <= cell["size"] <= measured["proseSize"]
                for cell in measured["cells"]
            )
            output = Path(os.environ.get("ML4T_CAPACITY_SCREENSHOTS", str(tmp_path)))
            output.mkdir(parents=True, exist_ok=True)
            page.locator("table.capacity-table").first.locator("xpath=../..").screenshot(
                path=str(output / f"capacity-{width}.png")
            )
        finally:
            browser.close()
