"""Prepared liquidity reaches public reports with honest coverage and scale."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
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


def test_complete_flat_asset_observation_is_not_bridged_by_a_liquidation_curve():
    dates = pd.date_range("2025-01-13", periods=4, tz="UTC")
    positions = pl.from_pandas(
        pd.DataFrame(
            {
                "date": dates[[0, 0, 1, 1, 2, 3, 3]],
                "asset": ["A", "B", "A", "B", "B", "A", "B"],
                "value": [10000.0] * 7,
            }
        )
    )
    account = pd.DataFrame({"equity": [20000.0] * 4, "cash": [0.0, 0.0, 10000.0, 0.0]}, index=dates)
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.0] * 4, index=dates),
        positions=positions,
        valuations=account,
        complete_snapshots=True,
        calendar=None,
    )
    market = pl.from_pandas(
        pd.DataFrame(
            {
                "date": dates.repeat(2),
                "asset": ["A", "B"] * 4,
                "price": [10.0] * 8,
                "volume": [10000.0] * 8,
            }
        )
    )
    result = analysis.compute_capacity(
        market, currency="USD", price_basis="raw", volume_basis="raw", lookback=1
    )
    figure = analysis.create_tear_sheet(capacity=result).figures["Lagged Market Liquidation"]
    trace = next(trace for trace in figure.data if trace.name == "A")
    assert len(trace.x) == 4
    assert list(trace.y) == [None, 1.0, 0.0, 1.0]


def test_actual_capacity_render_does_not_collapse_distinct_nanosecond_observations(tmp_path):
    import shutil

    playwright_api = pytest.importorskip("playwright.sync_api")
    executable = shutil.which("chromium") or shutil.which("google-chrome")
    if executable is None:
        pytest.skip("Actual Chrome/Chromium required for rendered ns-key checks")
    dates = pd.to_datetime(
        [
            "2025-01-13T20:00:00.000000001Z",
            "2025-01-13T20:00:00.000000002Z",
            "2025-01-13T20:00:00.000000003Z",
        ]
    )
    price = pd.Series([100.0, 101.0, 99.99], index=dates)
    analysis = PortfolioAnalysis.from_frames(
        price.pct_change().fillna(0.0),
        positions=pd.DataFrame({"A": price * 100, "cash": 0.0}, index=dates),
        complete_snapshots=True,
        calendar=None,
        periods_per_year=12,
    )
    market = pl.from_pandas(
        pd.DataFrame(
            {"date": dates, "asset": ["A"] * 3, "price": price.to_numpy(), "volume": [10000.0] * 3}
        )
    )
    result = analysis.compute_capacity(
        market, currency="USD", price_basis="raw", volume_basis="raw", lookback=1
    )
    path = tmp_path / "nanoseconds.html"
    analysis.create_tear_sheet(capacity=result).save_html(path, include_plotlyjs=True)
    with playwright_api.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=executable, headless=True)
        try:
            page = browser.new_page(viewport={"width": 375, "height": 900})
            page.goto(path.as_uri())
            container = page.locator(".plot-container").filter(
                has=page.get_by_role("heading", name="Lagged Market Liquidation", exact=True)
            )
            graph = container.locator(".js-plotly-plot")
            graph.locator(".main-svg").first.wait_for()
            actual = graph.evaluate("""p=>({keys:p.data[0].x,
                renderedPositions:p.calcdata[0].filter(point=>Number.isFinite(point.y)).map(point=>point.x)})""")
        finally:
            browser.close()
    assert actual["keys"][1].endswith("000000002+00:00")
    assert len(actual["renderedPositions"]) == 2
    assert len(set(actual["renderedPositions"])) == 2


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
