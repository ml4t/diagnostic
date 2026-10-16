"""A later active year must not erase observed flat or tiny-return years from reports."""

import shutil
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import (
    create_portfolio_dashboard,
    plot_monthly_returns_heatmap,
)


class _BuyOnce(Strategy):
    def __init__(self):
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            broker.submit_order("A", 4)
            self.submitted = True


def _actual_account(terminal_year, increment, *, active=True):
    dates = [
        datetime(2023, 12, 28, 15, tzinfo=UTC),
        datetime(2023, 12, 29, 15, tzinfo=UTC),
        datetime(terminal_year, 1, 2, 15, tzinfo=UTC),
        datetime(terminal_year, 1, 3, 15, tzinfo=UTC),
    ]
    opens = [100.0, 100.0, 110.0 if active else 100.0, 90.0 if active else 100.0]
    closes = [100.0, 100.0 + increment, *opens[2:]]
    prices = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["A"] * 4,
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [10000.0] * 4,
        }
    )
    result = Engine(DataFeed(prices_df=prices), _BuyOnce(), BacktestConfig(initial_cash=1000)).run()
    fills = result.to_fills_dataframe()
    assert fills.height == 1 and fills["quantity"][0] == 4 and fills["price"][0] == 100
    equity = result.to_equity_dataframe()
    np.testing.assert_allclose(
        equity["equity"],
        [1000, 1000 + 4 * increment, 1040 if active else 1000, 960 if active else 1000],
        rtol=0,
        atol=1e-12,
    )
    analysis = PortfolioAnalysis(result.to_returns_series(), dates=equity["timestamp"])
    return analysis


@pytest.mark.parametrize("terminal_year", [2024, 2025])
@pytest.mark.parametrize("increment", [0.0, 1e-7])
def test_actual_zero_or_tiny_year_remains_observed_without_inventing_gap_years(
    terminal_year, increment
):
    analysis = _actual_account(terminal_year, increment)
    first_return = (1000 + 4 * increment) / 1000 - 1
    if increment:
        assert 0 < first_return < 1e-8
    fig = plot_monthly_returns_heatmap(analysis)
    assert list(fig.data[0].y) == [2023, terminal_year]
    cells = np.asarray(fig.data[0].z, dtype=float)
    assert cells.shape == (2, 13)
    np.testing.assert_allclose(cells[0, [11, 12]], first_return, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        cells[1, [0, 12]], 960 / (1000 + 4 * increment) - 1, rtol=0, atol=1e-15
    )
    assert np.isnan(cells[0, :11]).all()
    assert np.isnan(cells[1, 1:12]).all()
    flat = plot_monthly_returns_heatmap(_actual_account(terminal_year, 0.0, active=False))
    assert list(flat.data[0].y) == [2023, terminal_year]
    np.testing.assert_array_equal(np.asarray(flat.data[0].z)[0, [11, 12]], [0, 0])
    np.testing.assert_array_equal(np.asarray(flat.data[0].z)[1, [0, 12]], [0, 0])


@pytest.mark.parametrize("width", [1280, 375])
def test_real_html_preserves_observed_zero_year_and_missing_months(width, tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium required for actual report checks")
    playwright = pytest.importorskip("playwright.sync_api")
    report = create_portfolio_dashboard(
        _actual_account(2024, 0.0), include_positions=False, include_transactions=False
    )
    path = tmp_path / "observed-years.html"
    report.save_html(path, include_plotlyjs=True)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            context = browser.new_context(
                viewport={"width": width, "height": 1000},
                is_mobile=width == 375,
                has_touch=width == 375,
                offline=True,
            )
            page = context.new_page()
            page.goto(path.as_uri(), wait_until="load")
            page.wait_for_function(
                "Array.from(document.querySelectorAll('.js-plotly-plot')).every(p=>p._fullLayout)"
            )
            panel = page.locator(".plot-container").filter(
                has=page.get_by_role("heading", name="Monthly Returns Heatmap", exact=True)
            )
            actual = panel.locator(".js-plotly-plot").evaluate("""p=>({
                years:Array.from(p._fullData[0].y),
                cells:Array.from(p._fullData[0].z,r=>Array.from(r)),
                labels:Array.from(p.querySelectorAll('.ytick text'))
                    .map(t=>({text:t.textContent,top:t.getBoundingClientRect().top}))
                    .sort((a,b)=>a.top-b.top).map(t=>t.text)
            })""")
            assert actual["years"] == [2023, 2024]
            assert actual["labels"] == ["2023", "2024"]
            assert actual["cells"][0][11] == 0
            assert actual["cells"][1][0] == pytest.approx(-0.04)
            assert np.isnan(np.asarray(actual["cells"][0][:11], dtype=float)).all()
            assert np.isnan(np.asarray(actual["cells"][1][1:12], dtype=float)).all()
        finally:
            browser.close()
