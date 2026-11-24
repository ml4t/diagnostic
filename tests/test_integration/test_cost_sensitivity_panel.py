"""The cost-sensitivity panel scales the costs an executed backtest actually paid.

Every test runs the public path a user takes: ``Engine.run()`` ->
``analyze_backtest_result`` -> tearsheet or ``plot_cost_sensitivity``. The run pays
4 bps commission plus 3 bps slippage per fill, so its realized cost (7 bps of traded
notional) differs from any default the panel could fall back on.
"""

import contextlib
import json
import re
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

import numpy as np
import plotly
import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import analyze_backtest_result
from ml4t.diagnostic.visualization.backtest import (
    generate_backtest_tearsheet,
    plot_cost_sensitivity,
)

COMMISSION = 0.0004
SLIPPAGE = 0.0003
REALIZED_BPS = (COMMISSION + SLIPPAGE) * 10_000


@cache
def _prices(bars: str) -> pl.DataFrame:
    if bars == "daily":
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(800)]
        drift, vol = 0.0008, 0.01
    else:
        # Six hourly bars per weekday, aggregated by the profile to daily periods.
        days = [datetime(2024, 1, 2, tzinfo=UTC) + timedelta(days=i) for i in range(400)]
        timestamps = [
            day.replace(hour=15) + timedelta(hours=h)
            for day in days
            if day.weekday() < 5
            for h in range(6)
        ]
        drift, vol = 0.0003, 0.004
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(timestamps))))
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": "A",
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1e6,
        }
    )


class _RoundTrips(Strategy):
    """Buy 95% of equity, sell a few bars later, repeat."""

    def __init__(self, period: int, hold: int):
        self.i = 0
        self.period = period
        self.hold = hold

    def on_data(self, timestamp, data, context, broker):
        step = self.i % self.period
        self.i += 1
        if step == 0:
            broker.submit_order("A", int(0.95 * broker.get_account_value() / data["A"]["close"]))
        elif step == self.hold and broker.get_position("A") is not None:
            broker.close_position("A")


@cache
def _run(bars: str, multiple: float = 1.0):
    period, hold = (5, 4) if bars == "daily" else (9, 5)
    config = BacktestConfig(
        initial_cash=100_000,
        commission_rate=COMMISSION * multiple,
        slippage_rate=SLIPPAGE * multiple,
    )
    return Engine(DataFeed(prices_df=_prices(bars)), _RoundTrips(period, hold), config).run()


def _reported_sharpe(result) -> float:
    return analyze_backtest_result(result).performance["metrics"]["sharpe_ratio"]


@cache
def _panel(bars: str, scale: int | None = None) -> tuple[list[dict], dict]:
    """Return the cost-sensitivity traces and layout from the default tearsheet."""
    html = generate_backtest_tearsheet(
        profile=analyze_backtest_result(_run(bars)), n_trials=1, periods_per_year=scale
    )
    marker = re.search(r'<(?:div|details)\b[^>]*\bdata-section="cost_sensitivity"[^>]*>', html)
    assert marker, "cost_sensitivity section missing"
    tail = html[marker.end() :]
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', tail)
    assert match
    decoder = json.JSONDecoder()
    data, offset = decoder.raw_decode(tail[match.end() :])
    layout, _ = decoder.raw_decode(tail[match.end() + offset :].lstrip()[1:].lstrip())
    return data, layout


def _trace(data: list[dict], name: str) -> tuple[list[float], list[float]]:
    trace = next(item for item in data if item["name"] == name)
    return trace["x"], trace["y"]


@pytest.mark.parametrize(("bars", "scale"), [("daily", None), ("intraday", None), ("daily", 365)])
def test_current_point_reproduces_the_reported_net_sharpe_and_cagr(bars, scale):
    """Guards against net returns read as gross, a default 10 bps "Current", a CAGR
    approximated from the arithmetic mean, and a scale other than the report's."""
    result = _run(bars)
    profile = analyze_backtest_result(result)
    if scale is not None:
        profile = replace(profile, annualization_factor=scale)
    performance = profile.performance["metrics"]
    data, layout = _panel(bars, scale)
    current = next(item for item in layout["annotations"] if item.get("text") == "Current")

    x, cagr = _trace(data, "CAGR (%)")
    panel = {
        "sharpe": current["y"],
        "cagr_pct": cagr[x.index(current["x"])],
        "cost_bps": current["x"],
    }
    reported = {
        "sharpe": pytest.approx(performance["sharpe_ratio"], rel=1e-12),
        "cagr_pct": pytest.approx(performance["cagr"] * 100, rel=1e-12),
        "cost_bps": pytest.approx(REALIZED_BPS),
    }
    assert panel == reported
    if bars == "daily" and scale is None:
        assert current["y"] == pytest.approx(result.metrics["sharpe"], rel=1e-12)


@pytest.mark.parametrize("bars", ["daily", "intraday"])
def test_each_point_matches_a_backtest_rerun_at_that_cost(bars):
    """Guards against the zero-cost point showing the net Sharpe and against costs
    charged at an assumed turnover instead of the fills the run made."""
    x, sharpe = _trace(_panel(bars)[0], "Sharpe Ratio")
    assert x[0] == 0
    for cost_bps, panel_sharpe in zip(x, sharpe, strict=True):
        rerun = _run(bars, cost_bps / REALIZED_BPS)
        # The panel does not compound the cost difference or re-round share counts,
        # which moves the Sharpe ratio by under 1% at five times the realized cost.
        expected = _reported_sharpe(rerun)
        assert panel_sharpe == pytest.approx(expected, rel=0.01, abs=0.01), cost_bps


def test_breakeven_lies_between_reruns_whose_sharpe_changes_sign():
    """Guards against a breakeven read off a curve that double-counts costs."""
    data, layout = _panel("daily")
    label = next(
        item["text"] for item in layout["annotations"] if item.get("text", "").startswith("Break")
    )
    breakeven = float(re.fullmatch(r"Breakeven ([\d.]+) bps", label).group(1))
    x, _ = _trace(data, "Sharpe Ratio")
    below = max(level for level in x if level < breakeven)
    above = min(level for level in x if level > breakeven)

    assert _reported_sharpe(_run("daily", below / REALIZED_BPS)) > 0
    assert _reported_sharpe(_run("daily", above / REALIZED_BPS)) < 0


def test_returns_only_tearsheet_omits_the_cost_panel():
    """Guards against charging assumed costs to returns of unknown cost content."""
    returns = analyze_backtest_result(_run("daily")).daily_returns.to_numpy()
    html = generate_backtest_tearsheet(returns=returns, n_trials=1)

    assert not re.search(r'<(?:div|details)\b[^>]*\bdata-section="cost_sensitivity"', html)


def test_plot_requires_the_caller_to_state_what_the_returns_contain():
    """Guards against silently treating any series as gross at 252 trades a year."""
    profile = analyze_backtest_result(_run("daily"))
    returns = profile.daily_returns

    with pytest.raises(ValueError, match="realized_costs"):
        plot_cost_sensitivity(returns)
    with pytest.raises(ValueError, match="trades_per_year"):
        plot_cost_sensitivity(returns, base_costs_bps=10.0)
    with pytest.raises(ValueError, match="trades_per_year"):
        plot_cost_sensitivity(returns, realized_costs=profile.cost_returns, trades_per_year=12)

    fig = plot_cost_sensitivity(returns, realized_costs=profile.cost_returns)
    assert list(fig.data[0].x) == [0, 0.5, 1, 1.5, 2, 3, 5]
    assert fig.data[0].y[2] == pytest.approx(profile.performance["metrics"]["sharpe_ratio"])


def test_declared_gross_costs_compound_cagr_from_the_adjusted_series():
    """Guards against CAGR approximated as (1 + mean) ** periods - 1."""
    gross = analyze_backtest_result(_run("daily", 0.0)).daily_returns.to_numpy()
    fig = plot_cost_sensitivity(gross, base_costs_bps=10.0, trades_per_year=52)

    drag = 10.0 / 10_000 * 52 / 252
    for multiple, cagr in zip([0, 0.5, 1, 1.5, 2, 3, 5], fig.data[1].y, strict=True):
        adjusted = gross - multiple * drag
        expected = np.prod(1 + adjusted) ** (252 / len(adjusted)) - 1
        assert cagr == pytest.approx(expected * 100, rel=1e-9)


@pytest.mark.native_browser
@pytest.mark.parametrize("viewport", [1400, 390])
def test_cost_panel_axis_starts_at_zero_and_fits_its_column(tmp_path, viewport):
    """Guards against an x axis autoscaled below 0 bps and a clipped CAGR subplot."""
    from choreographer.browsers.chromium import Chromium
    from playwright import sync_api as playwright

    destination = tmp_path / "tearsheet.html"
    destination.write_text(
        generate_backtest_tearsheet(profile=analyze_backtest_result(_run("daily")), n_trials=1)
    )
    local_plotly = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    chrome = Chromium.find_browser(skip_local=False) or shutil.which("chromium")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=chrome or runtime.chromium.executable_path,
            args=["--no-sandbox"],
        )
        try:
            page = browser.new_page(viewport={"width": viewport, "height": 1000})
            page.route("https://cdn.plot.ly/**", lambda route: route.fulfill(path=local_plotly))
            page.goto(destination.as_uri(), wait_until="load")
            page.locator('.workspace-tab:has-text("Trading")').first.click()
            page.wait_for_function(
                """() => document.querySelector(
                    '.report-section[data-section="cost_sensitivity"] .js-plotly-plot'
                )?._fullLayout?.xaxis2"""
            )
            # Activating the tab calls Plotly.Plots.resize, which is debounced, so the
            # figure keeps its hidden-panel width briefly. Wait for it to settle; a panel
            # that never fits times out here and the assertion below reports the overflow.
            with contextlib.suppress(playwright.TimeoutError):
                page.wait_for_function(
                    """() => {
                        const container = document.querySelector(
                            '.report-section[data-section="cost_sensitivity"] .chart-container'
                        );
                        return container.scrollWidth <= container.clientWidth;
                    }""",
                    timeout=3000,
                )
            geometry = page.evaluate(
                """() => {
                    const section = '.report-section[data-section="cost_sensitivity"]';
                    const container = document.querySelector(section + ' .chart-container');
                    const plot = container.querySelector('.js-plotly-plot');
                    const layout = plot._fullLayout;
                    return {
                        x_min: [layout.xaxis.range[0], layout.xaxis2.range[0]],
                        overflow_x: container.scrollWidth - container.clientWidth,
                        overflow_y: container.scrollHeight - container.clientHeight,
                    };
                }"""
            )
        finally:
            browser.close()

    rendered = {
        "x_axis_starts_at_or_above_zero": min(geometry["x_min"]) >= 0,
        "fits_width": geometry["overflow_x"] <= 0,
        "fits_height": geometry["overflow_y"] <= 0,
    }
    assert rendered == dict.fromkeys(rendered, True), geometry
