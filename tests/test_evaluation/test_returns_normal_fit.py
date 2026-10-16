"""Real account returns must not imply a normal fit without sample dispersion."""

import math
import shutil
import warnings
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import (
    create_portfolio_dashboard,
    plot_returns_distribution,
)


class _BuyOnce(Strategy):
    def __init__(self):
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            broker.submit_order("A", 4)
            self.submitted = True


def _produced_returns(case):
    prices, tail = {
        "flat": ([100.0] * 5, 4),
        "constant_positive": ([100.0, 100.0, 125.0, 152.5, 182.75], 3),
        "single": ([100.0], 1),
        "varying": ([100.0, 100.0, 110.0, 90.0], 4),
        "tiny_varying": ([100.0, 100.0, 100.0000000001, 100.0], 4),
    }[case]
    dates = [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6, 7, 8]][: len(prices)]
    bars = pl.DataFrame(
        dict(
            timestamp=dates,
            asset=["A"] * len(prices),
            **dict.fromkeys(["open", "high", "low", "close"], prices),
            volume=[10000.0] * len(prices),
        )
    )
    result = Engine(DataFeed(prices_df=bars), _BuyOnce(), BacktestConfig(initial_cash=1000)).run()
    account = result.to_equity_dataframe().tail(tail)
    if case == "constant_positive":
        np.testing.assert_array_equal(account["equity"], [1100, 1210, 1331])
        np.testing.assert_array_equal(account["return"], [0.1, 0.1, 0.1])
    return PortfolioAnalysis(account["return"], dates=account["timestamp"])


@pytest.mark.parametrize("case", ["flat", "constant_positive", "single"])
def test_real_degenerate_returns_explain_unavailable_fit_without_nan_or_roundoff_curve(case):
    analysis = _produced_returns(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_returns_distribution(analysis)
    assert not [trace for trace in fig.data if trace.name.startswith("Normal")]
    np.testing.assert_array_equal(fig.data[0].x, analysis.returns)
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    availability = fig.layout.meta["normal_fit"]
    assert availability["status"] == "unavailable"
    assert ("two observations" if case == "single" else "zero dispersion") in availability["reason"]
    assert any("Normal fit unavailable" in note.text for note in fig.layout.annotations)
    assert "Normal fit unavailable" in fig.layout.meta["description"]


def test_disabling_normal_fit_is_not_reported_as_unavailable():
    fig = plot_returns_distribution(_produced_returns("flat"), show_normal=False)
    assert len(fig.data) == 1
    assert fig.layout.meta["normal_fit"] == {"status": "not_requested", "reason": None}
    assert not any("unavailable" in note.text for note in fig.layout.annotations)


def test_real_varying_returns_retain_unannualized_sample_normal_density():
    analysis = _produced_returns("varying")
    # Independent account endpoints: 1000, 1000, 1040, 960.
    sample = [0.0, 0.0, 0.04, -80 / 1040]
    np.testing.assert_allclose(analysis.returns, sample, rtol=0, atol=1e-15)
    mean = math.fsum(sample) / 4
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in sample) / 3)
    fig = plot_returns_distribution(analysis)
    (curve,) = [trace for trace in fig.data if trace.name.startswith("Normal")]
    assert np.isfinite(curve.y).all()
    assert curve.x[0] == min(sample) and curve.x[-1] == max(sample)
    expected = [
        math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi)) for x in curve.x
    ]
    np.testing.assert_allclose(curve.y, expected, rtol=1e-13)
    assert fig.layout.meta["normal_fit"] == {"status": "available", "reason": None}


def test_tiny_genuine_account_changes_are_not_suppressed_by_a_dispersion_tolerance():
    analysis = _produced_returns("tiny_varying")
    assert 0 < np.ptp(analysis.returns) < 1e-10
    fig = plot_returns_distribution(analysis)
    (curve,) = [trace for trace in fig.data if trace.name.startswith("Normal")]
    assert np.isfinite(curve.y).all() and max(curve.y) > 0
    assert fig.layout.meta["normal_fit"] == {"status": "available", "reason": None}


@pytest.mark.parametrize("case", ["flat", "constant_positive"])
def test_dashboard_html_and_export_share_the_actual_unavailable_fit_reason(case, tmp_path):
    report = create_portfolio_dashboard(
        _produced_returns(case), include_positions=False, include_transactions=False
    )
    info = report.availability["normal_fit"]
    assert info["status"] == "unavailable" and "zero dispersion" in info["reason"]
    assert report.to_dict()["availability"]["normal_fit"] == info
    output = tmp_path / "normal-fit.html"
    report.save_html(output, include_plotlyjs=True)
    assert info["reason"] in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("width", [1280, 375])
def test_actual_html_displays_fit_reason_without_clipping_or_shrinking_notes(width, tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium required for actual report checks")
    playwright = pytest.importorskip("playwright.sync_api")
    report = create_portfolio_dashboard(
        _produced_returns("flat"), include_positions=False, include_transactions=False
    )
    output = tmp_path / "unavailable-fit.html"
    report.save_html(output, include_plotlyjs=True)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            page = browser.new_page(
                viewport={"width": width, "height": 1000},
                is_mobile=width == 375,
                has_touch=width == 375,
                offline=True,
            )
            page.goto(output.as_uri(), wait_until="load")
            page.wait_for_function(
                "Array.from(document.querySelectorAll('.js-plotly-plot')).every(p=>p._fullLayout)"
            )
            panel = page.locator(".plot-container").filter(
                has=page.get_by_role("heading", name="Returns Distribution", exact=True)
            )
            assert "zero dispersion" in panel.inner_text()
            actual = panel.locator(".js-plotly-plot").evaluate("""p=>({
                normal:p._fullData.filter(t=>t.name.startsWith('Normal')).length,
                annotations:Array.from(p.querySelectorAll('.annotation-text'))
                    .filter(t=>t.textContent.includes('Normal fit unavailable'))
                    .map(t=>({text:t.textContent,size:parseFloat(getComputedStyle(t).fontSize),
                        left:t.getBoundingClientRect().left,right:t.getBoundingClientRect().right})),
                body:parseFloat(getComputedStyle(document.body).fontSize),
                overflow:document.documentElement.scrollWidth>innerWidth
            })""")
            assert actual["normal"] == 0 and actual["body"] == 16
            (note,) = actual["annotations"]
            assert note["size"] >= 13 and "Zero observed dispersion" in note["text"]
            assert 0 <= note["left"] < note["right"] <= width
            assert not actual["overflow"]
        finally:
            browser.close()
