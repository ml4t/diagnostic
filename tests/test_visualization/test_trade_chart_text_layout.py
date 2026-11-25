"""Text on the tearsheet's trade charts must stay readable once rendered.

Guards three defects seen in a tearsheet built from a real ml4t-backtest run:

- execution quality: the Median and Mean labels sat on the histogram bars;
- trade duration: the Median label overlapped the x tick labels, and the Winners/Losers
  histogram had no legend;
- MFE/MAE: the y-axis title was longer than the plot and clipped at both ends, and the
  Edge ratio box covered a quadrant label.

The geometry only exists after Plotly lays the figures out, so the checks run in Chromium.
"""

from __future__ import annotations

import contextlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly
import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import analyze_backtest_result
from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

SECTIONS = ("execution_quality", "duration", "mfe_mae")


class _Rotate(Strategy):
    """Hold the asset with the best 5-bar return, re-ranked every 5 bars."""

    def __init__(self) -> None:
        self.bar = 0
        self.closes: dict[str, list[float]] = {}

    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            self.closes.setdefault(asset, []).append(bar["close"])
        self.bar += 1
        if self.bar < 6 or self.bar % 5:
            return
        pick = max(self.closes, key=lambda a: self.closes[a][-1] / self.closes[a][-6])
        for asset in list(broker.get_positions()):
            if asset != pick:
                broker.close_position(asset)
        if broker.get_position(pick) is None:
            broker.submit_order(pick, int(0.95 * broker.get_account_value() / data[pick]["close"]))


def _tearsheet(path: Path) -> None:
    rng = np.random.default_rng(7)
    start = datetime(2022, 1, 3)
    rows = []
    for asset in ("AAA", "BBB", "CCC", "DDD"):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 500)))
        for i, c in enumerate(close):
            o = c * (1 + rng.normal(0, 0.004))
            rows.append(
                {
                    "timestamp": start + timedelta(days=i),
                    "asset": asset,
                    "open": o,
                    "high": max(o, c) * 1.004,
                    "low": min(o, c) * 0.996,
                    "close": c,
                    "volume": 1e6,
                }
            )
    config = BacktestConfig(initial_cash=100_000, commission_rate=0.0005, slippage_rate=0.0005)
    result = Engine(DataFeed(prices_df=pl.DataFrame(rows)), _Rotate(), config).run()
    path.write_text(
        generate_backtest_tearsheet(
            profile=analyze_backtest_result(result), template="full", n_trials=1
        ),
        encoding="utf-8",
    )


@pytest.mark.native_browser
def test_trade_chart_text_does_not_collide_or_clip(tmp_path):
    from choreographer.browsers.chromium import Chromium
    from playwright import sync_api as playwright

    destination = tmp_path / "tearsheet.html"
    _tearsheet(destination)
    local_plotly = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    chrome = Chromium.find_browser(skip_local=False) or shutil.which("chromium")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=chrome or runtime.chromium.executable_path,
            args=["--no-sandbox"],
        )
        try:
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            page.route("https://cdn.plot.ly/**", lambda route: route.fulfill(path=local_plotly))
            page.goto(destination.as_uri(), wait_until="load")
            page.locator('.workspace-tab:has-text("Trading")').first.click()
            for section in SECTIONS:
                page.wait_for_function(
                    f"""() => document.querySelector(
                        '.report-section[data-section="{section}"] .js-plotly-plot'
                    )?._fullLayout"""
                )
            # Tab activation resizes the figures through a debounced call; wait until each
            # figure has taken its container's width before measuring.
            with contextlib.suppress(playwright.TimeoutError):
                page.wait_for_function(
                    """(sections) => sections.every((s) => {
                        const plot = document.querySelector(
                            `.report-section[data-section="${s}"] .js-plotly-plot`);
                        return Math.abs(plot._fullLayout.width - plot.clientWidth) <= 1;
                    })""",
                    arg=list(SECTIONS),
                    timeout=3000,
                )
            geometry = page.evaluate(
                """(sections) => {
                    const box = (el) => {
                        const r = el.getBoundingClientRect();
                        return {left: r.left, right: r.right, top: r.top, bottom: r.bottom,
                                text: el.textContent};
                    };
                    const out = {};
                    for (const s of sections) {
                        const plot = document.querySelector(
                            `.report-section[data-section="${s}"] .js-plotly-plot`);
                        const svg = plot.querySelector('.main-svg');
                        const q = (sel) => [...plot.querySelectorAll(sel)]
                            .filter((el) => el.getBoundingClientRect().width > 0)
                            .map(box);
                        out[s] = {
                            svg: box(svg),
                            annotations: q('.annotation-text-g text'),
                            ticks: q('.xtick text, .ytick text'),
                            titles: q('.g-xtitle text, .g-ytitle text'),
                            bars: q('.trace.bars path'),
                            legend: q('.legend .legendtext').map((b) => b.text),
                            visible_traces: plot._fullData.filter(
                                (t) => t.visible === true && t.type === 'histogram').length,
                        };
                    }
                    return out;
                }""",
                list(SECTIONS),
            )
        finally:
            browser.close()

    def overlap(a, b, pad=1.0):
        return (
            a["left"] < b["right"] - pad
            and b["left"] < a["right"] - pad
            and a["top"] < b["bottom"] - pad
            and b["top"] < a["bottom"] - pad
        )

    problems = []
    for section, g in geometry.items():
        anns = g["annotations"]
        for i, a in enumerate(anns):
            problems += [
                f"{section}: '{a['text']}' overlaps '{b['text']}'"
                for b in anns[i + 1 :]
                if overlap(a, b)
            ]
            problems += [
                f"{section}: '{a['text']}' overlaps tick '{t['text']}'"
                for t in g["ticks"]
                if overlap(a, t)
            ]
            if section != "mfe_mae":  # scatter quadrant labels sit among points by design
                problems += [
                    f"{section}: '{a['text']}' sits on a bar" for b in g["bars"] if overlap(a, b)
                ]
        svg = g["svg"]
        for t in g["titles"]:
            if t["top"] < svg["top"] or t["bottom"] > svg["bottom"] or t["left"] < svg["left"]:
                problems.append(f"{section}: axis title '{t['text']}' is clipped")
        if g["visible_traces"] > 1 and len(g["legend"]) < g["visible_traces"]:
            problems.append(f"{section}: {g['visible_traces']} histograms but legend {g['legend']}")

    assert all(g["annotations"] or g["titles"] for g in geometry.values()), geometry
    assert problems == [], "\n".join(problems)
