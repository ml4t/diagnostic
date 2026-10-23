"""Complete saved return reports must remain usable offline at phone width."""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import pyfolio as pf


@pytest.mark.parametrize("width", [375, 1280])
def test_whole_returns_report_keeps_actual_values_readable_and_errors_absent(width, tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("Actual Chrome/Chromium required for offline report qualification")
    playwright = pytest.importorskip("playwright.sync_api")
    dates = pd.date_range("2025-01-01", periods=4, tz="UTC")
    prices = pd.Series([100.0, 110.0, 99.0, 103.95], index=dates)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    report = pf.create_returns_tear_sheet(returns, return_fig=True)
    target = tmp_path / "returns.html"
    report.save_html(target, include_plotlyjs=True)
    errors = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            context = browser.new_context(viewport={"width": width, "height": 1000}, offline=True)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(target.as_uri(), wait_until="load")
            page.wait_for_function("""() => {
                const plots=[...document.querySelectorAll('.js-plotly-plot')];
                return plots.length===14 && plots.every(p=>p._fullLayout && p.calcdata);
            }""")
            actual = page.evaluate("""() => ({
                width:innerWidth,scroll:document.documentElement.scrollWidth,
                prose:parseFloat(getComputedStyle(document.querySelector('.metrics-summary p')).fontSize),
                proseFamily:getComputedStyle(document.querySelector('.metrics-summary p')).fontFamily,
                plots:[...document.querySelectorAll('body > .plot-container')].map(section=>{
                    const p=section.querySelector('.js-plotly-plot');
                    const first=p.data[0] || {};
                    const b=p.getBoundingClientRect();
                    const labels=[...p.querySelectorAll('.xtick text,.ytick text,.legendtext,.xtitle,.ytitle,.cell-text,.heatmaplayer text,.annotation-text')];
                    const heatmap=[...p.querySelectorAll('.heatmaplayer text')];
                    return {name:section.querySelector('h2').textContent,
                        sizes:labels.map(t=>parseFloat(getComputedStyle(t).fontSize)),
                        families:[...new Set(labels.map(t=>getComputedStyle(t).fontFamily))],
                        clipped:labels.some(t=>{const a=t.getBoundingClientRect();return a.left<b.left-2 || a.right>b.right+2 || a.top<b.top-2 || a.bottom>b.bottom+2}),
                        tableFit:first.type!=='table' || p._fullLayout._size.h >= first.cells.values[0].length * first.cells.height + first.header.height,
                        heatmapLabels:heatmap.length,
                        heatmapFit:heatmap.every(t=>t.getBoundingClientRect().width <= p._fullLayout._size.w / p.data[0].x.length),
                        y:first.y,table:first.type==='table' ? first.cells.values : null};
                })
            })""")
            screenshots = os.environ.get("ML4T_RETURNS_SCREENSHOTS")
            if screenshots:
                directory = Path(screenshots)
                directory.mkdir(parents=True, exist_ok=True)
                for number in (0, 2, 3, 10, 13):
                    page.locator("body > .plot-container").nth(number).screenshot(
                        path=str(directory / f"returns-{width}-{number}.png")
                    )
        finally:
            browser.close()
    assert not errors
    assert actual["scroll"] <= actual["width"] == width
    print(
        {
            "width": width,
            "prose_size": actual["prose"],
            "prose_family": actual["proseFamily"],
            "label_sizes": sorted({size for plot in actual["plots"] for size in plot["sizes"]}),
            "label_families": sorted(
                {family for plot in actual["plots"] for family in plot["families"]}
            ),
        }
    )
    for plot in actual["plots"]:
        assert not plot["clipped"], plot["name"]
        assert plot["tableFit"], plot["name"]
        assert plot["heatmapFit"], plot["name"]
        if plot["name"] == "Monthly returns":
            assert plot["heatmapLabels"] > 0
        assert all(14 <= size <= actual["prose"] * 1.1 for size in plot["sizes"]), plot["name"]
    np.testing.assert_allclose(actual["plots"][2]["y"], [1.1, 0.99, 1.0395])
    assert "3.95%" in actual["plots"][0]["table"][1]
