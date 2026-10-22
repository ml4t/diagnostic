"""Actual offline attribution reports must keep readable values, themes and labels."""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import tears


@pytest.mark.parametrize("width", [375, 1280])
@pytest.mark.parametrize("theme", ["default", "dark"])
def test_offline_sheet_renders_full_tables_paths_and_unclipped_typography(width, theme, tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Actual Chrome/Chromium required for report browser qualification")
    playwright = pytest.importorskip("playwright.sync_api")
    dates = pd.DatetimeIndex(["2025-07-07T20:00Z", "2025-07-08T20:00Z"])
    prices = pd.Series([100.0, 103.0, 101.97], index=dates[:1].shift(-1, freq="B").append(dates))
    returns = prices.pct_change(fill_method=None).iloc[1:]
    positions = pd.DataFrame({"A": [500.0, 250.0], "B": [500.0, 750.0], "cash": 0.0}, index=dates)
    factors = pd.DataFrame({"market": [0.01, -0.02], "style": [0.02, 0.01]}, index=dates)
    loadings = pd.DataFrame(
        {"market": [2.0, 1.0, 1.0, 1.0], "style": [0.0, 0.5, 2.0, 0.0]},
        index=pd.MultiIndex.from_product([dates, ["A", "B"]], names=["dt", "ticker"]),
    )
    path = tmp_path / "attribution.html"
    tears.create_perf_attrib_tear_sheet(
        returns,
        positions,
        factors,
        loadings,
        factor_partitions=None,
        theme=theme,
        output_file=path,
        periods_per_year=12,
    )
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            context = browser.new_context(viewport={"width": width, "height": 1000}, offline=True)
            page = context.new_page()
            page.goto(path.as_uri(), wait_until="load")
            page.wait_for_function("""() => {
                const plots=Array.from(document.querySelectorAll('.js-plotly-plot'));
                return plots.length===3 && plots.every(p=>p._fullLayout &&
                    Math.abs(p._fullLayout.width-p.getBoundingClientRect().width)<1);
            }""")
            actual = page.evaluate("""() => {
                const overlap=(a,b)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>1 &&
                    Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1;
                const prose=getComputedStyle(document.querySelector('.section-text p'));
                return {width:innerWidth,scroll:document.documentElement.scrollWidth,
                    proseSize:parseFloat(prose.fontSize),proseFamily:prose.fontFamily,
                    background:getComputedStyle(document.body).backgroundColor,
                    tables:document.querySelectorAll('.attribution-statistics table').length,
                    plots:Array.from(document.querySelectorAll('.js-plotly-plot')).map(p=>{
                        const box=p.getBoundingClientRect();
                        const labels=Array.from(p.querySelectorAll('.xtick text,.ytick text,.legendtext,.xtitle,.ytitle'));
                        const axes=Array.from(p.querySelectorAll('.xtick text,.xtitle'));
                        return {text:labels.map(t=>({size:parseFloat(getComputedStyle(t).fontSize),family:getComputedStyle(t).fontFamily})),
                            clipped:labels.some(t=>{const b=t.getBoundingClientRect();return b.left<box.left-2 || b.right>box.right+2 || b.top<box.top-2 || b.bottom>box.bottom+2}),
                            overlap:Array.from(p.querySelectorAll('.legendtext')).some(t=>axes.some(a=>overlap(t.getBoundingClientRect(),a.getBoundingClientRect()))),
                            traces:p.data.map((t,i)=>({name:t.name,y:p.calcdata[i].map(point=>point.y)}))};
                    })};
            }""")
            screenshots = os.environ.get("ML4T_ATTRIBUTION_SCREENSHOTS")
            if screenshots:
                target = Path(screenshots)
                target.mkdir(parents=True, exist_ok=True)
                page.locator(".section-text").scroll_into_view_if_needed()
                page.screenshot(
                    path=str(target / f"attribution-{theme}-{width}.png"), full_page=True
                )
        finally:
            browser.close()
    assert actual["scroll"] <= actual["width"] == width
    assert actual["tables"] == 2 and len(actual["plots"]) == 3
    assert actual["background"] == ("rgb(10, 22, 40)" if theme == "dark" else "rgb(250, 250, 249)")
    for plot in actual["plots"]:
        assert not plot["clipped"] and not plot["overlap"]
        for label in plot["text"]:
            assert 14 <= label["size"] <= actual["proseSize"] * 1.1
            assert label["family"] == actual["proseFamily"]
    np.testing.assert_allclose(actual["plots"][0]["traces"][0]["y"], [0.03, 0.0197], atol=1e-15)
