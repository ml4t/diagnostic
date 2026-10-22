"""Saved position reports must keep real values and private identities usable offline."""

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ml4t.diagnostic.compat import pyfolio as pf


@pytest.mark.parametrize("width", [375, 1280])
@pytest.mark.parametrize("hidden", [False, True])
def test_whole_positions_report_keeps_actual_panels_readable_without_leaking_hidden_ids(
    width, hidden, tmp_path
):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("Actual Chrome/Chromium required for offline report qualification")
    playwright = pytest.importorskip("playwright.sync_api")
    dates = pd.bdate_range("2025-01-29", periods=5, tz="UTC")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=dates)
    positions = pd.DataFrame(
        {"PRIVATE_ASSET_A": [60.0] * 5, "PRIVATE_ASSET_B": [-20.0] * 5, "cash": [60.0] * 5},
        index=dates,
    )
    report = pf.create_position_tear_sheet(
        returns,
        positions,
        sector_mappings={"PRIVATE_ASSET_A": "Technology", "PRIVATE_ASSET_B": "Finance"},
        hide_positions=hidden,
        estimate_intraday=False,
        return_fig=True,
    )
    target = tmp_path / "positions.html"
    report.save_html(target, include_plotlyjs=True)
    errors, external = [], []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000}, offline=True)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "request",
                lambda request: (
                    external.append(request.url) if request.url.startswith("http") else None
                ),
            )
            page.goto(target.as_uri(), wait_until="load")
            page.wait_for_function(
                """count => {
                const plots=[...document.querySelectorAll('.js-plotly-plot')];
                return plots.length===count && plots.every(p=>p._fullLayout && p.calcdata);
            }""",
                arg=7 if hidden else 10,
            )
            actual = page.evaluate("""() => ({
                width:innerWidth,scroll:document.documentElement.scrollWidth,
                prose:parseFloat(getComputedStyle(document.querySelector('.metrics-summary p')).fontSize),
                proseFamily:getComputedStyle(document.querySelector('.metrics-summary p')).fontFamily,
                plots:[...document.querySelectorAll('body > .plot-container')].map(section=>{
                    const p=section.querySelector('.js-plotly-plot');
                    const b=p.getBoundingClientRect(), first=p.data[0] || {};
                    const labels=[...p.querySelectorAll('.xtick text,.ytick text,.legendtext,.xtitle,.ytitle,.cell-text,.annotation-text,.gtitle')];
                    return {name:section.querySelector('h2').textContent,
                        sizes:labels.map(t=>parseFloat(getComputedStyle(t).fontSize)),
                        families:[...new Set(labels.map(t=>getComputedStyle(t).fontFamily))],
                        clipped:labels.filter(t=>{const a=t.getBoundingClientRect();return a.left<b.left-2 || a.right>b.right+2 || a.top<b.top-2 || a.bottom>b.bottom+2}).map(t=>t.textContent),
                        tableFit:first.type!=='table' || p._fullLayout._size.h >= first.cells.values[0].length * first.cells.height + first.header.height,
                        y:first.y,table:first.type==='table' ? first.cells.values : null,
                        payload:JSON.stringify(p.data)};
                })
            })""")
            screenshots = os.environ.get("ML4T_RETURNS_SCREENSHOTS")
            if screenshots:
                directory = Path(screenshots)
                directory.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(directory / f"positions-{width}-{hidden}-prose.png"))
                for number in (1, 4, 6):
                    page.locator("body > .plot-container").nth(number).screenshot(
                        path=str(directory / f"positions-{width}-{hidden}-{number}.png")
                    )
        finally:
            browser.close()
    assert not errors and not external
    assert actual["scroll"] <= actual["width"] == width
    for panel in actual["plots"]:
        assert not panel["clipped"], (panel["name"], panel["clipped"])
        assert panel["tableFit"], panel["name"]
        assert all(14 <= size <= actual["prose"] * 1.1 for size in panel["sizes"]), panel["name"]
        if hidden:
            assert "PRIVATE_ASSET" not in panel["payload"]
    assert actual["plots"][0]["y"] == [0.6] * 5
    print(json.dumps({key: value for key, value in actual.items() if key != "plots"}))
    print(
        json.dumps(
            {
                panel["name"]: {"sizes": panel["sizes"], "families": panel["families"]}
                for panel in actual["plots"]
            }
        )
    )
