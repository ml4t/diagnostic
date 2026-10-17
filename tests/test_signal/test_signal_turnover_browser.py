"""Actual report readers must see every selected lag and legible tables/charts."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_signal_turnover_report import captured_turnover

from ml4t.diagnostic.results.signal_results import SignalTearSheet


@pytest.mark.parametrize("profile", ["rotating", "constant"])
@pytest.mark.parametrize("width", [375, 1200])
@pytest.mark.parametrize("mode", ["standalone", "dashboard"])
def test_all_recorded_lags_are_visible_readable_and_bound_to_the_owned_price_sample(
    profile, width, mode, tmp_path
):
    from plotly.offline import get_plotlyjs

    from ml4t.diagnostic.visualization.signal import (
        SignalDashboard,
        generate_signal_turnover_report,
    )

    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("Actual Chrome/Chromium required for report qualification")
    playwright = pytest.importorskip("playwright.sync_api")
    _, component, dates = captured_turnover(profile)
    path = tmp_path / "turnover.html"
    if mode == "standalone":
        generate_signal_turnover_report(component, output_file=path)
    else:
        sheet = SignalTearSheet(
            turnover_analysis=component,
            n_assets=4,
            n_dates=12,
            date_range=(dates[0].isoformat(), dates[-1].isoformat()),
        )
        path.write_text(SignalDashboard().generate(sheet))
    errors = []
    evidence = Path("artifacts/turnover-reports")
    evidence.mkdir(parents=True, exist_ok=True)
    prefix = f"{mode}-{profile}-{width}"
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000}, offline=True)
            page.route(
                "https://cdn.plot.ly/*",
                lambda route: route.fulfill(
                    content_type="application/javascript", body=get_plotlyjs()
                ),
            )
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(path.as_uri(), wait_until="networkidle")
            if mode == "dashboard":
                page.get_by_role("button", name="Turnover", exact=True).click()
            scope = "#turnover" if mode == "dashboard" else "main"
            page.wait_for_selector(f"{scope} .js-plotly-plot")
            measured = page.locator(scope).evaluate("""e=>{
                const prose=e.querySelector('p'),style=getComputedStyle(prose);
                const plots=Array.from(e.querySelectorAll('.js-plotly-plot')).map(p=>{
                    const r=p.getBoundingClientRect(),title=p.querySelector('.gtitle'),t=title.getBoundingClientRect();
                    return {kind:p._fullLayout.meta.turnover_method?'turnover':'rank',lag:p._fullLayout.meta.lag,width:r.width,layout_width:p._fullLayout.width,left:r.left,right:r.right,title:{left:t.left-r.left,right:t.right-r.left,bottom:t.bottom-r.top},plot_top:p._fullLayout._size.t,
                        legend:Array.from(p.querySelectorAll('.legendtext')).map(n=>n.textContent),
                        traces:p._fullData.map(d=>({name:d.name,x:d.x,y:d.y,custom:d.customdata})),
                        legendInside:Array.from(p.querySelectorAll('.legendtext')).every(n=>{const b=n.getBoundingClientRect();return b.left>=r.left&&b.right<=r.right&&b.top>=r.top&&b.bottom<=r.bottom}),
                        annotationsInside:Array.from(p.querySelectorAll('.annotation-text')).every(n=>{const b=n.getBoundingClientRect();return b.left>=r.left&&b.right<=r.right&&b.top>=r.top&&b.bottom<=r.bottom}),
                        availabilityInsidePlot:Array.from(p.querySelectorAll('.annotation-text')).filter(n=>n.textContent.includes('unavailable')&&!n.textContent.startsWith('CI')).every(n=>{const b=n.getBoundingClientRect(),s=p._fullLayout._size;return b.left>=r.left+s.l&&b.right<=r.left+s.l+s.w}),
                        text:Array.from(p.querySelectorAll('text')).filter(n=>n.getBoundingClientRect().width>0).map(n=>({text:n.textContent,role:n.closest('.annotation')?'note':'label',size:parseFloat(getComputedStyle(n).fontSize),family:getComputedStyle(n).fontFamily}))};
                });
                const tables=Array.from(e.querySelectorAll('table')).map(t=>({rows:t.querySelectorAll('tbody tr').length,font:parseFloat(getComputedStyle(t.querySelector('td')||t.querySelector('th')).fontSize),headers:Array.from(t.querySelectorAll('th')).map(n=>n.textContent),client:t.parentElement.clientWidth,scroll:t.parentElement.scrollWidth}));
                return {viewport:innerWidth,scroll:document.documentElement.scrollWidth,body_font:parseFloat(style.fontSize),body_family:style.fontFamily,plots,tables};
            }""")
            for index, plot in enumerate(page.locator(f"{scope} .js-plotly-plot").all()):
                container = (
                    plot.locator("xpath=ancestor::section[1]")
                    if mode == "standalone"
                    else plot.locator("xpath=..")
                )
                container.screenshot(path=str(evidence / f"{prefix}-section-{index}.png"))
            page.locator(scope).screenshot(path=str(evidence / f"{prefix}-whole.png"))
            for table in page.locator(f"{scope} table").all():
                table.evaluate("t=>{t.parentElement.scrollLeft=t.parentElement.scrollWidth}")
                bounds = table.evaluate(
                    """t=>{const h=t.querySelector('thead tr').lastElementChild.getBoundingClientRect(),r=t.parentElement.getBoundingClientRect();return {left:h.left,right:h.right,container_left:r.left,container_right:r.right}}"""
                )
                assert (
                    bounds["container_left"] - 1
                    <= bounds["left"]
                    < bounds["right"]
                    <= bounds["container_right"] + 1
                )
        finally:
            browser.close()
    (evidence / f"{prefix}.json").write_text(json.dumps(measured, indent=2) + "\n")
    assert not errors
    assert measured["viewport"] == width and measured["scroll"] <= width
    assert [(plot["kind"], plot["lag"]) for plot in measured["plots"]] == [
        (kind, lag) for kind in ["turnover", "rank"] for lag in [1, 2, 5]
    ]
    for plot in measured["plots"]:
        assert abs(plot["layout_width"] - plot["width"]) <= 1
        assert 0 <= plot["left"] < plot["right"] <= width
        assert 0 <= plot["title"]["left"] < plot["title"]["right"] <= plot["width"]
        assert plot["title"]["bottom"] <= plot["plot_top"]
        assert plot["legend"] == [trace["name"] for trace in plot["traces"]]
        assert plot["legendInside"] and plot["annotationsInside"] and plot["availabilityInsidePlot"]
        for text in plot["text"]:
            minimum = 13 if text["role"] == "note" and "unavailable" in text["text"].lower() else 14
            assert minimum <= text["size"] <= measured["body_font"]
        for trace in plot["traces"]:
            assert [pd.Timestamp(value).value for value in trace["x"]] == [
                day.value for day in dates
            ]
            lag = plot["lag"]
            assert all(value is None for value in trace["y"][:lag])
            if plot["kind"] == "rank" and profile == "constant":
                assert all(value is None for value in trace["y"])
            else:
                expected = (
                    0
                    if profile == "constant"
                    else (1 if lag == 2 else 0.5)
                    if plot["kind"] == "turnover"
                    else (-1 if lag == 2 else 0)
                )
                np.testing.assert_allclose(trace["y"][lag:], expected, atol=1e-12)
    assert measured["tables"] and all(table["rows"] > 0 for table in measured["tables"])
    assert all(14 <= table["font"] <= measured["body_font"] for table in measured["tables"])
    if mode == "standalone":
        assert [table["rows"] for table in measured["tables"]] == [
            6 if profile == "rotating" else 3,
            3,
            72 if profile == "rotating" else 36,
            36,
        ]
        assert measured["tables"][0]["headers"] == ["lag", "quantile", "turnover", "n_dates"]
