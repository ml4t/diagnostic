"""Actual information reports keep every horizon, visible legend and readable leaf label."""

import json
import os
import runpy
import shutil
from pathlib import Path

import pytest


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("width", [375, 1200])
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("theme", ["default", "dark", "print", "presentation"])
def test_complete_information_panels_render_actual_samples_without_hidden_or_clipped_labels(
    grouped, width, constant, theme, tmp_path
):
    playwright = pytest.importorskip("playwright.sync_api")
    import numpy as np
    import pandas as pd
    from plotly.offline import get_plotlyjs

    from ml4t.diagnostic.signal import analyze_signal_information, prepare_signal_panel
    from ml4t.diagnostic.visualization.signal import generate_signal_information_report

    chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if chrome is None:
        pytest.skip("Actual Chrome is required for rendered information-report qualification")
    recipe = Path(__file__).resolve().parents[2] / "examples/legacy_migration_comparison.py"
    factor, forward, groups = runpy.run_path(str(recipe))["_signal_information_inputs"]()
    if constant:
        dates = pd.bdate_range("2025-01-02", periods=30, tz="UTC")
        assets = [f"asset-{index:02}" for index in range(12)]
        prices = pd.DataFrame(
            64 * (1 + np.arange(1, 13)[None, :] / 64) ** np.arange(30)[:, None],
            index=dates,
            columns=assets,
        )
        factor = (
            pd.DataFrame(np.tile(np.arange(1, 13), (25, 1)), index=dates[:25], columns=assets)
            .stack()
            .rename("factor")
        )
        factor.index = factor.index.set_names(["date", "asset"])
        forward = pd.DataFrame(
            {f"{period}D": (prices.shift(-period) / prices - 1).stack() for period in [1, 5]}
        ).reindex(factor.index)
        groups = {asset: "small" if index < 6 else "large" for index, asset in enumerate(assets)}
    result = analyze_signal_information(
        prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0),
        group_adjust=not constant,
        by_group=grouped,
    )
    target = tmp_path / "information.html"
    generate_signal_information_report(result, output_file=target, theme=theme)
    errors = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": width, "height": 1100})
            page.route(
                "https://cdn.plot.ly/*",
                lambda route: route.fulfill(
                    content_type="application/javascript", body=get_plotlyjs()
                ),
            )
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(target.as_uri(), wait_until="networkidle")
            page.wait_for_selector("#plot-6 .gtitle")
            measured = page.evaluate("""() => {
                const plots=Array.from(document.querySelectorAll('.js-plotly-plot'));
                const visible=Array.from(document.querySelectorAll('svg text,td,th')).filter(node=>{const b=node.getBoundingClientRect();return b.width>0&&b.height>0&&getComputedStyle(node).visibility!=='hidden'});
                const prose=document.querySelector('.section-text');
                return {width:innerWidth,scroll:document.documentElement.scrollWidth,
                    panels:plots.map(plot=>{const bounds=plot.getBoundingClientRect();return {
                        title:plot.layout.title.text,
                        titleInside:(()=>{const title=plot.querySelector('.gtitle').getBoundingClientRect();return title.left>=bounds.left&&title.right<=bounds.right})(),
                        annotationOverlap:(()=>{const boxes=Array.from(plot.querySelectorAll('.annotation-text')).map(node=>node.getBoundingClientRect());return boxes.some((a,i)=>boxes.slice(i+1).some(b=>a.left<b.right&&b.left<a.right&&a.top<b.bottom&&b.top<a.bottom))})(),
                        annotationsInside:Array.from(plot.querySelectorAll('.annotation-text')).every(node=>{const box=node.getBoundingClientRect();return box.left>=bounds.left&&box.right<=bounds.right&&box.top>=bounds.top&&box.bottom<=bounds.bottom}),
                        annotationsOverlapDecorations:(()=>{const boxes=Array.from(plot.querySelectorAll('.annotation-text')).map(node=>node.getBoundingClientRect()),decorations=Array.from(plot.querySelectorAll('.g-xtitle,.g-ytitle,.legendtext')).map(node=>node.getBoundingClientRect());return boxes.some(a=>decorations.some(b=>a.left<b.right&&b.left<a.right&&a.top<b.bottom&&b.top<a.bottom))})(),
                        availabilityLegendGap:(()=>{const note=Array.from(plot.querySelectorAll('.annotation-text')).find(node=>node.textContent.includes('KDE unavailable')),labels=Array.from(plot.querySelectorAll('.legendtext'));return note&&labels.length?Math.min(...labels.map(label=>label.getBoundingClientRect().top-note.getBoundingClientRect().bottom)):null})(),
                        heatmapLabels:(()=>{const boxes=Array.from(plot.querySelectorAll('.heatmaplayer text')).filter(node=>node.textContent).map(node=>node.getBoundingClientRect());return {count:boxes.length,overlap:boxes.some((a,i)=>boxes.slice(i+1).some(b=>a.left<b.right&&b.left<a.right&&a.top<b.bottom&&b.top<a.bottom))}})(),
                        traces:plot.data.map(trace=>({name:trace.name,x:trace.x,y:trace.y})),
                        legends:Array.from(plot.querySelectorAll('.legendtext')).map(node=>{const box=node.getBoundingClientRect();return {text:node.textContent,inside:box.left>=bounds.left&&box.right<=bounds.right&&box.top>=bounds.top&&box.bottom<=bounds.bottom}})
                    }}),
                    labelSizes:visible.map(node=>parseFloat(getComputedStyle(node).fontSize)),
                    labelFamilies:Array.from(new Set(visible.map(node=>getComputedStyle(node).fontFamily))),
                    titleSizes:Array.from(document.querySelectorAll('.gtitle')).map(node=>parseFloat(getComputedStyle(node).fontSize)),
                    proseSize:parseFloat(getComputedStyle(prose).fontSize),proseFamily:getComputedStyle(prose).fontFamily};
            }""")
            table = page.locator("#section-0 table")
            table.evaluate("node => {node.parentElement.scrollLeft=node.parentElement.scrollWidth}")
            table_geometry = table.locator("th").last.evaluate(
                """node => {const container=node.closest('table').parentElement,
                    parent=container.getBoundingClientRect(),cell=node.getBoundingClientRect(),
                    range=document.createRange();range.selectNodeContents(node);
                    const text=range.getBoundingClientRect();return {
                        parent:[parent.left,parent.right],cell:[cell.left,cell.right],
                        text:[text.left,text.right],scrollLeft:container.scrollLeft,
                        scrollWidth:container.scrollWidth,clientWidth:container.clientWidth,
                        visible:getComputedStyle(node).visibility!=='hidden'&&text.width>0&&text.left>=parent.left&&text.right<=parent.right};} """
            )
            measured["tableGeometry"] = table_geometry
            if os.environ.get("GITHUB_ACTIONS") == "true":
                output = Path(os.environ["GITHUB_WORKSPACE"]) / "artifacts/information-reports"
                output.mkdir(parents=True, exist_ok=True)
                prefix = (
                    f"information-{theme}-constant-{int(constant)}-grouped-{int(grouped)}-{width}"
                )
                page.screenshot(path=str(output / f"{prefix}.png"), full_page=True)
                for index in range(len(measured["panels"]) + 1):
                    page.locator(f"#section-{index}").screenshot(
                        path=str(output / f"{prefix}-section-{index}.png")
                    )
                (output / f"{prefix}.json").write_text(json.dumps(measured, indent=2))
        finally:
            browser.close()
    assert not errors
    assert table_geometry["visible"], table_geometry
    assert measured["width"] == measured["scroll"] == width
    assert len(measured["panels"]) == (7 if grouped else 8)
    assert min(measured["labelSizes"]) >= 14
    assert max(measured["titleSizes"]) <= measured["proseSize"]
    assert all(panel["titleInside"] for panel in measured["panels"]), measured
    assert not any(panel["annotationOverlap"] for panel in measured["panels"]), measured
    assert all(panel["annotationsInside"] for panel in measured["panels"]), measured
    assert not any(panel["annotationsOverlapDecorations"] for panel in measured["panels"]), measured
    assert all(
        panel["availabilityLegendGap"] is None or panel["availabilityLegendGap"] >= 14
        for panel in measured["panels"]
    ), measured
    if not grouped:
        assert all(panel["heatmapLabels"]["count"] > 0 for panel in measured["panels"][-2:])
        assert not any(panel["heatmapLabels"]["overlap"] for panel in measured["panels"][-2:]), (
            measured
        )
    assert all(label["inside"] for panel in measured["panels"] for label in panel["legends"])
    for position in [0, 3]:
        assert len(measured["panels"][position]["traces"][0]["x"]) == (25 if constant else 32)
        assert [label["text"] for label in measured["panels"][position]["legends"]] == [
            "Observed IC",
            "22-Observation Rolling Mean",
        ]
