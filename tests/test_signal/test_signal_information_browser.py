"""Actual information reports keep every horizon, visible legend and readable leaf label."""

import json
import os
import runpy
import shutil
from pathlib import Path

import pytest


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("width", [375, 1200])
def test_complete_information_panels_render_actual_samples_without_hidden_or_clipped_labels(
    grouped, width, tmp_path
):
    playwright = pytest.importorskip("playwright.sync_api")
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
    result = analyze_signal_information(
        prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0),
        group_adjust=True,
        by_group=grouped,
    )
    target = tmp_path / "information.html"
    generate_signal_information_report(result, output_file=target)
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
            final_header_visible = table.locator("th").last.evaluate(
                "node => {const box=node.getBoundingClientRect(),parent=node.closest('table').parentElement.getBoundingClientRect();return box.left>=parent.left&&box.right<=parent.right}"
            )
            assert final_header_visible
            if os.environ.get("GITHUB_ACTIONS") == "true":
                output = Path(os.environ["GITHUB_WORKSPACE"]) / "artifacts/information-reports"
                output.mkdir(parents=True, exist_ok=True)
                prefix = f"information-grouped-{int(grouped)}-{width}"
                page.screenshot(path=str(output / f"{prefix}.png"), full_page=True)
                for index in range(len(measured["panels"]) + 1):
                    page.locator(f"#section-{index}").screenshot(
                        path=str(output / f"{prefix}-section-{index}.png")
                    )
                (output / f"{prefix}.json").write_text(json.dumps(measured, indent=2))
        finally:
            browser.close()
    assert not errors
    assert measured["width"] == measured["scroll"] == width
    assert len(measured["panels"]) == (7 if grouped else 8)
    assert min(measured["labelSizes"]) >= 14
    assert max(measured["titleSizes"]) <= measured["proseSize"]
    assert all(label["inside"] for panel in measured["panels"] for label in panel["legends"])
    for position in [0, 3]:
        assert len(measured["panels"][position]["traces"][0]["x"]) == 32
        assert [label["text"] for label in measured["panels"][position]["legends"]] == [
            "Observed IC",
            "22-Observation Rolling Mean",
        ]
