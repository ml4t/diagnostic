"""Supplied chart workflows must remain readable and keep distinct time keys."""

import importlib.util
import json
import os
import runpy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("width", [375, 1200])
@pytest.mark.parametrize("route", ["group_bar", "wealth", "event_paths", "event_counts"])
def test_direct_chart_preserves_visible_labels_legend_errors_and_exact_time_positions(
    width, route, tmp_path
):
    playwright = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("plotly")
    from ml4t.diagnostic.compat.alphalens import performance, plotting
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    if route == "group_bar":
        recipe = Path(__file__).with_name("test_alphalens_plotting.py")
        data = runpy.run_path(str(recipe))["_panel"]()
        data["group"] = data["group"].astype(str).replace({"red": "information_technology"})
        means, _ = performance.mean_return_by_quantile(data, by_group=True)
        figures = plotting.plot_quantile_returns_bar(means, by_group=True)
    elif route == "event_paths":
        recipe = Path(__file__).with_name("test_alphalens_event_performance.py")
        events, values = runpy.run_path(str(recipe))["_inputs"]()
        average = performance.average_cumulative_return_by_quantile(
            events, values, periods_before=1, periods_after=1, demeaned=False
        )
        figures = [plotting.plot_quantile_average_cumulative_return(average, std_bar=True)]
    else:
        dates = pd.date_range("2025-02-03", periods=3, freq="ns", tz="UTC")
        periodic = pd.Series([0.01, 0.02, -0.03], index=dates)
        if route == "wealth":
            figures = [plotting.plot_cumulative_returns(periodic, "1ns")]
        else:
            events = periodic.copy()
            events.index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "asset"])
            figures = [plotting.plot_events_distribution(events, num_bars=2)]
    target = tmp_path / "direct-charts.html"
    combine_figures_to_html(
        figures,
        output_file=target,
        include_plotlyjs=True,
        title="Supplied analytical observations",
        sections=[
            {
                "title": f"Supplied panel {index + 1}",
                "figure_index": index,
                "text": "These charts retain actual supplied observations. Sample deviations are descriptive, not confidence intervals. Exact time keys remain available on hover.",
            }
            for index in range(len(figures))
        ],
    )
    errors = []
    with playwright.sync_playwright() as runtime:
        executable = None
        if importlib.util.find_spec("choreographer"):
            from choreographer.browsers.chromium import Chromium

            executable = Chromium.find_browser(skip_local=False)
        executable = executable or shutil.which("google-chrome") or shutil.which("chromium")
        if executable is None and Path(runtime.chromium.executable_path).is_file():
            executable = runtime.chromium.executable_path
        assert executable is not None, "Provision an actual Chrome/Chromium browser"
        browser = runtime.chromium.launch(executable_path=executable, headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("http://**/*", lambda connection: connection.abort())
            page.route("https://**/*", lambda connection: connection.abort())
            page.goto(target.as_uri())
            page.wait_for_function(
                "[...document.querySelectorAll('.js-plotly-plot')].length>0 && [...document.querySelectorAll('.js-plotly-plot')].every(p=>p._fullLayout)"
            )
            measured = page.evaluate("""() => {
                const prose=getComputedStyle(document.querySelector('.section-text'));
                return {proseSize:parseFloat(prose.fontSize),proseFamily:prose.fontFamily,overflow:document.documentElement.scrollWidth>innerWidth,
                    panels:[...document.querySelectorAll('.js-plotly-plot')].map(plot=>{
                        const box=plot.getBoundingClientRect();
                        const leaves=[...plot.querySelectorAll('.gtitle,.gtitle .line,.xtick text,.ytick text,.xtitle,.ytitle,.legendtext')].filter(n=>{const b=n.getBoundingClientRect();return b.width>0&&b.height>0});
                        const ticks=[...plot.querySelectorAll('.xtick text')].map(n=>n.getBoundingClientRect()).sort((a,b)=>a.left-b.left);
                        return {title:plot.layout.title.text,clipped:leaves.filter(n=>{const b=n.getBoundingClientRect();return b.left<box.left-1||b.right>box.right+1||b.top<box.top-1||b.bottom>box.bottom+1}).map(n=>n.textContent),
                            labels:leaves.map(n=>({size:parseFloat(getComputedStyle(n).fontSize),family:getComputedStyle(n).fontFamily})),
                            tickOverlap:ticks.some((a,i)=>ticks.slice(i+1).some(b=>a.left<b.right&&b.left<a.right)),
                            legends:[...plot.querySelectorAll('.legendtext')].map(n=>n.textContent),
                            distinctPositions:new Set(plot.calcdata[0].map(point=>point.x)).size,
                            errors:[...plot.querySelectorAll('.errorbar path.yerror')].map(n=>n.getAttribute('d'))};
                    })};
            }""")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                artifact = Path(os.environ["GITHUB_WORKSPACE"]) / "artifacts/alphalens-direct"
                artifact.mkdir(parents=True, exist_ok=True)
                page.locator("#section-0").screenshot(path=str(artifact / f"{route}-{width}.png"))
        finally:
            browser.close()
    assert not errors and not measured["overflow"]
    for panel in measured["panels"]:
        assert not panel["clipped"] and not panel["tickOverlap"], panel
        assert panel["legends"], "Supplied horizons/quantiles must have visible labels"
        assert all(
            14 <= label["size"] <= measured["proseSize"]
            and label["family"] == measured["proseFamily"]
            for label in panel["labels"]
        ), panel
        if route in {"wealth", "event_counts"}:
            assert panel["distinctPositions"] == 3, (
                "Browser dates must not collapse ns observations"
            )
        if route == "event_paths":
            # Two quantiles have two defined deviations each; singleton errors stay absent.
            assert len(panel["errors"]) == 4
            assert all("NaN" not in segment for segment in panel["errors"])
    print(json.dumps({"route": route, "width": width, "rendered": measured}))
