"""Prepared regime/cone reports must render actual results and reject another account."""

import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _account(values=(0.1, 0.1, 0.9, -0.8)):
    dates = pd.date_range("2025-01-13", periods=len(values), tz="UTC")
    return PortfolioAnalysis.from_frames(pd.Series(values, index=dates), calendar=None), dates


def test_native_report_embeds_prepared_stress_live_cones_and_exact_owned_exports(tmp_path):
    analysis, dates = _account()
    stress = analysis.compute_stress(
        {"example <period>": (dates[0], dates[1]), "unobserved": ("1990", "1990")}
    )
    live = analysis.compute_live_split(live_start=dates[2])
    cones = analysis.compute_wealth_cones(
        live_start=dates[2], horizon=3, n_samples=4, initial_capital=1000
    )
    report = analysis.create_tear_sheet(stress=stress, live=live, wealth_cones=cones)
    assert report.availability["stress"]["status"] == "available"
    assert report.availability["live"]["status"] == "available"
    assert report.availability["wealth_cones"]["status"] == "available"
    exported = report.to_dict()
    for name, result in (("stress", stress), ("live", live), ("wealth_cones", cones)):
        restored = type(result).from_json(json.dumps(exported[name]))
        assert restored.metadata == result.metadata
    chart = report.figures["Training-only wealth cones"]
    actual = next(trace for trace in chart.data if trace.name == "Actual live wealth")
    np.testing.assert_allclose(actual.y, [1900, 380])
    assert chart.layout.xaxis.title.text == "Forward observation"
    assert chart.layout.meta["method"] == "iid"
    target = tmp_path / "report.html"
    report.save_html(target, include_plotlyjs=True)
    html = target.read_text(encoding="utf-8")
    assert "example &lt;period&gt;" in html
    assert "unobserved" in html and "not quantiles" in html
    assert "training-only" in html.lower()
    assert report.wealth_cones.paths.equals(cones.paths)


@pytest.mark.parametrize("component", ["stress", "live", "wealth_cones"])
def test_prepared_report_rejects_another_accounts_values_before_replacing_output(
    component, tmp_path
):
    analysis, dates = _account()
    other, _ = _account((0.1, 0.2, 0.9, -0.8))
    prepared = {
        "stress": analysis.compute_stress({"window": (dates[0], dates[1])}),
        "live": analysis.compute_live_split(live_start=dates[2]),
        "wealth_cones": analysis.compute_wealth_cones(live_start=dates[2], horizon=3, n_samples=4),
    }[component]
    target = tmp_path / "report.html"
    target.write_text("existing report", encoding="utf-8")
    with pytest.raises(ValueError, match="exact analysis"):
        other.create_tear_sheet(**{component: prepared}).save_html(target)
    assert target.read_text(encoding="utf-8") == "existing report"


def test_absent_regime_cone_requests_are_not_silently_computed():
    analysis, _ = _account()
    report = analysis.create_tear_sheet()
    assert report.availability["stress"]["status"] == "not_requested"
    assert report.availability["live"]["status"] == "not_requested"
    assert report.availability["wealth_cones"]["status"] == "not_requested"
    assert not any("cone" in name.lower() for name in report.figures)
    assert report.to_dict()["wealth_cones"] is None


@pytest.mark.parametrize("width", [375, 1280])
def test_offline_regime_cone_report_keeps_readable_labels_and_real_live_values(width, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    analysis, dates = _account()
    cones = analysis.compute_wealth_cones(
        live_start=dates[2], horizon=3, n_samples=4, initial_capital=1000
    )
    stress = analysis.compute_stress({"selected": (dates[0], dates[1]), "absent": ("1990", "1990")})
    target = tmp_path / "report.html"
    report = analysis.create_tear_sheet(stress=stress, wealth_cones=cones)
    report.save_html(target, include_plotlyjs=True)
    with playwright.sync_playwright() as runtime:
        executable = None
        if importlib.util.find_spec("choreographer") is not None:
            from choreographer.browsers.chromium import Chromium

            executable = Chromium.find_browser(skip_local=False)
        executable = executable or shutil.which("google-chrome") or shutil.which("chromium")
        if executable is None and Path(runtime.chromium.executable_path).is_file():
            executable = runtime.chromium.executable_path
        assert executable is not None, "An actual Chrome/Chromium browser must be provisioned"
        browser = runtime.chromium.launch(executable_path=executable, headless=True)
        page = browser.new_page(viewport={"width": width, "height": 1000})
        page.route("http://**/*", lambda route: route.abort())
        page.route("https://**/*", lambda route: route.abort())
        page.goto(target.as_uri(), timeout=90_000)
        container = page.locator(".plot-container").filter(
            has=page.locator("h2", has_text="Training-only wealth cones")
        )
        graph = container.locator(".js-plotly-plot")
        graph.wait_for()
        page.wait_for_function(
            "count => {const plots = [...document.querySelectorAll('.js-plotly-plot')]; "
            "return plots.length === count && plots.every(e => e._fullLayout);}",
            arg=len(report.figures),
        )
        evidence = graph.evaluate("""e => {
            const box = e.getBoundingClientRect();
            const labels = Array.from(e.querySelectorAll('.legendtext,.xtick text,.ytick text,.xtitle,.ytitle'));
            return {width:box.width,bodySize:parseFloat(getComputedStyle(document.body).fontSize),
                labels:labels.map(t=>{const b=t.getBoundingClientRect();return {text:t.textContent,size:parseFloat(getComputedStyle(t).fontSize),left:b.left-box.left,right:b.right-box.left,top:b.top-box.top,bottom:b.bottom-box.top}}),
                height:box.height,actual:e.data.find(t=>t.name==='Actual live wealth').y};
        }""")
        evidence.update(browser_executable=executable, browser_version=browser.version)
        print(
            json.dumps(
                {
                    "viewport": width,
                    "browser_executable": executable,
                    "browser_version": browser.version,
                }
            )
        )
        assert evidence["actual"] == pytest.approx([1900, 380])
        assert evidence["bodySize"] == 16
        assert evidence["labels"]
        legend_names = graph.locator(".legendtext").all_text_contents()
        assert "Actual live wealth" in legend_names
        assert "Quantile 0.025" in legend_names
        assert "Mean +2 std" in legend_names
        for label in evidence["labels"]:
            assert 14 <= label["size"] <= evidence["bodySize"], label
            assert -1 <= label["left"] and label["right"] <= evidence["width"] + 1, label
            assert -1 <= label["top"] and label["bottom"] <= evidence["height"] + 1, label
        assert page.locator("body").evaluate("e=>e.scrollWidth<=innerWidth")
        assert (
            container.locator(".figure-note").evaluate(
                "e=>parseFloat(getComputedStyle(e).fontSize)"
            )
            == 13
        )
        if os.environ.get("ML4T_REGIMES_SCREENSHOTS"):
            output = Path(os.environ["ML4T_REGIMES_SCREENSHOTS"])
            output.mkdir(parents=True, exist_ok=True)
            container.screenshot(path=str(output / f"cones-{width}.png"))
            (output / f"cones-{width}.json").write_text(
                json.dumps(evidence, indent=2), encoding="utf-8"
            )
        browser.close()
