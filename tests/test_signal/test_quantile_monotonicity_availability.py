"""Undefined native quantile scores must not become negative analytical verdicts."""

import json
import math
import re
import shutil
import subprocess

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import analyze_prepared_signal, analyze_signal, prepare_signal_panel
from ml4t.diagnostic.visualization.signal import SignalDashboard, plot_quantile_returns_bar


def _inputs(profile):
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    assets = (
        ["A", "B", "C"]
        if profile in {"increasing", "decreasing", "non_monotonic"}
        else ["A", "B", "C", "D"]
    )
    future = {
        "increasing": [0.125, 0.25, 0.5],
        "decreasing": [0.5, 0.25, 0.125],
        "non_monotonic": [0.5, 0.125, 0.25],
    }.get(profile, [0.25] * 4 if profile.startswith("constant") else [0.25, -0.125, 0.5, 0.125])
    prices = {}
    for asset, past, outcome in zip(
        assets, [1.125, 1.25, 1.5, 2.0][: len(assets)], future, strict=True
    ):
        first = 128 * past
        second = first * (1 + outcome)
        third = second * past
        prices[asset] = [128.0, first, second, third, third * (1 + outcome)]
    prices = pd.DataFrame(prices, index=dates)
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[[1, 3]]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    marks = (
        prices.stack(future_stack=True).rename("price").rename_axis(["date", "asset"]).reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=[1], group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1"]]
        .rename(columns={"fwd_ret_1": "1D"})
    )
    buckets = {"one": 1, "two": 2, "constant_three": 3, "constant_four": 4}.get(profile, 3)
    panel = prepare_signal_panel(factor, forward, quantiles=buckets, max_loss=0)
    return factor.reset_index(), marks, panel, buckets


def _model(profile, route="owned"):
    factor, marks, panel, buckets = _inputs(profile)
    if route == "owned":
        result = panel.analyze(min_assets=2)
    elif route == "prepared":
        result = analyze_prepared_signal(
            panel.data, periods=(1,), quantiles=buckets, min_assets=2, compute_turnover_flag=False
        )
    else:
        result = analyze_signal(
            factor,
            marks,
            periods=(1,),
            quantiles=buckets,
            quantile_method="qcut",
            filter_zscore=None,
            min_assets=2,
            compute_turnover_flag=False,
        )
    return result.to_quantile_result()


@pytest.mark.parametrize("profile", ["one", "two", "constant_three", "constant_four"])
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_real_undefined_scores_remain_nullable_in_views_and_unavailable_in_bar_summary(
    profile, route
):
    model = _model(profile, route)
    assert math.isnan(model.rank_correlation["1D"])
    assert model.is_monotonic["1D"] is False
    view = model.get_dataframe("spread")
    assert view["is_monotonic"].dtype == pl.Boolean
    assert view["is_monotonic"].to_list() == [None]
    assert view["monotonicity_direction"].to_list() == ["unavailable"]
    assert model.get_monotonicity(1) == (None, "unavailable")
    assert "Monotonic:           N/A" in model.summary()
    figure = plot_quantile_returns_bar(model, period="1D")
    assert any("Monotonic: N/A" in annotation.text for annotation in figure.layout.annotations)
    assert not any("✗ No" in annotation.text for annotation in figure.layout.annotations)


@pytest.mark.parametrize(
    "profile,expected,direction,score",
    [
        ("increasing", True, "increasing", 1.0),
        ("decreasing", True, "decreasing", -1.0),
        ("non_monotonic", False, "none", -0.5),
    ],
)
@pytest.mark.parametrize("route", ["raw", "prepared", "owned"])
def test_defined_native_decisions_and_directions_do_not_change(
    profile, expected, direction, score, route
):
    model = _model(profile, route)
    assert model.rank_correlation["1D"] == pytest.approx(score, abs=1e-14)
    assert model.is_monotonic["1D"] is expected
    assert model.get_dataframe("spread")["is_monotonic"].to_list() == [expected]
    assert model.get_dataframe("spread")["monotonicity_direction"].to_list() == [direction]
    assert f"Monotonic:           {expected} ({direction})" in model.summary()
    figure = plot_quantile_returns_bar(model, period="1D")
    label = f"✓ {direction}" if expected else "✗ No"
    assert any(f"Monotonic: {label}" in annotation.text for annotation in figure.layout.annotations)


@pytest.mark.parametrize(
    "profile",
    ["one", "two", "constant_three", "constant_four", "increasing", "decreasing", "non_monotonic"],
)
def test_actual_json_roundtrip_keeps_raw_flag_compatibility_and_availability_aware_access(profile):
    model = _model(profile)
    before = model.model_dump_json()
    restored = type(model).model_validate_json(before)
    assert restored.is_monotonic == model.is_monotonic
    assert restored.monotonicity_direction == model.monotonicity_direction
    expected = (
        (None, "unavailable")
        if not math.isfinite(model.rank_correlation["1D"])
        else (model.is_monotonic["1D"], model.monotonicity_direction["1D"])
    )
    assert restored.get_monotonicity("1D") == expected
    exported = restored.get_dataframe("spread")
    exported.replace_column(
        exported.get_column_index("is_monotonic"),
        pl.Series("is_monotonic", [not bool(expected[0])], dtype=pl.Boolean),
    )
    assert restored.get_monotonicity(1) == expected
    assert restored.model_dump_json() == before


def test_unknown_period_is_not_misreported_as_unavailable_analysis():
    model = _model("two")
    with pytest.raises(ValueError, match="Period"):
        model.get_monotonicity("7D")


@pytest.mark.parametrize("width", [1280, 375])
def test_actual_report_browser_displays_unavailable_summary_card_and_bar_annotation(
    width, tmp_path
):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium required for real report checks")
    _, _, panel, _ = _inputs("two")
    html = SignalDashboard().generate(panel.analyze(min_assets=2).to_tear_sheet())
    probe = """<script>document.addEventListener('DOMContentLoaded',()=>{
    const card=Array.from(document.querySelectorAll('#summary .metric-card')).find(node=>node.querySelector('.metric-label').textContent==='Monotonic');
    const summary=card.querySelector('.metric-value').textContent;
    Array.from(document.querySelectorAll('button')).find(node=>node.textContent==='Quantile Analysis').click();
    const marker=document.createElement('output');marker.id='monotonicity-browser-check';
    marker.textContent=JSON.stringify({summary,annotations:Array.from(document.querySelectorAll('#quantile .annotation-text')).map(node=>node.textContent)});document.body.appendChild(marker);});</script>"""
    path = tmp_path / "report.html"
    path.write_text(html.replace("</body>", probe + "</body>"))
    completed = subprocess.run(
        [
            chrome,
            "--headless",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--host-resolver-rules=MAP * ~NOTFOUND",
            f"--user-data-dir={tmp_path / 'browser-profile'}",
            f"--window-size={width},1000",
            "--virtual-time-budget=1500",
            "--dump-dom",
            path.as_uri(),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    match = re.search(r'<output id="monotonicity-browser-check">([^<]+)</output>', completed.stdout)
    assert match, completed.stderr
    measured = json.loads(match[1])
    assert measured["summary"] == "N/A"
    assert any("Monotonic: N/A" in text for text in measured["annotations"])
