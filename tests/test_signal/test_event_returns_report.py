"""The complete event-returns workflow must retain global, quantile and group panels."""

from itertools import product

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.diagnostic.signal import EventReturnsResult, analyze_event_returns, prepare_signal_panel
from ml4t.diagnostic.visualization.signal import (
    generate_event_returns_report,
    plot_event_returns,
)


def _inputs():
    dates = pd.bdate_range("2025-01-02", periods=8, tz="UTC")
    prices = pd.DataFrame(
        np.arange(100.0, 106.0)
        * np.exp(np.arange(8)[:, None] * np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.005])),
        index=dates,
        columns=list("ABCDEF"),
    )
    factor = prices.iloc[[0, 2, 4, 6]].stack().rename("factor")
    factor.index.names = ["date", "asset"]
    forward = prices.pct_change(fill_method=None).shift(-1).stack().rename("1D").to_frame()
    forward.index.names = ["date", "asset"]
    groups = pd.Series(
        [
            "small" if asset in ("AB" if day < dates[4] else "AC") else "large"
            for day, asset in factor.index
        ],
        index=factor.index,
        name="group",
    )
    return prepare_signal_panel(factor, forward, quantiles=2, groups=groups, max_loss=0), prices


@pytest.mark.parametrize("std_bar,by_group", list(product([False, True], repeat=2)))
def test_event_returns_report_keeps_overview_when_optional_panels_are_requested(std_bar, by_group):
    panel, prices = _inputs()
    before = prices.copy(deep=True)
    result = panel.event_returns(
        prices, before=1, after=2, input_kind="levels", demeaned=False, by_group=by_group
    )
    figures = plot_event_returns(result, std_bar=std_bar)
    assert len(figures) == 1 + (2 if std_bar else 0) + (2 if by_group else 0)
    assert figures[0].layout.title.text == "All quantiles"
    assert len(figures[0].data) == 2
    assert all(not trace.error_y.visible for trace in figures[0].data)
    if std_bar:
        assert [fig.layout.title.text for fig in figures[1:3]] == ["Quantile 1", "Quantile 2"]
        assert all(len(fig.data) == 1 and fig.data[0].error_y.visible for fig in figures[1:3])
    if by_group:
        assert [fig.layout.title.text for fig in figures[-2:]] == ["Group: large", "Group: small"]
        assert all(not trace.error_y.visible for fig in figures[-2:] for trace in fig.data)
    assert result.overall.paths.height == 96
    assert result.overall.paths.filter(pl.col("status") == "window_clipped").height > 0
    pd.testing.assert_frame_equal(prices, before)
    assert result.overall.event_counts["n_events"].sum() == 24


@pytest.mark.parametrize(
    "demeaned,group_adjust,std_bar,by_group", list(product([False, True], repeat=4))
)
@pytest.mark.parametrize("input_kind", ["levels", "returns"])
def test_complete_event_returns_report_matches_actual_reference_sheet_panels_and_samples(
    demeaned, group_adjust, std_bar, by_group, input_kind, monkeypatch
):
    tears = pytest.importorskip("alphalens.tears")
    import matplotlib.pyplot as plt

    panel, prices = _inputs()
    if input_kind == "returns":
        values = prices.pct_change(fill_method=None)
        reference_values = (1 + values.fillna(0)).cumprod()
    else:
        values = prices
        reference_values = prices
    frame = (
        panel.data.to_pandas()
        .rename(columns={"quantile": "factor_quantile"})
        .set_index(["date", "asset"])
    )
    captured = []
    original = tears.plotting.plot_quantile_average_cumulative_return

    def capture(actual, *args, **kwargs):
        axes = original(actual, *args, **kwargs)
        captured.append((actual.copy(deep=True), kwargs.copy(), axes))
        return axes

    monkeypatch.setattr(tears.plotting, "plot_quantile_average_cumulative_return", capture)
    tears.create_event_returns_tear_sheet(
        frame,
        reference_values,
        avgretplot=(1, 2),
        long_short=demeaned,
        group_neutral=group_adjust,
        std_bar=std_bar,
        by_group=by_group,
    )
    result = analyze_event_returns(
        panel.data,
        values,
        input_kind=input_kind,
        before=1,
        after=2,
        demeaned=demeaned,
        group_adjust=group_adjust,
        by_group=by_group,
    )
    figures = plot_event_returns(result, std_bar=std_bar)
    assert len(captured) == 1 + int(std_bar) + (2 if by_group else 0)
    assert captured[0][1]["by_quantile"] is False and captured[0][1]["std_bar"] is False
    if std_bar:
        assert captured[1][1]["by_quantile"] is True and captured[1][1]["std_bar"] is True
    for table, options, axes in captured:
        group = options.get("title")
        native = result.overall if group is None else result.groups
        assert native is not None
        statistics = native.statistics
        if group is not None:
            statistics = statistics.filter(pl.col("group") == group)
        for row in statistics.iter_rows(named=True):
            for metric in ["mean", "std"]:
                expected = table.loc[(row["quantile"], metric), row["offset"]]
                actual = np.nan if row[metric] is None else row[metric]
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        axis_list = axes if isinstance(axes, list) else [axes]
        for axis in axis_list:
            for line in axis.lines:
                label = line.get_label()
                if not label.startswith("Quantile "):
                    continue
                quantile = int(label.split()[-1])
                values = statistics.filter(pl.col("quantile") == quantile).sort("offset")
                # Reference plotting scales the supplied levels by10,000, not event-zero returns.
                offsets = line.get_xdata()
                if table.columns.dtype == object:
                    labels = dict(zip(axis.get_xticks(), axis.get_xticklabels(), strict=True))
                    offsets = [int(labels[position].get_text()) for position in offsets]
                    assert offsets == list(table.columns)
                coordinates = sorted(zip(offsets, line.get_ydata() / 10000, strict=True))
                np.testing.assert_allclose(
                    coordinates, values.select("offset", "mean").to_numpy(), equal_nan=True
                )
    assert len(figures) == 1 + (2 if std_bar else 0) + (2 if by_group else 0)
    for figure in figures:
        if demeaned or group_adjust:
            expected_units = "Level difference" if input_kind == "levels" else "Wealth difference"
        else:
            expected_units = "Supplied levels" if input_kind == "levels" else "Wealth (start = 1)"
        assert figure.layout.yaxis.title.text == expected_units
        sample = result.overall.statistics
        if figure.layout.title.text.startswith("Group: "):
            group = figure.layout.title.text.removeprefix("Group: ")
            sample = result.groups.statistics.filter(pl.col("group") == group)
        for trace in figure.data:
            quantile = int(trace.name.removeprefix("Q"))
            expected = sample.filter(pl.col("quantile") == quantile).sort("offset")
            np.testing.assert_allclose(trace.x, expected["offset"])
            np.testing.assert_allclose(trace.y, expected["mean"], equal_nan=True)
            if trace.error_y.visible:
                np.testing.assert_allclose(trace.error_y.array, expected["std"], equal_nan=True)
    plt.close("all")


def test_event_returns_json_owns_both_samples_and_rejects_mismatched_inputs():
    panel, prices = _inputs()
    result = panel.event_returns(prices, input_kind="levels", before=1, after=2, by_group=True)
    restored = EventReturnsResult.from_json(result.to_json())
    assert_frame_equal(restored.overall.paths, result.overall.paths)
    assert_frame_equal(restored.groups.observations, result.groups.observations)
    copy = restored.overall.statistics
    copy[0, "mean"] = 999
    assert restored.overall.statistics["mean"][0] != 999
    changed = panel.event_returns(prices * 2, input_kind="levels", before=1, after=2, by_group=True)
    with pytest.raises(ValueError, match="same input"):
        EventReturnsResult(result.overall, changed.groups)
    with pytest.raises(ValueError, match="combine groups"):
        EventReturnsResult(result.groups)


def test_complete_event_returns_html_does_not_overwrite_existing_output_after_render_failure(
    tmp_path, monkeypatch
):
    from ml4t.diagnostic.visualization import report_generation

    panel, prices = _inputs()
    result = panel.event_returns(prices, input_kind="levels", before=1, after=2, by_group=True)
    target = tmp_path / "events.html"
    target.write_text("user-owned")

    def broken(*args, **kwargs):
        raise RuntimeError("report composition failed")

    with monkeypatch.context() as context:
        context.setattr(report_generation, "combine_figures_to_html", broken)
        with pytest.raises(RuntimeError, match="composition failed"):
            generate_event_returns_report(result, output_file=target)
    assert target.read_text() == "user-owned"
    assert not list(tmp_path.glob(".ml4t-events-*"))
    assert generate_event_returns_report(result, output_file=target) == str(target)
    assert target.read_text().count('class="plot-container"') == 5


def test_group_names_are_display_data_not_html_or_filenames():
    panel, prices = _inputs()
    events = panel.data.with_columns(
        pl.col("group").replace({"small": "<script>alert(1)</script>"})
    )
    result = analyze_event_returns(events, prices, input_kind="levels", by_group=True)
    html = generate_event_returns_report(result)
    assert '<h2 class="section-title">Group: <script>' not in html
    assert "Group: &lt;script&gt;" in html


def test_demeaned_wealth_report_does_not_claim_paths_start_at_one():
    dates = pd.date_range("2025-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.0] * 3, "B": [0.0] * 3}, index=dates)
    events = pd.DataFrame({"date": [dates[1]] * 2, "asset": ["A", "B"]})
    result = analyze_event_returns(events, returns, input_kind="returns", before=1, after=1)
    figure = plot_event_returns(result, std_bar=False)[0]
    assert list(figure.data[0].y) == [0, 0, 0]
    assert figure.layout.yaxis.title.text == "Wealth difference"


@pytest.mark.parametrize("width", [375, 1200])
def test_complete_event_returns_report_renders_all_panels_readably_in_actual_browser(
    width, tmp_path
):
    import json
    import os
    import shutil
    from pathlib import Path

    playwright = pytest.importorskip("playwright.sync_api")
    from plotly.offline import get_plotlyjs

    chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if chrome is None:
        pytest.skip("actual Chrome is required for rendered report verification")
    panel, prices = _inputs()
    result = panel.event_returns(prices, input_kind="levels", before=1, after=2, by_group=True)
    target = tmp_path / "events.html"
    generate_event_returns_report(result, output_file=target)
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
            page.wait_for_selector("#plot-4 .gtitle")
            measured = page.evaluate("""() => {
                const plots=Array.from(document.querySelectorAll('.js-plotly-plot'));
                const labels=Array.from(document.querySelectorAll('.xtick text,.ytick text,.legendtext,.xtitle,.ytitle'));
                return {width:innerWidth,scroll:document.documentElement.scrollWidth,
                    panels:plots.map(plot=>({title:plot.layout.title.text,traces:plot.data.map(trace=>({name:trace.name,x:trace.x,y:trace.y,error:trace.error_y}))})),
                    labelSizes:labels.map(node=>parseFloat(getComputedStyle(node).fontSize)),
                    proseSize:parseFloat(getComputedStyle(document.querySelector('.section-text')).fontSize),
                    titleSizes:Array.from(document.querySelectorAll('.gtitle')).map(node=>parseFloat(getComputedStyle(node).fontSize))};
            }""")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                evidence = Path(os.environ["GITHUB_WORKSPACE"]) / "artifacts/event-reports"
                evidence.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(evidence / f"event-returns-{width}.png"), full_page=True)
                for index in range(5):
                    page.locator(f"#section-{index}").screenshot(
                        path=str(evidence / f"event-returns-{width}-section-{index}.png")
                    )
                (evidence / f"event-returns-{width}.json").write_text(
                    json.dumps(measured, indent=2)
                )
        finally:
            browser.close()
    assert not errors
    assert measured["width"] == measured["scroll"] == width
    assert len(measured["panels"]) == 5
    assert min(measured["labelSizes"]) >= 14
    assert max(measured["titleSizes"]) <= measured["proseSize"]
    expected = plot_event_returns(result)
    for actual, figure in zip(measured["panels"], expected, strict=True):
        assert actual["title"] == figure.layout.title.text
        for trace, requested in zip(actual["traces"], figure.data, strict=True):
            assert trace["name"] == requested.name
            np.testing.assert_allclose(trace["x"], requested.x)
            np.testing.assert_allclose(trace["y"], requested.y, equal_nan=True)
    print("EVENT_RETURNS_BROWSER " + json.dumps(measured))
