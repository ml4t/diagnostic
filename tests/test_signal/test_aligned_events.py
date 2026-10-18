"""Aligned levels must not become CAR or silently change event/date weights."""

from itertools import product

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.diagnostic.signal import (
    AlignedEventResult,
    align_event_paths,
    analyze_signal_events,
    prepare_signal_panel,
)


def _constant():
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    prices = pd.DataFrame({"A": [10.0] * 5, "B": [100.0] * 5}, index=dates)
    factor = prices.loc[dates[[0, 2, 4]]].stack().rename("factor")
    factor.index.names = ["date", "asset"]
    return factor, prices


def test_event_levels_are_not_rebased_and_clipped_windows_do_not_invent_observations():
    factor, prices = _constant()
    result = analyze_signal_events(
        factor, prices, before=1, after=1, input_kind="levels", demeaned=False
    )
    np.testing.assert_allclose(result.statistics["mean"], [55.0, 55.0, 55.0])
    assert result.statistics["n_obs"].to_list() == [2, 3, 2]
    assert result.statistics["std"].to_list() == [0.0, 0.0, 0.0]
    assert result.paths.filter(pl.col("status") == "window_clipped").height == 4
    assert result.event_counts["n_events"].to_list() == [2, 2, 2]
    assert result.metadata["event_zero_rebased"] is False
    assert result.metadata["units"] == "supplied_levels"


def test_simple_returns_use_the_shared_full_clock_wealth_product_with_nan_as_flat():
    dates = pd.date_range("2025-01-01", periods=4)
    values = pd.DataFrame({"A": [0.1, np.nan, -1.0, 0.2]}, index=dates)
    events = pd.DataFrame({"date": dates[[1, 3]], "asset": ["A", "A"]})
    result = analyze_signal_events(
        events, values, input_kind="returns", before=1, after=1, demeaned=False
    )
    np.testing.assert_allclose(
        result.paths["raw_value"].to_numpy(), [1.1, 1.1, 0.0, 0.0, 0.0, np.nan], equal_nan=True
    )
    assert result.metadata["units"] == "wealth_starting_at_one"
    assert result.paths["status"][-1] == "window_clipped"


def test_date_averaging_is_not_asset_weighting_and_repeated_events_keep_their_ids():
    dates = pd.date_range("2025-01-01", periods=2)
    values = pd.DataFrame({"A": [10.0, 100.0], "B": [20.0, 200.0], "C": [30.0, 300.0]}, index=dates)
    events = pd.DataFrame({"date": [dates[0]] * 3 + [dates[1]], "asset": ["A", "B", "C", "A"]})
    kwargs = {"input_kind": "levels", "before": 0, "after": 0, "demeaned": False}
    equal_dates = analyze_signal_events(events, values, **kwargs)
    equal_events = analyze_signal_events(events, values, **kwargs, mean_by_date=False)
    assert equal_dates.statistics["mean"][0] == 60
    assert equal_events.statistics["mean"][0] == 40
    assert equal_dates.observations["n_assets"].to_list() == [3, 1]
    duplicated = pd.concat([events.iloc[:1], events.iloc[:2]], ignore_index=True)
    duplicate_result = analyze_signal_events(duplicated, values, **kwargs, mean_by_date=False)
    assert duplicate_result.paths["event_id"].n_unique() == 3
    assert duplicate_result.statistics["mean"][0] == pytest.approx(40 / 3)


def _prepared():
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
    panel = prepare_signal_panel(factor, forward, quantiles=2, groups=groups, max_loss=0)
    return panel, prices


@pytest.mark.parametrize("demeaned,group_adjust,by_group", list(product([False, True], repeat=3)))
def test_price_produced_panel_matches_actual_alphalens_event_options_and_intermediate_samples(
    demeaned, group_adjust, by_group
):
    reference = pytest.importorskip("alphalens.performance")
    panel, prices = _prepared()
    frame = (
        panel.data.to_pandas()
        .rename(columns={"quantile": "factor_quantile"})
        .set_index(["date", "asset"])
    )
    expected = reference.average_cumulative_return_by_quantile(
        frame, prices, 1, 2, demeaned=demeaned, group_adjust=group_adjust, by_group=by_group
    )
    result = analyze_signal_events(
        panel.data,
        prices,
        before=1,
        after=2,
        input_kind="levels",
        demeaned=demeaned,
        group_adjust=group_adjust,
        by_group=by_group,
    )
    for row in result.statistics.iter_rows(named=True):
        for statistic in ["mean", "std"]:
            key = (
                (row["quantile"], statistic, row["group"])
                if by_group
                else (row["quantile"], statistic)
            )
            wanted = expected.loc[key, row["offset"]]
            if pd.isna(wanted):
                assert row[statistic] is None
            else:
                assert row[statistic] == pytest.approx(wanted, rel=2e-12, abs=2e-12)
    event_frame = panel.data.to_pandas()
    for row in result.paths.iter_rows(named=True):
        source = row["source_date"]
        if source is None:
            assert row["value"] is None
            continue
        event_date = pd.Timestamp(row["event_date"])
        peers = event_frame[event_frame.date == event_date]
        if group_adjust:
            peers = peers[peers.group == row["group"]]
        benchmark = (
            prices.loc[source, peers.asset.tolist()].mean() if (demeaned or group_adjust) else 0.0
        )
        assert row["raw_value"] == prices.loc[source, row["asset"]]
        assert row["benchmark"] == pytest.approx(benchmark)
        assert row["value"] == pytest.approx(row["raw_value"] - benchmark)
    assert result.paths.height == panel.data.height * 4
    assert result.event_counts["n_events"].sum() == panel.data.height


def test_polars_long_values_and_json_preserve_nanoseconds_timezone_and_ownership():
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(i, "ns") for i in range(3)]
    )
    prices = pd.DataFrame({"A": [10.0, np.nan, 30.0]}, index=dates)
    events = pd.DataFrame({"date": dates, "asset": ["A"] * 3})
    original_prices, original_events = prices.copy(deep=True), events.copy(deep=True)
    long = pl.from_pandas(prices.rename_axis("time").reset_index()).unpivot(
        index="time", variable_name="symbol", value_name="level"
    )
    result = analyze_signal_events(
        pl.from_pandas(events).rename({"date": "time", "asset": "symbol"}),
        long,
        date_col="time",
        asset_col="symbol",
        value_col="level",
        before=1,
        after=1,
        input_kind="levels",
        demeaned=False,
    )
    restored = AlignedEventResult.from_json(result.to_json())
    for name in ["paths", "observations", "statistics", "event_counts"]:
        assert_frame_equal(getattr(result, name), getattr(restored, name))
    assert result.paths["event_date"].dtype == pl.Datetime("ns", "UTC")
    assert (
        result.paths["event_date"].cast(pl.Int64).unique().sort().to_list() == dates.asi8.tolist()
    )
    view = result.paths
    view[0, "value"] = 999.0
    assert result.paths["value"][0] is None
    metadata = result.metadata
    metadata["input_kind"] = "changed"
    assert result.metadata["input_kind"] == "levels"
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(events, original_events)


@pytest.mark.parametrize(
    "setting",
    [
        {"before": -1},
        {"after": True},
        {"input_kind": "prices"},
        {"group_adjust": True},
        {"by_group": True},
        {"mean_by_date": 1},
    ],
)
def test_invalid_event_options_reject_before_mutating_caller_input(setting):
    events, prices = _constant()
    original_events, original_prices = events.copy(deep=True), prices.copy(deep=True)
    with pytest.raises((ValueError, TypeError)):
        analyze_signal_events(events, prices, **{"input_kind": "levels", **setting})
    pd.testing.assert_series_equal(events, original_events)
    pd.testing.assert_frame_equal(prices, original_prices)


def test_absent_clock_event_and_nan_levels_remain_unavailable_not_zero():
    events, prices = _constant()
    frame = events.rename("factor").reset_index().iloc[:1]
    frame["date"] = pd.Timestamp("2030-01-01", tz="UTC")
    paths = align_event_paths(frame, prices, input_kind="levels", before=1, after=1)
    assert paths["status"].to_list() == ["event_not_on_clock"] * 3
    assert paths["value"].null_count() == 3
    assert paths["source_date"].dtype == pl.Datetime("ns", "UTC")


@pytest.mark.parametrize(
    "input_kind,mean_by_date,demeaned",
    list(product(["levels", "returns"], [False, True], [False, True])),
)
def test_common_start_options_match_real_reference_missingness_and_sample_counts(
    input_kind, mean_by_date, demeaned
):
    reference = pytest.importorskip("alphalens.performance")
    events, prices = _constant()
    prices.iloc[1, 1] = np.nan
    values = prices if input_kind == "levels" else prices.pct_change(fill_method=None)
    expected = reference.common_start_returns(
        events,
        values,
        2,
        1,
        cumulative=input_kind == "levels",
        mean_by_date=mean_by_date,
        demean_by=events if demeaned else None,
    ).reindex(range(-2, 2))
    result = analyze_signal_events(
        events,
        values,
        before=2,
        after=1,
        input_kind=input_kind,
        mean_by_date=mean_by_date,
        demeaned=demeaned,
    )
    np.testing.assert_allclose(
        result.statistics["mean"].to_numpy(),
        expected.mean(axis=1).to_numpy(),
        equal_nan=True,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.statistics["std"].to_numpy(),
        expected.std(axis=1).to_numpy(),
        equal_nan=True,
        atol=1e-12,
    )
    assert result.statistics["n_obs"].to_list() == expected.count(axis=1).to_list()


def test_repeated_event_ids_have_an_explicit_weighting_difference_from_legacy_asset_deduplication():
    reference = pytest.importorskip("alphalens.performance")
    factor, prices = _constant()
    events = pd.concat([factor.iloc[:1], factor.iloc[:2]])
    expected = reference.common_start_returns(
        events, prices, 0, 0, cumulative=True, mean_by_date=True
    )
    assert expected.iloc[0, 0] == 55
    result = analyze_signal_events(
        events, prices, input_kind="levels", before=0, after=0, demeaned=False
    )
    assert result.statistics["mean"][0] == 40
    assert result.paths["event_id"].n_unique() == 3
    assert (
        result.metadata["duplicate_event_policy"]
        == "retain each event_id; no implicit deduplication"
    )


def test_nullable_supplied_values_do_not_turn_missing_levels_into_zero():
    events, prices = _constant()
    values = prices.astype("Float64")
    values.iloc[2, 0] = pd.NA
    result = analyze_signal_events(
        events, values, input_kind="levels", before=0, after=0, demeaned=False
    )
    assert (
        result.paths.filter((pl.col("asset") == "A") & (pl.col("status") == "missing_value")).height
        == 1
    )


@pytest.mark.parametrize("bucket", [np.inf, -np.inf, np.nan, 1.5, True])
def test_invalid_legacy_bucket_labels_are_rejected_before_publishing_event_statistics(bucket):
    factor, prices = _constant()
    events = factor.to_frame().assign(factor_quantile=bucket)
    original = events.copy(deep=True)
    with pytest.raises(ValueError, match="quantiles must be positive integers"):
        analyze_signal_events(
            events, prices, input_kind="levels", before=0, after=0, demeaned=False
        )
    pd.testing.assert_frame_equal(events, original)


def _report_result():
    dates = pd.date_range("2025-01-01", periods=2)
    prices = pd.DataFrame({"A": [10.0, 20.0], "B": [100.0, 140.0]}, index=dates)
    events = prices.stack().rename("factor").rename_axis(["date", "asset"])
    return analyze_signal_events(
        events, prices, input_kind="levels", before=0, after=0, demeaned=False
    )


def test_event_plot_preserves_level_units_nonzero_sample_deviation_and_effective_counts():
    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.signal import (
        plot_aligned_event_paths,
        plot_event_count_distribution,
    )

    result = _report_result()
    before = result.to_json()
    figure = plot_aligned_event_paths(result, std_multiplier=0.5)
    assert list(figure.data[0].x) == [0]
    np.testing.assert_allclose(figure.data[0].y, [67.5])
    np.testing.assert_allclose(figure.data[0].error_y.array, [8.838834764831844])
    assert list(map(list, figure.data[0].customdata)) == [[2, 2]]
    assert figure.layout.yaxis.title.text == "Supplied levels"
    assert figure.data[0].meta["uncertainty"] == "sample_std"
    counts = plot_event_count_distribution(result)
    assert list(counts.data[0].y) == [2, 2]
    assert result.to_json() == before


def test_event_report_validates_before_writing_and_cleans_up_after_real_filesystem_failure(
    tmp_path,
):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.signal import generate_aligned_event_report

    result = _report_result()
    target = tmp_path / "report.html"
    target.write_text("existing report")
    with pytest.raises(ValueError, match="std_multiplier"):
        generate_aligned_event_report(result, output_file=target, std_multiplier=-1)
    assert target.read_text() == "existing report"
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    sentinel = directory / "preserved"
    sentinel.write_text("user-owned")
    with pytest.raises(OSError):
        generate_aligned_event_report(result, output_file=directory)
    assert sentinel.read_text() == "user-owned"
    assert not list(tmp_path.glob(".ml4t-events-*"))
    assert generate_aligned_event_report(result, output_file=target) == str(target)
    assert "Aligned event paths" in target.read_text()
    assert "standard deviation, not SE" in target.read_text()


@pytest.mark.parametrize("width", [375, 1200])
def test_real_aligned_event_report_browser_preserves_data_and_readable_labels(
    width, tmp_path, monkeypatch
):
    import json
    import os
    import shutil
    from pathlib import Path

    playwright = pytest.importorskip("playwright.sync_api")
    from plotly.offline import get_plotlyjs

    from ml4t.diagnostic.visualization.signal import event_plots, generate_aligned_event_report

    chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if chrome is None:
        pytest.skip("actual Chrome is required for rendered report verification")
    target = tmp_path / "events.html"
    generate_aligned_event_report(_report_result(), output_file=target, std_multiplier=0.5)
    previous = tmp_path / "previous-auto-ticks.html"
    if width == 375:
        original_plot = event_plots.plot_event_count_distribution

        def previous_auto_ticks(*args, **kwargs):
            figure = original_plot(*args, **kwargs)
            figure.update_xaxes(tickangle=None, nticks=None, automargin=None)
            return figure

        with monkeypatch.context() as context:
            context.setattr(event_plots, "plot_event_count_distribution", previous_auto_ticks)
            generate_aligned_event_report(_report_result(), output_file=previous)
    overlap_script = """() => {
        const plot=document.getElementById('plot-1');
        const legend=plot.querySelector('.legend').getBoundingClientRect();
        return Array.from(plot.querySelectorAll('.xtick text')).some(node=>{
            const tick=node.getBoundingClientRect();
            return tick.width>0 && tick.height>0 && tick.left<legend.right &&
                tick.right>legend.left && tick.top<legend.bottom && tick.bottom>legend.top;
        });
    }"""
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
            if width == 375:
                page.goto(previous.as_uri(), wait_until="networkidle")
                page.wait_for_selector("#plot-1 .xtick text")
                assert page.evaluate(overlap_script), "previous automatic ticks must expose overlap"
                if os.environ.get("GITHUB_ACTIONS") == "true":
                    evidence = Path(os.environ["GITHUB_WORKSPACE"]) / "artifacts/event-reports"
                    evidence.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(evidence / "events-375-previous.png"), full_page=True)
            page.goto(target.as_uri(), wait_until="networkidle")
            page.wait_for_selector(".js-plotly-plot .gtitle")
            measured = page.evaluate("""() => {
                const plot=document.getElementById('plot-0');
                const labels=Array.from(document.querySelectorAll('.xtick text,.ytick text,.legendtext,.xtitle,.ytitle'));
                const prose=Array.from(document.querySelectorAll('.section-text')).find(node=>node.textContent.includes('Offsets count'));
                return {width:innerWidth,scroll:document.documentElement.scrollWidth,
                    mean:plot.data[0].y,error:plot.data[0].error_y.array,
                    labelSizes:labels.map(node=>parseFloat(getComputedStyle(node).fontSize)),
                    labelFamilies:labels.map(node=>getComputedStyle(node).fontFamily),
                    proseSize:parseFloat(getComputedStyle(prose).fontSize),
                    proseFamily:getComputedStyle(prose).fontFamily,
                    titles:Array.from(document.querySelectorAll('.gtitle')).map(node=>parseFloat(getComputedStyle(node).fontSize))};
            }""")
            measured["eventCountTickLegendOverlap"] = page.evaluate(overlap_script)
            measured["previousAutoTicksOverlapControl"] = width == 375
            page.screenshot(path=str(tmp_path / f"events-{width}.png"), full_page=True)
            if os.environ.get("GITHUB_ACTIONS") == "true":
                evidence = Path(os.environ["GITHUB_WORKSPACE"]) / "artifacts/event-reports"
                evidence.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(evidence / f"events-{width}.png"), full_page=True)
                (evidence / f"events-{width}.json").write_text(json.dumps(measured, indent=2))
        finally:
            browser.close()
    assert not errors
    assert measured["width"] == width and measured["scroll"] <= width
    assert measured["eventCountTickLegendOverlap"] is False
    assert measured["mean"] == [67.5]
    np.testing.assert_allclose(measured["error"], [8.838834764831844])
    assert measured["labelSizes"] and min(measured["labelSizes"]) >= 14
    assert max(measured["titles"]) <= measured["proseSize"]
    print("EVENT_BROWSER " + json.dumps(measured))
