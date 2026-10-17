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
                np.testing.assert_allclose(line.get_ydata() / 10000, values["mean"], equal_nan=True)
    assert len(figures) == 1 + (2 if std_bar else 0) + (2 if by_group else 0)
    for figure in figures:
        expected_units = "Supplied levels" if input_kind == "levels" else "Wealth (start = 1)"
        assert figure.layout.yaxis.title.text == expected_units
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
