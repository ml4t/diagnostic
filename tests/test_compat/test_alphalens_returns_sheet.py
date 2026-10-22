"""Familiar return sheets must use the real prepared samples and declared definitions."""

import runpy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _panel():
    return runpy.run_path(str(Path(__file__).with_name("test_alphalens_information_sheet.py")))[
        "_price_panel"
    ]()


@pytest.mark.parametrize("by_group", [False, True])
def test_familiar_returns_sheet_keeps_global_panels_groups_and_atomic_failure(by_group, tmp_path):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.compat.alphalens import tears

    data = _panel()
    before = data.copy(deep=True)
    target = tmp_path / "returns.html"
    figures = tears.create_returns_tear_sheet(
        data, by_group=by_group, return_fig=True, output_file=target, set_context=False
    )
    assert len(figures) == (12 if by_group else 8)
    assert target.read_text().count('class="plot-container"') == len(figures)
    assert (
        len([figure for figure in figures if figure.layout.meta["analysis"] == "factor_wealth"])
        == 1
    )
    assert {figure.layout.meta["return_column"] for figure in figures} == {"1D", "2D"}
    spread = next(
        figure for figure in figures if figure.layout.meta["analysis"] == "quantile_spread"
    )
    assert spread.layout.meta["exposure_scale"] == 0.5
    assert spread.layout.meta["rolling_window_observations"] == 22
    assert spread.layout.meta["bandwidth"] == 0.5
    pd.testing.assert_frame_equal(data, before)
    target.write_text("retained report")
    with pytest.raises(TypeError, match="long_short"):
        tears.create_returns_tear_sheet(data, long_short="yes", output_file=target)
    assert target.read_text() == "retained report"


@pytest.mark.parametrize("long_short", [False, True])
@pytest.mark.parametrize("group_neutral", [False, True])
@pytest.mark.parametrize("by_group", [False, True])
def test_familiar_returns_sheet_matches_actual_reference_panels_and_exposure(
    long_short, group_neutral, by_group, monkeypatch
):
    reference = pytest.importorskip("alphalens.tears")
    performance = pytest.importorskip("alphalens.performance")
    plotting = pytest.importorskip("alphalens.plotting")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    profile = version("alphalens-reloaded")
    if profile not in {"0.5.0", "0.4.6"}:
        pytest.skip("Corrected and frozen references are qualified independently")
    from ml4t.diagnostic.compat.alphalens import tears

    data = _panel()
    before = data.copy(deep=True)
    bars, distributions, spreads, displayed = [], [], [], []
    original_bar = plotting.plot_quantile_returns_bar
    original_violin = plotting.plot_quantile_returns_violin
    original_spread = plotting.plot_mean_quantile_returns_spread_time_series

    def draw_bar(means, *args, **kwargs):
        bars.append(means.copy(deep=True))
        return original_bar(means, *args, **kwargs)

    def draw_violin(means, *args, **kwargs):
        distributions.append(means.copy(deep=True))
        return original_violin(means, *args, **kwargs)

    def draw_spread(means, *args, **kwargs):
        spreads.append((means.copy(deep=True), kwargs["std_err"].copy(deep=True)))
        return original_spread(means, *args, **kwargs)

    def show():
        for identifier in pyplot.get_fignums():
            figure = pyplot.figure(identifier)
            figure.canvas.draw()
            displayed.append(
                sum(
                    len(axis.lines) + len(axis.patches) + len(axis.collections)
                    for axis in figure.axes
                )
            )

    monkeypatch.setattr(plotting, "plot_quantile_returns_bar", draw_bar)
    monkeypatch.setattr(plotting, "plot_quantile_returns_violin", draw_violin)
    monkeypatch.setattr(plotting, "plot_mean_quantile_returns_spread_time_series", draw_spread)
    monkeypatch.setattr(pyplot, "show", show)
    try:
        reference.create_returns_tear_sheet(
            data,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            set_context=False,
        )
    finally:
        pyplot.close("all")
    assert displayed and all(count > 0 for count in displayed)
    figures = tears.create_returns_tear_sheet(
        data,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        reference_version=profile,
        return_fig=True,
        set_context=False,
    )
    for column in ("1D", "2D"):
        selected = [figure for figure in figures if figure.layout.meta["return_column"] == column]
        mean = next(
            figure for figure in selected if figure.layout.meta["analysis"] == "quantile_mean"
        )
        np.testing.assert_allclose(mean.data[0].y, bars[0][column], rtol=1e-12, atol=1e-12)
        violin = next(
            figure
            for figure in selected
            if figure.layout.meta["analysis"] == "quantile_distribution"
        )
        for bucket, trace in enumerate(violin.data, start=1):
            np.testing.assert_allclose(
                trace.y,
                distributions[0].xs(bucket, level="factor_quantile")[column],
                rtol=1e-12,
                atol=1e-12,
            )
        spread = next(
            figure for figure in selected if figure.layout.meta["analysis"] == "quantile_spread"
        )
        np.testing.assert_allclose(spread.data[0].y, spreads[0][0][column], rtol=1e-12, atol=1e-12)
        upper = spreads[0][0][column] + 0.5 * spreads[0][1][column]
        lower = spreads[0][0][column] - 0.5 * spreads[0][1][column]
        np.testing.assert_allclose(
            spread.data[2].y, [*upper, *lower.iloc[::-1]], rtol=1e-12, atol=1e-12
        )
        if by_group:
            groups = [
                figure
                for figure in selected
                if figure.layout.meta["analysis"] == "group_quantile_mean"
            ]
            assert len(groups) == 2
            for group in groups:
                expected = bars[1].xs(group.layout.meta["group"], level="group")[column]
                np.testing.assert_allclose(group.data[0].y, expected, rtol=1e-12, atol=1e-12)
    factor_returns = performance.factor_returns(data, long_short, group_neutral)
    factor_path = next(
        figure for figure in figures if figure.layout.meta["analysis"] == "factor_wealth"
    )
    np.testing.assert_allclose(
        factor_path.data[0].y,
        performance.cumulative_returns(factor_returns["1D"]),
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(data, before)
