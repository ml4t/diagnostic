"""Familiar report calls must preserve real prepared samples and supplied level units."""

import itertools
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.alphalens import tears, utils


def _inputs():
    dates = pd.bdate_range("2025-01-02", periods=14, tz="UTC")
    assets = list("ABCDEF")
    changes = np.array([np.roll([-0.04, -0.02, 0.01, 0.02, 0.04, 0.06], day) for day in range(13)])
    prices = pd.DataFrame(
        np.vstack([np.full(6, 100.0), 100 * np.cumprod(1 + changes, axis=0)]),
        index=dates,
        columns=assets,
    )
    factor = prices.pct_change(fill_method=None).iloc[1:9].stack(future_stack=True)
    factor.index = factor.index.set_names(["date", "asset"])
    panel = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2),
        quantiles=2,
        groupby=dict(zip(assets, ["left"] * 3 + ["right"] * 3, strict=True)),
        filter_zscore=None,
        max_loss=0,
    )
    return panel, prices


@pytest.mark.parametrize("route", ["turnover", "event_returns", "event_study"])
def test_price_preparation_to_familiar_reports_retains_every_panel_and_input(route, tmp_path):
    pytest.importorskip("plotly")
    panel, prices = _inputs()
    before, original_prices = panel.copy(deep=True), prices.copy(deep=True)
    args = (panel,) if route == "turnover" else (panel, prices)
    options = {"turnover_periods": ["1D", "2D"]} if route == "turnover" else {"avgretplot": (2, 3)}
    if route == "event_returns":
        options["by_group"] = True
    if route == "event_study":
        options["n_bars"] = 5
    target = tmp_path / (route + ".html")
    figures = getattr(tears, f"create_{route}_tear_sheet")(
        *args, **options, output_file=target, return_fig=True, set_context=False
    )
    expected = {"turnover": 4, "event_returns": 5, "event_study": 6}[route]
    assert len(figures) == expected and all(len(figure.data) for figure in figures)
    assert target.read_text(encoding="utf-8").count('class="plot-container"') == expected
    pd.testing.assert_frame_equal(panel, before)
    pd.testing.assert_frame_equal(prices, original_prices)
    assert len(panel) == 48 and list(utils.get_forward_returns_columns(panel.columns)) == [
        "1D",
        "2D",
    ]


def test_event_return_adapter_preserves_supplied_levels_instead_of_compounding_them():
    pytest.importorskip("plotly")
    dates = pd.bdate_range("2025-01-02", periods=3, tz="UTC")
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [20.0, 22.0, 24.0]}, index=dates)
    factor = pd.Series(
        1.0,
        index=pd.MultiIndex.from_product([dates[:2], ["A", "B"]], names=["date", "asset"]),
    )
    panel = utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1,), quantiles=1, filter_zscore=None, max_loss=0
    )
    figures = tears.create_event_returns_tear_sheet(
        panel, prices, avgretplot=(0, 1), long_short=False, std_bar=False, return_fig=True
    )
    assert len(figures) == 1
    np.testing.assert_allclose(figures[0].data[0].y, [15.75, 17.25], rtol=0, atol=1e-14)
    assert figures[0].layout.meta["units"] == "supplied_levels"


@pytest.mark.parametrize("route", ["turnover", "event_returns", "event_study"])
def test_invalid_prepared_sample_cannot_overwrite_existing_familiar_report(route, tmp_path):
    pytest.importorskip("plotly")
    panel, prices = _inputs()
    panel.iloc[0, panel.columns.get_loc("1D")] = np.nan
    target = tmp_path / "retained.html"
    target.write_text("Retain the previous report", encoding="utf-8")
    args = (panel,) if route == "turnover" else (panel, prices)
    with pytest.raises(ValueError, match="complete and finite"):
        getattr(tears, f"create_{route}_tear_sheet")(*args, output_file=target)
    assert target.read_text(encoding="utf-8") == "Retain the previous report"


@pytest.fixture
def reference_workflow(monkeypatch):
    pytest.importorskip("plotly")
    reference = pytest.importorskip("alphalens.tears")
    performance = pytest.importorskip("alphalens.performance")
    profile = version("alphalens-reloaded")
    if profile not in {"0.5.0", "0.4.6"}:
        pytest.skip("Corrected and frozen reference profiles are qualified separately")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plotted = {"lines": [], "bars": []}
    original_show = plt.show

    def observe_actual_render(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            figure.canvas.draw()
            for axis in figure.axes:
                plotted["lines"].extend(
                    np.asarray(line.get_ydata(), dtype=float) for line in axis.lines
                )
                plotted["bars"].append([float(rect.get_height()) for rect in axis.patches])
        return original_show(*args, **kwargs)

    monkeypatch.setattr(plt, "show", observe_actual_render)
    try:
        yield reference, performance, profile, plotted
    finally:
        plt.close("all")


@pytest.fixture
def shared_reference_inputs(reference_workflow):
    from alphalens import utils as reference_utils

    native, prices = _inputs()
    factor = prices.pct_change(fill_method=None).iloc[1:9].stack(future_stack=True)
    factor.index = factor.index.set_names(["date", "asset"])
    reference = reference_utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2),
        quantiles=2,
        groupby=dict(zip(prices.columns, ["left"] * 3 + ["right"] * 3, strict=True)),
        filter_zscore=None,
        max_loss=0,
    )
    pd.testing.assert_frame_equal(native, reference, check_exact=False, rtol=1e-13, atol=1e-14)
    return native, reference, prices


def _aligned_offsets(trace, target):
    actual = pd.Index(trace.x)
    assert actual.is_unique and target.index.is_unique
    assert all(
        isinstance(label, int | np.integer) and not isinstance(label, bool | np.bool_)
        for label in [*actual, *target.index]
    )
    assert sorted(actual.tolist()) == sorted(target.index.tolist())
    return target.reindex(actual).to_numpy()


def _contains_rendered_line(plotted, expected):
    expected = np.asarray(expected, dtype=float)
    return any(
        line.shape == expected.shape
        and np.allclose(line, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        for line in plotted["lines"]
    )


@pytest.mark.parametrize("periods", [None, ["2D", "1D"]])
def test_actual_reference_turnover_sheet_preserves_default_custom_lags_and_dated_samples(
    periods, reference_workflow, shared_reference_inputs
):
    reference, performance, profile, plotted = reference_workflow
    native, data, _ = shared_reference_inputs
    reference.create_turnover_tear_sheet(data, turnover_periods=periods, set_context=False)
    assert plotted["lines"], "The actual whole reference sheet must render"
    figures = tears.create_turnover_tear_sheet(
        native, turnover_periods=periods, reference_version=profile, return_fig=True
    )
    lags = [1, 2] if periods is None else [2, 1]
    assert [figure.layout.meta["lag"] for figure in figures] == lags * 2
    for figure in figures:
        lag = figure.layout.meta["lag"]
        rank = "turnover_method" not in figure.layout.meta
        for trace in figure.data:
            target = (
                performance.factor_rank_autocorrelation(data, lag)
                if rank
                else performance.quantile_turnover(
                    data["factor_quantile"], int(trace.name[1:]), lag
                )
            )
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(trace.x), target.index, check_names=False, exact=False
            )
            np.testing.assert_allclose(
                np.asarray(trace.y, dtype=float),
                target.to_numpy(),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
            assert _contains_rendered_line(plotted, target.to_numpy())


@pytest.mark.parametrize(
    "long_short,group_neutral,by_group,std_bar", list(itertools.product([False, True], repeat=4))
)
def test_actual_reference_event_return_sheet_keeps_groups_deviations_and_level_samples(
    long_short, group_neutral, by_group, std_bar, reference_workflow, shared_reference_inputs
):
    reference, performance, profile, plotted = reference_workflow
    native, data, prices = shared_reference_inputs
    options = {
        "avgretplot": (2, 3),
        "long_short": long_short,
        "group_neutral": group_neutral,
        "by_group": by_group,
        "std_bar": std_bar,
    }
    reference.create_event_returns_tear_sheet(data, prices, **options, set_context=False)
    assert plotted["lines"], "The actual whole reference sheet must render"
    figures = tears.create_event_returns_tear_sheet(
        native, prices, **options, reference_version=profile, return_fig=True
    )
    expected = performance.average_cumulative_return_by_quantile(
        data, prices, 2, 3, demeaned=long_short, group_adjust=group_neutral
    )
    assert len(figures) == 1 + (2 if std_bar else 0) + (2 if by_group else 0)
    for trace in figures[0].data:
        target = expected.loc[(int(trace.name[1:]), "mean")]
        np.testing.assert_allclose(
            np.asarray(trace.y, dtype=float),
            _aligned_offsets(trace, target),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        assert _contains_rendered_line(plotted, target.to_numpy() * 10000)
    if std_bar:
        for figure in figures[1:3]:
            quantile = int(figure.layout.title.text.split()[-1])
            trace = figure.data[0]
            np.testing.assert_allclose(
                np.asarray(trace.error_y.array, dtype=float),
                _aligned_offsets(trace, expected.loc[(quantile, "std")]),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
    if by_group:
        groups = performance.average_cumulative_return_by_quantile(
            data, prices, 2, 3, demeaned=long_short, group_adjust=group_neutral, by_group=True
        )
        for figure in figures[-2:]:
            group = figure.layout.title.text.removeprefix("Group: ")
            for trace in figure.data:
                target = groups.xs(group, level="group").loc[(int(trace.name[1:]), "mean")]
                np.testing.assert_allclose(
                    np.asarray(trace.y, dtype=float),
                    _aligned_offsets(trace, target),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
                assert not trace.error_y.visible


@pytest.mark.parametrize("rate,paths,reverse", list(itertools.product([False, True], repeat=3)))
def test_actual_reference_event_study_keeps_bins_raw_rate_samples_and_first_horizon(
    rate, paths, reverse, reference_workflow, shared_reference_inputs
):
    from alphalens import utils as reference_utils

    reference, performance, profile, plotted = reference_workflow
    native, _, prices = shared_reference_inputs
    columns = ["2D", "1D"] if reverse else ["1D", "2D"]
    data = native[[*columns, "factor", "group", "factor_quantile"]].copy()
    window = (2, 3) if paths else None
    reference.create_event_study_tear_sheet(
        data, prices, avgretplot=window, rate_of_ret=rate, n_bars=5, set_context=False
    )
    assert plotted["bars"], "The actual whole reference sheet must render"
    figures = tears.create_event_study_tear_sheet(
        data,
        prices,
        avgretplot=window,
        rate_of_ret=rate,
        n_bars=5,
        reference_version=profile,
        return_fig=True,
    )
    assert len(figures) == (6 if paths else 3)
    expected, _ = performance.mean_return_by_quantile(data, demeaned=False)
    daily, _ = performance.mean_return_by_quantile(data, by_date=True, demeaned=False)
    if rate:
        expected = expected.apply(reference_utils.rate_of_return, axis=0, base_period=columns[0])
        daily = daily.apply(reference_utils.rate_of_return, axis=0, base_period=columns[0])
    bars, violin = figures[-2:]
    assert [trace.name for trace in bars.data] == columns
    for trace in bars.data:
        np.testing.assert_allclose(
            np.asarray(trace.y, dtype=float),
            expected[trace.name].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    for trace in violin.data:
        np.testing.assert_allclose(
            np.asarray(trace.y, dtype=float),
            daily[trace.name].dropna().to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    counts = np.asarray(figures[0].data[0].y, dtype=float)
    assert counts.sum() == len(data)
    assert any(
        len(heights) == len(counts) and np.array_equal(heights, counts)
        for heights in plotted["bars"]
    )
    if paths:
        target = performance.average_cumulative_return_by_quantile(
            data, prices, 2, 3, demeaned=False
        )
        for trace in figures[1].data:
            np.testing.assert_allclose(
                np.asarray(trace.y, dtype=float),
                _aligned_offsets(trace, target.loc[(int(trace.name[1:]), "mean")]),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
