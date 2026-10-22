"""The whole familiar positions report must expose real samples and hide identities."""

import json
import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
from plotly.utils import PlotlyJSONEncoder

from ml4t.diagnostic.compat import pyfolio as pf


def _inputs():
    dates = pd.bdate_range("2025-01-29", periods=5, tz="UTC")
    prices = pd.Series(
        [100.0, 101, 99, 103, 102, 104], index=pd.bdate_range("2025-01-28", periods=6, tz="UTC")
    )
    returns = prices.pct_change(fill_method=None).iloc[1:]
    positions = pd.DataFrame(
        {
            "PRIVATE_ASSET_A": [60.0, 0, 80, 60, -20],
            "PRIVATE_ASSET_B": [-20.0, 0, np.nan, -10, 10],
            "cash": [60.0, 100, 20, np.nan, -40],
        },
        index=dates,
    )
    return returns, positions


def test_price_and_supplied_positions_produce_the_complete_original_manifest_without_reconstructing_returns(
    tmp_path,
):
    returns, positions = _inputs()
    before, returns_before = positions.copy(deep=True), returns.copy(deep=True)
    report = pf.create_position_tear_sheet(
        returns, positions, estimate_intraday=False, return_fig=True
    )
    assert set(report.figures) == {
        "Exposure",
        "Top positions",
        "Position concentration",
        "Holdings",
        "Long and short holdings",
        "Gross leverage",
        "Top long positions",
        "Top short positions",
        "Top absolute positions",
    }
    assert report.metrics is None
    assert report.availability["positions"]["snapshot_kind"] == "supplied"
    np.testing.assert_allclose(report.figures["Exposure"].data[0].y, [0.6, 0, 0.8, 1.2, -0.2])
    np.testing.assert_allclose(report.figures["Exposure"].data[1].y, [-0.2, 0, 0, -0.2, 0.4])
    np.testing.assert_allclose(report.figures["Exposure"].data[2].y, [0.4, 0, 0.8, 1, 0.2])
    np.testing.assert_allclose(report.figures["Gross leverage"].data[0].y, [0.8, 0, 0.8, 1.4, -0.6])
    assert report.figures["Holdings"].data[0].y == (2, 0, 1, 2, 2)
    assert report.figures["Long and short holdings"].data[0].y == (1, 0, 1, 1, 1)
    assert report.figures["Long and short holdings"].data[1].y == (1, 0, 0, 1, 1)
    report.save_html(tmp_path / "positions.html", include_plotlyjs=True)
    assert "PRIVATE_ASSET_A" in (tmp_path / "positions.html").read_text()
    pd.testing.assert_frame_equal(positions, before)
    pd.testing.assert_series_equal(returns, returns_before)


@pytest.mark.parametrize(
    "mode, top_chart, tables", [(0, True, False), (1, False, True), (2, True, True)]
)
def test_top_position_selection_controls_actual_manifest_instead_of_allocating_blank_panels(
    mode, top_chart, tables
):
    returns, positions = _inputs()
    report = pf.create_position_tear_sheet(
        returns, positions, show_and_plot_top_pos=mode, estimate_intraday=False, return_fig=True
    )
    assert ("Top positions" in report.figures) is top_chart
    assert ("Top long positions" in report.figures) is tables


def test_hidden_positions_are_absent_from_html_export_and_actual_notebook_payloads(
    tmp_path, capsys
):
    returns, positions = _inputs()
    report = pf.create_position_tear_sheet(
        returns,
        positions,
        show_and_plot_top_pos=2,
        hide_positions=True,
        sector_mappings={"PRIVATE_ASSET_A": "PRIVATE_ASSET_A"},
        estimate_intraday=False,
        return_fig=True,
    )
    target = tmp_path / "private.html"
    report.save_html(target, include_plotlyjs=True)
    payloads = [target.read_text(), json.dumps(report.to_dict(), cls=PlotlyJSONEncoder)]
    import plotly.io as pio

    previous = pio.renderers.default
    try:
        pio.renderers.default = "plotly_mimetype"
        for figure in report.figures.values():
            bundle = figure._repr_mimebundle_()
            assert "application/vnd.plotly.v1+json" in bundle
            payloads.append(json.dumps(bundle, cls=PlotlyJSONEncoder))
    finally:
        pio.renderers.default = previous
    assert "Top long positions" not in report.figures
    assert all("PRIVATE_ASSET" not in payload for payload in payloads)
    assert "PRIVATE_ASSET" not in capsys.readouterr().out
    assert positions.columns.tolist() == ["PRIVATE_ASSET_A", "PRIVATE_ASSET_B", "cash"]


def test_sector_report_discloses_legacy_exclusion_and_normalization_instead_of_claiming_native_full_nav():
    returns, positions = _inputs()
    with pytest.warns(UserWarning, match="PRIVATE_ASSET_B"):
        report = pf.create_position_tear_sheet(
            returns,
            positions,
            sector_mappings={"PRIVATE_ASSET_A": "Technology"},
            estimate_intraday=False,
            return_fig=True,
        )
    assert report.figures["Sector allocations"].data[0].y[0] == 0.5
    assert report.availability["sectors"]["denominator"] == "mapped_sector_values_plus_cash"


def test_duplicate_intraday_estimate_candidates_keep_separate_observation_identity_and_exact_time():
    returns, positions = _inputs()
    selected = positions.iloc[[0, 0, 1]].copy()
    report = pf.create_position_tear_sheet(
        returns, selected, estimate_intraday=False, return_fig=True
    )
    figure = report.figures["Exposure"]
    assert len(figure.data[0].x) == len(set(figure.data[0].x)) == 3
    assert list(figure.data[0].customdata)[:2] == [positions.index[0].isoformat()] * 2


def test_familiar_holdings_count_does_not_turn_a_missing_allocation_into_an_observed_position():
    returns, positions = _inputs()
    positions.iloc[0] = [100, -100, 0]
    report = pf.create_position_tear_sheet(
        returns, positions, estimate_intraday=False, return_fig=True
    )
    assert report.figures["Holdings"].data[0].y[0] == 2
    assert report.availability["positions"]["nonfinite_nav_fractions"] > 0
    assert not np.isfinite(report.figures["Gross leverage"].data[0].y[0])


def test_invalid_top_display_selector_leaves_inputs_unchanged():
    returns, positions = _inputs()
    original = positions.copy(deep=True)
    with pytest.raises(ValueError, match="show_and_plot"):
        pf.create_position_tear_sheet(returns, positions, show_and_plot_top_pos=3, return_fig=True)
    pd.testing.assert_frame_equal(positions, original)


def _intraday_inputs():
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    returns = pd.Series([0.0, 0.005], index=dates)
    positions = pd.DataFrame({"A": [0.0, 0], "cash": [1000.0, 1005.0]}, index=dates)
    transactions = pd.DataFrame(
        {"symbol": ["A"] * 5, "amount": [1.0, -2, 1, 1, -1], "price": [100.0, 100, 100, 100, 105]},
        index=pd.to_datetime(
            [
                "2025-07-07T14:00Z",
                "2025-07-07T15:00Z",
                "2025-07-07T16:00Z",
                "2025-07-08T14:00Z",
                "2025-07-08T15:00Z",
            ]
        ),
    )
    return returns, positions, transactions


@pytest.mark.parametrize("estimate", [True, "infer", False, None])
def test_intraday_selector_preserves_actual_estimated_candidates_and_does_not_relabel_them_observed(
    estimate,
):
    import warnings

    returns, positions, transactions = _intraday_inputs()
    original = positions.copy(deep=True)
    with warnings.catch_warnings(record=True) as messages:
        report = pf.create_position_tear_sheet(
            returns,
            positions,
            transactions=transactions,
            estimate_intraday=estimate,
            return_fig=True,
        )
    inferred = estimate in (True, "infer")
    assert report.availability["positions"]["snapshot_kind"] == (
        "estimated_intraday" if inferred else "supplied"
    )
    assert report.availability["positions"]["n_observations"] == (3 if inferred else 2)
    assert report.availability["positions"]["duplicate_time_observations"] == int(inferred)
    if inferred:
        np.testing.assert_allclose(report.figures["Exposure"].data[0].y[:2], [0.1, 0])
        np.testing.assert_allclose(report.figures["Exposure"].data[1].y[:2], [0, -0.1])
    if estimate == "infer":
        assert any("intraday" in str(message.message).lower() for message in messages)
    pd.testing.assert_frame_equal(positions, original)


def test_standalone_hidden_top_chart_never_exposes_original_trace_names_or_tables():
    returns, positions = _inputs()
    figure = pf.plotting.show_and_plot_top_positions(
        returns, pf.pos.get_percent_alloc(positions), hide_positions=True, show_and_plot=1
    )
    assert len(figure.data) == 2
    assert all("PRIVATE_ASSET" not in trace.name for trace in figure.data)


def test_all_cash_report_distinguishes_zero_exposure_from_undefined_position_concentration():
    returns, positions = _inputs()
    report = pf.create_position_tear_sheet(
        returns, positions[["cash"]].fillna(100), estimate_intraday=False, return_fig=True
    )
    np.testing.assert_allclose(report.figures["Exposure"].data[0].y, 0)
    assert all(value == 0 for value in report.figures["Holdings"].data[0].y)
    assert report.figures["Top positions"].layout.annotations[0].text == "No asset holdings to rank"
    assert (
        report.figures["Position concentration"].layout.annotations[0].text
        == "No finite sample values"
    )


def _assert_filled_reference_sample(trace, collection, dates):
    import matplotlib.dates as mdates

    vertices = np.concatenate([path.vertices for path in collection.get_paths()])
    for actual, instant in zip(trace.y, mdates.date2num(dates.to_pydatetime()), strict=True):
        values = np.unique(vertices[vertices[:, 0] == instant, 1])
        values = values[values != 0]
        assert actual == pytest.approx(values[0] if len(values) else 0)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("sectors", [False, True])
def test_actual_reference_entire_positions_sheet_and_standalone_traces_execute_with_matched_options(
    mode, sectors
):
    reference = pytest.importorskip("pyfolio.tears")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    assert version("pyfolio-reloaded") == "0.10.0"
    returns, positions = _inputs()
    mapping = {"PRIVATE_ASSET_A": "Technology", "PRIVATE_ASSET_B": "Finance"} if sectors else None
    options = {
        "show_and_plot_top_pos": mode,
        "sector_mappings": mapping,
        "estimate_intraday": False,
        "return_fig": True,
    }
    try:
        original = reference.create_position_tear_sheet(returns, positions, **options)
        original.set_dpi(30)
        original.canvas.draw()
        actual = pf.create_position_tear_sheet(returns, positions, **options)
        assert len(original.axes) == 6 + sectors
        np.testing.assert_allclose(
            actual.figures["Exposure"].data[2].y,
            original.axes[0].lines[0].get_ydata(),
            equal_nan=True,
        )
        for number in range(2):
            _assert_filled_reference_sample(
                actual.figures["Exposure"].data[number],
                original.axes[0].collections[number],
                positions.index,
            )
            _assert_filled_reference_sample(
                actual.figures["Long and short holdings"].data[number],
                original.axes[4].collections[number],
                positions.index,
            )
        np.testing.assert_allclose(
            actual.figures["Gross leverage"].data[0].y,
            original.axes[5].lines[0].get_ydata(),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual.figures["Holdings"].data[0].y,
            original.axes[3].lines[0].get_ydata(),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual.figures["Holdings"].data[1].y,
            original.axes[3].lines[1].get_ydata(),
            equal_nan=True,
        )
        for number in range(4):
            np.testing.assert_allclose(
                actual.figures["Position concentration"].data[number].y,
                original.axes[2].lines[number].get_ydata(),
                equal_nan=True,
            )
        if mode != 1:
            for number in range(len(original.axes[1].lines)):
                np.testing.assert_allclose(
                    actual.figures["Top positions"].data[number].y,
                    original.axes[1].lines[number].get_ydata(),
                    equal_nan=True,
                )
        if sectors:
            for number in range(len(original.axes[6].lines)):
                np.testing.assert_allclose(
                    actual.figures["Sector allocations"].data[number].y,
                    original.axes[6].lines[number].get_ydata(),
                    equal_nan=True,
                )
    finally:
        pyplot.close("all")


@pytest.mark.parametrize("estimate", [True, "infer", False, None])
def test_actual_reference_position_report_intraday_selection_keeps_matching_original_trace_membership(
    estimate,
):
    reference = pytest.importorskip("pyfolio.tears")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    returns, positions, transactions = _intraday_inputs()
    options = {"transactions": transactions, "estimate_intraday": estimate, "return_fig": True}
    try:
        original = reference.create_position_tear_sheet(returns, positions, **options)
        original.set_dpi(30)
        original.canvas.draw()
        actual = pf.create_position_tear_sheet(returns, positions, **options)
        for panel, axis in (("Gross leverage", 5), ("Holdings", 3)):
            np.testing.assert_allclose(
                actual.figures[panel].data[0].y,
                original.axes[axis].lines[0].get_ydata(),
                equal_nan=True,
            )
    finally:
        pyplot.close("all")


def test_mandatory_positions_report_reference_is_present_and_runs_actual_sheet():
    if os.environ.get("ML4T_REQUIRED_PYFOLIO_REFERENCE") is None:
        pytest.skip("Actual reference workflow is mandatory only in its explicit lane")
    import pyfolio.tears as reference

    assert version("pyfolio-reloaded") == os.environ["ML4T_REQUIRED_PYFOLIO_REFERENCE"] == "0.10.0"
    assert reference.create_position_tear_sheet is not None
