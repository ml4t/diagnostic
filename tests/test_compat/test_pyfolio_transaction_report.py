"""Complete transaction reports must use the actual fills and selected denominator."""

import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import pyfolio as pf


def _inputs():
    dates = pd.date_range("2025-07-07", periods=3, tz="America/New_York")
    returns = pd.Series([0.01, -0.02, 0.03], index=dates)
    positions = pd.DataFrame({"A": [200.0, 400, 600], "cash": [800.0, 600, 400]}, index=dates)
    transactions = pd.DataFrame(
        {"symbol": ["A"] * 3, "amount": [1.0, -1, 2], "price": [100.0] * 3},
        index=pd.to_datetime(
            ["2025-07-07T15:00Z", "2025-07-07T16:00Z", "2025-07-09T15:00Z"]
        ).tz_convert("America/New_York"),
    )
    return returns, positions, transactions


def test_existing_public_turnover_calculation_already_matches_independent_fill_values():
    returns, positions, transactions = _inputs()
    figure = pf.plotting.plot_turnover(returns, transactions, positions)
    np.testing.assert_allclose(figure.data[0].y, [2, 0, 0.4])


@pytest.mark.parametrize(
    "denominator, expected", [("AGB", [2.0, 0, 0.4]), ("portfolio_value", [0.2, 0, 0.2])]
)
def test_complete_transaction_manifest_retains_per_panel_samples_and_does_not_infer_trades(
    denominator, expected, tmp_path
):
    inputs = _inputs()
    originals = [value.copy(deep=True) for value in inputs]
    report = pf.create_txn_tear_sheet(
        *inputs, turnover_denom=denominator, estimate_intraday=False, return_fig=True
    )
    assert set(report.figures) == {
        "Turnover",
        "Daily volume",
        "Turnover distribution",
        "Transaction timing",
    }
    np.testing.assert_allclose(report.figures["Turnover"].data[0].y, expected)
    np.testing.assert_allclose(report.figures["Turnover distribution"].data[0].x, expected)
    np.testing.assert_allclose(report.figures["Daily volume"].data[0].y, [2, 2])
    assert report.availability["transactions"]["n_actual_fills"] == 3
    assert report.availability["transactions"]["turnover_denom"] == denominator
    report.save_html(tmp_path / "transactions.html", include_plotlyjs=True)
    for actual, original in zip(inputs, originals, strict=True):
        if isinstance(actual, pd.Series):
            pd.testing.assert_series_equal(actual, original)
        else:
            pd.testing.assert_frame_equal(actual, original)


def test_transaction_report_passes_explicit_annualization_to_all99_additional_slippage_scenarios():
    returns, positions, transactions = _inputs()
    report = pf.create_txn_tear_sheet(
        returns,
        positions,
        transactions,
        unadjusted_returns=returns,
        estimate_intraday=False,
        return_fig=True,
        periods_per_year=3,
    )
    assert len(report.figures) == 6
    figure = report.figures["Slippage sensitivity"]
    assert list(figure.data[0].x) == list(range(1, 100))
    assert figure.layout.meta["periods_per_year"] == 3
    assert figure.data[0].y[9] == pytest.approx(1.0098 * 0.98 * 1.0298 - 1)
    np.testing.assert_allclose(
        report.figures["Slippage sweep"].data[0].y,
        [1.00994, 1.00994 * 0.98, 1.00994 * 0.98 * 1.02994],
    )


def test_empty_actual_ledger_is_not_replaced_with_rebalances_in_transaction_report():
    returns, positions, transactions = _inputs()
    report = pf.create_txn_tear_sheet(
        returns, positions, transactions.iloc[:0], estimate_intraday=False, return_fig=True
    )
    assert report.availability["transactions"]["n_actual_fills"] == 0
    assert len(report.figures["Daily volume"].data[0].x) == 0
    assert report.figures["Transaction timing"].layout.meta["included_fills"] == 0
    np.testing.assert_allclose(report.figures["Turnover"].data[0].y, 0)


def test_actual_utc_fills_and_new_york_position_labels_share_the_declared_book_civil_clock():
    returns, positions, transactions = _inputs()
    utc_fills = transactions.tz_convert("UTC")
    before = utc_fills.copy(deep=True)
    report = pf.create_txn_tear_sheet(
        returns, positions, utc_fills, estimate_intraday=False, return_fig=True
    )
    np.testing.assert_allclose(report.figures["Turnover"].data[0].y, [2, 0, 0.4])
    assert report.availability["transactions"]["civil_day_timezone"] == "America/New_York"
    pd.testing.assert_frame_equal(utc_fills, before)


def test_actual_reference_mixed_zone_daily_alignment_loses_known_fill_turnover():
    reference = pytest.importorskip("pyfolio.txn")
    returns, positions, transactions = _inputs()
    actual = reference.get_turnover(positions, transactions.tz_convert("UTC"))
    assert len(actual) == 5
    np.testing.assert_allclose(actual, 0)
    report = pf.create_txn_tear_sheet(
        returns, positions, transactions.tz_convert("UTC"), estimate_intraday=False, return_fig=True
    )
    np.testing.assert_allclose(report.figures["Turnover"].data[0].y, [2, 0, 0.4])


def test_book_clock_alignment_keeps_day_membership_across_the_new_york_dst_change():
    dates = pd.DatetimeIndex(["2025-03-07", "2025-03-10", "2025-03-11"], tz="America/New_York")
    positions = pd.DataFrame({"A": [100.0] * 3, "cash": [900.0] * 3}, index=dates)
    fills = pd.DataFrame(
        {"amount": [1.0, 1], "price": [100.0, 100]},
        index=pd.to_datetime(["2025-03-07T20:59:00.000000001Z", "2025-03-10T19:59:00.000000001Z"]),
    )
    original = fills.copy(deep=True)
    report = pf.create_txn_tear_sheet(
        pd.Series([0.0] * 3, index=dates),
        positions,
        fills,
        turnover_denom="portfolio_value",
        estimate_intraday=False,
        return_fig=True,
    )
    np.testing.assert_allclose(report.figures["Turnover"].data[0].y, [0.1, 0.1, 0])
    assert report.figures["Transaction timing"].layout.meta["included_fills"] == 2
    pd.testing.assert_frame_equal(fills, original)


def test_naive_fill_instants_are_not_assigned_an_unstated_timezone_for_an_aware_book():
    returns, positions, fills = _inputs()
    fills.index = fills.index.tz_localize(None)
    original = fills.copy(deep=True)
    with pytest.raises(ValueError, match="timezone"):
        pf.create_txn_tear_sheet(
            returns, positions, fills, estimate_intraday=False, return_fig=True
        )
    pd.testing.assert_frame_equal(fills, original)


def test_transaction_report_invalid_denominator_preserves_actual_input_ledger():
    inputs = _inputs()
    before = inputs[2].copy(deep=True)
    with pytest.raises(ValueError, match="denominator|turnover_denom"):
        pf.create_txn_tear_sheet(*inputs, turnover_denom="unknown", return_fig=True)
    pd.testing.assert_frame_equal(before, inputs[2])


@pytest.mark.parametrize("denominator", ["AGB", "portfolio_value"])
@pytest.mark.parametrize("slippage", [False, True])
def test_actual_reference_complete_transaction_sheet_matches_fills_samples_and_optional_slippage(
    denominator, slippage
):
    reference = pytest.importorskip("pyfolio.tears")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    assert version("pyfolio-reloaded") == "0.10.0"
    returns, positions, transactions = _inputs()
    options = {
        "turnover_denom": denominator,
        "unadjusted_returns": returns if slippage else None,
        "estimate_intraday": False,
        "return_fig": True,
    }
    try:
        original = reference.create_txn_tear_sheet(returns, positions, transactions, **options)
        original.set_dpi(30)
        original.canvas.draw()
        actual = pf.create_txn_tear_sheet(returns, positions, transactions, **options)
        assert len(original.axes) == len(actual.figures) == 4 + 2 * slippage
        for panel, axis in (("Turnover", 0), ("Daily volume", 1)):
            for number, trace in enumerate(actual.figures[panel].data):
                np.testing.assert_allclose(
                    trace.y, original.axes[axis].lines[number].get_ydata(), rtol=1e-12
                )
        np.testing.assert_allclose(
            actual.figures["Transaction timing"].data[0].y,
            [patch.get_height() for patch in original.axes[3].patches],
            rtol=1e-12,
        )
        if slippage:
            for number, trace in enumerate(actual.figures["Slippage sweep"].data):
                np.testing.assert_allclose(
                    trace.y, original.axes[4].lines[number].get_ydata(), rtol=1e-12
                )
            np.testing.assert_allclose(
                actual.figures["Slippage sensitivity"].data[0].y,
                original.axes[5].lines[0].get_ydata(),
                rtol=1e-12,
            )
    finally:
        pyplot.close("all")


def test_mandatory_transaction_report_reference_cannot_silently_skip_when_absent():
    required = os.environ.get("ML4T_REQUIRED_PYFOLIO_REFERENCE")
    if required is None:
        pytest.skip("Actual reference workflow is mandatory only in its explicit lane")
    import pyfolio.tears as reference

    assert version("pyfolio-reloaded") == required == "0.10.0"
    assert callable(reference.create_txn_tear_sheet)
