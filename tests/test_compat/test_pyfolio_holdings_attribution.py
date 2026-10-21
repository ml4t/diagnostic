"""Familiar dollar books must share attribution without renormalizing missing assets."""

import warnings

import numpy as np
import pandas as pd
import pytest
from test_empyrical_holdings_attribution import inputs

from ml4t.diagnostic.compat.pyfolio import perf_attrib as pf


def account_inputs():
    returns, weights, factors, loadings = inputs()
    positions = weights.unstack("ticker") * 1000
    positions["cash"] = 0.0
    return returns, positions, factors, loadings


def test_dollar_and_fractional_account_books_retain_independent_full_exposures():
    returns, positions, factors, loadings = account_inputs()
    dollars = pf.perf_attrib(returns, positions, factors, loadings)
    fractional = pf.perf_attrib(returns, positions / 1000, factors, loadings, pos_in_dollars=False)
    for left, right in zip(dollars, fractional, strict=True):
        pd.testing.assert_frame_equal(left, right)
    np.testing.assert_allclose(dollars[0], [[1.5, 0.25], [1.0, 0.5], [1.5, 0.25]])
    np.testing.assert_allclose(dollars[1]["common_returns"], [0.02, -0.015, 0.015])


def test_uncovered_asset_cannot_change_the_full_account_nav_denominator():
    returns, positions, factors, loadings = account_inputs()
    loadings = loadings.loc[loadings.index.get_level_values("ticker") == "A"]
    with pytest.warns(UserWarning, match="uncovered"):
        exposures, table = pf.perf_attrib(returns, positions, factors, loadings)
    np.testing.assert_allclose(exposures["market"], [1.0, 0.25, 1.0])
    np.testing.assert_allclose(table["common_returns"], [0.01, 0.0, 0.005], atol=1e-15)
    with pytest.warns(UserWarning, match="uncovered"):
        fractional = pf.perf_attrib(
            returns, positions / 1000, factors, loadings, pos_in_dollars=False
        )
    for left, right in zip((exposures, table), fractional, strict=True):
        pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize(
    "case", ["dollars", "fractions", "missing_row", "missing_day", "high_turnover"]
)
def test_actual_pyfolio_preprocessing_tables_statistics_and_caller_ownership(case):
    reference = pytest.importorskip("pyfolio.perf_attrib")
    returns, positions, factors, loadings = account_inputs()
    dollars = case != "fractions"
    transactions = None
    if not dollars:
        positions /= 1000
    if case == "missing_row":
        loadings = loadings.iloc[[0, 2, 3, 4, 5]]
    elif case == "missing_day":
        loadings = loadings.loc[loadings.index.get_level_values("dt") != returns.index[1]]
    elif case == "high_turnover":
        transactions = pd.DataFrame(
            {"symbol": ["A"], "amount": [100.0], "price": [100.0]}, index=returns.index[:1]
        )
    original = [frame.copy(deep=True) for frame in (returns, positions, factors, loadings)]
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        actual = pf.perf_attrib(returns, positions, factors, loadings, transactions, dollars)
    if case == "high_turnover":
        assert any("High turnover" in str(item.message) for item in observed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = reference.perf_attrib(
            *[frame.copy(deep=True) for frame in original],
            transactions=transactions,
            pos_in_dollars=dollars,
        )
        expected_stats = reference.create_perf_attrib_stats(expected[1], expected[0])
    for got, wanted in zip(actual, expected, strict=True):
        pd.testing.assert_frame_equal(got, wanted, rtol=1e-12, atol=1e-15)
    actual_stats = pf.create_perf_attrib_stats(actual[1], actual[0])
    pd.testing.assert_series_equal(actual_stats[0], expected_stats[0], rtol=1e-12, atol=1e-15)
    pd.testing.assert_frame_equal(actual_stats[1], expected_stats[1], rtol=1e-12, atol=1e-15)
    for before, after in zip(original, (returns, positions, factors, loadings), strict=True):
        if isinstance(before, pd.Series):
            pd.testing.assert_series_equal(before, after)
        else:
            pd.testing.assert_frame_equal(before, after)


def test_component_statistics_keep_explicit_annualization_and_named_nonadditive_compounding():
    exposures, table = pf.perf_attrib(*account_inputs())
    summary, _ = pf.create_perf_attrib_stats(table, exposures, periods_per_year=12)
    expected_growth = 1.01 * 1.005 * 1.005
    assert summary["Annualized Specific Return"] == pytest.approx(expected_growth**4 - 1)
    assert summary["Cumulative Specific Return"] == pytest.approx(expected_growth - 1)
    assert summary["Cumulative Specific Return"] != pytest.approx(
        0.01 + 1.03 * 0.005 + (1.03 * 0.99) * 0.005
    )


def test_reference_missing_asset_normalization_error_is_explicitly_corrected_not_tolerance_matched():
    reference = pytest.importorskip("pyfolio.perf_attrib")
    returns, positions, factors, loadings = account_inputs()
    loadings = loadings.loc[loadings.index.get_level_values("ticker") == "A"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = reference.perf_attrib(
            returns.copy(), positions.copy(), factors.copy(), loadings.copy()
        )
        current = pf.perf_attrib(returns, positions, factors, loadings)
    np.testing.assert_allclose(legacy[0]["market"], [2.0, 1.0, 2.0])
    np.testing.assert_allclose(current[0]["market"], [1.0, 0.25, 1.0])
    np.testing.assert_allclose(positions.sum(axis=1), [1000.0, 1000.0, 1000.0])


@pytest.mark.parametrize("shape", ["wide_dollars", "wide_fractions", "stacked_fractions"])
def test_actual_exposure_entrypoint_preserves_each_stacking_and_position_unit_option(shape):
    reference = pytest.importorskip("pyfolio.perf_attrib")
    _, positions, _, loadings = account_inputs()
    dollars = shape == "wide_dollars"
    stacked = shape == "stacked_fractions"
    if not dollars:
        positions /= 1000
    if stacked:
        positions = positions.drop(columns="cash").stack(future_stack=True)
        positions.index = positions.index.set_names(["dt", "ticker"])
    original = positions.copy(deep=True)
    actual = pf.compute_exposures(positions, loadings, not stacked, dollars)
    expected = reference.compute_exposures(positions.copy(), loadings.copy(), not stacked, dollars)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-15)
    if stacked:
        pd.testing.assert_series_equal(positions, original)
    else:
        pd.testing.assert_frame_equal(positions, original)


def test_printable_public_statistics_return_owned_tables_without_ipython(capsys):
    summary, factors = pf.show_perf_attrib_stats(*account_inputs(), periods_per_year=12)
    text = capsys.readouterr().out
    assert "Annualized Specific Return" in text and "Average Risk Factor Exposure" in text
    assert summary["Annualized Specific Return"] == pytest.approx((1.01 * 1.005 * 1.005) ** 4 - 1)
    np.testing.assert_allclose(factors["Average Risk Factor Exposure"], [4 / 3, 1 / 3])


def test_missing_account_cash_cannot_silently_be_treated_as_a_complete_nav_book():
    returns, positions, factors, loadings = account_inputs()
    with pytest.raises(ValueError, match="cash"):
        pf.perf_attrib(returns, positions.drop(columns="cash"), factors, loadings)


@pytest.mark.parametrize("reverse_loadings", [False, True])
def test_numeric_and_text_asset_identifiers_remain_distinct_across_independent_loading_order(
    reverse_loadings,
):
    returns, positions, factors, loadings = account_inputs()
    positions = positions.rename(columns={"A": 1, "B": "1"})
    loadings = loadings.rename(index={"A": 1, "B": "1"}, level="ticker")
    if reverse_loadings:
        loadings = loadings.iloc[[1, 0, 3, 2, 5, 4]]
    before = positions.copy(deep=True), loadings.copy(deep=True)
    exposures, table = pf.perf_attrib(returns, positions, factors, loadings)
    np.testing.assert_allclose(exposures, [[1.5, 0.25], [1.0, 0.5], [1.5, 0.25]])
    np.testing.assert_allclose(table["common_returns"], [0.02, -0.015, 0.015])
    pd.testing.assert_frame_equal(positions, before[0])
    pd.testing.assert_frame_equal(loadings, before[1])


def test_actual_reference_mixed_asset_labels_use_the_same_joint_identity_mapping():
    reference = pytest.importorskip("pyfolio.perf_attrib")
    returns, positions, factors, loadings = account_inputs()
    positions = positions.rename(columns={"A": 1, "B": "1"})
    loadings = loadings.rename(index={"A": 1, "B": "1"}, level="ticker").iloc[[1, 0, 3, 2, 5, 4]]
    actual = pf.perf_attrib(returns, positions, factors, loadings)
    expected = reference.perf_attrib(returns, positions, factors, loadings)
    for got, wanted in zip(actual, expected, strict=True):
        pd.testing.assert_frame_equal(got, wanted, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("invalid", [None, ""])
def test_internal_identity_transport_does_not_turn_missing_asset_labels_into_valid_codes(invalid):
    from ml4t.diagnostic.compat import empyrical as ep

    returns, positions, factors, loadings = inputs()
    positions = positions.rename(index={"A": invalid}, level="ticker")
    loadings = loadings.rename(index={"A": invalid}, level="ticker")
    with pytest.raises(ValueError, match="asset IDs.*nonmissing.*nonempty"):
        ep.perf_attrib(returns, positions, factors, loadings)
