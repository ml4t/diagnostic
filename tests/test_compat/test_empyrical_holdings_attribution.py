"""Supplied holdings retain factor identity, reference missingness and caller ownership."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep


def inputs():
    dates = pd.bdate_range("2025-07-07", periods=3, tz="UTC", name="date")
    keys = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["dt", "ticker"])
    returns = pd.Series([0.03, -0.01, 0.02], index=dates)
    weights = pd.Series([0.5, 0.5, 0.25, 0.75, 0.5, 0.5], index=keys)
    loadings = pd.DataFrame(
        {"market": [2, 1, 1, 1, 2, 1], "style": [0, 0.5, 2, 0, 0, 0.5]},
        index=keys,
        dtype=float,
    )
    factors = pd.DataFrame(
        {"style": [0.02, 0.01, 0.03], "market": [0.01, -0.02, 0.005]},
        index=dates,
    )
    return returns, weights, factors, loadings


def test_familiar_holdings_workflow_preserves_independent_exposures_and_attribution_identities():
    returns, weights, factors, loadings = inputs()
    exposures, table = ep.perf_attrib(returns, weights, factors, loadings)
    np.testing.assert_allclose(exposures, [[1.5, 0.25], [1.0, 0.5], [1.5, 0.25]])
    np.testing.assert_allclose(table["common_returns"], [0.02, -0.015, 0.015])
    np.testing.assert_allclose(table["specific_returns"], [0.01, 0.005, 0.005])
    np.testing.assert_allclose(table["tilt_returns"], [0.02, -7 / 300, 1 / 60])
    assert list(table.columns) == [
        "market",
        "style",
        "total_returns",
        "common_returns",
        "specific_returns",
        "tilt_returns",
        "timing_returns",
    ]
    assert exposures.index.name == "dt" and table.index.name is None
    pd.testing.assert_frame_equal(
        exposures, ep.compute_exposures(weights, loadings), check_freq=False
    )


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "missing_loading_cell",
        "missing_loading_row",
        "missing_weight_day",
        "all_zero_weights",
        "empty_weights",
        "empty_loadings",
        "missing_factor_return",
        "missing_portfolio_return",
        "missing_factor_day",
    ],
)
def test_real_reference_attribution_keeps_keys_columns_masks_and_periodic_values(case):
    reference = pytest.importorskip("empyrical")
    returns, weights, factors, loadings = inputs()
    if case == "missing_loading_cell":
        loadings.iloc[1, 0] = np.nan
    elif case == "missing_loading_row":
        loadings = loadings.iloc[[0, 2, 3, 4, 5]]
    elif case == "missing_weight_day":
        weights = weights.iloc[[0, 1, 4, 5]]
    elif case == "all_zero_weights":
        weights[:] = 0
    elif case == "empty_weights":
        weights = weights.iloc[:0]
    elif case == "empty_loadings":
        loadings = loadings.iloc[:0]
    elif case == "missing_factor_return":
        factors.iloc[1, 0] = np.nan
    elif case == "missing_portfolio_return":
        returns.iloc[1] = np.nan
    elif case == "missing_factor_day":
        factors = factors.iloc[[0, 2]]
    original = [frame.copy(deep=True) for frame in (returns, weights, factors, loadings)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = ep.perf_attrib(returns, weights, factors, loadings)
        expected = reference.perf_attrib(*[frame.copy(deep=True) for frame in original])
        for got, wanted in zip(actual, expected, strict=True):
            pd.testing.assert_frame_equal(got, wanted, rtol=1e-12, atol=1e-15)
        pd.testing.assert_frame_equal(
            ep.compute_exposures(weights, loadings),
            reference.compute_exposures(weights.copy(), loadings.copy()),
            rtol=1e-12,
            atol=1e-15,
        )
    for before, after in zip(original, (returns, weights, factors, loadings), strict=True):
        if isinstance(before, pd.Series):
            pd.testing.assert_series_equal(before, after)
        else:
            pd.testing.assert_frame_equal(before, after)


def test_legacy_uncovered_loadings_warn_instead_of_claiming_complete_zero_exposure():
    returns, weights, factors, loadings = inputs()
    loadings.iloc[1, 0] = np.nan
    with pytest.warns(UserWarning, match="uncovered"):
        ep.perf_attrib(returns, weights, factors, loadings)
