"""Rolling compounding must not replace observed losses with residual wealth."""

import math

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def assert_rolling_products(returns, window):
    result = PortfolioAnalysis(returns).compute_rolling_metrics(
        windows=[window], metrics=["returns"]
    )
    actual = result.returns[window].to_numpy()
    expected = np.full(len(returns), np.nan)
    for end in range(window, len(returns) + 1):
        sample = np.asarray(returns)[end - window : end]
        expected[end - 1] = math.prod(1 + float(value) for value in sample) - 1
        if -1 in sample:
            assert actual[end - 1] == -1
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
    return result


@pytest.mark.parametrize("nav", [[100.0, 110.0, 0.0], [100.0, 0.0], [100.0, 100.0, 105.0, 0.0]])
@pytest.mark.parametrize("window", [1, 2, 3])
def test_returns_produced_by_an_observed_zero_nav_cannot_be_clipped(nav, window):
    values = pd.Series(nav, index=pd.bdate_range("2025-01-02", periods=len(nav)))
    returns = values.pct_change(fill_method=None).iloc[1:]
    result = assert_rolling_products(returns, window)
    assert result.dates.to_list() == returns.index.to_pydatetime().tolist()
    if len(returns) >= window:
        assert result.returns[window][-1] == -1


def test_later_declared_return_window_recovers_its_product_after_total_loss_leaves():
    result = assert_rolling_products(np.array([0.1, -1.0, 0.2, 0.3]), 2)
    assert result.returns[2].to_list()[:3] == [None, -1.0, -1.0]
    assert result.returns[2][-1] == pytest.approx(0.56)


@pytest.mark.parametrize("window", [1, 2, 4, 8])
def test_near_total_losses_keep_the_supplied_values_rather_than_a_log_clipping_floor(window):
    returns = np.array([0.1, -0.999999, 0.2, 0.015, 0.3, -0.99995, -0.2, 0.15])
    assert_rolling_products(returns, window)


@pytest.mark.parametrize("window", [1, 2, 3])
def test_repeated_total_losses_require_every_loss_to_leave_before_ordinary_compounding(window):
    assert_rolling_products(np.array([0.1, -1.0, -1.0, 0.2, 0.3, 0.1]), window)


@pytest.mark.parametrize("window", [1, 63, 252])
def test_ordinary_long_return_windows_preserve_independent_product_precision(window):
    returns = np.random.default_rng(124).normal(0.0001, 0.005, 1000)
    assert_rolling_products(returns, window)


@pytest.mark.parametrize("invalid", [-1.01, np.nan, np.inf, -np.inf])
def test_unsupported_or_missing_return_producers_are_rejected_instead_of_clipped(invalid):
    with pytest.raises(ValueError, match="finite|below -1"):
        PortfolioAnalysis([0.1, invalid]).compute_rolling_metrics(windows=[2], metrics=["returns"])
