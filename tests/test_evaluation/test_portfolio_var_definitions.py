"""Gaussian return risk and normal cash-loss quantiles have different units."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.evaluation.portfolio_analysis import metrics


def test_gaussian_return_var_uses_sample_deviation_without_becoming_an_empirical_quantile():
    returns = pd.Series([-0.02, np.nan, 0.0, 0.02], name="strategy")
    original = returns.copy(deep=True)
    assert metrics.gaussian_value_at_risk(returns, sigma=2) == pytest.approx(-0.04)
    assert metrics.value_at_risk(returns, confidence=0.95) == pytest.approx(-0.018)
    pd.testing.assert_series_equal(returns, original)


def test_normal_cash_var_scales_the_lower_return_quantile_without_clipping_gains():
    assert metrics.normal_cash_value_at_risk(1000, 0.5, mean=-0.01, std=0.02) == pytest.approx(10)
    assert metrics.normal_cash_value_at_risk(1000, 0.5, mean=0.01, std=0.02) == pytest.approx(-10)
    assert metrics.normal_cash_value_at_risk(1000, 0.975, mean=0, std=0.02) == pytest.approx(
        39.19927969080108
    )


@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0, 3.0])
def test_published_pyfolio_gaussian_return_workflow_matches_the_native_route(sigma):
    reference = pytest.importorskip("pyfolio.timeseries")
    returns = pd.Series([-0.02, np.nan, 0.0, 0.02])
    assert metrics.gaussian_value_at_risk(returns, sigma=sigma) == pytest.approx(
        reference.value_at_risk(returns, sigma=sigma)
    )


@pytest.mark.parametrize("confidence", [0.5, 0.95, 0.975])
def test_published_pyfolio_normal_cash_workflow_matches_its_actual_units(confidence):
    reference = pytest.importorskip("pyfolio.timeseries")
    expected = reference.var_cov_var_normal(1000, confidence, mu=0.001, sigma=0.02)
    assert metrics.normal_cash_value_at_risk(
        1000, confidence, mean=0.001, std=0.02
    ) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("period", ["weekly", "monthly", "yearly"])
def test_pyfolio_period_aggregation_reuses_the_qualified_civil_product_adapter(period):
    reference = pytest.importorskip("pyfolio.timeseries")
    from ml4t.diagnostic.compat.empyrical import aggregate_returns

    dates = pd.to_datetime(["2019-12-30", "2020-01-02", "2020-02-03", "2020-02-04"], utc=True)
    returns = pd.Series([0.02, -0.01, 0.03, 0.01], index=dates)
    grouped = aggregate_returns(returns, period)
    actual = metrics.gaussian_value_at_risk(grouped, sigma=1.5)
    assert actual == pytest.approx(
        reference.value_at_risk(returns, period=period, sigma=1.5), nan_ok=True
    )


@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf, True])
def test_invalid_gaussian_sigma_cannot_silently_change_the_risk_definition(sigma):
    with pytest.raises((TypeError, ValueError), match="sigma"):
        metrics.gaussian_value_at_risk(np.array([-0.01, 0.01]), sigma=sigma)


@pytest.mark.parametrize("values", [[], [0.01], [np.nan, np.nan]])
def test_insufficient_gaussian_sample_is_undefined_without_inventing_dispersion(values):
    assert np.isnan(metrics.gaussian_value_at_risk(np.array(values)))


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", np.nan),
        ("std", 0),
        ("std", -0.02),
        ("mean", np.inf),
        ("portfolio_value", True),
    ],
)
def test_invalid_normal_model_parameters_are_rejected_instead_of_becoming_plausible_cash_risk(
    keyword, value
):
    arguments = {"portfolio_value": 1000, "confidence": 0.95, "mean": 0.0, "std": 0.02}
    arguments[keyword] = value
    with pytest.raises(ValueError, match=keyword):
        metrics.normal_cash_value_at_risk(**arguments)
