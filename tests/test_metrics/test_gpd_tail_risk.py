"""Tail risk distinguishes fitted excesses, full sample probability and failed fits."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from ml4t.diagnostic import metrics
from ml4t.diagnostic.compat import empyrical as ep


def portfolio_returns():
    returns = np.r_[
        np.repeat(0.002, 40), -np.array([0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2])
    ]
    prices = pd.Series(
        np.r_[100.0, 100 * np.cumprod(1 + returns)],
        index=pd.bdate_range("2025-01-02", periods=49, tz="UTC"),
    )
    return prices.pct_change(fill_method=None).iloc[1:]


def test_fitted_public_portfolio_tail_matches_independent_survival_quantile_and_integral():
    returns = portfolio_returns()
    original = returns.copy(deep=True)
    result = metrics.estimate_gpd_tail_risk(returns, threshold=0.02, tail_probability=0.02)
    assert result.status == "ok" and result.method == "peaks_over_threshold"
    assert result.n_observations == result.n_finite == 48
    assert result.n_exceedances == 8 and result.n_excluded == 0
    assert result.shape_lower_bound == -1 and result.shape_at_lower_bound
    assert result.shape == pytest.approx(-1, abs=1e-6)
    assert result.scale == pytest.approx(0.18, rel=1e-6)
    shape, scale = result.shape, result.scale
    x = result.value_at_risk - result.threshold
    survival = math.exp(-x / scale) if shape == 0 else (1 + shape * x / scale) ** (-1 / shape)
    assert survival * 8 / 48 == pytest.approx(0.02, rel=1e-10)
    if shape < 1:
        upper = -scale / shape if shape < 0 else math.inf
        integrated = quad(
            lambda y: (
                math.exp(-y / scale) if shape == 0 else (1 + shape * y / scale) ** (-1 / shape)
            ),
            x,
            upper,
        )[0]
        assert result.expected_shortfall == pytest.approx(
            result.value_at_risk + integrated / survival, rel=1e-7, abs=1e-10
        )
    else:
        assert np.isposinf(result.expected_shortfall)
    pd.testing.assert_series_equal(returns, original)


def test_appending_gains_changes_unconditional_risk_without_changing_the_fitted_tail():
    values = portfolio_returns().to_numpy()
    first = metrics.estimate_gpd_tail_risk(values, threshold=0.02, tail_probability=0.02)
    second = metrics.estimate_gpd_tail_risk(
        np.r_[values, np.repeat(0.01, 100)], threshold=0.02, tail_probability=0.02
    )
    assert first.status == second.status == "ok"
    assert first.n_exceedances == second.n_exceedances == 8
    assert first.shape == second.shape and first.scale == second.scale
    assert second.value_at_risk < first.value_at_risk


@pytest.mark.parametrize("count", [0, 1, 2])
def test_insufficient_exceedances_are_undefined_instead_of_safe_zero_risk(count):
    values = np.r_[np.repeat(0.01, 40), -np.arange(1, count + 1) * 0.1]
    result = metrics.estimate_gpd_tail_risk(values, threshold=0.02)
    assert result.status == "insufficient_exceedances" and result.n_exceedances == count
    assert np.isnan(result.value_at_risk) and np.isnan(result.expected_shortfall)


def test_constant_excesses_do_not_claim_a_valid_fitted_distribution():
    result = metrics.estimate_gpd_tail_risk(
        np.r_[np.repeat(0.01, 40), np.repeat(-0.3, 5)], threshold=0.2
    )
    assert result.status == "degenerate_tail" and result.n_exceedances == 5
    assert np.isnan(result.value_at_risk) and np.isnan(result.expected_shortfall)


def test_nonfinite_input_exclusions_preserve_the_finite_sample_denominator():
    values = np.r_[portfolio_returns().to_numpy(), np.nan, np.inf, -np.inf]
    result = metrics.estimate_gpd_tail_risk(values, threshold=0.02, tail_probability=0.02)
    assert result.status == "ok" and result.n_observations == 51
    assert result.n_finite == 48 and result.n_excluded == 3 and result.n_exceedances == 8


@pytest.mark.parametrize("shape", [0.0, 0.5, 1.0, 1.5])
def test_public_quantile_and_es_helpers_include_the_exponential_limit_and_infinite_mean(shape):
    threshold, scale, probability = 0.2, 0.1, 0.01
    ratio = probability / 0.2
    expected = (
        threshold - scale * math.log(ratio)
        if shape == 0
        else threshold + scale * (ratio**-shape - 1) / shape
    )
    value = ep.gpd_var_calculator(threshold, scale, shape, probability, 100, 20)
    assert value == pytest.approx(expected, rel=1e-12)
    shortfall = ep.gpd_es_calculator(value, threshold, scale, shape)
    if shape >= 1:
        assert np.isposinf(shortfall)
    else:
        assert shortfall == pytest.approx((expected + scale - shape * threshold) / (1 - shape))


@pytest.mark.parametrize("scale", [0.0, -0.1, np.nan, np.inf])
def test_invalid_scale_cannot_produce_a_valid_tail_risk_estimate(scale):
    with pytest.raises(ValueError, match="scale"):
        ep.gpd_var_calculator(0.2, scale, 0.5, 0.01, 100, 20)


def test_likelihood_helpers_share_the_actual_exponential_and_shape_one_density_contract():
    values = np.array([0.0, 2.0, 6.0])
    assert ep.gpd_loglikelihood_scale_only(2.0, values) == pytest.approx(-3 * math.log(2) - 4)
    assert ep.gpd_loglikelihood_scale_and_shape(2.0, 1.0, values) == pytest.approx(-9 * math.log(2))
    for factory in (ep.gpd_loglikelihood_factory, ep.gpd_loglikelihood_scale_and_shape_factory):
        assert factory(values)([2.0, 1.0]) == pytest.approx(9 * math.log(2))
    assert ep.gpd_loglikelihood_scale_only_factory(values)(2.0) == pytest.approx(
        3 * math.log(2) + 4
    )
    assert ep.gpd_loglikelihood([2.0, 0.0], values) == pytest.approx(3 * math.log(2) + 4)


def test_probability_above_fitted_tail_is_not_extrapolated_into_an_unmodeled_body():
    result = metrics.estimate_gpd_tail_risk(
        portfolio_returns(), threshold=0.02, tail_probability=0.5
    )
    assert result.status == "probability_outside_fitted_tail"
    assert np.isnan(result.value_at_risk) and np.isnan(result.expected_shortfall)


@pytest.mark.parametrize("labelled", [False, True])
def test_actual_reference_workflow_preserves_nondefault_probability_and_five_output_fields(
    labelled,
):
    reference = pytest.importorskip("empyrical")
    values = np.array([0.01, -0.02, 0.03, -0.04, 0.02, -0.01, -0.03, 0.015, 0.005, -0.025])
    if labelled:
        values = pd.Series(values, index=pd.bdate_range("2025-01-02", periods=10, tz="UTC"))
    for probability in (0.01, 0.05):
        actual = ep.gpd_risk_estimates(values, var_p=probability)
        expected = reference.gpd_risk_estimates(values, var_p=probability)
        assert np.shape(actual) == (5,)
        assert isinstance(actual, pd.Series) == labelled
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-10)


def test_actual_heavy_tail_fit_keeps_corrected_reference_infinite_expected_shortfall():
    reference = pytest.importorskip("empyrical")
    values = -0.1 * np.random.default_rng(2026).pareto(0.5, 2048)
    expected = np.asarray(reference.gpd_risk_estimates(values))
    result = metrics.estimate_gpd_tail_risk(values, method="empyrical_raw_tail")
    assert result.status == "ok" and result.shape > 1
    assert result.threshold == expected[0] and result.scale > 0 and result.value_at_risk > 0
    assert np.isposinf(result.expected_shortfall) and np.isposinf(expected[4])
    np.testing.assert_allclose(
        [result.scale, result.shape, result.value_at_risk], expected[1:4], rtol=1e-5, atol=1e-10
    )


def test_real_optimizer_budget_failure_cannot_be_reported_as_a_successful_or_zero_risk_fit(
    monkeypatch,
):
    from ml4t.diagnostic.metrics import tail_risk

    original = tail_risk.optimize.minimize
    observed = []

    def run_with_exhausted_budget(*args, **kwargs):
        kwargs["options"] = {**(kwargs.get("options") or {}), "maxiter": 1}
        actual = original(*args, **kwargs)
        assert not actual.success and actual.nit == 1
        observed.append(actual)
        return actual

    monkeypatch.setattr(tail_risk.optimize, "minimize", run_with_exhausted_budget)
    result = metrics.estimate_gpd_tail_risk(portfolio_returns(), threshold=0.02)
    assert len(observed) == 1 and result.status == "fit_failed"
    assert result.n_finite == 48 and result.n_exceedances == 8
    assert np.isnan(result.scale) and np.isnan(result.shape)
    assert np.isnan(result.value_at_risk) and np.isnan(result.expected_shortfall)


@pytest.mark.parametrize("labelled", [False, True])
def test_familiar_no_loss_fit_remains_undefined_instead_of_reference_zero_risk(labelled):
    values = np.repeat(0.01, 10)
    if labelled:
        values = pd.Series(values, index=pd.bdate_range("2025-01-02", periods=10))
    original = values.copy()
    assert np.isnan(ep.gpd_risk_estimates(values)).all()
    np.testing.assert_array_equal(values, original)


def test_familiar_likelihood_retains_its_nonnegative_shape_model_without_restricting_native_pot():
    data = np.array([0.0, 0.1, 0.2])
    assert np.isneginf(ep.gpd_loglikelihood_scale_and_shape(1.0, -0.25, data))
    assert np.isposinf(ep.gpd_loglikelihood([1.0, -0.25], data))
    native = metrics.estimate_gpd_tail_risk(portfolio_returns(), threshold=0.02)
    assert native.status == "ok" and native.shape < 0


def test_failed_raw_tail_fit_reports_the_threshold_that_actually_selected_its_sample():
    changes = np.array([2e-10, 4e-10, 8e-10])
    prices = pd.Series(
        100 * np.cumprod(np.r_[1.0, 1 - changes]),
        index=pd.bdate_range("2025-01-02", periods=4, tz="UTC"),
    )
    values = prices.pct_change(fill_method=None).iloc[1:]
    result = metrics.estimate_gpd_tail_risk(values, method="empyrical_raw_tail")
    assert result.status == "fit_failed" and result.n_losses == 3
    assert result.n_exceedances == int((-values >= result.threshold).sum())
    assert result.threshold > 1e-9
    assert np.isnan(result.value_at_risk) and np.isnan(result.expected_shortfall)


@pytest.mark.parametrize("count", [1, 2])
def test_familiar_method_does_not_discard_real_losses_in_a_long_mostly_profitable_sample(count):
    values = np.r_[np.repeat(0.01, 40), -0.01 * np.arange(1, count + 1)]
    fitted = metrics.estimate_gpd_tail_risk(values, method="empyrical_raw_tail")
    assert fitted.status == "ok" and fitted.n_losses == count
    assert fitted.n_exceedances == int((-values >= fitted.threshold).sum())
    assert fitted.probability_denominator == count
    assert (
        fitted.value_at_risk > fitted.threshold and fitted.expected_shortfall > fitted.value_at_risk
    )
    assert np.isfinite(ep.gpd_risk_estimates(values)).all()
