"""Explicit metric definitions prevent plausible but incompatible return analyses."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep
from ml4t.diagnostic.evaluation.portfolio_analysis import metrics as native


def test_fixed_count_tail_omits_missing_rows_before_sizing_without_changing_native_ties():
    returns = pd.Series([-0.2, -0.1, np.nan, -0.1, 0.1], name="strategy")
    original = returns.copy(deep=True)
    assert ep.conditional_value_at_risk(returns, cutoff=0.5) == pytest.approx(-0.15)
    assert native.conditional_var(returns, confidence=0.5) == pytest.approx(-2 / 15)
    pd.testing.assert_series_equal(returns, original)


def test_empirical_var_propagates_missingness_but_native_default_omits_it():
    returns = np.array([-0.2, -0.1, np.nan, -0.1, 0.1])
    assert np.isnan(ep.value_at_risk(returns, cutoff=0.25))
    assert ep.value_at_risk(returns[~np.isnan(returns)], cutoff=0.25) == pytest.approx(-0.125)
    assert native.value_at_risk(returns, confidence=0.75) == pytest.approx(-0.125)


def test_absolute_tail_preserves_native_signed_numerator_and_zero_denominator_difference():
    losses = np.array([-0.4, -0.3, -0.2, -0.1])
    assert ep.tail_ratio(losses) == pytest.approx(0.115 / 0.385)
    assert native.tail_ratio(losses) == pytest.approx(-0.115 / 0.385)
    assert np.isposinf(ep.tail_ratio(np.array([0.0] * 20 + [0.1, 0.2])))


def test_log_return_stability_uses_observed_rows_and_preserves_native_wealth_default():
    returns = np.array([0.1, np.nan, 0.1, 0.1, 0.1])
    assert ep.stability_of_timeseries(returns) == pytest.approx(1.0)
    assert native.stability_of_timeseries(returns) < 1
    assert np.isnan(ep.stability_of_timeseries(np.array([np.nan, np.nan])))


def test_omega_converts_annual_hurdle_but_keeps_periodic_risk_free_and_no_loss_convention():
    returns = np.array([0.3, -0.1, 0.2, np.nan])
    assert ep.omega_ratio(
        returns, risk_free=0.01, required_return=0.21, annualization=2
    ) == pytest.approx(0.28 / 0.21)
    assert np.isnan(ep.omega_ratio(np.array([0.1, 0.2])))
    assert np.isposinf(native.omega_ratio(np.array([0.1, 0.2])))
    assert np.isnan(ep.omega_ratio(returns, required_return=-1))
    assert np.isnan(ep.omega_ratio(returns, required_return=-1, annualization=1))


@pytest.mark.parametrize(
    "name", ["tail_ratio", "stability_of_timeseries", "conditional_value_at_risk"]
)
def test_pooled_legacy_metric_shapes_are_explicit_not_invented_columnwise_results(name):
    reference = pytest.importorskip("empyrical")
    values = np.array([[-0.2, 0.1], [np.nan, 0.3], [-0.1, 0.2]])
    assert getattr(ep, name)(values) == pytest.approx(getattr(reference, name)(values), nan_ok=True)


@pytest.mark.parametrize(
    "name",
    [
        "omega_ratio",
        "tail_ratio",
        "stability_of_timeseries",
        "value_at_risk",
        "conditional_value_at_risk",
    ],
)
@pytest.mark.parametrize(
    "values", [[-0.2, -0.1, -0.1, 0.1], [0.1, 0.2], [np.nan, -0.2, 0.1], [], [np.nan, np.nan]]
)
def test_corrected_published_definition_workflows_match_each_actual_missingness_contract(
    name, values
):
    reference = pytest.importorskip("empyrical")
    returns = pd.Series(values, dtype=float)
    assert getattr(ep, name)(returns) == pytest.approx(
        getattr(reference, name)(returns.copy()), nan_ok=True
    )


def test_supplied_single_beta_array_uses_the_same_public_regression_workflow():
    benchmark = np.array([-0.02, 0.0, 0.02])
    returns = 0.0003 + 1.2 * benchmark
    assert ep.alpha_aligned(returns, benchmark, _beta=np.array([1.2])) == pytest.approx(
        1.0003**252 - 1
    )


def test_infinite_sign_subset_observation_is_not_silently_dropped_to_a_finite_fit():
    benchmark = np.array([0.01, 0.02, 0.03])
    returns = np.array([0.02, 0.03, np.inf])
    assert np.isposinf(ep.up_alpha_beta(returns, benchmark)).all()


def test_published_cross_moment_infinity_propagation_is_not_a_finite_fit_claim():
    reference = pytest.importorskip("empyrical")
    returns = np.array([0.02, 0.03, np.inf])
    benchmark = np.array([0.01, 0.02, 0.03])
    np.testing.assert_equal(
        ep.up_alpha_beta(returns, benchmark), reference.up_alpha_beta(returns, benchmark)
    )


def test_fixed_tail_count_preserves_original_cutoff_without_complement_rounding():
    returns = np.arange(-10.0, 1.0) / 10
    assert ep.conditional_value_at_risk(returns, cutoff=0.1) == pytest.approx(-0.95)


def test_log_stability_does_not_salvage_a_prefix_when_the_later_return_is_outside_log_domain():
    assert np.isnan(ep.stability_of_timeseries(np.array([0.1, 0.1, -1.1])))


def test_simple_stat_registry_runs_the_complete_published_metric_workflow_in_order():
    reference = pytest.importorskip("empyrical.stats")
    returns = np.array([-0.02, 0.03, -0.01, 0.01, 0.02])
    assert [function.__name__ for function in ep.SIMPLE_STAT_FUNCS] == [
        function.__name__ for function in reference.SIMPLE_STAT_FUNCS
    ]
    actual = [function(returns.copy()) for function in ep.SIMPLE_STAT_FUNCS]
    expected = [function(returns.copy()) for function in reference.SIMPLE_STAT_FUNCS]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def _run_factor_registry(registry, returns, benchmark):
    return {
        function.__name__: (
            function(returns.copy(), var_p=0.01)
            if function.__name__ == "gpd_risk_estimates"
            else function(returns.copy(), benchmark.copy())
        )
        for function in registry
    }


def test_factor_registry_supports_paired_metrics_and_univariate_five_field_tail_workflow():
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.02, -0.01, -0.03, 0.015, 0.005, -0.025])
    benchmark = np.array([0.02, -0.01, 0.025, -0.03, 0.01, -0.02, -0.025, 0.01, 0.005, -0.015])
    original = returns.copy(), benchmark.copy()
    actual = _run_factor_registry(ep.FACTOR_STAT_FUNCS, returns, benchmark)
    assert list(actual) == [
        "excess_sharpe",
        "alpha",
        "beta",
        "beta_fragility_heuristic",
        "gpd_risk_estimates",
        "capture",
        "up_capture",
        "down_capture",
    ]
    expected_beta = np.cov(returns, benchmark)[0, 1] / np.var(benchmark, ddof=1)
    assert actual["beta"] == pytest.approx(expected_beta)
    assert actual["alpha"] == pytest.approx(
        (1 + returns.mean() - expected_beta * benchmark.mean()) ** 252 - 1
    )
    tail = np.asarray(actual["gpd_risk_estimates"])
    assert tail.shape == (5,) and np.isfinite(tail).all() and (tail > 0).all()
    np.testing.assert_array_equal(returns, original[0])
    np.testing.assert_array_equal(benchmark, original[1])


def test_factor_registry_executes_every_actual_reference_signature_in_original_order():
    reference = pytest.importorskip("empyrical.stats")
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.02, -0.01, -0.03, 0.015, 0.005, -0.025])
    benchmark = np.array([0.02, -0.01, 0.025, -0.03, 0.01, -0.02, -0.025, 0.01, 0.005, -0.015])
    actual = _run_factor_registry(ep.FACTOR_STAT_FUNCS, returns, benchmark)
    expected = _run_factor_registry(reference.FACTOR_STAT_FUNCS, returns, benchmark)
    assert list(actual) == list(expected)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5, atol=1e-10)
