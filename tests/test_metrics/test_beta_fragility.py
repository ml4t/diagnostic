"""Fragility uses real paired samples and the actual middle benchmark coordinate."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic import metrics
from ml4t.diagnostic.compat import empyrical as ep


def price_returns():
    dates = pd.bdate_range("2025-01-02", periods=4, tz="UTC")
    strategy = pd.Series([100, 99.96, 99.96, 99.870036], index=dates)
    market = pd.Series([100, 98, 98, 100.94], index=dates)
    return ep.simple_returns(strategy), ep.simple_returns(market)


def test_dated_price_workflow_interpolates_at_the_actual_middle_benchmark_value():
    returns, benchmark = price_returns()
    original = returns.copy(deep=True)
    result = metrics.beta_fragility(returns, benchmark.iloc[::-1])
    assert result.fragility == pytest.approx(-0.0006, abs=1e-15)
    assert result.n_observations == result.n_pairs == 3 and result.n_excluded == 0
    assert result.selected_ordinals == (0, 1, 2)
    assert result.selected_keys == tuple(returns.index)
    assert result.endpoint_weights == pytest.approx((0.6, 0.4))
    assert result.benchmark_points == pytest.approx((-0.02, 0.0, 0.03))
    assert ep.beta_fragility_heuristic(returns, benchmark.iloc[::-1]) == pytest.approx(
        -0.0006, abs=1e-15
    )
    pd.testing.assert_series_equal(returns, original)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_insufficient_usable_pairs_are_counted_without_fabricating_a_third_point(count):
    benchmark = np.array([-0.02, -0.01, 0, 0.01, 0.02])
    values = 0.01 + 2 * benchmark
    values[count:] = np.nan
    result = metrics.beta_fragility(values, benchmark)
    assert np.isnan(result.fragility) and result.n_pairs == count and result.n_excluded == 5 - count
    assert result.selected_ordinals is None and result.status == "insufficient_pairs"
    assert np.isnan(ep.beta_fragility_heuristic(values, benchmark))


@pytest.mark.parametrize(
    "factor_values,gap",
    [
        ([-0.02, 0, 0.03], -0.0006),
        ([-0.02, -0.01, 0.01, 0.03], -0.0006),
        ([-0.04, -0.02, 0, 0.01, 0.03], -0.0012),
        ([-0.05, -0.03, -0.02, 0, 0.01, 0.02, 0.04], -0.002),
    ],
)
@pytest.mark.parametrize("name", ["beta_fragility_heuristic", "beta_fragility_heuristic_aligned"])
def test_asymmetric_affine_and_concave_samples_match_independent_endpoint_interpolation(
    factor_values, gap, name
):
    benchmark = np.array(factor_values)
    assert getattr(ep, name)(0.01 + 2 * benchmark, benchmark) == pytest.approx(0, abs=1e-15)
    assert getattr(ep, name)(-(benchmark**2), benchmark) == pytest.approx(gap, abs=1e-15)
    assert metrics.beta_fragility(-(benchmark**2), benchmark).fragility == pytest.approx(
        gap, abs=1e-15
    )


def test_constant_benchmark_ties_use_declared_input_order_not_permutation_invariance():
    values = np.array([0, 0.1, 0.2, 0.4, 0.9])
    benchmark = np.zeros(5)
    result = metrics.beta_fragility(values, benchmark)
    assert result.selected_ordinals == (0, 2, 4) and result.constant_benchmark
    assert result.endpoint_weights == (0.5, 0.5)
    assert result.fragility == pytest.approx(0.25)
    reordered = metrics.beta_fragility(values[[1, 2, 0, 3, 4]], benchmark)
    assert reordered.fragility == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["beta_fragility_heuristic", "beta_fragility_heuristic_aligned"])
@pytest.mark.parametrize("tied", [False, True])
def test_actual_published_fragility_workflow_matches_values_and_declared_reference_ties(name, tied):
    reference = pytest.importorskip("empyrical")
    benchmark = np.repeat([-0.02, 0, 0.03], [14, 13, 13]) if tied else np.linspace(-0.03, 0.04, 40)
    values = -(benchmark**2) + np.arange(40) * 0.001
    dates = pd.bdate_range("2025-01-02", periods=40, tz="UTC")
    returns, factor = pd.Series(values, index=dates), pd.Series(benchmark, index=dates)
    assert getattr(ep, name)(returns, factor) == pytest.approx(
        getattr(reference, name)(returns.copy(), factor.copy()), abs=1e-15
    )
    native = metrics.beta_fragility(returns, factor)
    stable = np.argsort(benchmark, kind="stable")
    assert native.selected_ordinals == tuple(int(stable[index]) for index in (0, 20, 39))


def test_disjoint_dated_inputs_have_zero_pairs_instead_of_positional_matches():
    returns = pd.Series([0.01, 0.02, 0.03], index=pd.bdate_range("2025-01-02", periods=3, tz="UTC"))
    benchmark = pd.Series(
        [0.01, 0.02, 0.03], index=pd.bdate_range("2025-02-03", periods=3, tz="UTC")
    )
    result = metrics.beta_fragility(returns, benchmark)
    assert result.n_observations == 6 and result.n_pairs == 0 and np.isnan(result.fragility)


def test_zero_price_recovery_exposes_finite_native_policy_and_legacy_infinity_propagation():
    reference = pytest.importorskip("empyrical")
    strategy = pd.Series([0, 100, 99.96, 99.96, 99.870036], dtype=float)
    market = pd.Series([100, 97, 95.06, 95.06, 97.9118], dtype=float)
    values, benchmark = ep.simple_returns(strategy), ep.simple_returns(market)
    assert np.isposinf(values.iloc[0])
    result = metrics.beta_fragility(values, benchmark)
    assert result.n_pairs == 3 and result.n_excluded == 1
    assert result.fragility == pytest.approx(-0.0006, abs=1e-15)
    assert np.isposinf(ep.beta_fragility_heuristic(values, benchmark))
    assert np.isposinf(reference.beta_fragility_heuristic(values, benchmark))


def test_even_sample_uses_the_upper_observed_middle_not_an_average_of_coordinates():
    factor = np.array([-0.04, -0.01, 0.02, 0.05])
    result = metrics.beta_fragility(-(factor**2), factor)
    assert result.selected_ordinals == (0, 2, 3)
    assert result.benchmark_points[1] == 0.02
    assert result.fragility == pytest.approx(-0.0018, abs=1e-15)


def test_finite_coordinate_range_overflow_preserves_independent_symmetric_weights():
    values, factor = np.array([-0.01, 0.02, 0.03]), np.array([-1e308, 0, 1e308])
    result = metrics.beta_fragility(values, factor)
    assert result.endpoint_weights == (0.5, 0.5) and not result.constant_benchmark
    assert result.fragility == pytest.approx(-0.01, abs=1e-15)
    assert ep.beta_fragility_heuristic(values, factor) == pytest.approx(-0.01, abs=1e-15)


def test_infinite_benchmark_coordinates_do_not_masquerade_as_a_constant_finite_design():
    values, factor = np.array([-0.01, 0.02, 0.03]), np.full(3, np.inf)
    result = metrics.beta_fragility(values, factor)
    assert result.n_pairs == 0 and result.n_excluded == 3 and np.isnan(result.fragility)
    assert np.isnan(ep.beta_fragility_heuristic(values, factor))


@pytest.mark.parametrize("container", [list, np.array, pl.Series])
def test_native_supplied_returns_keep_positional_pairs_and_do_not_require_legacy_packages(
    container,
):
    returns, benchmark = price_returns()
    values, factor = container(returns.to_numpy()), container(benchmark.to_numpy())
    result = metrics.beta_fragility(values, factor)
    assert result.fragility == pytest.approx(-0.0006, abs=1e-15)
    assert result.selected_keys == result.selected_ordinals == (0, 1, 2)
    np.testing.assert_array_equal(np.asarray(values), returns.to_numpy())
    np.testing.assert_array_equal(np.asarray(factor), benchmark.to_numpy())


@pytest.mark.parametrize("shape", [(3, 1), (1, 3)])
def test_native_rejects_multi_axis_inputs_without_changing_supplied_returns(shape):
    values = np.array([0.01, 0.02, 0.03]).reshape(shape)
    original = values.copy()
    with pytest.raises(ValueError, match="equal-length one-dimensional"):
        metrics.beta_fragility(values, np.array([-0.02, 0.0, 0.03]))
    np.testing.assert_array_equal(values, original)


def test_native_rejects_unequal_positional_samples_instead_of_truncating_them():
    with pytest.raises(ValueError, match="equal-length one-dimensional"):
        metrics.beta_fragility([0.01, 0.02, 0.03], [-0.02, 0.0, 0.02, 0.03])


def test_float32_price_pipeline_preserves_observed_coordinates_in_float64_interpolation():
    from fractions import Fraction

    prices = pd.Series(np.array([100, 101.111, 100.875, 105.617], dtype=np.float32))
    market = pd.Series(np.array([100, 97.725, 97.931, 100.045], dtype=np.float32))
    values, factor = ep.simple_returns(prices), ep.simple_returns(market)
    assert values.dtype == factor.dtype == np.float32
    result = metrics.beta_fragility(values, factor)
    order = (0, 1, 2)
    xs = [Fraction(float(factor.iloc[index])) for index in order]
    ys = [Fraction(float(values.iloc[index])) for index in order]
    independent = ((xs[2] - xs[1]) * ys[0] + (xs[1] - xs[0]) * ys[2]) / (xs[2] - xs[0]) - ys[1]
    assert result.selected_ordinals == order
    assert result.fragility == pytest.approx(float(independent), rel=0, abs=1e-15)
    assert ep.beta_fragility_heuristic(values, factor) == result.fragility
