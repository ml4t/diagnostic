"""Subtracting rates must not erase variance present in the original inputs."""

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep
from ml4t.diagnostic.core.numba_utils import rolling_sharpe_numba
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import information_ratio
from ml4t.diagnostic.metrics import periodic_sharpe_ratio, sharpe_ratio


def exact_sharpe(returns, rates, *, scale=1, ddof=1):
    rates = np.broadcast_to(rates, np.shape(returns))
    values = [
        Fraction(float(value)) - Fraction(float(rate))
        for value, rate in zip(returns, rates, strict=True)
        if not math.isnan(float(value) - float(rate))
    ]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - ddof)
    return float(mean) / math.sqrt(float(variance)) * math.sqrt(scale)


def test_native_wider_return_dtype_is_not_demoted_before_calculating_variance():
    value = np.longdouble(1) / 1024
    values = np.full(10, value, dtype=np.longdouble)
    values[-1] = np.nextafter(value, np.longdouble(np.inf))
    first = Fraction(1, 1024)
    delta = first * Fraction(float(np.finfo(np.longdouble).eps))
    exact = [first] * 9 + [first + delta]
    mean = sum(exact) / 10
    variance = sum((item - mean) ** 2 for item in exact) / 9
    actual = periodic_sharpe_ratio(values)
    assert np.isfinite(actual)
    assert actual == pytest.approx(float(mean) / math.sqrt(float(variance)), rel=2e-14)


@pytest.mark.parametrize("route", ["compat", "excess", "information"])
@pytest.mark.parametrize("shift", [0, 3, 9])
def test_large_paired_benchmark_observation_does_not_erase_small_active_variation(route, shift):
    rates = np.array([0.1] + [0.0001] * 9)
    values = rates + 1e-6
    values[-1] = np.nextafter(values[-1], np.inf)
    values, rates = np.roll(values, shift), np.roll(rates, shift)
    if route == "compat":
        actual = ep.sharpe_ratio(values, risk_free=rates)
    elif route == "excess":
        actual = ep.excess_sharpe(values, rates)
    else:
        actual = information_ratio(values, rates)
    expected = exact_sharpe(values, rates, scale=1 if route == "excess" else 252)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("length", [10, 252])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
    "route",
    [
        "periodic",
        "annual",
        "compat",
        "excess",
        "information",
        "native_rolling",
        "numba",
        "rolling",
        "rolling_excess",
    ],
)
def test_rate_subtraction_does_not_erase_observed_one_ulp_variance(route, sign, length):
    values = np.full(length, sign * 1e-6)
    values[-1] = np.nextafter(values[-1], np.inf)
    rate, scale, ddof = 0.0001, 252, 1
    if route == "periodic":
        scale = 1
        actual = periodic_sharpe_ratio(values, periodic_risk_free_rate=rate)
    elif route == "annual":
        rate = (1.05 ** (1 / 252)) - 1
        actual = sharpe_ratio(values, risk_free_rate=0.05)
    elif route == "compat":
        actual = ep.sharpe_ratio(values, risk_free=rate)
    elif route == "excess":
        scale = 1
        actual = ep.excess_sharpe(values, np.full(length, rate))
    elif route == "information":
        actual = information_ratio(values, np.full(length, rate))
    elif route == "native_rolling":
        rate = (1.05 ** (1 / 252)) - 1
        actual = (
            PortfolioAnalysis(values, risk_free=0.05)
            .compute_rolling_metrics(windows=[length], metrics=["sharpe"])
            .sharpe[length][-1]
        )
    elif route == "numba":
        rate, ddof = 0.05 / 252, 0
        actual = rolling_sharpe_numba(values, length, risk_free_rate=0.05)[-1]
    elif route == "rolling":
        actual = ep.roll_sharpe_ratio(values, window=length, risk_free=rate)[-1]
    else:
        scale = 1
        actual = ep.roll_excess_sharpe(values, np.full(length, rate), window=length)[-1]
    expected = exact_sharpe(values, rate, scale=scale, ddof=ddof)
    assert actual is not None and np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("kind", ["scalar", "column_series", "frame", "array"])
def test_columnwise_rates_preserve_variance_labels_missingness_and_output_ownership(
    kind,
):
    dates = pd.date_range("2025-07-07", periods=10)
    first = np.full(10, 1e-6)
    first[-1] = np.nextafter(first[-1], np.inf)
    frame = pd.DataFrame({"first": first, "second": -first}, index=dates)
    frame.iloc[0, 1] = np.nan
    if kind == "scalar":
        rates = 0.0001
        expected_rates = np.full((10, 2), rates)
    elif kind == "column_series":
        rates = pd.Series({"second": -0.0001, "first": 0.0001})
        expected_rates = np.tile([0.0001, -0.0001], (10, 1))
    else:
        rates = pd.DataFrame({"second": -0.0001, "first": 0.0001}, index=dates)
        rates.loc[dates[1], "first"] = np.nan
        expected_rates = np.tile([0.0001, -0.0001], (10, 1))
        expected_rates[1, 0] = np.nan
        if kind == "array":
            rates = expected_rates.copy()
    before = frame.copy(deep=True)
    rate_before = rates.copy() if not np.isscalar(rates) else rates
    output = np.full(2, 123.0)
    actual = ep.sharpe_ratio(frame, risk_free=rates, out=output)
    assert actual is output
    expected = [exact_sharpe(frame.iloc[:, i], expected_rates[:, i], scale=252) for i in range(2)]
    np.testing.assert_allclose(actual, expected, rtol=2e-14)
    pd.testing.assert_frame_equal(frame, before)
    if isinstance(rates, pd.DataFrame):
        pd.testing.assert_frame_equal(rates, rate_before)
    elif isinstance(rates, pd.Series):
        pd.testing.assert_series_equal(rates, rate_before)
    else:
        np.testing.assert_equal(rates, rate_before)


@pytest.mark.parametrize("route", ["compat", "excess", "information"])
def test_real_rate_variation_is_not_erased_by_larger_constant_returns(route):
    returns = pd.Series(np.full(10, 0.001), index=list("abcdefghij"))
    rates = pd.Series(np.full(10, 1e-6), index=returns.index)
    rates.iloc[-1] = np.nextafter(rates.iloc[-1], np.inf)
    if route == "compat":
        actual = ep.sharpe_ratio(returns, risk_free=rates.iloc[::-1])
    elif route == "excess":
        actual = ep.excess_sharpe(returns, rates.iloc[::-1])
    else:
        actual = information_ratio(returns, rates)
    expected = exact_sharpe(returns, rates, scale=1 if route == "excess" else 252)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-14)


def test_bootstrap_rate_subtraction_preserves_finite_nonconstant_samples():
    value = 1e-6
    returns = np.tile([value, np.nextafter(value, np.inf)], 10)
    periodic = 1.05 ** (1 / 252) - 1
    rng = np.random.RandomState(7)
    expected = [
        exact_sharpe(rng.choice(returns, len(returns)), periodic, scale=252) for _ in range(32)
    ]
    actual = sharpe_ratio(
        returns,
        risk_free_rate=0.05,
        confidence_intervals=True,
        bootstrap_samples=32,
        random_state=7,
    )
    assert actual["sharpe"] == pytest.approx(exact_sharpe(returns, periodic, scale=252), rel=2e-14)
    assert actual["lower_ci"] == pytest.approx(np.percentile(expected, 2.5), rel=2e-14)
    assert actual["upper_ci"] == pytest.approx(np.percentile(expected, 97.5), rel=2e-14)


@pytest.mark.parametrize("route", ["compat", "excess", "information"])
def test_unpaired_large_prefix_cannot_erase_variance_in_the_actual_paired_sample(route):
    values = np.full(11, 1e-6)
    values[0] = 0.1
    values[-1] = np.nextafter(values[-1], np.inf)
    rates = np.full(11, 0.0001)
    rates[0] = np.nan
    if route == "compat":
        actual = ep.sharpe_ratio(values, risk_free=rates)
    elif route == "excess":
        actual = ep.excess_sharpe(values, rates)
    else:
        actual = information_ratio(values, rates)
    expected = exact_sharpe(values, rates, scale=1 if route == "excess" else 252)
    assert actual == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("route", ["periodic", "compat", "excess", "information"])
def test_genuinely_constant_active_returns_keep_the_existing_zero_dispersion_conventions(route):
    values = np.full(10, 1e-6)
    if route == "periodic":
        actual = periodic_sharpe_ratio(values, periodic_risk_free_rate=0.0001)
    elif route == "compat":
        actual = ep.sharpe_ratio(values, risk_free=0.0001)
    elif route == "excess":
        actual = ep.excess_sharpe(values, np.full(10, 0.0001))
    else:
        actual = information_ratio(values, np.full(10, 0.0001))
    assert np.isnan(actual) if route == "information" else actual == -np.inf


@pytest.mark.parametrize("route", ["compat", "excess"])
def test_undefined_infinity_difference_is_omitted_without_poisoning_other_pairs(route):
    values = pd.Series([np.inf, 0.01, 0.02, -0.01])
    rates = pd.Series([np.inf, 0.001, 0.001, 0.001])
    with np.errstate(invalid="ignore"):
        actual = (
            ep.sharpe_ratio(values, risk_free=rates)
            if route == "compat"
            else ep.excess_sharpe(values, rates)
        )
    expected = exact_sharpe(values, rates, scale=252 if route == "compat" else 1)
    assert actual == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("route", ["periodic", "compat"])
@pytest.mark.parametrize("rate", [np.nan, np.inf, -np.inf])
def test_undefined_scalar_rates_do_not_become_infinite_sharpe_after_centering(route, rate):
    with np.errstate(invalid="ignore"):
        result = (
            periodic_sharpe_ratio(np.array([0.01, 0.02, 0.03]), periodic_risk_free_rate=rate)
            if route == "periodic"
            else ep.sharpe_ratio(np.array([0.01, 0.02, 0.03]), risk_free=rate)
        )
    assert np.isnan(result)
