"""Constant decimal samples must not acquire dispersion from reduction rounding."""

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.compat import empyrical as ep
from ml4t.diagnostic.compat.pyfolio.timeseries import rolling_sharpe
from ml4t.diagnostic.core.numba_utils import rolling_sharpe_numba
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.metrics import periodic_sharpe_ratio, sharpe_ratio


def calculate(route, returns):
    if route == "periodic":
        return periodic_sharpe_ratio(returns)
    if route == "native":
        return sharpe_ratio(returns)
    if route == "compat":
        return ep.sharpe_ratio(returns)
    return ep.excess_sharpe(returns, np.zeros(len(returns)))


@pytest.mark.parametrize("route", ["periodic", "native", "compat", "excess"])
@pytest.mark.parametrize("length", [10, 252, 1000])
@pytest.mark.parametrize("value", [0.001, -0.001])
def test_identical_decimal_observations_have_exact_zero_dispersion(route, length, value):
    assert calculate(route, np.full(length, value)) == np.copysign(np.inf, value)


@pytest.mark.parametrize("route", ["periodic", "native", "compat", "excess"])
@pytest.mark.parametrize("length", [10, 252, 1000])
@pytest.mark.parametrize("value", [0.001, -0.001])
def test_one_ulp_variation_uses_real_sample_variance_not_a_tolerance(route, length, value):
    values = np.full(length, value)
    values[-1] = np.nextafter(value, np.inf)
    exact = [Fraction.from_float(float(x)) for x in values]
    mean = sum(exact) / length
    variance = sum((x - mean) ** 2 for x in exact) / (length - 1)
    scale = 252 if route in {"native", "compat"} else 1
    expected = float(mean) / math.sqrt(float(variance)) * math.sqrt(scale)
    actual = calculate(route, values)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("container", [np.array, pd.Series, pl.Series])
@pytest.mark.parametrize("value", [0.001, -0.001, 0.0])
def test_native_missing_observations_do_not_invent_dispersion(container, value):
    values = container([np.nan] + [value] * 10 + [np.nan])
    actual = periodic_sharpe_ratio(values)
    if value == 0:
        assert np.isnan(actual)
    else:
        assert actual == np.copysign(np.inf, value)


@pytest.mark.parametrize("route", ["periodic", "native", "compat", "excess"])
@pytest.mark.parametrize("values", [[], [np.nan, np.nan], [0.001, np.nan], [0.0] * 10])
def test_zero_or_insufficient_usable_observations_remain_undefined(route, values):
    assert np.isnan(calculate(route, np.asarray(values)))


def test_risk_free_adjustment_precedes_constant_excess_classification():
    values = np.full(252, 0.001)
    assert periodic_sharpe_ratio(values, periodic_risk_free_rate=0.002) == -np.inf
    assert np.isnan(periodic_sharpe_ratio(values, periodic_risk_free_rate=0.001))
    assert sharpe_ratio(values, risk_free_rate=0.5) == -np.inf
    assert ep.sharpe_ratio(values, risk_free=0.002) == -np.inf
    rates = pd.Series([0.125, 0.25, 0.375, 0.5], index=list("dcba"))
    returns = (rates + 0.125).sort_index()
    assert ep.sharpe_ratio(returns, risk_free=rates) == np.inf


def test_compat_columns_missingness_and_output_buffer_keep_existing_contract():
    frame = pd.DataFrame({"gain": [0.001] * 10, "loss": [-0.001] * 10, "zero": [0.0] * 10})
    frame.loc[0, "gain"] = np.nan
    frame.loc[1, "loss"] = np.nan
    frame["single"] = np.nan
    frame.loc[0, "single"] = 0.001
    original = frame.copy(deep=True)
    output = np.empty(4)
    assert ep.sharpe_ratio(frame, out=output) is output
    np.testing.assert_equal(output, [np.inf, -np.inf, np.nan, np.nan])
    np.testing.assert_equal(ep.sharpe_ratio(frame.to_numpy()), output)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("value", [0.001, -0.001])
def test_legacy_rolling_routes_keep_infinity_labels_and_missing_window_rules(value):
    values = pd.Series(np.full(12, value), index=pd.date_range("2026-01-01", periods=12))
    for result in [
        ep.roll_sharpe_ratio(values, window=10),
        ep.roll_excess_sharpe(values, values * 0, window=10),
    ]:
        pd.testing.assert_index_equal(result.index, values.index[9:])
        np.testing.assert_equal(result.to_numpy(), np.full(3, np.copysign(np.inf, value)))
    complete = rolling_sharpe(values, 10)
    assert complete.iloc[:9].isna().all()
    np.testing.assert_equal(complete.iloc[9:].to_numpy(), np.full(3, np.copysign(np.inf, value)))
    values.iloc[5] = np.nan
    assert rolling_sharpe(values, 10).isna().all()
    assert np.isinf(ep.roll_sharpe_ratio(values, window=10)).all()


@pytest.mark.parametrize("value", [0.001, -0.001, 1e-12, 0.125])
def test_native_rolling_keeps_its_undefined_flat_window_convention(value):
    values = np.full(12, value)
    result = PortfolioAnalysis(values).compute_rolling_metrics(windows=[10], metrics=["sharpe"])
    assert result.sharpe[10].is_null().all()
    assert np.isnan(rolling_sharpe_numba(values, 10)[9:]).all()


@pytest.mark.parametrize("backend", ["native", "numba"])
def test_native_rolling_one_ulp_dispersion_matches_exact_observed_values(backend):
    values = np.full(10, 0.001)
    values[-1] = np.nextafter(values[-1], np.inf)
    exact = [Fraction.from_float(float(x)) for x in values]
    mean = sum(exact) / len(exact)
    denominator = len(exact) - (backend == "native")
    variance = sum((x - mean) ** 2 for x in exact) / denominator
    expected = float(mean) / math.sqrt(float(variance)) * math.sqrt(252)
    if backend == "native":
        result = PortfolioAnalysis(values).compute_rolling_metrics(windows=[10], metrics=["sharpe"])
        actual = result.sharpe[10][-1]
    else:
        actual = rolling_sharpe_numba(values, 10)[-1]
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-14)
