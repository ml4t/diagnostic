"""Joint rolling factor fits must retain their actual windows and missing-data sample."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import timeseries


def sample():
    dates = pd.bdate_range("2025-01-02", periods=11, tz="UTC")
    market = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.04, 0.02, -0.03, 0.01, 0, 0.02])
    style = np.array([0.02, 0.01, -0.01, 0.03, 0.02, -0.02, 0.01, 0.04, -0.03, 0.02, 0.01])
    factors = pd.DataFrame({"market": market, "style": style}, index=dates)
    returns = pd.Series(0.001 + 1.5 * market - 0.75 * style, index=dates)
    return returns, factors


def test_joint_fit_retains_inclusive_endpoints_and_drops_return_gaps_before_window_selection():
    returns, factors = sample()
    returns.iloc[2] = np.nan
    before = returns.copy(deep=True), factors.copy(deep=True)
    actual = timeseries.rolling_regression(returns, factors, rolling_window=3)
    assert actual.index.equals(returns.dropna().index)
    assert actual.index.name == "dt"
    assert actual.iloc[:3].isna().all().all()
    np.testing.assert_allclose(actual.iloc[3:], np.tile([0.001, 1.5, -0.75], (7, 1)), atol=1e-12)
    pd.testing.assert_series_equal(returns, before[0])
    pd.testing.assert_frame_equal(factors, before[1])


@pytest.mark.parametrize("threshold", [0, 0.2, 0.25, 1])
def test_actual_reference_joint_windows_apply_per_factor_threshold_then_complete_case_fit(
    threshold,
):
    reference = pytest.importorskip("pyfolio.timeseries")
    assert version("pyfolio-reloaded") == "0.10.0"
    returns, factors = sample()
    returns.iloc[2] = np.nan
    returns.iloc[7] += 0.013
    factors.iloc[4, 0] = np.nan
    factors.iloc[6, 1] = np.nan
    factors = factors.iloc[::-1]
    expected = reference.rolling_regression(returns, factors, 3, threshold).astype(float)
    actual = timeseries.rolling_regression(returns, factors, 3, threshold)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "mode", ["constant", "collinear", "single_observation", "empty", "overlong"]
)
def test_actual_reference_degenerate_and_short_joint_fits_retain_minimum_norm_and_warmup(mode):
    reference = pytest.importorskip("pyfolio.timeseries")
    assert version("pyfolio-reloaded") == "0.10.0"
    returns, factors = sample()
    window, threshold = 3, 0.1
    if mode == "constant":
        factors["style"] = 1.0
    elif mode == "collinear":
        factors["style"] = factors["market"] * 2
    elif mode == "single_observation":
        factors.iloc[:4] = np.nan
        window, threshold = 1, 1.0
    elif mode == "empty":
        returns, factors = returns.iloc[:0], factors.iloc[:0]
    else:
        window = 50
    expected = reference.rolling_regression(returns, factors, window, threshold).astype(float)
    actual = timeseries.rolling_regression(returns, factors, window, threshold)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-11, atol=1e-12)


def test_missing_factor_dates_are_rejected_instead_of_silently_compressing_the_fit_clock():
    returns, factors = sample()
    with pytest.raises(KeyError):
        timeseries.rolling_regression(returns, factors.iloc[1:], rolling_window=3)


@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, np.nan, np.inf])
def test_invalid_joint_fit_missing_fraction_is_rejected_without_changing_inputs(invalid):
    returns, factors = sample()
    before = returns.copy(deep=True), factors.copy(deep=True)
    with pytest.raises(ValueError, match="nan_threshold"):
        timeseries.rolling_regression(returns, factors, nan_threshold=invalid)
    pd.testing.assert_series_equal(returns, before[0])
    pd.testing.assert_frame_equal(factors, before[1])


def test_collinear_factor_coefficients_use_centered_minimum_norm_instead_of_separate_betas():
    dates = pd.date_range("2025-01-01", periods=4)
    factors = pd.DataFrame({"first": [0, 1, 2, 3], "second": [0, 2, 4, 6]}, index=dates)
    returns = pd.Series([1, 6, 11, 16], index=dates)
    actual = timeseries.rolling_regression(returns, factors, rolling_window=3)
    np.testing.assert_allclose(actual.iloc[-1], [1, 1, 2], atol=1e-12)


def test_executable_joint_factor_recipe_runs_through_public_api(capsys):
    import json
    import runpy
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "examples/pyfolio_rolling_regression.py"
    runpy.run_path(str(example), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["fitted_windows"] == 7
    np.testing.assert_allclose(list(result["last_coefficients"].values()), [0.001, 1.5, -0.75])


def test_familiar_metric_namespace_computes_real_values_without_importing_the_legacy_package():
    returns = pd.Series([0.1, -0.1])
    assert timeseries.max_drawdown(returns) == pytest.approx(-0.1)
    assert timeseries.annual_return(returns, period="yearly") == pytest.approx(np.sqrt(0.99) - 1)
    assert timeseries.cum_returns(returns).iloc[-1] == pytest.approx(-0.01)
    with pytest.raises(AttributeError, match="no_such_metric"):
        timeseries.no_such_metric


@pytest.mark.parametrize(
    "metric",
    [
        "max_drawdown",
        "annual_return",
        "annual_volatility",
        "calmar_ratio",
        "omega_ratio",
        "sortino_ratio",
        "downside_risk",
        "sharpe_ratio",
        "alpha_beta",
        "alpha",
        "beta",
        "stability_of_timeseries",
        "tail_ratio",
        "common_sense_ratio",
        "cum_returns",
        "aggregate_returns",
        "normalize",
        "var_cov_var_normal",
    ],
)
def test_actual_reference_metric_signatures_delegate_to_existing_definitions_and_keep_labels(
    metric,
):
    reference = pytest.importorskip("pyfolio.timeseries")
    assert version("pyfolio-reloaded") == "0.10.0"
    returns, factors = sample()
    args, kwargs = [returns], {}
    if metric in ("alpha_beta", "alpha", "beta"):
        args.append(factors["market"])
    elif metric == "aggregate_returns":
        args.append("monthly")
    elif metric == "omega_ratio":
        kwargs["annual_return_threshhold"] = 0.03
    elif metric == "normalize":
        kwargs["starting_value"] = 100.0
    elif metric == "var_cov_var_normal":
        args, kwargs = [1000.0, 0.95], {"mu": 0.001, "sigma": 0.02}
    actual, expected = (
        getattr(timeseries, metric)(*args, **kwargs),
        getattr(reference, metric)(*args, **kwargs),
    )
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
