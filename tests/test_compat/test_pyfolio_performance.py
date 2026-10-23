"""Full portfolio reports need the actual legacy statistic definitions and samples."""

import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import timeseries


def _returns():
    prices = pd.Series(
        [100.0, 110.0, 99.0, 103.95], index=pd.date_range("2025-02-03", periods=4, tz="UTC")
    )
    return prices.pct_change(fill_method=None).iloc[1:]


def test_actual_reference_is_present_and_executes_when_mandatory():
    required = os.environ.get("ML4T_REQUIRED_PYFOLIO_REFERENCE")
    if required is None:
        pytest.skip("The installed reference lane explicitly requires its real oracle")
    import pyfolio.timeseries as reference

    assert required == version("pyfolio-reloaded") == "0.10.0"
    actual = reference.perf_stats(_returns())
    assert actual["Cumulative returns"] == pytest.approx(0.0395)
    assert actual["Daily value at risk"] == pytest.approx(1 / 60 - 2 * np.sqrt(13 / 1200))


def test_report_performance_table_distinguishes_gaussian_var_and_explicit_annualization():
    returns = _returns()
    original = returns.copy(deep=True)
    actual = timeseries.perf_stats(returns)
    assert len(actual) == 13
    assert actual["Cumulative returns"] == pytest.approx(0.0395)
    assert actual["Max drawdown"] == pytest.approx(-0.1)
    assert actual["Annual return"] == pytest.approx(1.0395**84 - 1)
    assert actual["Daily value at risk"] == pytest.approx(1 / 60 - 2 * np.sqrt(13 / 1200))
    assert timeseries.value_at_risk(returns, sigma=0) == pytest.approx(1 / 60)
    scaled = timeseries.perf_stats(returns, periods_per_year=12)
    assert scaled["Annual return"] == pytest.approx(1.0395**4 - 1)
    assert scaled["Annual volatility"] == pytest.approx(np.sqrt(13 / 100))
    pd.testing.assert_series_equal(original, returns)


def test_familiar_gross_leverage_preserves_zero_negative_nav_and_original_asset_identifiers():
    positions = pd.DataFrame(
        {101: [-100.0, 0, 100], 203: [50.0, 0, -100], "cash": [150.0, 0, -100]},
        index=_returns().index,
    )
    before = positions.copy(deep=True)
    actual = timeseries.gross_lev(positions)
    np.testing.assert_allclose(actual, [1.5, np.nan, -2.0], equal_nan=True)
    stats = timeseries.perf_stats(_returns(), positions=positions)
    assert stats["Gross leverage"] == pytest.approx(-0.25)
    pd.testing.assert_frame_equal(positions, before)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("with_benchmark", [False, True])
@pytest.mark.parametrize("holdings", [False, True])
def test_actual_reference_performance_table_keeps_each_metrics_sample_and_original_order(
    missing, with_benchmark, holdings
):
    reference = pytest.importorskip("pyfolio.timeseries")
    assert version("pyfolio-reloaded") == "0.10.0"
    returns = pd.concat([_returns()] * 3, ignore_index=True)
    returns.index = pd.bdate_range("2025-02-03", periods=len(returns), tz="UTC")
    if missing:
        returns.iloc[2] = np.nan
    factor = (
        pd.Series(np.linspace(-0.03, 0.04, len(returns)), index=returns.index)
        if with_benchmark
        else None
    )
    positions = (
        pd.DataFrame(
            {101: np.linspace(-100.0, 100, len(returns)), "cash": 150.0}, index=returns.index
        )
        if holdings
        else None
    )
    expected = reference.perf_stats(returns, factor_returns=factor, positions=positions)
    actual = timeseries.perf_stats(returns, factor_returns=factor, positions=positions)
    pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def test_seeded_iid_summary_bootstrap_pairs_benchmark_rows_without_mutating_global_rng():
    returns = _returns()
    state = np.random.get_state()
    first = timeseries.calc_bootstrap(np.mean, returns, n_samples=7, random_seed=10)
    second = timeseries.calc_bootstrap(np.mean, returns, n_samples=7, random_seed=10)
    np.testing.assert_array_equal(first, second)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    benchmark = returns * 2
    paired = timeseries.calc_bootstrap(
        lambda sample, factor: np.mean(factor - 2 * sample),
        returns,
        factor_returns=benchmark,
        n_samples=7,
        random_seed=10,
    )
    np.testing.assert_array_equal(paired, np.zeros(7))
    raw = timeseries.perf_stats_bootstrap(returns, return_stats=False, n_samples=7, random_seed=10)
    assert raw.shape == (7, 13)
    summary = timeseries.perf_stats_bootstrap(returns, n_samples=7, random_seed=10)
    assert summary.index.tolist() == raw.columns.tolist()
    assert summary.columns.tolist() == ["mean", "median", "5%", "95%"]


def test_actual_reference_iid_sampler_matches_local_seeded_stream_and_statistic_arguments():
    reference = pytest.importorskip("pyfolio.timeseries")
    returns = _returns()
    state = np.random.get_state()
    try:
        np.random.seed(10)
        expected = reference.calc_bootstrap(np.quantile, returns, 0.25, n_samples=7)
    finally:
        np.random.set_state(state)
    actual = timeseries.calc_bootstrap(np.quantile, returns, 0.25, n_samples=7, random_seed=10)
    np.testing.assert_array_equal(actual, expected)


def test_actual_reference_complete_bootstrap_table_retains_all_draws_metric_order_and_pairing():
    reference = pytest.importorskip("pyfolio.timeseries")
    generator = np.random.default_rng(20)
    dates = pd.bdate_range("2025-02-03", periods=33, tz="UTC")
    prices = pd.Series(100 * np.cumprod(1 + generator.normal(0.001, 0.02, 33)), index=dates)
    benchmark_prices = pd.Series(
        100 * np.cumprod(1 + generator.normal(0.001, 0.015, 33)), index=dates
    )
    returns = prices.pct_change(fill_method=None).iloc[1:]
    factor = benchmark_prices.pct_change(fill_method=None).iloc[1:]
    state = np.random.get_state()
    try:
        np.random.seed(17)
        expected = reference.perf_stats_bootstrap(
            returns, factor_returns=factor, return_stats=False
        )
    finally:
        np.random.set_state(state)
    actual = timeseries.perf_stats_bootstrap(
        returns, factor_returns=factor, return_stats=False, random_seed=17
    )
    assert actual.shape == expected.shape == (1000, 15)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
