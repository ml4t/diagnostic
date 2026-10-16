"""Prevent observed cash sessions from disappearing from uncertainty samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.uncertainty import (
    compute_backtest_uncertainty,
    compute_paired_uncertainty,
    compute_reality_check,
    compute_selection_adjustment,
    pick_block_length,
)


def _observed_nav_returns(raw: np.ndarray) -> tuple[pd.Series, pd.Series]:
    nav = pd.Series(
        np.r_[100.0, 100.0 * np.cumprod(1.0 + raw)],
        index=pd.bdate_range("2025-01-02", periods=len(raw) + 1, tz="UTC"),
    )
    return nav, nav.pct_change(fill_method=None).iloc[1:]


@pytest.mark.parametrize("representation", ["pandas", "polars", "frame", "array", "list"])
@pytest.mark.parametrize("periods_per_year", [12, 252])
def test_uncertainty_keeps_delayed_trade_cash_observations(representation, periods_per_year):
    _, returns = _observed_nav_returns(
        np.r_[np.zeros(20), np.random.default_rng(11).normal(0.001, 0.01, 80)]
    )
    inputs = {
        "pandas": returns,
        "polars": pl.Series("daily_return", returns.to_numpy()),
        "frame": pl.DataFrame({"timestamp": returns.index, "daily_return": returns.to_numpy()}),
        "array": returns.to_numpy(),
        "list": returns.tolist(),
    }
    result = compute_backtest_uncertainty(
        inputs[representation], periods_per_year=periods_per_year, n_boot=25, block_length=5
    )

    assert len(returns) == 100
    assert (returns.iloc[:20] == 0.0).all()
    assert result.n_observations == 100
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)
    assert result.sharpe == pytest.approx(expected)


@pytest.mark.parametrize("periods_per_year", [12, 252])
def test_uncertainty_point_metrics_use_cash_session_denominators(periods_per_year):
    nav, returns = _observed_nav_returns(
        np.r_[np.zeros(20), np.random.default_rng(11).normal(0.001, 0.01, 80)]
    )
    original = returns.copy(deep=True)
    result = compute_backtest_uncertainty(
        returns, periods_per_year=periods_per_year, n_boot=25, block_length=5
    )
    ordinary = PortfolioAnalysis(returns, periods_per_year=periods_per_year).compute_summary_stats()
    mean = returns.mean()
    std = returns.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (periods_per_year / 100) - 1.0
    drawdown = (nav / nav.cummax() - 1.0).min()

    assert result.sharpe == pytest.approx(mean / std * np.sqrt(periods_per_year))
    assert result.sharpe == pytest.approx(ordinary.sharpe_ratio)
    assert result.sortino == pytest.approx(mean / downside * np.sqrt(periods_per_year))
    assert result.volatility == pytest.approx(std * np.sqrt(periods_per_year))
    assert result.annualized_return == pytest.approx(annual_return)
    assert result.max_drawdown == pytest.approx(drawdown)
    assert result.calmar == pytest.approx(annual_return / abs(drawdown))
    pd.testing.assert_series_equal(returns, original)


def test_selection_does_not_favor_a_variant_by_removing_its_cash_history():
    _, delayed = _observed_nav_returns(
        np.r_[np.zeros(20), np.random.default_rng(11).normal(0.001, 0.01, 80)]
    )
    noise = np.random.default_rng(12).normal(size=100)
    steady_raw = (noise - noise.mean()) / noise.std(ddof=1) * 0.01 + 0.02 / np.sqrt(252)
    _, steady = _observed_nav_returns(steady_raw)
    expected_delayed = delayed.mean() / delayed.std(ddof=1) * np.sqrt(252)
    expected_steady = steady.mean() / steady.std(ddof=1) * np.sqrt(252)
    assert expected_delayed == pytest.approx(1.9101470013537822)
    assert expected_steady == pytest.approx(2.0)

    result = compute_selection_adjustment({"delayed": delayed, "steady": steady})

    assert result.leader == "steady"
    assert result.leader_sharpe == pytest.approx(expected_steady)
    assert result.n_variants == 2


def test_auto_block_uses_all_eight_observations_not_three_active_returns():
    _, returns = _observed_nav_returns(np.array([0.0] * 5 + [0.01, -0.005, 0.02]))

    # The existing short-sample rule selects floor(8 / 3), not floor(3 / 3).
    assert pick_block_length(returns) == 2
    result = compute_backtest_uncertainty(returns, n_boot=25)
    assert result.n_observations == 8
    assert result.bootstrap_block_length == 2


def test_paired_auto_block_keeps_observed_zero_differences_before_divergence():
    _, baseline = _observed_nav_returns(np.array([0.0] * 3 + [0.01, -0.01, 0.02, -0.01, 0.01]))
    _, challenger = _observed_nav_returns(np.array([0.0] * 3 + [0.01, -0.01, 0.03, -0.015, 0.03]))

    result = compute_paired_uncertainty(challenger, baseline, n_boot=25)

    assert result.n_observations == 8
    assert result.bootstrap_block_length == 2
    expected = (
        challenger.mean() / challenger.std(ddof=1) - baseline.mean() / baseline.std(ddof=1)
    ) * np.sqrt(252)
    assert result.sharpe_diff == pytest.approx(expected)


@pytest.mark.parametrize("count", [4, 100])
def test_all_cash_observations_remain_observed_without_fabricating_a_return(count):
    _, returns = _observed_nav_returns(np.zeros(count))

    result = compute_backtest_uncertainty(returns, n_boot=25, block_length=1)

    assert result.n_observations == count
    assert result.sharpe == result.annualized_return == result.max_drawdown == 0.0


def test_explicit_dated_slice_changes_the_window_without_implicit_value_selection():
    _, returns = _observed_nav_returns(
        np.r_[np.zeros(20), np.random.default_rng(11).normal(0.001, 0.01, 80)]
    )
    selected = returns.loc[returns.index[20] :]

    result = compute_backtest_uncertainty(selected, n_boot=25, block_length=5)

    assert len(selected) == result.n_observations == 80
    expected = selected.mean() / selected.std(ddof=1) * np.sqrt(252)
    assert result.sharpe == pytest.approx(expected)
    assert result.sharpe == pytest.approx(2.1368210611826055)


def test_reality_check_keeps_cash_history_when_active_starts_differ():
    _, baseline = _observed_nav_returns(np.zeros(100))
    _, challenger = _observed_nav_returns(
        np.r_[np.zeros(20), np.random.default_rng(11).normal(0.001, 0.01, 80)]
    )

    result = compute_reality_check({"delayed": challenger}, baseline, n_bootstrap=25, block_size=5)

    assert result.best_strategy == "delayed"
    assert result.n_strategies == 1
    assert np.isfinite(result.test_statistic)
