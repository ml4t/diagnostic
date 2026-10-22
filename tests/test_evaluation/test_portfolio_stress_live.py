"""Regime membership and units must survive the actual public account workflow."""

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioRegimeResult


def _account(*, with_fills=False, scale=252):
    dates = pd.date_range("2025-01-13", periods=5, tz="UTC")
    fills = None
    if with_fills:
        fills = pd.DataFrame(
            {
                "timestamp": [dates[3] - pd.Timedelta(1, unit="ns"), dates[3]],
                "asset": ["A", "A"],
                "quantity": [1.0, -1.0],
                "price": [100.0, 101.0],
            }
        )
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, -0.02, 0.03, -0.04, 0.05], index=dates),
        transactions=fills,
        calendar=None,
        transaction_timestamp_col="timestamp",
        periods_per_year=scale,
    )
    return analysis, dates


def test_stress_membership_reuses_native_statistics_and_keeps_unobserved_windows_unknown():
    analysis, dates = _account()
    result = analysis.compute_stress(
        {
            "observed": (dates[1], dates[3]),
            "absent": (
                dates[-1] + pd.Timedelta(1, unit="D"),
                dates[-1] + pd.Timedelta(3, unit="D"),
            ),
        }
    )
    assert result.samples["return"].to_list() == [-0.02, 0.03, -0.04]
    assert pd.DatetimeIndex(result.samples["date"].to_pandas()).equals(dates[1:4])
    actual = result.statistics.filter(pl.col("period") == "observed").row(0, named=True)
    assert actual["total_return"] == pytest.approx(-0.030976)
    assert actual["max_drawdown"] == pytest.approx(-0.04)
    assert actual["annual_volatility"] == pytest.approx((0.0013 * 252) ** 0.5)
    absent = result.statistics.filter(pl.col("period") == "absent").row(0, named=True)
    assert absent["n_observations"] == 0 and absent["total_return"] is None
    assert result.coverage.filter(pl.col("period") == "absent")["status"].to_list() == [
        "unobserved"
    ]


def test_live_split_assigns_boundary_once_and_preserves_actual_nanosecond_execution_keys():
    analysis, dates = _account(with_fills=True)
    result = analysis.compute_live_split(live_start=dates[3])
    assert result.samples.filter(pl.col("regime") == "pre_live")["return"].to_list() == [
        0.01,
        -0.02,
        0.03,
    ]
    assert result.samples.filter(pl.col("regime") == "live")["return"].to_list() == [-0.04, 0.05]
    before = result.transactions.filter(pl.col("regime") == "pre_live")
    live = result.transactions.filter(pl.col("regime") == "live")
    assert before["timestamp"].cast(pl.Int64).to_list() == [dates[3].value - 1]
    assert live["timestamp"].cast(pl.Int64).to_list() == [dates[3].value]
    assert before["quantity"].to_list() == [1.0] and live["quantity"].to_list() == [-1.0]
    assert result.metadata["account"]["calendar"] is None


def test_common_explicit_scale_reaches_each_regime_without_annualizing_period_returns():
    annual, dates = _account()
    monthly, _ = _account(scale=12)
    periods = {"window": (dates[1], dates[3])}
    first, second = annual.compute_stress(periods), monthly.compute_stress(periods)
    assert first.statistics["total_return"].to_list() == second.statistics["total_return"].to_list()
    assert first.statistics["annual_volatility"][0] == pytest.approx(
        second.statistics["annual_volatility"][0] * 21**0.5
    )


def test_returned_regime_frames_and_json_cannot_mutate_owned_input_or_result():
    analysis, dates = _account(with_fills=True)
    original = analysis.input_data.to_json()
    result = analysis.compute_live_split(live_start=dates[3])
    samples = result.samples
    samples.replace_column(samples.columns.index("return"), pl.Series("return", [0.0] * 5))
    assert result.samples["return"].to_list() == [0.01, -0.02, 0.03, -0.04, 0.05]
    restored = PortfolioRegimeResult.from_json(result.to_json())
    assert restored.samples.equals(result.samples)
    assert restored.transactions.equals(result.transactions)
    assert analysis.input_data.to_json() == original


def test_invalid_window_or_timezone_fails_without_mutating_account_snapshots():
    analysis, dates = _account(with_fills=True)
    original = analysis.input_data.to_json()
    with pytest.raises(ValueError, match="ends before"):
        analysis.compute_stress({"bad": (dates[3], dates[1])})
    with pytest.raises(ValueError, match="timezone"):
        analysis.compute_live_split(live_start=pd.Timestamp("2025-01-16T00:00"))
    assert analysis.input_data.to_json() == original
