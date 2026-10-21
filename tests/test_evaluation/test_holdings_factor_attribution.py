"""Accepted two-period account and loading inputs have independently known attribution."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import factor


def inputs():
    dates = pd.DatetimeIndex(["2025-07-07T20:00Z", "2025-07-08T20:00Z"])
    starts = pd.DatetimeIndex(["2025-07-03T20:00Z", "2025-07-07T20:00Z"])
    returns = pl.DataFrame({"timestamp": dates, "period_start": starts, "return": [0.03, -0.01]})
    keys = {"timestamp": dates.repeat(2), "asset": ["A", "B", "A", "B"]}
    known = starts.repeat(2) - pd.Timedelta(nanoseconds=1)
    weights = pl.DataFrame({**keys, "weight": [0.5, 0.5, 0.25, 0.75], "known_at": known})
    loadings = pl.DataFrame(
        {**keys, "market": [2.0, 1.0, 1.0, 1.0], "style": [0.0, 0.5, 2.0, 0.0], "known_at": known}
    )
    factors = pl.DataFrame({"timestamp": dates, "style": [0.02, 0.01], "market": [0.01, -0.02]})
    return returns, weights, factors, loadings


def run(frames, **kwargs):
    return factor.compute_holdings_attribution(
        *frames, factor_names=["market", "style"], complete_snapshots=True, **kwargs
    )


def test_declared_beginning_weights_reconcile_exposures_returns_and_linked_wealth():
    frames = inputs()
    before = [frame.clone() for frame in frames]
    result = run(frames)
    np.testing.assert_allclose(
        result.exposures.select("market", "style"), [[1.5, 0.25], [1.0, 0.5]]
    )
    np.testing.assert_allclose(result.contributions["common_returns"], [0.02, -0.015])
    np.testing.assert_allclose(result.contributions["specific_returns"], [0.01, 0.005])
    np.testing.assert_allclose(result.contributions["tilt_returns"], [0.02, -0.02125])
    np.testing.assert_allclose(result.contributions["timing_returns"], [0.0, 0.00625], atol=1e-15)
    np.testing.assert_allclose(result.cumulative["total_returns"], [0.03, 0.0197])
    np.testing.assert_allclose(result.cumulative["common_returns"], [0.02, 0.00455])
    np.testing.assert_allclose(result.cumulative["specific_returns"], [0.01, 0.01515])
    assert result.cumulative_method == "wealth_linked"
    assert all(original.equals(actual) for original, actual in zip(before, frames, strict=True))


def test_explicit_cash_contribution_is_removed_from_the_specific_residual_exactly_once():
    returns, weights, factors, loadings = inputs()
    weights = weights.with_columns(pl.col("weight") * 0.9)
    cash = returns.select("timestamp").with_columns(pl.Series("cash_contribution", [0.001, 0.002]))
    result = run((returns, weights, factors, loadings), cash_contributions=cash)
    np.testing.assert_allclose(result.contributions["common_returns"], [0.018, -0.0135])
    np.testing.assert_allclose(result.contributions["cash_returns"], [0.001, 0.002])
    np.testing.assert_allclose(result.contributions["specific_returns"], [0.011, 0.0015])


@pytest.mark.parametrize("frame_index", [1, 3])
def test_future_weight_or_loading_availability_cannot_enter_a_prior_return_interval(frame_index):
    frames = list(inputs())
    starts = frames[0]["period_start"].gather([0, 0, 1, 1])
    future = starts.cast(pl.Datetime("ns", "UTC")) + pl.duration(nanoseconds=1)
    frames[frame_index] = frames[frame_index].with_columns(future.alias("known_at"))
    assert (frames[frame_index]["known_at"].to_pandas() > starts.to_pandas()).all()
    with pytest.raises(ValueError, match="known_at|available|beginning"):
        run(frames)


def test_missing_nonzero_asset_loading_cannot_disappear_through_an_inner_join():
    returns, weights, factors, loadings = inputs()
    loadings = loadings.filter(
        ~((pl.col("asset") == "B") & (pl.col("timestamp") == returns["timestamp"][0]))
    )
    with pytest.raises(ValueError, match="cover|loading"):
        run((returns, weights, factors, loadings))


def test_an_absent_weight_snapshot_is_not_assumed_to_be_a_complete_all_cash_account():
    frames = inputs()
    with pytest.raises(ValueError, match="complete_snapshots"):
        factor.compute_holdings_attribution(*frames, factor_names=["market", "style"])


def test_missing_whole_weight_observation_cannot_turn_into_an_invented_cash_only_day():
    returns, weights, factors, loadings = inputs()
    weights = weights.filter(pl.col("timestamp") == returns["timestamp"][0])
    with pytest.raises(ValueError, match="snapshot|observation"):
        run((returns, weights, factors, loadings))


def test_overlapping_realized_intervals_cannot_be_linked_as_nonoverlapping_account_returns():
    returns, weights, factors, loadings = inputs()
    start = returns["period_start"][0]
    returns = returns.with_columns(
        pl.lit(start).cast(returns["period_start"].dtype).alias("period_start")
    )
    known = pl.lit(start).cast(pl.Datetime("ns", "UTC")) - pl.duration(nanoseconds=1)
    weights = weights.with_columns(known.alias("known_at"))
    loadings = loadings.with_columns(known.alias("known_at"))
    with pytest.raises(ValueError, match="overlap"):
        run((returns, weights, factors, loadings))


def test_observed_all_cash_timeline_needs_no_invented_asset_or_factor_loading_rows():
    returns, weights, factors, loadings = inputs()
    cash = returns.select("timestamp", pl.col("return").alias("cash_contribution"))
    result = run(
        (returns, weights.head(0), factors, loadings.head(0)),
        cash_contributions=cash,
        observed_snapshot_dates=returns["timestamp"],
    )
    np.testing.assert_array_equal(result.exposures.select("market", "style"), np.zeros((2, 2)))
    np.testing.assert_array_equal(result.contributions["specific_returns"], [0.0, 0.0])
    np.testing.assert_allclose(result.cumulative["cash_returns"], [0.03, 0.0197])


def test_existing_portfolio_analysis_consumes_explicit_beginning_weights_without_fitting_betas():
    from ml4t.diagnostic.evaluation import PortfolioAnalysis

    returns, weights, factors, loadings = inputs()
    analysis = PortfolioAnalysis(
        returns["return"],
        dates=returns["timestamp"],
        initial_timestamp=returns["period_start"][0],
        periods_per_year=12,
    )
    result = analysis.compute_holdings_attribution(
        weights,
        factors,
        loadings,
        factor_names=["market", "style"],
        complete_snapshots=True,
    )
    np.testing.assert_allclose(result.contributions["common_returns"], [0.02, -0.015])
    np.testing.assert_allclose(result.cumulative["total_returns"], [0.03, 0.0197])
    assert result.timing == "beginning_known_at"


def test_portfolio_attribution_cannot_guess_the_first_return_interval_from_calendar_cadence():
    from ml4t.diagnostic.evaluation import PortfolioAnalysis

    returns, weights, factors, loadings = inputs()
    analysis = PortfolioAnalysis(returns["return"], dates=returns["timestamp"])
    with pytest.raises(ValueError, match="period_starts|initial_timestamp"):
        analysis.compute_holdings_attribution(weights, factors, loadings, complete_snapshots=True)
    result = analysis.compute_holdings_attribution(
        weights,
        factors,
        loadings,
        period_starts=returns["period_start"],
        complete_snapshots=True,
    )
    np.testing.assert_allclose(result.contributions["common_returns"], [0.02, -0.015])


def test_bad_holdings_attribution_inputs_leave_existing_portfolio_observations_and_cache_unchanged():
    from ml4t.diagnostic.evaluation import PortfolioAnalysis

    returns, weights, factors, loadings = inputs()
    analysis = PortfolioAnalysis(
        returns["return"],
        dates=returns["timestamp"],
        initial_timestamp=returns["period_start"][0],
    )
    before = analysis.returns
    analysis.compute_summary_stats()
    cached = analysis._metrics_cache
    loadings = loadings.with_columns(pl.col("timestamp").alias("known_at"))
    with pytest.raises(ValueError, match="known_at"):
        analysis.compute_holdings_attribution(weights, factors, loadings, complete_snapshots=True)
    np.testing.assert_array_equal(analysis.returns, before)
    assert analysis._metrics_cache is cached
