"""A drawdown query must not inherit another call's selection or threshold."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def observations():
    return pd.Series(
        [0.1, -0.05, 0.06, -0.2, 0.3, -0.03, 0.04],
        index=pd.bdate_range("2025-01-02", periods=7, tz="America/New_York"),
    )


def assert_episodes(result, dates, top_n, threshold):
    # Explicit wealth path: 1.1,1.045,1.1077,.88616,1.152008,1.11744776,1.1621456704.
    expected = [(-0.2, 2, 3, 4), (-0.05, 0, 1, 2), (-0.03, 4, 5, 6)]
    selected = [row for row in expected if -row[0] > threshold]
    assert result.num_drawdowns == len(selected)
    assert len(result.top_drawdowns) == min(top_n, len(selected))
    np.testing.assert_allclose(
        result.underwater_curve, [0, -0.05, 0, -0.2, 0, -0.03, 0], atol=1e-14
    )
    assert result.max_drawdown == pytest.approx(-0.2)
    assert result.current_drawdown == pytest.approx(0)
    for episode, (depth, peak, valley, recovery) in zip(result.top_drawdowns, selected[:top_n]):
        assert episode.depth == pytest.approx(depth)
        assert episode.peak_date == dates[peak]
        assert episode.valley_date == dates[valley]
        assert episode.recovery_date == dates[recovery]
        assert episode.duration_days == 1
        assert episode.recovery_days == 1
    expected_average = np.mean([row[0] for row in selected]) if selected else 0.0
    assert result.avg_drawdown == pytest.approx(expected_average)


@pytest.mark.parametrize("first", range(5))
def test_query_argument_changes_cannot_return_a_prior_call_selection_or_threshold(first):
    values = observations()
    analysis = PortfolioAnalysis(values)
    queries = [(1, 0.01), (3, 0.01), (5, 0.1), (0, 0.04), (2, 0.04)]
    queries = queries[first:] + queries[:first]
    for top_n, threshold in queries + queries[::-1]:
        result = analysis.compute_drawdown_analysis(top_n=top_n, threshold=threshold)
        assert_episodes(result, values.index, top_n, threshold)
        fresh = PortfolioAnalysis(values).compute_drawdown_analysis(top_n, threshold)
        assert result.top_drawdowns == fresh.top_drawdowns
        assert result.num_drawdowns == fresh.num_drawdowns


def test_mutating_a_query_result_cannot_poison_any_cached_argument_pair():
    values = observations()
    analysis = PortfolioAnalysis(values)
    result = analysis.compute_drawdown_analysis(top_n=1, threshold=0.01)
    result.top_drawdowns[0].depth = -0.99
    result.top_drawdowns.clear()
    result.underwater_curve[3] = -0.99
    for top_n, threshold in [(1, 0.01), (3, 0.01), (5, 0.1)]:
        assert_episodes(
            analysis.compute_drawdown_analysis(top_n, threshold), values.index, top_n, threshold
        )


@pytest.mark.parametrize("primed", [False, True])
@pytest.mark.parametrize("top_n", [-1, 1.5, True, "1", None, np.bool_(True)])
def test_invalid_top_n_is_rejected_even_after_a_valid_query(primed, top_n):
    analysis = PortfolioAnalysis(observations())
    if primed:
        analysis.compute_drawdown_analysis()
    with pytest.raises((TypeError, ValueError), match="top_n"):
        analysis.compute_drawdown_analysis(top_n=top_n)


@pytest.mark.parametrize("primed", [False, True])
@pytest.mark.parametrize("threshold", [np.nan, np.inf, -0.01, 1.01, True, ".1", None])
def test_invalid_threshold_is_rejected_even_after_a_valid_query(primed, threshold):
    analysis = PortfolioAnalysis(observations())
    if primed:
        analysis.compute_drawdown_analysis()
    with pytest.raises((TypeError, ValueError), match="threshold"):
        analysis.compute_drawdown_analysis(threshold=threshold)


def test_numpy_scalar_queries_and_a_unit_threshold_preserve_valid_overview_results():
    values = observations()
    analysis = PortfolioAnalysis(values)
    assert_episodes(
        analysis.compute_drawdown_analysis(np.int64(2), np.float64(0.04)), values.index, 2, 0.04
    )
    result = analysis.compute_drawdown_analysis(top_n=0, threshold=1)
    assert_episodes(result, values.index, 0, 1)
    assert isinstance(result.underwater_curve, pl.Series)
