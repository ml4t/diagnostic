"""Drawdown must include the capital held before the first observed return."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis import max_drawdown
from ml4t.diagnostic.evaluation.uncertainty import compute_backtest_uncertainty
from ml4t.diagnostic.metrics import maximum_drawdown
from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet
from ml4t.diagnostic.visualization.portfolio import plot_drawdown_underwater


@pytest.mark.parametrize(
    ("nav", "expected"),
    [
        ([100, 80, 88, 92.4], [-0.2, -0.12, -0.076]),
        ([100, 90, 80, 70], [-0.1, -0.2, -0.3]),
        ([100, 110, 104.5, 115], [0, -0.05, 0]),
        ([100, 110, 120, 130], [0, 0, 0]),
    ],
)
def test_nav_pipeline_preserves_initial_capital_in_summary_episodes_and_underwater(nav, expected):
    equity = pd.Series(nav, index=pd.bdate_range("2025-01-01", periods=len(nav)))
    returns = equity.pct_change().iloc[1:]
    analysis = PortfolioAnalysis(returns)
    minimum = min(expected)
    assert analysis.compute_summary_stats().max_drawdown == pytest.approx(minimum)
    details = analysis.compute_drawdown_analysis()
    assert details.max_drawdown == pytest.approx(minimum)
    np.testing.assert_allclose(details.underwater_curve, expected, atol=1e-14)
    assert details.dates.to_list() == list(returns.index.to_pydatetime())
    figure = plot_drawdown_underwater(analysis)
    np.testing.assert_allclose(figure.data[0].y, expected, atol=1e-14)
    assert len(figure.data[0].x) == len(returns)
    assert max_drawdown(returns) == pytest.approx(minimum)
    assert maximum_drawdown(returns)["max_drawdown"] == pytest.approx(minimum)


def test_initial_peak_time_is_unavailable_without_inventing_a_trading_observation():
    returns = pd.Series([-0.2, 0.1, 0.05], index=pd.bdate_range("2025-01-02", periods=3))
    episode = PortfolioAnalysis(returns).compute_drawdown_analysis().top_drawdowns[0]
    assert episode.peak_date is None
    assert episode.valley_date == returns.index[0]
    assert episode.duration_days == 1  # Legacy field name counts return observations.
    assert maximum_drawdown(returns)["peak_date"] == -1


def test_core_drawdown_peak_index_does_not_name_first_return_as_initial_capital():
    result = maximum_drawdown(np.array([-0.2, 0.1, 0.05]))
    assert result["max_drawdown"] == pytest.approx(-0.2)
    assert result["peak_date"] == -1
    assert result["max_drawdown_duration"] == 1


def test_initial_valuation_timestamp_is_retained_when_the_producer_supplies_it():
    nav = pd.Series([100, 80, 100, 90, 100], index=pd.bdate_range("2025-01-01", periods=5))
    returns = nav.pct_change().iloc[1:]
    analysis = PortfolioAnalysis(returns, initial_timestamp=nav.index[0])
    episodes = analysis.compute_drawdown_analysis().top_drawdowns
    assert len(episodes) == 2
    assert episodes[0].peak_date == nav.index[0]
    assert episodes[0].valley_date == nav.index[1]
    assert episodes[0].recovery_date == nav.index[2]
    assert episodes[0].depth == pytest.approx(-0.2)
    assert episodes[1].peak_date == nav.index[2]
    assert episodes[1].depth == pytest.approx(-0.1)


@pytest.mark.parametrize("initial", ["2025-01-02", "2025-01-03"])
def test_initial_valuation_cannot_be_at_or_after_the_first_return(initial):
    returns = pd.Series([-0.2, 0.1], index=pd.bdate_range("2025-01-02", periods=2))
    with pytest.raises(ValueError, match="initial_timestamp"):
        PortfolioAnalysis(returns, initial_timestamp=pd.Timestamp(initial))


@pytest.mark.parametrize("initial", [pd.NaT, pd.Timestamp("2025-01-01", tz="UTC")])
def test_initial_valuation_rejects_missing_or_incompatible_timezone(initial):
    returns = pd.Series([-0.2, 0.1], index=pd.bdate_range("2025-01-02", periods=2))
    with pytest.raises(ValueError, match="initial_timestamp"):
        PortfolioAnalysis(returns, initial_timestamp=initial)


def test_initial_timestamp_requires_real_return_dates_instead_of_mixing_time_and_row_indices():
    with pytest.raises(ValueError, match="initial_timestamp"):
        PortfolioAnalysis([-0.2, 0.1], initial_timestamp=pd.Timestamp("2025-01-01"))


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("returns", [[-1.0], [-1.0, 0, 0.1], [0.1, -1.0, 0]])
def test_total_loss_stays_at_minus_one_without_zero_high_water_or_division_errors(returns):
    analysis = PortfolioAnalysis(returns)
    assert analysis.compute_summary_stats().max_drawdown == -1
    details = analysis.compute_drawdown_analysis()
    assert details.max_drawdown == -1
    assert details.underwater_curve[-1] == -1
    assert max_drawdown(returns) == -1


def test_uncertainty_point_drawdown_and_calmar_include_initial_capital():
    returns = np.array([-0.2, 0.1, 0.05, 0.0])
    result = compute_backtest_uncertainty(returns, periods_per_year=4, n_boot=30, block_length=2)
    assert result.max_drawdown == pytest.approx(-0.2)
    assert result.calmar == pytest.approx(-0.076 / 0.2)
    assert PortfolioAnalysis(
        returns, periods_per_year=4
    ).compute_summary_stats().calmar_ratio == pytest.approx(-0.076 / 0.2)


def test_raw_return_report_has_consistent_initial_drawdown_magnitude_curve_and_peak_availability():
    metrics = {}
    html = generate_backtest_tearsheet(
        returns=np.array([-0.2, 0.1, 0.05]), metrics=metrics, template="full"
    )
    assert metrics["max_drawdown"] == pytest.approx(0.2)
    assert "Initial capital (time unavailable)" in html
    assert "-20.0%" in html


def test_raw_chart_cannot_disagree_with_correct_precomputed_initial_drawdown():
    html = generate_backtest_tearsheet(
        returns=np.array([-0.2, 0.1, 0.05]), metrics={"max_drawdown": 0.2}, template="full"
    )
    section = html[html.index('data-section="overview_snapshot"') :]
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', section)
    data, _ = json.JSONDecoder().raw_decode(section[match.end() :])
    curve = next(trace for trace in data if trace.get("name") == "Drawdown")
    np.testing.assert_allclose(curve["y"], [-0.2, -0.12, -0.076])


def test_empty_low_level_drawdown_is_unavailable_without_a_reduction_error():
    assert np.isnan(max_drawdown([]))
