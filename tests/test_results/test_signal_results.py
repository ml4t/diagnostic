"""Tests for SignalResults module result classes.

These tests focus on the result class methods:
- get_dataframe()
- summary()
- list_available_dataframes()
- save_html(), save_json(), save_png()
- to_dashboard(), to_dict()
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import pytest

from ml4t.diagnostic.results.signal_results import (
    IRtcResult,
    QuantileAnalysisResult,
    RASICResult,
    SignalICResult,
    SignalTearSheet,
    TurnoverAnalysisResult,
    _figure_from_data,
    _normalize_period,
    _validate_dict_keys_match,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_ic_result() -> SignalICResult:
    """Create a sample SignalICResult for testing."""
    return SignalICResult(
        ic_by_date={
            "1D": [0.05, 0.03, 0.04, 0.02, 0.06],
            "5D": [0.08, 0.06, 0.07, 0.05, 0.09],
            "21D": [0.10, 0.12, 0.09, 0.11, 0.13],
        },
        dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
        ic_mean={"1D": 0.04, "5D": 0.07, "21D": 0.11},
        ic_std={"1D": 0.015, "5D": 0.015, "21D": 0.015},
        ic_t_stat={"1D": 2.67, "5D": 4.67, "21D": 7.33},
        ic_p_value={"1D": 0.02, "5D": 0.001, "21D": 0.0001},
        ic_positive_pct={"1D": 1.0, "5D": 1.0, "21D": 1.0},
        ic_ir={"1D": 2.67, "5D": 4.67, "21D": 7.33},
        ic_t_stat_hac={"1D": 2.5, "5D": 4.5, "21D": 7.0},
        ic_p_value_hac={"1D": 0.03, "5D": 0.002, "21D": 0.0002},
        ras_adjusted_ic={"1D": 0.025, "5D": 0.055, "21D": 0.095},
        ras_significant={"1D": True, "5D": True, "21D": True},
        ras_complexity=0.015,
    )


@pytest.fixture
def sample_ic_result_no_ras() -> SignalICResult:
    """Create SignalICResult without RAS fields."""
    return SignalICResult(
        ic_by_date={"1D": [0.05, 0.03]},
        dates=["2020-01-01", "2020-01-02"],
        ic_mean={"1D": 0.04},
        ic_std={"1D": 0.015},
        ic_t_stat={"1D": 2.67},
        ic_p_value={"1D": 0.02},
        ic_positive_pct={"1D": 1.0},
        ic_ir={"1D": 2.67},
    )


@pytest.fixture
def sample_ras_result() -> RASICResult:
    """Create a sample RASICResult for testing."""
    return RASICResult(
        n_signals=5,
        n_samples=100,
        delta=0.05,
        kappa=0.1,
        n_simulations=10000,
        rademacher_complexity=0.02,
        massart_bound=0.03,
        observed_ic={"sig_1": 0.08, "sig_2": 0.05, "sig_3": 0.03, "sig_4": 0.01, "sig_5": -0.02},
        adjusted_ic={"sig_1": 0.04, "sig_2": 0.01, "sig_3": -0.01, "sig_4": -0.03, "sig_5": -0.06},
        is_significant={
            "sig_1": True,
            "sig_2": True,
            "sig_3": False,
            "sig_4": False,
            "sig_5": False,
        },
        n_significant=2,
        any_significant=True,
        data_snooping_term=0.04,
        estimation_error_term=0.03,
    )


@pytest.fixture
def sample_quantile_result() -> QuantileAnalysisResult:
    """Create a sample QuantileAnalysisResult for testing."""
    return QuantileAnalysisResult(
        n_quantiles=5,
        quantile_labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        periods=["1D", "5D"],
        mean_returns={
            "1D": {"Q1": -0.005, "Q2": -0.001, "Q3": 0.001, "Q4": 0.003, "Q5": 0.008},
            "5D": {"Q1": -0.015, "Q2": -0.003, "Q3": 0.002, "Q4": 0.010, "Q5": 0.025},
        },
        std_returns={
            "1D": {"Q1": 0.02, "Q2": 0.02, "Q3": 0.02, "Q4": 0.02, "Q5": 0.02},
            "5D": {"Q1": 0.04, "Q2": 0.04, "Q3": 0.04, "Q4": 0.04, "Q5": 0.04},
        },
        count_by_quantile={"Q1": 100, "Q2": 100, "Q3": 100, "Q4": 100, "Q5": 100},
        spread_mean={"1D": 0.013, "5D": 0.040},
        spread_std={"1D": 0.005, "5D": 0.015},
        spread_t_stat={"1D": 2.6, "5D": 2.67},
        spread_p_value={"1D": 0.02, "5D": 0.015},
        spread_ci_lower={"1D": 0.003, "5D": 0.01},
        spread_ci_upper={"1D": 0.023, "5D": 0.07},
        is_monotonic={"1D": True, "5D": True},
        monotonicity_direction={"1D": "increasing", "5D": "increasing"},
        rank_correlation={"1D": 1.0, "5D": 1.0},
        cumulative_returns={
            "1D": {
                "Q1": [1.0, 0.995, 0.990],
                "Q2": [1.0, 0.999, 0.998],
                "Q3": [1.0, 1.001, 1.002],
                "Q4": [1.0, 1.003, 1.006],
                "Q5": [1.0, 1.008, 1.016],
            },
            "5D": {
                "Q1": [1.0, 0.985, 0.970],
                "Q2": [1.0, 0.997, 0.994],
                "Q3": [1.0, 1.002, 1.004],
                "Q4": [1.0, 1.010, 1.020],
                "Q5": [1.0, 1.025, 1.051],
            },
        },
        cumulative_dates=["2020-01-01", "2020-01-02", "2020-01-03"],
    )


@pytest.fixture
def sample_quantile_result_no_cumulative() -> QuantileAnalysisResult:
    """Create QuantileAnalysisResult without cumulative returns."""
    return QuantileAnalysisResult(
        n_quantiles=3,
        quantile_labels=["Q1", "Q2", "Q3"],
        periods=["1D"],
        mean_returns={"1D": {"Q1": -0.005, "Q2": 0.001, "Q3": 0.008}},
        std_returns={"1D": {"Q1": 0.02, "Q2": 0.02, "Q3": 0.02}},
        count_by_quantile={"Q1": 100, "Q2": 100, "Q3": 100},
        spread_mean={"1D": 0.013},
        spread_std={"1D": 0.005},
        spread_t_stat={"1D": 2.6},
        spread_p_value={"1D": 0.02},
        spread_ci_lower={"1D": 0.003},
        spread_ci_upper={"1D": 0.023},
        is_monotonic={"1D": True},
        monotonicity_direction={"1D": "increasing"},
        rank_correlation={"1D": 1.0},
    )


@pytest.fixture
def sample_turnover_result() -> TurnoverAnalysisResult:
    """Create a sample TurnoverAnalysisResult for testing."""
    return TurnoverAnalysisResult(
        quantile_turnover={
            "1D": {"Q1": 0.15, "Q2": 0.12, "Q3": 0.10, "Q4": 0.12, "Q5": 0.15},
            "5D": {"Q1": 0.25, "Q2": 0.20, "Q3": 0.18, "Q4": 0.20, "Q5": 0.25},
        },
        mean_turnover={"1D": 0.128, "5D": 0.216},
        top_quantile_turnover={"1D": 0.15, "5D": 0.25},
        bottom_quantile_turnover={"1D": 0.15, "5D": 0.25},
        autocorrelation={
            "1D": [0.8, 0.65, 0.53, 0.43, 0.35],
            "5D": [0.7, 0.55, 0.42, 0.32, 0.24],
        },
        autocorrelation_lags=[1, 2, 3, 4, 5],
        mean_autocorrelation={"1D": 0.552, "5D": 0.446},
        half_life={"1D": 3.0, "5D": 2.5},
    )


@pytest.fixture
def sample_irtc_result() -> IRtcResult:
    """Create a sample IRtcResult for testing."""
    return IRtcResult(
        cost_per_trade=0.001,
        ir_gross={"1D": 0.5, "5D": 0.8},
        ir_tc={"1D": 0.35, "5D": 0.65},
        implied_cost={"1D": 0.0015, "5D": 0.0012},
        breakeven_cost={"1D": 0.003, "5D": 0.004},
        cost_drag={"1D": 0.30, "5D": 0.1875},
    )


@pytest.fixture
def sample_tear_sheet(
    sample_ic_result: SignalICResult,
    sample_quantile_result: QuantileAnalysisResult,
    sample_turnover_result: TurnoverAnalysisResult,
    sample_irtc_result: IRtcResult,
) -> SignalTearSheet:
    """Create a sample SignalTearSheet for testing."""
    return SignalTearSheet(
        signal_name="test_signal",
        n_assets=30,
        n_dates=100,
        date_range=("2020-01-01", "2020-04-10"),
        ic_analysis=sample_ic_result,
        quantile_analysis=sample_quantile_result,
        turnover_analysis=sample_turnover_result,
        ir_tc_analysis=sample_irtc_result,
        figures={
            "ic_time_series": '{"data": [], "layout": {"title": "IC Time Series"}}',
            "quantile_returns": '{"data": [], "layout": {"title": "Quantile Returns"}}',
        },
    )


@pytest.fixture
def minimal_tear_sheet() -> SignalTearSheet:
    """Create a minimal SignalTearSheet without component results."""
    return SignalTearSheet(
        signal_name="minimal_signal",
        n_assets=10,
        n_dates=50,
        date_range=("2020-01-01", "2020-02-20"),
    )


# =============================================================================
# SignalICResult Tests
# =============================================================================


class TestSignalICResultGetDataframe:
    """Tests for SignalICResult.get_dataframe method."""

    def test_get_dataframe_default(self, sample_ic_result: SignalICResult) -> None:
        """Test get_dataframe with default (ic_by_date)."""
        df = sample_ic_result.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "date" in df.columns
        assert "ic_1D" in df.columns
        assert "ic_5D" in df.columns
        assert "ic_21D" in df.columns
        assert len(df) == 5  # 5 dates

    def test_get_dataframe_ic_by_date(self, sample_ic_result: SignalICResult) -> None:
        """Test get_dataframe with name='ic_by_date'."""
        df = sample_ic_result.get_dataframe("ic_by_date")

        assert isinstance(df, pl.DataFrame)
        assert "date" in df.columns
        assert len(df) == 5

    def test_get_dataframe_summary(self, sample_ic_result: SignalICResult) -> None:
        """Test get_dataframe with name='summary'."""
        df = sample_ic_result.get_dataframe("summary")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "ic_mean" in df.columns
        assert "ic_std" in df.columns
        assert "ic_t_stat" in df.columns
        assert "ic_p_value" in df.columns
        assert "ic_positive_pct" in df.columns
        assert "ic_ir" in df.columns
        # RAS columns should be present
        assert "ras_adjusted_ic" in df.columns
        assert "ras_significant" in df.columns
        assert len(df) == 3  # 3 periods

    def test_get_dataframe_summary_without_ras(
        self, sample_ic_result_no_ras: SignalICResult
    ) -> None:
        """Test get_dataframe summary when RAS not present."""
        df = sample_ic_result_no_ras.get_dataframe("summary")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        # RAS columns should NOT be present
        assert "ras_adjusted_ic" not in df.columns
        assert "ras_significant" not in df.columns

    def test_get_dataframe_invalid_name(self, sample_ic_result: SignalICResult) -> None:
        """Test get_dataframe with invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown DataFrame name"):
            sample_ic_result.get_dataframe("invalid_name")


class TestSignalICResultListDataframes:
    """Tests for SignalICResult.list_available_dataframes method."""

    def test_list_available_dataframes(self, sample_ic_result: SignalICResult) -> None:
        """Test list_available_dataframes returns expected list."""
        available = sample_ic_result.list_available_dataframes()

        assert isinstance(available, list)
        assert "ic_by_date" in available
        assert "summary" in available


class TestSignalICResultSummary:
    """Tests for SignalICResult.summary method."""

    def test_summary_basic(self, sample_ic_result: SignalICResult) -> None:
        """Test summary returns formatted string."""
        summary = sample_ic_result.summary()

        assert isinstance(summary, str)
        assert "IC Analysis Summary" in summary
        assert "Period: 1D" in summary
        assert "Period: 5D" in summary
        assert "Period: 21D" in summary
        assert "Mean IC:" in summary
        assert "RAS IC:" in summary

    def test_summary_without_ras(self, sample_ic_result_no_ras: SignalICResult) -> None:
        """Test summary without RAS fields."""
        summary = sample_ic_result_no_ras.summary()

        assert isinstance(summary, str)
        assert "IC Analysis Summary" in summary
        # Should not have RAS section
        assert "RAS IC:" not in summary


# =============================================================================
# RASICResult Tests
# =============================================================================


class TestRASICResultGetDataframe:
    """Tests for RASICResult.get_dataframe method."""

    def test_get_dataframe_default(self, sample_ras_result: RASICResult) -> None:
        """Test get_dataframe with default (signals)."""
        df = sample_ras_result.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "signal" in df.columns
        assert "observed_ic" in df.columns
        assert "adjusted_ic" in df.columns
        assert "is_significant" in df.columns
        assert len(df) == 5  # 5 signals

    def test_get_dataframe_signals(self, sample_ras_result: RASICResult) -> None:
        """Test get_dataframe with name='signals'."""
        df = sample_ras_result.get_dataframe("signals")

        assert isinstance(df, pl.DataFrame)
        assert "signal" in df.columns
        assert len(df) == 5

    def test_get_dataframe_summary(self, sample_ras_result: RASICResult) -> None:
        """Test get_dataframe with name='summary'."""
        df = sample_ras_result.get_dataframe("summary")

        assert isinstance(df, pl.DataFrame)
        assert "metric" in df.columns
        assert "value" in df.columns
        # Check expected metrics are present
        metrics = df["metric"].to_list()
        assert "n_signals" in metrics
        assert "n_samples" in metrics
        assert "rademacher_complexity" in metrics
        assert "massart_bound" in metrics
        assert "n_significant" in metrics

    def test_get_dataframe_invalid_name(self, sample_ras_result: RASICResult) -> None:
        """Test get_dataframe with invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown DataFrame name"):
            sample_ras_result.get_dataframe("invalid_name")


class TestRASICResultListDataframes:
    """Tests for RASICResult.list_available_dataframes method."""

    def test_list_available_dataframes(self, sample_ras_result: RASICResult) -> None:
        """Test list_available_dataframes returns expected list."""
        available = sample_ras_result.list_available_dataframes()

        assert isinstance(available, list)
        assert "signals" in available
        assert "summary" in available


class TestRASICResultSummary:
    """Tests for RASICResult.summary method."""

    def test_summary_with_significant(self, sample_ras_result: RASICResult) -> None:
        """Test summary when there are significant signals."""
        summary = sample_ras_result.summary()

        assert isinstance(summary, str)
        assert "RAS IC Analysis Summary" in summary
        assert "Signals Tested:" in summary
        assert "5" in summary  # n_signals
        assert "Time Periods:" in summary
        assert "100" in summary  # n_samples
        assert "Rademacher Complexity:" in summary
        assert "Significant Signals:" in summary
        assert "sig_1" in summary  # First significant signal

    def test_summary_without_significant(self) -> None:
        """Test summary when there are no significant signals."""
        result = RASICResult(
            n_signals=3,
            n_samples=50,
            delta=0.05,
            kappa=0.1,
            n_simulations=1000,
            rademacher_complexity=0.05,
            massart_bound=0.06,
            observed_ic={"sig_1": 0.02, "sig_2": 0.01, "sig_3": 0.0},
            adjusted_ic={"sig_1": -0.03, "sig_2": -0.04, "sig_3": -0.05},
            is_significant={"sig_1": False, "sig_2": False, "sig_3": False},
            n_significant=0,
            any_significant=False,
            data_snooping_term=0.10,
            estimation_error_term=0.05,
        )

        summary = result.summary()

        assert isinstance(summary, str)
        assert "Significant Signals:" in summary
        assert "0 / 3" in summary
        # Should not have "Significant signals:" section since none are significant


# =============================================================================
# QuantileAnalysisResult Tests
# =============================================================================


class TestQuantileAnalysisResultGetDataframe:
    """Tests for QuantileAnalysisResult.get_dataframe method."""

    def test_get_dataframe_default(self, sample_quantile_result: QuantileAnalysisResult) -> None:
        """Test get_dataframe with default (mean_returns)."""
        df = sample_quantile_result.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "quantile" in df.columns
        assert "mean_return" in df.columns
        assert "std_return" in df.columns
        # 5 quantiles × 2 periods = 10 rows
        assert len(df) == 10

    def test_get_dataframe_mean_returns(
        self, sample_quantile_result: QuantileAnalysisResult
    ) -> None:
        """Test get_dataframe with name='mean_returns'."""
        df = sample_quantile_result.get_dataframe("mean_returns")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "quantile" in df.columns

    def test_get_dataframe_spread(self, sample_quantile_result: QuantileAnalysisResult) -> None:
        """Test get_dataframe with name='spread'."""
        df = sample_quantile_result.get_dataframe("spread")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "spread_mean" in df.columns
        assert "spread_std" in df.columns
        assert "spread_t_stat" in df.columns
        assert "spread_p_value" in df.columns
        assert "spread_ci_lower" in df.columns
        assert "spread_ci_upper" in df.columns
        assert "is_monotonic" in df.columns
        assert "monotonicity_direction" in df.columns
        assert "rank_correlation" in df.columns
        assert len(df) == 2  # 2 periods

    def test_get_dataframe_cumulative(self, sample_quantile_result: QuantileAnalysisResult) -> None:
        """Test get_dataframe with name='cumulative'."""
        df = sample_quantile_result.get_dataframe("cumulative")

        assert isinstance(df, pl.DataFrame)
        assert "date" in df.columns
        # Should have columns like "1D_Q1", "1D_Q2", etc.
        assert "1D_Q1" in df.columns
        assert "5D_Q5" in df.columns
        assert len(df) == 3  # 3 dates

    def test_get_dataframe_cumulative_not_available(
        self, sample_quantile_result_no_cumulative: QuantileAnalysisResult
    ) -> None:
        """Test get_dataframe cumulative when not available raises ValueError."""
        with pytest.raises(ValueError, match="Cumulative returns not available"):
            sample_quantile_result_no_cumulative.get_dataframe("cumulative")

    def test_get_dataframe_invalid_name(
        self, sample_quantile_result: QuantileAnalysisResult
    ) -> None:
        """Test get_dataframe with invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown DataFrame name"):
            sample_quantile_result.get_dataframe("invalid_name")


class TestQuantileAnalysisResultListDataframes:
    """Tests for QuantileAnalysisResult.list_available_dataframes method."""

    def test_list_available_dataframes_with_cumulative(
        self, sample_quantile_result: QuantileAnalysisResult
    ) -> None:
        """Test list_available_dataframes with cumulative returns."""
        available = sample_quantile_result.list_available_dataframes()

        assert isinstance(available, list)
        assert "mean_returns" in available
        assert "spread" in available
        assert "cumulative" in available

    def test_list_available_dataframes_without_cumulative(
        self, sample_quantile_result_no_cumulative: QuantileAnalysisResult
    ) -> None:
        """Test list_available_dataframes without cumulative returns."""
        available = sample_quantile_result_no_cumulative.list_available_dataframes()

        assert isinstance(available, list)
        assert "mean_returns" in available
        assert "spread" in available
        assert "cumulative" not in available


class TestQuantileAnalysisResultSummary:
    """Tests for QuantileAnalysisResult.summary method."""

    def test_summary(self, sample_quantile_result: QuantileAnalysisResult) -> None:
        """Test summary returns formatted string."""
        summary = sample_quantile_result.summary()

        assert isinstance(summary, str)
        assert "Quantile Analysis Summary" in summary
        assert "Period: 1D" in summary
        assert "Period: 5D" in summary
        assert "Q1" in summary
        assert "Q5" in summary
        assert "Spread (Top-Bottom):" in summary
        assert "Monotonic:" in summary


# =============================================================================
# TurnoverAnalysisResult Tests
# =============================================================================


class TestTurnoverAnalysisResultGetDataframe:
    """Tests for TurnoverAnalysisResult.get_dataframe method."""

    def test_get_dataframe_default(self, sample_turnover_result: TurnoverAnalysisResult) -> None:
        """Test get_dataframe with default (turnover)."""
        df = sample_turnover_result.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "quantile" in df.columns
        assert "turnover" in df.columns
        # 5 quantiles × 2 periods = 10 rows
        assert len(df) == 10

    def test_get_dataframe_turnover(self, sample_turnover_result: TurnoverAnalysisResult) -> None:
        """Test get_dataframe with name='turnover'."""
        df = sample_turnover_result.get_dataframe("turnover")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "quantile" in df.columns
        assert "turnover" in df.columns

    def test_get_dataframe_autocorrelation(
        self, sample_turnover_result: TurnoverAnalysisResult
    ) -> None:
        """Test get_dataframe with name='autocorrelation'."""
        df = sample_turnover_result.get_dataframe("autocorrelation")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "lag" in df.columns
        assert "autocorrelation" in df.columns
        # 5 lags × 2 periods = 10 rows
        assert len(df) == 10

    def test_get_dataframe_summary(self, sample_turnover_result: TurnoverAnalysisResult) -> None:
        """Test get_dataframe with name='summary'."""
        df = sample_turnover_result.get_dataframe("summary")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "mean_turnover" in df.columns
        assert "top_turnover" in df.columns
        assert "bottom_turnover" in df.columns
        assert "mean_autocorrelation" in df.columns
        assert "half_life" in df.columns
        assert len(df) == 2  # 2 periods

    def test_get_dataframe_invalid_name(
        self, sample_turnover_result: TurnoverAnalysisResult
    ) -> None:
        """Test get_dataframe with invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown DataFrame name"):
            sample_turnover_result.get_dataframe("invalid_name")

    def test_get_dataframe_empty_turnover(self) -> None:
        """Test get_dataframe returns empty DataFrame when no data."""
        result = TurnoverAnalysisResult(
            quantile_turnover={},
            mean_turnover={},
            top_quantile_turnover={},
            bottom_quantile_turnover={},
            autocorrelation={},
            autocorrelation_lags=[],
            mean_autocorrelation={},
            half_life={},
        )

        df = result.get_dataframe("turnover")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


class TestTurnoverAnalysisResultListDataframes:
    """Tests for TurnoverAnalysisResult.list_available_dataframes method."""

    def test_list_available_dataframes(
        self, sample_turnover_result: TurnoverAnalysisResult
    ) -> None:
        """Test list_available_dataframes returns expected list."""
        available = sample_turnover_result.list_available_dataframes()

        assert isinstance(available, list)
        assert "turnover" in available
        assert "autocorrelation" in available
        assert "summary" in available


class TestTurnoverAnalysisResultSummary:
    """Tests for TurnoverAnalysisResult.summary method."""

    def test_summary(self, sample_turnover_result: TurnoverAnalysisResult) -> None:
        """Test summary returns formatted string."""
        summary = sample_turnover_result.summary()

        assert isinstance(summary, str)
        assert "Turnover Analysis Summary" in summary
        assert "Period: 1D" in summary
        assert "Period: 5D" in summary
        assert "Mean Turnover:" in summary
        assert "Top Quantile:" in summary
        assert "Bottom Quantile:" in summary
        assert "Mean Autocorrelation:" in summary
        assert "Signal Half-Life:" in summary

    def test_summary_with_none_half_life(self) -> None:
        """Test summary with None half_life."""
        result = TurnoverAnalysisResult(
            quantile_turnover={"1D": {"Q1": 0.15}},
            mean_turnover={"1D": 0.15},
            top_quantile_turnover={"1D": 0.15},
            bottom_quantile_turnover={"1D": 0.15},
            autocorrelation={"1D": [0.5, 0.3, 0.2]},
            autocorrelation_lags=[1, 2, 3],
            mean_autocorrelation={"1D": 0.33},
            half_life={"1D": None},
        )

        summary = result.summary()

        assert isinstance(summary, str)
        # Should not have half-life line when None
        assert "Signal Half-Life:" not in summary


# =============================================================================
# IRtcResult Tests
# =============================================================================


class TestIRtcResultGetDataframe:
    """Tests for IRtcResult.get_dataframe method."""

    def test_get_dataframe(self, sample_irtc_result: IRtcResult) -> None:
        """Test get_dataframe returns DataFrame with all columns."""
        df = sample_irtc_result.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "ir_gross" in df.columns
        assert "ir_tc" in df.columns
        assert "implied_cost" in df.columns
        assert "breakeven_cost" in df.columns
        assert "cost_drag" in df.columns
        assert len(df) == 2  # 2 periods


class TestIRtcResultListDataframes:
    """Tests for IRtcResult.list_available_dataframes method."""

    def test_list_available_dataframes(self, sample_irtc_result: IRtcResult) -> None:
        """Test list_available_dataframes returns expected list."""
        available = sample_irtc_result.list_available_dataframes()

        assert isinstance(available, list)
        assert "primary" in available


class TestIRtcResultSummary:
    """Tests for IRtcResult.summary method."""

    def test_summary(self, sample_irtc_result: IRtcResult) -> None:
        """Test summary returns formatted string."""
        summary = sample_irtc_result.summary()

        assert isinstance(summary, str)
        assert "Transaction-Cost Adjusted IR Summary" in summary
        assert "Cost per Trade:" in summary
        assert "0.001" in summary or "10 bps" in summary
        assert "IR_gross" in summary
        assert "IR_tc" in summary


# =============================================================================
# SignalTearSheet Tests
# =============================================================================


class TestSignalTearSheetGetDataframe:
    """Tests for SignalTearSheet.get_dataframe method."""

    def test_get_dataframe_default(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test get_dataframe with default (summary)."""
        df = sample_tear_sheet.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_get_dataframe_summary(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test get_dataframe with name='summary'."""
        df = sample_tear_sheet.get_dataframe("summary")

        assert isinstance(df, pl.DataFrame)
        # Check expected summary metrics
        metrics = df["metric"].to_list()
        assert "signal_name" in metrics
        assert "n_assets" in metrics
        assert "n_dates" in metrics

    def test_get_dataframe_ic_routing(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test get_dataframe routes to IC analysis."""
        df = sample_tear_sheet.get_dataframe("ic_summary")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns

    def test_get_dataframe_quantile_routing(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test get_dataframe routes to quantile analysis."""
        df = sample_tear_sheet.get_dataframe("quantile_spread")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "spread_mean" in df.columns

    def test_get_dataframe_turnover_routing(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test get_dataframe routes to turnover analysis."""
        df = sample_tear_sheet.get_dataframe("turnover_summary")

        assert isinstance(df, pl.DataFrame)
        assert "period" in df.columns
        assert "mean_turnover" in df.columns

    def test_get_dataframe_ic_not_available(self, minimal_tear_sheet: SignalTearSheet) -> None:
        """Test get_dataframe raises when IC not available."""
        with pytest.raises(ValueError, match="IC analysis not available"):
            minimal_tear_sheet.get_dataframe("ic_summary")

    def test_get_dataframe_quantile_not_available(
        self, minimal_tear_sheet: SignalTearSheet
    ) -> None:
        """Test get_dataframe raises when quantile not available."""
        with pytest.raises(ValueError, match="Quantile analysis not available"):
            minimal_tear_sheet.get_dataframe("quantile_spread")

    def test_get_dataframe_turnover_not_available(
        self, minimal_tear_sheet: SignalTearSheet
    ) -> None:
        """Test get_dataframe raises when turnover not available."""
        with pytest.raises(ValueError, match="Turnover analysis not available"):
            minimal_tear_sheet.get_dataframe("turnover_summary")

    def test_get_dataframe_invalid_name(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test get_dataframe raises for invalid name."""
        with pytest.raises(ValueError, match="Unknown DataFrame name"):
            sample_tear_sheet.get_dataframe("invalid_name")


class TestSignalTearSheetListDataframes:
    """Tests for SignalTearSheet.list_available_dataframes method."""

    def test_list_available_dataframes_full(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test list_available_dataframes with all components."""
        available = sample_tear_sheet.list_available_dataframes()

        assert isinstance(available, list)
        assert "summary" in available
        # Should include prefixed component dataframes
        assert any(n.startswith("ic_") for n in available)
        assert any(n.startswith("quantile_") for n in available)
        assert any(n.startswith("turnover_") for n in available)

    def test_list_available_dataframes_minimal(self, minimal_tear_sheet: SignalTearSheet) -> None:
        """Test list_available_dataframes with no components."""
        available = minimal_tear_sheet.list_available_dataframes()

        assert isinstance(available, list)
        assert "summary" in available
        # Should not include component dataframes
        assert not any(n.startswith("ic_") for n in available)


class TestSignalTearSheetSummary:
    """Tests for SignalTearSheet.summary method."""

    def test_summary_full(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test summary with all components."""
        summary = sample_tear_sheet.summary()

        assert isinstance(summary, str)
        assert "Signal Analysis Tear Sheet: test_signal" in summary
        assert "Assets:" in summary
        assert "Dates:" in summary
        assert "IC Analysis" in summary
        assert "Quantile Analysis" in summary
        assert "Turnover Analysis" in summary
        assert "IR_tc Analysis" in summary

    def test_summary_minimal(self, minimal_tear_sheet: SignalTearSheet) -> None:
        """Test summary with no components."""
        summary = minimal_tear_sheet.summary()

        assert isinstance(summary, str)
        assert "Signal Analysis Tear Sheet: minimal_signal" in summary
        # Should not have component sections
        assert "IC Analysis" not in summary


class TestSignalTearSheetShow:
    """Tests for SignalTearSheet.show method."""

    def test_show_without_ipython(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test show prints summary when IPython not available."""
        # Should not raise, just print summary
        sample_tear_sheet.show()


class TestSignalTearSheetSaveHtml:
    """Tests for SignalTearSheet.save_html method."""

    def test_save_html_simple_layout(
        self, sample_tear_sheet: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_html with simple stacked layout."""
        output_path = tmp_path / "test_report.html"
        saved_path = sample_tear_sheet.save_html(output_path, use_dashboard=False)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

        content = saved_path.read_text()
        assert "test_signal" in content
        assert "Signal Analysis" in content
        assert "<html>" in content

    def test_save_html_creates_parent_dirs(
        self, sample_tear_sheet: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_html creates parent directories."""
        output_path = tmp_path / "nested" / "dirs" / "report.html"
        saved_path = sample_tear_sheet.save_html(output_path, use_dashboard=False)

        assert saved_path.exists()
        assert saved_path.parent.exists()


class TestSignalTearSheetSaveJson:
    """Tests for SignalTearSheet.save_json method."""

    def test_save_json_full(self, sample_tear_sheet: SignalTearSheet, tmp_path: Path) -> None:
        """Test save_json with all data."""
        import json

        output_path = tmp_path / "test_metrics.json"
        saved_path = sample_tear_sheet.save_json(output_path)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

        with open(saved_path) as f:
            data = json.load(f)

        assert "signal_name" in data
        assert data["signal_name"] == "test_signal"
        assert "n_assets" in data
        assert "n_dates" in data

    def test_save_json_exclude_figures(
        self, sample_tear_sheet: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_json with exclude_figures=True."""
        import json

        output_path = tmp_path / "test_compact.json"
        saved_path = sample_tear_sheet.save_json(output_path, exclude_figures=True)

        assert saved_path.exists()

        with open(saved_path) as f:
            data = json.load(f)

        assert "signal_name" in data
        # Figures should be excluded
        assert "figures" not in data or not data["figures"]

    def test_save_json_creates_parent_dirs(
        self, sample_tear_sheet: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_json creates parent directories."""
        output_path = tmp_path / "nested" / "dirs" / "metrics.json"
        saved_path = sample_tear_sheet.save_json(output_path)

        assert saved_path.exists()


@pytest.mark.slow
class TestSignalTearSheetSavePng:
    """Tests for SignalTearSheet.save_png method."""

    def test_save_png_requires_kaleido(
        self, sample_tear_sheet: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_png raises ImportError when kaleido not available."""
        # This test checks the error handling path
        # In practice, kaleido may or may not be installed
        try:
            import plotly.io as pio

            pio.kaleido.scope  # Check if kaleido is available
            # If we get here, kaleido is installed
            # Test actual functionality
            saved_paths = sample_tear_sheet.save_png(tmp_path)
            assert isinstance(saved_paths, list)
        except (ImportError, AttributeError):
            # Kaleido not installed - test error path
            with pytest.raises(ImportError, match="kaleido is required"):
                sample_tear_sheet.save_png(tmp_path)

    def test_save_png_specific_figures(
        self, sample_tear_sheet: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_png with specific figures list."""
        try:
            import plotly.io as pio

            pio.kaleido.scope
            # Kaleido available
            saved_paths = sample_tear_sheet.save_png(tmp_path, figures=["ic_time_series"])
            assert isinstance(saved_paths, list)
            assert len(saved_paths) <= 1
        except (ImportError, AttributeError):
            pytest.skip("kaleido not installed")


class TestSignalTearSheetToDashboard:
    """Tests for SignalTearSheet.to_dashboard method."""

    def test_to_dashboard_light_theme(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test to_dashboard with light theme."""
        from ml4t.diagnostic.visualization.signal.dashboard import SignalDashboard

        dashboard = sample_tear_sheet.to_dashboard(theme="light")

        assert isinstance(dashboard, SignalDashboard)

    def test_to_dashboard_dark_theme(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test to_dashboard with dark theme."""
        from ml4t.diagnostic.visualization.signal.dashboard import SignalDashboard

        dashboard = sample_tear_sheet.to_dashboard(theme="dark")

        assert isinstance(dashboard, SignalDashboard)


class TestSignalTearSheetToDict:
    """Tests for SignalTearSheet.to_dict method."""

    def test_to_dict_full(self, sample_tear_sheet: SignalTearSheet) -> None:
        """Test to_dict returns complete dictionary."""
        data = sample_tear_sheet.to_dict()

        assert isinstance(data, dict)
        assert "signal_name" in data
        assert "n_assets" in data
        assert "n_dates" in data
        assert "date_range" in data

    def test_to_dict_exclude_none(self, minimal_tear_sheet: SignalTearSheet) -> None:
        """Test to_dict with exclude_none=True."""
        data = minimal_tear_sheet.to_dict(exclude_none=True)

        assert isinstance(data, dict)
        assert "signal_name" in data
        # None values should be excluded
        assert data.get("ic_analysis") is None or "ic_analysis" not in data


# =============================================================================
# _normalize_period Helper Tests
# =============================================================================


class TestNormalizePeriod:
    """Tests for _normalize_period helper function."""

    def test_normalize_int(self) -> None:
        """Test normalizing integer periods."""
        assert _normalize_period(1) == "1D"
        assert _normalize_period(5) == "5D"
        assert _normalize_period(21) == "21D"
        assert _normalize_period(252) == "252D"

    def test_normalize_string_without_suffix(self) -> None:
        """Test normalizing strings without D suffix."""
        assert _normalize_period("1") == "1D"
        assert _normalize_period("21") == "21D"

    def test_normalize_string_with_suffix(self) -> None:
        """Test normalizing strings with D suffix."""
        assert _normalize_period("1D") == "1D"
        assert _normalize_period("21D") == "21D"

    def test_normalize_strips_whitespace(self) -> None:
        """Test that whitespace is stripped."""
        assert _normalize_period(" 21 ") == "21D"
        assert _normalize_period(" 21D ") == "21D"
        assert _normalize_period("  5  ") == "5D"


class TestSignalValidationHelpers:
    """Tests for helper utilities in results.signal_results.validation."""

    def test_validate_dict_keys_match_required_passes(self) -> None:
        data = {
            "ic_mean": {"1D": 0.1, "5D": 0.2},
            "ic_std": {"1D": 0.01, "5D": 0.02},
        }
        _validate_dict_keys_match(data, required_fields=["ic_mean", "ic_std"])

    def test_validate_dict_keys_match_required_missing_key_raises(self) -> None:
        data = {
            "ic_mean": {"1D": 0.1, "5D": 0.2},
            "ic_std": {"1D": 0.01},
        }
        with pytest.raises(ValueError, match="Key mismatch in 'ic_std'"):
            _validate_dict_keys_match(data, required_fields=["ic_mean", "ic_std"])

    def test_validate_dict_keys_match_required_none_raises(self) -> None:
        data = {
            "ic_mean": {"1D": 0.1},
            "ic_std": None,
        }
        with pytest.raises(ValueError, match="Required field 'ic_std' is None"):
            _validate_dict_keys_match(data, required_fields=["ic_mean", "ic_std"])

    def test_validate_dict_keys_match_optional_mismatch_raises(self) -> None:
        data = {
            "ic_mean": {"1D": 0.1, "5D": 0.2},
            "ic_std": {"1D": 0.01, "5D": 0.02},
            "ic_p_value": {"1D": 0.03},
        }
        with pytest.raises(ValueError, match="Key mismatch in 'ic_p_value'"):
            _validate_dict_keys_match(
                data,
                required_fields=["ic_mean", "ic_std"],
                optional_fields=["ic_p_value"],
            )

    def test_validate_dict_keys_match_empty_reference_returns(self) -> None:
        data = {
            "ic_mean": {},
            "ic_std": {"1D": 0.01},
        }
        _validate_dict_keys_match(data, required_fields=["ic_mean", "ic_std"])

    def test_validate_dict_keys_match_no_required_fields_returns(self) -> None:
        _validate_dict_keys_match({"ic_mean": {"1D": 0.1}}, required_fields=[])

    def test_figure_from_data_dict(self) -> None:
        fig = _figure_from_data({"data": [{"type": "bar", "x": ["A"], "y": [1]}]})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_figure_from_data_json_string(self) -> None:
        source = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        fig = _figure_from_data(source.to_json())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_figure_from_data_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected dict or str"):
            _figure_from_data(123)  # type: ignore[arg-type]


# =============================================================================
# Additional Edge Case Tests for Complete Coverage
# =============================================================================


class TestSignalICResultEdgeCases:
    """Additional edge case tests for SignalICResult."""

    @pytest.fixture
    def ic_result_no_hac(self) -> SignalICResult:
        """Create SignalICResult without HAC stats."""
        return SignalICResult(
            ic_by_date={"1D": [0.05, 0.03]},
            dates=["2020-01-01", "2020-01-02"],
            ic_mean={"1D": 0.04},
            ic_std={"1D": 0.015},
            ic_t_stat={"1D": 2.67},
            ic_p_value={"1D": 0.02},
            ic_positive_pct={"1D": 1.0},
            ic_ir={"1D": 2.67},
        )

    def test_get_t_stat_returns_none_for_missing(self, ic_result_no_hac: SignalICResult) -> None:
        """Test get_t_stat returns None for nonexistent period."""
        assert ic_result_no_hac.get_t_stat(100) is None
        assert ic_result_no_hac.get_t_stat("100D") is None

    def test_get_p_value_returns_none_for_missing(self, ic_result_no_hac: SignalICResult) -> None:
        """Test get_p_value returns None for nonexistent period."""
        assert ic_result_no_hac.get_p_value(100) is None

    def test_get_ir_returns_none_for_missing(self, ic_result_no_hac: SignalICResult) -> None:
        """Test get_ir returns None for nonexistent period."""
        assert ic_result_no_hac.get_ir(100) is None

    def test_get_stats_with_no_hac(self, ic_result_no_hac: SignalICResult) -> None:
        """Test get_stats when HAC and RAS fields are None."""
        from ml4t.diagnostic.results.signal_results import ICStats

        stats = ic_result_no_hac.get_stats(1)
        assert stats is not None
        assert isinstance(stats, ICStats)
        assert stats.t_stat_hac is None
        assert stats.p_value_hac is None
        assert stats.ras_adjusted is None
        assert stats.ras_significant is None

    def test_is_significant_returns_false_for_missing(
        self, ic_result_no_hac: SignalICResult
    ) -> None:
        """Test is_significant returns False for nonexistent period."""
        assert ic_result_no_hac.is_significant(100) is False

    def test_is_significant_uses_regular_pvalue_when_no_hac(
        self, ic_result_no_hac: SignalICResult
    ) -> None:
        """Test is_significant falls back to regular p-value when HAC not available."""
        # With use_hac=True but no HAC stats, should use regular p-value
        assert ic_result_no_hac.is_significant(1, alpha=0.05, use_hac=True) is True


class TestQuantileAnalysisResultEdgeCases:
    """Additional edge case tests for QuantileAnalysisResult."""

    @pytest.fixture
    def quantile_result(self) -> QuantileAnalysisResult:
        """Create QuantileAnalysisResult for testing."""
        return QuantileAnalysisResult(
            n_quantiles=3,
            quantile_labels=["Q1", "Q2", "Q3"],
            periods=["1D"],
            mean_returns={"1D": {"Q1": -0.005, "Q2": 0.001, "Q3": 0.008}},
            std_returns={"1D": {"Q1": 0.02, "Q2": 0.02, "Q3": 0.02}},
            count_by_quantile={"Q1": 100, "Q2": 100, "Q3": 100},
            spread_mean={"1D": 0.013},
            spread_std={"1D": 0.005},
            spread_t_stat={"1D": 2.6},
            spread_p_value={"1D": 0.02},
            spread_ci_lower={"1D": 0.003},
            spread_ci_upper={"1D": 0.023},
            is_monotonic={"1D": True},
            monotonicity_direction={"1D": "increasing"},
            rank_correlation={"1D": 1.0},
        )

    def test_get_top_quantile_return_missing_period(
        self, quantile_result: QuantileAnalysisResult
    ) -> None:
        """Test get_top_quantile_return for missing period."""
        assert quantile_result.get_top_quantile_return(100) is None

    def test_get_bottom_quantile_return_missing_period(
        self, quantile_result: QuantileAnalysisResult
    ) -> None:
        """Test get_bottom_quantile_return for missing period."""
        assert quantile_result.get_bottom_quantile_return(100) is None

    def test_is_spread_significant_missing_period(
        self, quantile_result: QuantileAnalysisResult
    ) -> None:
        """Test is_spread_significant for missing period returns False."""
        assert quantile_result.is_spread_significant(100) is False


class TestSignalTearSheetDashboard:
    """Tests for SignalTearSheet dashboard-related methods."""

    @pytest.fixture
    def tear_sheet_with_figures(self) -> SignalTearSheet:
        """Create SignalTearSheet with valid Plotly figure JSON."""
        import json

        # Create minimal valid Plotly figure JSON
        fig_json = json.dumps(
            {
                "data": [{"type": "scatter", "x": [1, 2, 3], "y": [4, 5, 6]}],
                "layout": {"title": "Test Plot"},
            }
        )

        return SignalTearSheet(
            signal_name="test_signal",
            n_assets=30,
            n_dates=100,
            date_range=("2020-01-01", "2020-04-10"),
            figures={"test_plot": fig_json},
        )

    def test_save_html_with_dashboard(
        self, tear_sheet_with_figures: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_html with use_dashboard=True."""
        output_path = tmp_path / "dashboard_report.html"
        saved_path = tear_sheet_with_figures.save_html(output_path, use_dashboard=True)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

    def test_save_html_with_theme(
        self, tear_sheet_with_figures: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_html with dark theme."""
        output_path = tmp_path / "dark_report.html"
        saved_path = tear_sheet_with_figures.save_html(
            output_path, use_dashboard=True, theme="dark"
        )

        assert saved_path.exists()


@pytest.mark.slow
class TestSignalTearSheetPngExport:
    """Tests for SignalTearSheet PNG export edge cases."""

    @pytest.fixture
    def tear_sheet_with_plotly_fig(self) -> SignalTearSheet:
        """Create SignalTearSheet with valid Plotly figure."""
        import json

        fig_json = json.dumps(
            {
                "data": [{"type": "scatter", "x": [1, 2, 3], "y": [4, 5, 6]}],
                "layout": {"title": "Test Plot", "width": 800, "height": 400},
            }
        )

        return SignalTearSheet(
            signal_name="test_signal",
            n_assets=30,
            n_dates=100,
            date_range=("2020-01-01", "2020-04-10"),
            figures={"test_plot": fig_json, "another_plot": fig_json},
        )

    def test_save_png_nonexistent_figure(
        self, tear_sheet_with_plotly_fig: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_png skips nonexistent figures."""
        try:
            import plotly.io as pio

            pio.kaleido.scope

            # Request non-existent figure
            saved_paths = tear_sheet_with_plotly_fig.save_png(
                tmp_path, figures=["nonexistent_figure"]
            )
            # Should return empty list since figure doesn't exist
            assert isinstance(saved_paths, list)
            assert len(saved_paths) == 0
        except (ImportError, AttributeError):
            pytest.skip("kaleido not installed")

    def test_save_png_all_figures(
        self, tear_sheet_with_plotly_fig: SignalTearSheet, tmp_path: Path
    ) -> None:
        """Test save_png exports all figures when none specified."""
        try:
            import plotly.io as pio

            pio.kaleido.scope

            saved_paths = tear_sheet_with_plotly_fig.save_png(tmp_path)
            assert isinstance(saved_paths, list)
            assert len(saved_paths) == 2  # Both figures
            for path in saved_paths:
                assert path.exists()
        except (ImportError, AttributeError):
            pytest.skip("kaleido not installed")
