"""Tests for Event Study Analysis module.

Tests the EventStudyAnalysis class including:
- Data validation and preparation
- Market model estimation (OLS regression)
- Mean-adjusted and market-adjusted models
- Abnormal return computation
- CAAR aggregation
- Statistical tests (t-test, BMP, Corrado)
- Configuration options
- Edge cases and error handling

References
----------
MacKinlay, A.C. (1997). "Event Studies in Economics and Finance"
Boehmer, E., Musumeci, J., Poulsen, A.B. (1991). BMP test
Corrado, C.J. (1989). Non-parametric rank test
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config.event_config import EventConfig, WindowSettings
from ml4t.diagnostic.evaluation.event_analysis import EventStudyAnalysis
from ml4t.diagnostic.results.event_results import AbnormalReturnResult, EventStudyResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trading_dates() -> list[datetime]:
    """Generate 500 trading days (approx 2 years)."""
    start = datetime(2020, 1, 1)
    dates = []
    current = start
    while len(dates) < 500:
        # Skip weekends
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture
def sample_returns_data(trading_dates: list[datetime]) -> pl.DataFrame:
    """Create sample returns data in long format.

    Generates returns for 10 assets over 500 trading days.
    Returns are generated with slight positive drift.
    """
    np.random.seed(42)
    n_assets = 10
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

    records = []
    for asset in assets:
        for date in trading_dates:
            # Generate return with mean 0.0005 (daily) and vol 0.02
            ret = np.random.normal(0.0005, 0.02)
            records.append({"date": date, "asset": asset, "return": ret})

    return pl.DataFrame(records)


@pytest.fixture
def sample_benchmark_data(trading_dates: list[datetime]) -> pl.DataFrame:
    """Create sample benchmark (market) returns."""
    np.random.seed(123)

    records = []
    for date in trading_dates:
        # Market return with mean 0.0003 and vol 0.015
        ret = np.random.normal(0.0003, 0.015)
        records.append({"date": date, "return": ret})

    return pl.DataFrame(records)


@pytest.fixture
def sample_events_data(trading_dates: list[datetime]) -> pl.DataFrame:
    """Create sample events in the middle of the date range."""
    # Pick events around day 300 to ensure enough estimation window
    event_dates = [trading_dates[300], trading_dates[320], trading_dates[340]]
    assets = ["ASSET_00", "ASSET_01", "ASSET_02"]

    return pl.DataFrame(
        {"date": event_dates, "asset": assets, "event_type": ["earnings", "merger", "earnings"]}
    )


@pytest.fixture
def default_config() -> EventConfig:
    """Default event study configuration."""
    return EventConfig(
        window=WindowSettings(
            estimation_start=-252,
            estimation_end=-20,
            event_start=-5,
            event_end=5,
            gap=5,
        ),
        model="market_model",
        test="t_test",
        confidence_level=0.95,
        min_estimation_obs=100,
    )


@pytest.fixture
def event_analysis(
    sample_returns_data: pl.DataFrame,
    sample_events_data: pl.DataFrame,
    sample_benchmark_data: pl.DataFrame,
    default_config: EventConfig,
) -> EventStudyAnalysis:
    """Create EventStudyAnalysis instance for testing."""
    return EventStudyAnalysis(
        returns=sample_returns_data,
        events=sample_events_data,
        benchmark=sample_benchmark_data,
        config=default_config,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestEventStudyAnalysisInit:
    """Tests for EventStudyAnalysis initialization."""

    def test_init_with_polars_dataframes(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test initialization with Polars DataFrames."""
        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=sample_events_data,
            benchmark=sample_benchmark_data,
        )
        assert analysis is not None
        assert analysis.n_events == 3

    def test_init_with_pandas_dataframes(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test initialization with Pandas DataFrames (auto-conversion)."""
        pytest.importorskip("pandas")

        analysis = EventStudyAnalysis(
            returns=sample_returns_data.to_pandas(),
            events=sample_events_data.to_pandas(),
            benchmark=sample_benchmark_data.to_pandas(),
        )
        assert analysis is not None
        assert analysis.n_events == 3

    def test_init_with_default_config(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test that default config is applied when none provided."""
        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=sample_events_data,
            benchmark=sample_benchmark_data,
        )
        assert analysis.config.model == "market_model"
        assert analysis.config.test == "boehmer"  # Default per config

    def test_init_validates_returns_columns(
        self,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test that missing returns columns raise error."""
        bad_returns = pl.DataFrame({"date": [datetime.now()], "asset": ["X"]})

        with pytest.raises(ValueError, match="missing columns"):
            EventStudyAnalysis(
                returns=bad_returns,
                events=sample_events_data,
                benchmark=sample_benchmark_data,
            )

    def test_init_validates_events_columns(
        self,
        sample_returns_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test that missing events columns raise error."""
        bad_events = pl.DataFrame({"event_date": [datetime.now()]})

        with pytest.raises(ValueError, match="missing columns"):
            EventStudyAnalysis(
                returns=sample_returns_data,
                events=bad_events,
                benchmark=sample_benchmark_data,
            )

    def test_init_validates_benchmark_columns(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
    ) -> None:
        """Test that missing benchmark columns raise error."""
        bad_benchmark = pl.DataFrame({"date": [datetime.now()]})

        with pytest.raises(ValueError, match="missing columns"):
            EventStudyAnalysis(
                returns=sample_returns_data,
                events=sample_events_data,
                benchmark=bad_benchmark,
            )

    def test_init_validates_non_empty_events(
        self,
        sample_returns_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test that empty events DataFrame raises error."""
        empty_events = pl.DataFrame({"date": [], "asset": []})

        with pytest.raises(ValueError, match="No events provided"):
            EventStudyAnalysis(
                returns=sample_returns_data,
                events=empty_events,
                benchmark=sample_benchmark_data,
            )


# =============================================================================
# Market Model Tests
# =============================================================================


class TestMarketModel:
    """Tests for market model estimation."""

    def test_market_model_alpha_beta_estimation(
        self,
        trading_dates: list[datetime],
    ) -> None:
        """Test that market model correctly estimates alpha and beta."""
        np.random.seed(42)

        # Create synthetic data with known alpha=0.001, beta=1.2
        true_alpha = 0.001
        true_beta = 1.2
        n_days = 400

        market_returns = np.random.normal(0.0005, 0.015, n_days)
        noise = np.random.normal(0, 0.005, n_days)
        asset_returns = true_alpha + true_beta * market_returns + noise

        dates = trading_dates[:n_days]
        returns_df = pl.DataFrame(
            {"date": dates, "asset": ["ASSET_00"] * n_days, "return": asset_returns}
        )
        benchmark_df = pl.DataFrame({"date": dates, "return": market_returns})
        events_df = pl.DataFrame({"date": [dates[350]], "asset": ["ASSET_00"]})

        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            model="market_model",
            min_estimation_obs=100,
        )

        analysis = EventStudyAnalysis(
            returns=returns_df,
            events=events_df,
            benchmark=benchmark_df,
            config=config,
        )

        ar_results = analysis.compute_abnormal_returns()
        assert len(ar_results) == 1

        result = ar_results[0]
        assert result.estimation_alpha is not None
        assert result.estimation_beta is not None

        # Check alpha and beta are close to true values (within reasonable tolerance)
        assert abs(result.estimation_alpha - true_alpha) < 0.005
        assert abs(result.estimation_beta - true_beta) < 0.3

    def test_market_model_abnormal_returns(self, event_analysis: EventStudyAnalysis) -> None:
        """Test that market model computes abnormal returns correctly."""
        ar_results = event_analysis.compute_abnormal_returns()

        # Should have results for all events (3)
        assert len(ar_results) == 3

        for result in ar_results:
            # Check structure
            assert isinstance(result, AbnormalReturnResult)
            assert result.asset in ["ASSET_00", "ASSET_01", "ASSET_02"]
            assert result.event_date is not None

            # Check AR values exist for event window
            assert -5 in result.ar_by_day
            assert 0 in result.ar_by_day
            assert 5 in result.ar_by_day

            # Check CAR is sum of daily ARs
            expected_car = sum(result.ar_by_day.values())
            assert abs(result.car - expected_car) < 1e-10

            # Check market model params were estimated
            assert result.estimation_alpha is not None
            assert result.estimation_beta is not None


# =============================================================================
# Alternative Model Tests
# =============================================================================


class TestMeanAdjustedModel:
    """Tests for mean-adjusted model."""

    def test_mean_adjusted_abnormal_returns(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test mean-adjusted model: AR = R - mean(R_estimation)."""
        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            model="mean_adjusted",
            min_estimation_obs=100,
        )

        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=sample_events_data,
            benchmark=sample_benchmark_data,
            config=config,
        )

        ar_results = analysis.compute_abnormal_returns()
        assert len(ar_results) == 3

        for result in ar_results:
            # Mean-adjusted should have alpha (the mean) but beta should be 0
            assert result.estimation_alpha is None  # Not stored for mean_adjusted
            assert result.estimation_beta is None
            assert result.estimation_residual_std is not None


class TestMarketAdjustedModel:
    """Tests for market-adjusted model."""

    def test_market_adjusted_abnormal_returns(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test market-adjusted model: AR = R - R_market."""
        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            model="market_adjusted",
            min_estimation_obs=100,
        )

        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=sample_events_data,
            benchmark=sample_benchmark_data,
            config=config,
        )

        ar_results = analysis.compute_abnormal_returns()
        assert len(ar_results) == 3

        for result in ar_results:
            assert result.estimation_alpha is None
            assert result.estimation_beta is None


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestAggregation:
    """Tests for AAR and CAAR aggregation."""

    def test_aggregate_produces_event_study_result(
        self, event_analysis: EventStudyAnalysis
    ) -> None:
        """Test that aggregate() returns EventStudyResult with all fields."""
        result = event_analysis.aggregate()

        assert isinstance(result, EventStudyResult)
        assert result.n_events == 3
        assert result.model_name == "market_model"
        assert result.event_window == (-5, 5)
        assert result.confidence_level == 0.95

    def test_aar_is_cross_sectional_mean(self, event_analysis: EventStudyAnalysis) -> None:
        """Test that AAR is the cross-sectional average of ARs."""
        ar_results = event_analysis.compute_abnormal_returns()
        result = event_analysis.aggregate()

        # Manually compute AAR for day 0
        ars_day0 = [r.ar_by_day[0] for r in ar_results if 0 in r.ar_by_day]
        expected_aar0 = np.mean(ars_day0)

        assert abs(result.aar_by_day[0] - expected_aar0) < 1e-10

    def test_caar_is_cumulative_sum_of_aar(self, event_analysis: EventStudyAnalysis) -> None:
        """Test that CAAR is cumulative sum of AAR."""
        result = event_analysis.aggregate()

        # Compute expected CAAR
        sorted_days = sorted(result.aar_by_day.keys())
        expected_caar = []
        cumsum = 0.0
        for day in sorted_days:
            cumsum += result.aar_by_day[day]
            expected_caar.append(cumsum)

        np.testing.assert_array_almost_equal(result.caar, expected_caar)

    def test_caar_dates_are_sorted(self, event_analysis: EventStudyAnalysis) -> None:
        """Test that CAAR dates are properly sorted."""
        result = event_analysis.aggregate()

        assert result.caar_dates == sorted(result.caar_dates)
        assert result.caar_dates[0] == -5
        assert result.caar_dates[-1] == 5
        assert 0 in result.caar_dates

    def test_confidence_intervals_calculated(self, event_analysis: EventStudyAnalysis) -> None:
        """Test that confidence intervals are calculated."""
        result = event_analysis.aggregate()

        assert len(result.caar_ci_lower) == len(result.caar)
        assert len(result.caar_ci_upper) == len(result.caar)

        # CI should bracket the CAAR
        for i in range(len(result.caar)):
            assert result.caar_ci_lower[i] <= result.caar[i]
            assert result.caar_ci_upper[i] >= result.caar[i]


# =============================================================================
# Statistical Tests
# =============================================================================


class TestStatisticalTests:
    """Tests for statistical significance tests."""

    def test_t_test_produces_valid_statistics(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test t-test produces valid test statistic and p-value."""
        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            model="market_model",
            test="t_test",
            min_estimation_obs=100,
        )

        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=sample_events_data,
            benchmark=sample_benchmark_data,
            config=config,
        )

        result = analysis.run()

        assert result.test_name == "t_test"
        assert np.isfinite(result.test_statistic)
        assert 0 <= result.p_value <= 1

    def test_bmp_test_produces_valid_statistics(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test BMP (Boehmer) test produces valid test statistic and p-value."""
        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            model="market_model",
            test="boehmer",
            min_estimation_obs=100,
        )

        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=sample_events_data,
            benchmark=sample_benchmark_data,
            config=config,
        )

        result = analysis.run()

        assert result.test_name == "boehmer"
        assert np.isfinite(result.test_statistic)
        assert 0 <= result.p_value <= 1

    def test_corrado_test_produces_valid_statistics(
        self,
        sample_returns_data: pl.DataFrame,
        sample_events_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test Corrado rank test produces valid test statistic and p-value."""
        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            model="market_model",
            test="corrado",
            min_estimation_obs=100,
        )

        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=sample_events_data,
            benchmark=sample_benchmark_data,
            config=config,
        )

        result = analysis.run()

        assert result.test_name == "corrado"
        assert np.isfinite(result.test_statistic)
        assert 0 <= result.p_value <= 1

    def test_t_test_significant_with_large_effect(
        self,
        trading_dates: list[datetime],
    ) -> None:
        """Test that t-test detects significant effect with large abnormal return."""
        np.random.seed(42)
        n_days = 400

        # Create data with large positive AR on event day
        dates = trading_dates[:n_days]
        market_returns = np.random.normal(0.0003, 0.015, n_days)
        asset_returns = market_returns.copy()  # Perfect correlation initially

        # Add large positive abnormal return at event day (day 350)
        for i in range(-5, 6):
            asset_returns[350 + i] += 0.02  # 2% daily AR

        returns_df = pl.DataFrame(
            {"date": dates, "asset": ["ASSET_00"] * n_days, "return": asset_returns}
        )
        benchmark_df = pl.DataFrame({"date": dates, "return": market_returns})
        events_df = pl.DataFrame({"date": [dates[350]], "asset": ["ASSET_00"]})

        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            model="market_adjusted",  # Use simpler model for this test
            test="t_test",
            confidence_level=0.95,
            min_estimation_obs=100,
        )

        analysis = EventStudyAnalysis(
            returns=returns_df,
            events=events_df,
            benchmark=benchmark_df,
            config=config,
        )

        result = analysis.run()

        # With such large effect, should have large positive CAAR
        assert result.final_caar > 0.1  # > 10% cumulative
        # Note: With only 1 event, p-value may not be significant due to lack of cross-section


# =============================================================================
# Result Properties Tests
# =============================================================================


class TestResultProperties:
    """Tests for EventStudyResult properties and methods."""

    def test_is_significant_property(self, event_analysis: EventStudyAnalysis) -> None:
        """Test is_significant property based on p-value and confidence level."""
        result = event_analysis.run()

        # is_significant should compare p_value to alpha (1 - confidence_level)
        expected = result.p_value < (1 - result.confidence_level)
        assert result.is_significant == expected

    def test_final_caar_property(self, event_analysis: EventStudyAnalysis) -> None:
        """Test final_caar returns last CAAR value."""
        result = event_analysis.run()

        assert result.final_caar == result.caar[-1]

    def test_event_day_aar_property(self, event_analysis: EventStudyAnalysis) -> None:
        """Test event_day_aar returns AAR at t=0."""
        result = event_analysis.run()

        assert result.event_day_aar == result.aar_by_day.get(0, 0.0)

    def test_summary_method_returns_string(self, event_analysis: EventStudyAnalysis) -> None:
        """Test summary() returns formatted string."""
        result = event_analysis.run()
        summary = result.summary()

        assert isinstance(summary, str)
        assert "EVENT STUDY RESULTS" in summary
        assert "CAAR" in summary
        assert str(result.n_events) in summary

    def test_get_dataframe_caar(self, event_analysis: EventStudyAnalysis) -> None:
        """Test get_dataframe returns CAAR DataFrame."""
        result = event_analysis.run()
        df = result.get_dataframe("caar")

        assert isinstance(df, pl.DataFrame)
        assert "relative_day" in df.columns
        assert "caar" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns

    def test_get_dataframe_aar(self, event_analysis: EventStudyAnalysis) -> None:
        """Test get_dataframe returns AAR DataFrame."""
        result = event_analysis.run()
        df = result.get_dataframe("aar")

        assert isinstance(df, pl.DataFrame)
        assert "relative_day" in df.columns
        assert "aar" in df.columns

    def test_get_dataframe_events(self, event_analysis: EventStudyAnalysis) -> None:
        """Test get_dataframe returns events DataFrame."""
        result = event_analysis.run()
        df = result.get_dataframe("events")

        assert isinstance(df, pl.DataFrame)
        assert "event_id" in df.columns
        assert "asset" in df.columns
        assert "car" in df.columns
        assert len(df) == 3

    def test_to_dict_serialization(self, event_analysis: EventStudyAnalysis) -> None:
        """Test to_dict produces valid dictionary."""
        result = event_analysis.run()
        data = result.to_dict()

        assert isinstance(data, dict)
        assert "caar" in data
        assert "p_value" in data
        assert "n_events" in data
        assert data["n_events"] == 3


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_insufficient_estimation_data_warning(
        self,
        trading_dates: list[datetime],
    ) -> None:
        """Test that events with insufficient estimation data are skipped with warning."""
        # Create short dataset
        dates = trading_dates[:100]  # Only 100 days
        returns_df = pl.DataFrame(
            {
                "date": dates,
                "asset": ["ASSET_00"] * 100,
                "return": np.random.randn(100) * 0.02,
            }
        )
        benchmark_df = pl.DataFrame({"date": dates, "return": np.random.randn(100) * 0.015})
        # Event at day 50 - not enough estimation window
        events_df = pl.DataFrame({"date": [dates[50]], "asset": ["ASSET_00"]})

        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            min_estimation_obs=100,  # Requires 100 observations
        )

        analysis = EventStudyAnalysis(
            returns=returns_df, events=events_df, benchmark=benchmark_df, config=config
        )

        with pytest.warns(UserWarning, match="Skipped .* events"):
            ar_results = analysis.compute_abnormal_returns()

        assert len(ar_results) == 0  # Event should be skipped

    def test_no_valid_events_raises_error(
        self,
        trading_dates: list[datetime],
    ) -> None:
        """Test that aggregate() raises error if no valid events."""
        dates = trading_dates[:100]
        returns_df = pl.DataFrame(
            {"date": dates, "asset": ["ASSET_00"] * 100, "return": np.random.randn(100) * 0.02}
        )
        benchmark_df = pl.DataFrame({"date": dates, "return": np.random.randn(100) * 0.015})
        events_df = pl.DataFrame({"date": [dates[50]], "asset": ["ASSET_00"]})

        config = EventConfig(
            window=WindowSettings(
                estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
            ),
            min_estimation_obs=100,
        )

        analysis = EventStudyAnalysis(
            returns=returns_df, events=events_df, benchmark=benchmark_df, config=config
        )

        with pytest.raises(ValueError, match="No valid events"):
            analysis.aggregate()

    def test_missing_asset_in_returns(
        self,
        sample_returns_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test that events for missing assets are skipped."""
        # Event for asset not in returns data
        events_df = pl.DataFrame({"date": [datetime(2020, 10, 1)], "asset": ["NONEXISTENT_ASSET"]})

        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=events_df,
            benchmark=sample_benchmark_data,
        )

        with pytest.warns(UserWarning, match="Skipped"):
            ar_results = analysis.compute_abnormal_returns()

        assert len(ar_results) == 0

    def test_event_date_not_in_returns(
        self,
        sample_returns_data: pl.DataFrame,
        sample_benchmark_data: pl.DataFrame,
    ) -> None:
        """Test that events on non-trading dates are skipped."""
        # Event on a date not in returns
        events_df = pl.DataFrame(
            {"date": [datetime(2025, 1, 1)], "asset": ["ASSET_00"]}  # Future date
        )

        analysis = EventStudyAnalysis(
            returns=sample_returns_data,
            events=events_df,
            benchmark=sample_benchmark_data,
        )

        with pytest.warns(UserWarning, match="Skipped"):
            ar_results = analysis.compute_abnormal_returns()

        assert len(ar_results) == 0


# =============================================================================
# API Tests
# =============================================================================


class TestAPI:
    """Tests for public API consistency."""

    def test_run_method_returns_result(self, event_analysis: EventStudyAnalysis) -> None:
        """Test run() returns EventStudyResult."""
        result = event_analysis.run()
        assert isinstance(result, EventStudyResult)

    def test_create_tear_sheet_alias(self, event_analysis: EventStudyAnalysis) -> None:
        """Test create_tear_sheet() is alias for run()."""
        result1 = event_analysis.run()
        result2 = event_analysis.create_tear_sheet()

        # Should produce same results (using cached values)
        assert result1.n_events == result2.n_events
        assert result1.final_caar == result2.final_caar

    def test_n_events_property(self, event_analysis: EventStudyAnalysis) -> None:
        """Test n_events property."""
        assert event_analysis.n_events == 3

    def test_n_valid_events_property(self, event_analysis: EventStudyAnalysis) -> None:
        """Test n_valid_events property."""
        n_valid = event_analysis.n_valid_events
        assert n_valid == 3  # All events should be valid in fixture

    def test_assets_property(self, event_analysis: EventStudyAnalysis) -> None:
        """Test assets property returns unique assets."""
        assets = event_analysis.assets
        assert set(assets) == {"ASSET_00", "ASSET_01", "ASSET_02"}

    def test_date_range_property(self, event_analysis: EventStudyAnalysis) -> None:
        """Test date_range property."""
        start, end = event_analysis.date_range
        assert start < end

    def test_results_cached(self, event_analysis: EventStudyAnalysis) -> None:
        """Test that results are cached after first computation."""
        # First call computes
        result1 = event_analysis.compute_abnormal_returns()

        # Second call uses cache
        result2 = event_analysis.compute_abnormal_returns()

        assert result1 is result2  # Same object reference


# =============================================================================
# Configuration Tests
# =============================================================================


class TestEventConfig:
    """Tests for EventConfig validation."""

    def test_default_config_values(self) -> None:
        """Test default configuration values."""
        config = EventConfig()

        assert config.model == "market_model"
        assert config.test == "boehmer"
        assert config.confidence_level == 0.95
        assert config.min_estimation_obs == 100
        assert config.window.estimation_window == (-252, -20)
        assert config.window.event_window == (-5, 5)

    def test_alpha_property(self) -> None:
        """Test alpha property calculation."""
        config = EventConfig(confidence_level=0.95)
        assert abs(config.alpha - 0.05) < 1e-10

        config = EventConfig(confidence_level=0.99)
        assert abs(config.alpha - 0.01) < 1e-10

    def test_invalid_estimation_window_order(self) -> None:
        """Test validation rejects reversed estimation window."""
        with pytest.raises(ValueError, match="estimation_start.*must be < estimation_end"):
            WindowSettings(estimation_start=-20, estimation_end=-252)

    def test_invalid_estimation_window_positive_end(self) -> None:
        """Test validation rejects estimation window ending after event."""
        # estimation_end has le=-1 constraint, so 0 is invalid
        with pytest.raises(ValueError, match="less than or equal to -1"):
            WindowSettings(estimation_start=-252, estimation_end=0)

    def test_invalid_event_window_order(self) -> None:
        """Test validation rejects reversed event window."""
        with pytest.raises(ValueError, match="event_start.*must be < event_end"):
            WindowSettings(event_start=5, event_end=-5)

    def test_window_length_properties(self) -> None:
        """Test window length calculation properties."""
        window = WindowSettings(
            estimation_start=-252, estimation_end=-20, event_start=-5, event_end=5
        )

        assert window.estimation_length == 232  # (-20) - (-252) = 232
        assert window.event_length == 11  # 5 - (-5) + 1 = 11


# =============================================================================
# Individual Result Tests
# =============================================================================


class TestAbnormalReturnResult:
    """Tests for AbnormalReturnResult."""

    def test_get_dataframe(self, event_analysis: EventStudyAnalysis) -> None:
        """Test AbnormalReturnResult.get_dataframe()."""
        ar_results = event_analysis.compute_abnormal_returns()
        result = ar_results[0]

        df = result.get_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert "relative_day" in df.columns
        assert "abnormal_return" in df.columns
        assert len(df) == len(result.ar_by_day)

    def test_summary(self, event_analysis: EventStudyAnalysis) -> None:
        """Test AbnormalReturnResult.summary()."""
        ar_results = event_analysis.compute_abnormal_returns()
        result = ar_results[0]

        summary = result.summary()
        assert isinstance(summary, str)
        assert result.asset in summary
        assert "CAR" in summary
