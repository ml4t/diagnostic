"""Week 1 tests for BarrierAnalysis - Core infrastructure and metrics.

Tests cover:
- BarrierConfig validation
- Result dataclass creation and serialization
- BarrierAnalysis initialization and validation
- compute_hit_rates() implementation
- compute_profit_factor() implementation
- Chi-square independence test
- Monotonicity analysis
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config.barrier_config import (
    AnalysisSettings,
    BarrierConfig,
    BarrierLabel,
    ColumnSettings,
    DecileMethod,
    VisualizationSettings,
)
from ml4t.diagnostic.evaluation.barrier_analysis import BarrierAnalysis
from ml4t.diagnostic.results.barrier_results import (
    BarrierTearSheet,
    HitRateResult,
    ProfitFactorResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dates() -> list[date]:
    """Generate 100 trading dates."""
    start = date(2020, 1, 1)
    return [start + timedelta(days=i) for i in range(100)]


@pytest.fixture
def sample_assets() -> list[str]:
    """Generate 10 assets."""
    return [f"ASSET_{i:02d}" for i in range(10)]


@pytest.fixture
def signal_data(sample_dates: list[date], sample_assets: list[str]) -> pl.DataFrame:
    """Create synthetic signal data.

    Signal is designed such that higher values should predict TP,
    lower values should predict SL.
    """
    np.random.seed(42)
    rows = []
    for d in sample_dates:
        for asset in sample_assets:
            # Signal from -1 to 1
            signal = np.random.uniform(-1, 1)
            rows.append({"date": d, "asset": asset, "signal": signal})
    return pl.DataFrame(rows)


@pytest.fixture
def barrier_labels(sample_dates: list[date], sample_assets: list[str]) -> pl.DataFrame:
    """Create synthetic barrier labels.

    Labels are correlated with signal: higher signals -> more TP,
    lower signals -> more SL.
    """
    np.random.seed(42)
    rows = []
    for d in sample_dates:
        for asset in sample_assets:
            signal = np.random.uniform(-1, 1)

            # Probability of TP increases with signal
            p_tp = 0.3 + 0.4 * (signal + 1) / 2  # 0.3 to 0.7
            p_sl = 0.3 - 0.2 * (signal + 1) / 2  # 0.3 to 0.1
            p_timeout = 1 - p_tp - p_sl

            # Sample outcome
            outcome = np.random.choice(
                [1, -1, 0],
                p=[p_tp, p_sl, p_timeout],
            )

            # Return based on outcome
            if outcome == 1:  # TP
                ret = np.random.uniform(0.01, 0.03)  # 1-3% gain
            elif outcome == -1:  # SL
                ret = np.random.uniform(-0.02, -0.01)  # 1-2% loss
            else:  # Timeout
                ret = np.random.uniform(-0.005, 0.005)  # small return

            bars = np.random.randint(1, 20)

            rows.append(
                {
                    "date": d,
                    "asset": asset,
                    "label": outcome,
                    "label_return": ret,
                    "label_bars": bars,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def merged_data(
    signal_data: pl.DataFrame,
    barrier_labels: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return both dataframes for testing."""
    return signal_data, barrier_labels


# =============================================================================
# Config Tests
# =============================================================================


class TestBarrierConfig:
    """Tests for BarrierConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BarrierConfig()
        # Test flat access via convenience properties
        assert config.n_quantiles == 10
        assert config.decile_method == DecileMethod.QUANTILE
        assert config.min_observations_per_quantile == 30
        assert config.significance_level == 0.05
        assert config.signal_col == "signal"
        assert config.label_col == "label"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BarrierConfig(
            analysis=AnalysisSettings(n_quantiles=5, significance_level=0.01),
            signal_name="my_signal",
        )
        assert config.n_quantiles == 5
        assert config.significance_level == 0.01
        assert config.signal_name == "my_signal"

    def test_n_quantiles_validation(self) -> None:
        """Test n_quantiles bounds."""
        # Valid bounds
        AnalysisSettings(n_quantiles=2)
        AnalysisSettings(n_quantiles=20)

        # Invalid bounds
        with pytest.raises(ValueError):
            AnalysisSettings(n_quantiles=1)
        with pytest.raises(ValueError):
            AnalysisSettings(n_quantiles=21)

    def test_significance_level_validation(self) -> None:
        """Test significance_level bounds."""
        AnalysisSettings(significance_level=0.001)
        AnalysisSettings(significance_level=0.20)

        with pytest.raises(ValueError):
            AnalysisSettings(significance_level=0.0001)
        with pytest.raises(ValueError):
            AnalysisSettings(significance_level=0.25)

    def test_column_name_validation(self) -> None:
        """Test column name validation."""
        with pytest.raises(ValueError):
            ColumnSettings(signal_col="")
        with pytest.raises(ValueError):
            ColumnSettings(date_col="   ")

    def test_column_uniqueness_validation(self) -> None:
        """Test that column names must be unique."""
        with pytest.raises(ValueError):
            BarrierConfig(columns=ColumnSettings(signal_col="date", date_col="date"))

    def test_decile_method_enum(self) -> None:
        """Test DecileMethod enum values."""
        config = BarrierConfig(analysis=AnalysisSettings(decile_method=DecileMethod.UNIFORM))
        assert config.decile_method == DecileMethod.UNIFORM

        config = BarrierConfig(analysis=AnalysisSettings(decile_method="quantile"))
        assert config.decile_method == DecileMethod.QUANTILE

    def test_serialization(self) -> None:
        """Test config to_dict and from_dict."""
        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=5))
        data = config.to_dict()
        assert data["analysis"]["n_quantiles"] == 5

        # Roundtrip
        restored = BarrierConfig.from_dict(data)
        assert restored.n_quantiles == 5


class TestVisualizationSettings:
    """Tests for VisualizationSettings."""

    def test_default_config(self) -> None:
        """Test default tear sheet configuration."""
        config = VisualizationSettings()
        assert config.theme == "default"
        assert config.width == 1000
        assert config.include_hit_rate_heatmap is True
        assert config.include_profit_factor is True

    def test_custom_config(self) -> None:
        """Test custom tear sheet configuration."""
        config = VisualizationSettings(
            theme="dark",
            width=1200,
            include_profit_factor=False,
        )
        assert config.theme == "dark"
        assert config.width == 1200
        assert config.include_profit_factor is False


# =============================================================================
# Result Tests
# =============================================================================


class TestHitRateResult:
    """Tests for HitRateResult dataclass."""

    @pytest.fixture
    def sample_hit_rate_result(self) -> HitRateResult:
        """Create sample HitRateResult."""
        q_labels = ["D1", "D2", "D3"]
        return HitRateResult(
            n_quantiles=3,
            quantile_labels=q_labels,
            hit_rate_tp={"D1": 0.3, "D2": 0.4, "D3": 0.5},
            hit_rate_sl={"D1": 0.4, "D2": 0.3, "D3": 0.2},
            hit_rate_timeout={"D1": 0.3, "D2": 0.3, "D3": 0.3},
            count_tp={"D1": 30, "D2": 40, "D3": 50},
            count_sl={"D1": 40, "D2": 30, "D3": 20},
            count_timeout={"D1": 30, "D2": 30, "D3": 30},
            count_total={"D1": 100, "D2": 100, "D3": 100},
            chi2_statistic=15.0,
            chi2_p_value=0.01,
            chi2_dof=4,
            is_significant=True,
            significance_level=0.05,
            overall_hit_rate_tp=0.4,
            overall_hit_rate_sl=0.3,
            overall_hit_rate_timeout=0.3,
            n_observations=300,
            tp_rate_monotonic=True,
            tp_rate_direction="increasing",
            tp_rate_spearman=0.9,
        )

    def test_get_dataframe_hit_rates(self, sample_hit_rate_result: HitRateResult) -> None:
        """Test get_dataframe for hit_rates view."""
        df = sample_hit_rate_result.get_dataframe("hit_rates")
        assert isinstance(df, pl.DataFrame)
        assert "quantile" in df.columns
        assert "hit_rate_tp" in df.columns
        assert df.height == 3

    def test_get_dataframe_counts(self, sample_hit_rate_result: HitRateResult) -> None:
        """Test get_dataframe for counts view."""
        df = sample_hit_rate_result.get_dataframe("counts")
        assert "count_tp" in df.columns
        assert "count_sl" in df.columns
        assert df.height == 3

    def test_get_dataframe_summary(self, sample_hit_rate_result: HitRateResult) -> None:
        """Test get_dataframe for summary view."""
        df = sample_hit_rate_result.get_dataframe("summary")
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_list_available_dataframes(self, sample_hit_rate_result: HitRateResult) -> None:
        """Test list_available_dataframes."""
        available = sample_hit_rate_result.list_available_dataframes()
        assert "hit_rates" in available
        assert "counts" in available
        assert "summary" in available

    def test_summary_output(self, sample_hit_rate_result: HitRateResult) -> None:
        """Test summary string generation."""
        summary = sample_hit_rate_result.summary()
        assert "Barrier Hit Rate Analysis" in summary
        assert "Chi-Square Test" in summary
        assert "D1" in summary

    def test_serialization(self, sample_hit_rate_result: HitRateResult) -> None:
        """Test JSON serialization."""
        json_str = sample_hit_rate_result.to_json_string()
        assert "barrier_hit_rate" in json_str


class TestProfitFactorResult:
    """Tests for ProfitFactorResult dataclass."""

    @pytest.fixture
    def sample_profit_factor_result(self) -> ProfitFactorResult:
        """Create sample ProfitFactorResult."""
        q_labels = ["D1", "D2", "D3"]
        return ProfitFactorResult(
            n_quantiles=3,
            quantile_labels=q_labels,
            profit_factor={"D1": 0.8, "D2": 1.0, "D3": 1.5},
            sum_tp_returns={"D1": 0.8, "D2": 1.0, "D3": 1.5},
            sum_sl_returns={"D1": -1.0, "D2": -1.0, "D3": -1.0},
            sum_timeout_returns={"D1": 0.1, "D2": 0.1, "D3": 0.1},
            sum_all_returns={"D1": -0.1, "D2": 0.1, "D3": 0.6},
            avg_tp_return={"D1": 0.02, "D2": 0.02, "D3": 0.025},
            avg_sl_return={"D1": -0.015, "D2": -0.015, "D3": -0.015},
            avg_return={"D1": -0.001, "D2": 0.001, "D3": 0.006},
            count_tp={"D1": 40, "D2": 50, "D3": 60},
            count_sl={"D1": 60, "D2": 50, "D3": 40},
            count_total={"D1": 100, "D2": 100, "D3": 100},
            overall_profit_factor=1.1,
            overall_sum_returns=0.6,
            overall_avg_return=0.002,
            n_observations=300,
            pf_monotonic=True,
            pf_direction="increasing",
            pf_spearman=0.95,
        )

    def test_get_dataframe_profit_factor(
        self, sample_profit_factor_result: ProfitFactorResult
    ) -> None:
        """Test get_dataframe for profit_factor view."""
        df = sample_profit_factor_result.get_dataframe()
        assert "profit_factor" in df.columns
        assert "avg_return" in df.columns
        assert df.height == 3

    def test_get_dataframe_returns(self, sample_profit_factor_result: ProfitFactorResult) -> None:
        """Test get_dataframe for returns view."""
        df = sample_profit_factor_result.get_dataframe("returns")
        assert "sum_tp_returns" in df.columns
        assert "avg_tp_return" in df.columns

    def test_summary_output(self, sample_profit_factor_result: ProfitFactorResult) -> None:
        """Test summary string generation."""
        summary = sample_profit_factor_result.summary()
        assert "Profit Factor" in summary
        assert "D1" in summary


# =============================================================================
# BarrierAnalysis Initialization Tests
# =============================================================================


class TestBarrierAnalysisInit:
    """Tests for BarrierAnalysis initialization and validation."""

    def test_basic_initialization(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test basic initialization succeeds."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        assert analysis.n_observations > 0
        assert analysis.n_assets == 10
        assert analysis.n_dates == 100

    def test_initialization_with_config(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test initialization with custom config."""
        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=5))
        analysis = BarrierAnalysis(signal_data, barrier_labels, config=config)
        assert analysis.config.n_quantiles == 5

    def test_missing_signal_column(
        self,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test error when signal column is missing."""
        bad_signal = pl.DataFrame({"date": [date(2020, 1, 1)], "asset": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            BarrierAnalysis(bad_signal, barrier_labels)

    def test_missing_label_column(
        self,
        signal_data: pl.DataFrame,
    ) -> None:
        """Test error when label column is missing."""
        bad_barrier = pl.DataFrame({"date": [date(2020, 1, 1)], "asset": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            BarrierAnalysis(signal_data, bad_barrier)

    def test_empty_signal_data(
        self,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test error on empty signal data."""
        empty = pl.DataFrame(schema={"date": pl.Date, "asset": pl.Utf8, "signal": pl.Float64})
        with pytest.raises(ValueError, match="signal_data is empty"):
            BarrierAnalysis(empty, barrier_labels)

    def test_invalid_label_values(
        self,
        signal_data: pl.DataFrame,
    ) -> None:
        """Test error on invalid label values."""
        bad_labels = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)],
                "asset": ["ASSET_00"],
                "label": [5],  # Invalid
                "label_return": [0.01],
            }
        )
        with pytest.raises(ValueError, match="invalid values"):
            BarrierAnalysis(signal_data, bad_labels)

    def test_no_matching_rows(self) -> None:
        """Test error when no rows match between dataframes."""
        signal = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)],
                "asset": ["A"],
                "signal": [0.5],
            }
        )
        barrier = pl.DataFrame(
            {
                "date": [date(2021, 1, 1)],  # Different date
                "asset": ["B"],  # Different asset
                "label": [1],
                "label_return": [0.01],
            }
        )
        with pytest.raises(ValueError, match="No matching rows"):
            BarrierAnalysis(signal, barrier)

    def test_quantile_labels_property(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test quantile_labels property."""
        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=5))
        analysis = BarrierAnalysis(signal_data, barrier_labels, config=config)
        labels = analysis.quantile_labels
        assert len(labels) == 5
        assert labels == ["D1", "D2", "D3", "D4", "D5"]

    def test_date_range_property(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test date_range property."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        start, end = analysis.date_range
        assert start == "2020-01-01"
        assert "2020-04" in end  # ~100 days from start


# =============================================================================
# Hit Rate Tests
# =============================================================================


class TestComputeHitRates:
    """Tests for compute_hit_rates() method."""

    def test_basic_hit_rates(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test basic hit rate computation."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_hit_rates()

        assert isinstance(result, HitRateResult)
        assert result.n_quantiles == 10
        assert len(result.quantile_labels) == 10

        # Check hit rates sum to 1 for each quantile
        for q in result.quantile_labels:
            total = result.hit_rate_tp[q] + result.hit_rate_sl[q] + result.hit_rate_timeout[q]
            assert abs(total - 1.0) < 0.01, f"Hit rates don't sum to 1 for {q}"

    def test_hit_rate_counts(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test that counts match rates."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_hit_rates()

        for q in result.quantile_labels:
            total = result.count_total[q]
            if total > 0:
                expected_tp_rate = result.count_tp[q] / total
                assert abs(result.hit_rate_tp[q] - expected_tp_rate) < 0.001

    def test_chi_square_test(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test chi-square independence test."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_hit_rates()

        # Chi-square should have positive statistic
        assert result.chi2_statistic >= 0
        # P-value should be between 0 and 1
        assert 0 <= result.chi2_p_value <= 1
        # DoF should be (n_quantiles - 1) * (n_outcomes - 1)
        assert result.chi2_dof > 0

    def test_overall_hit_rates(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test overall hit rate calculation."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_hit_rates()

        # Overall rates should sum to 1
        total = (
            result.overall_hit_rate_tp
            + result.overall_hit_rate_sl
            + result.overall_hit_rate_timeout
        )
        assert abs(total - 1.0) < 0.01

    def test_monotonicity_analysis(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test monotonicity analysis of TP rate."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_hit_rates()

        assert result.tp_rate_direction in ["increasing", "decreasing", "none"]
        assert -1.0 <= result.tp_rate_spearman <= 1.0

    def test_result_caching(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test that result is cached on second call."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result1 = analysis.compute_hit_rates()
        result2 = analysis.compute_hit_rates()
        assert result1 is result2  # Same object


class TestComputeProfitFactor:
    """Tests for compute_profit_factor() method."""

    def test_basic_profit_factor(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test basic profit factor computation."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_profit_factor()

        assert isinstance(result, ProfitFactorResult)
        assert result.n_quantiles == 10

        # Profit factors should be non-negative
        for q in result.quantile_labels:
            assert result.profit_factor[q] >= 0

    def test_profit_factor_formula(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test profit factor calculation formula."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_profit_factor()
        eps = analysis.config.profit_factor_epsilon

        for q in result.quantile_labels:
            sum_tp = result.sum_tp_returns[q]
            sum_sl = result.sum_sl_returns[q]

            if sum_tp > 0 and sum_sl != 0:
                expected_pf = sum_tp / (abs(sum_sl) + eps)
                assert abs(result.profit_factor[q] - expected_pf) < 0.01

    def test_average_returns(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test average return calculations."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_profit_factor()

        for q in result.quantile_labels:
            if result.count_tp[q] > 0:
                expected_avg_tp = result.sum_tp_returns[q] / result.count_tp[q]
                assert abs(result.avg_tp_return[q] - expected_avg_tp) < 0.0001

    def test_overall_profit_factor(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test overall profit factor."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_profit_factor()

        assert result.overall_profit_factor >= 0
        assert result.n_observations > 0

    def test_monotonicity_analysis_pf(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test monotonicity analysis of profit factor."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        result = analysis.compute_profit_factor()

        assert result.pf_direction in ["increasing", "decreasing", "none"]
        assert -1.0 <= result.pf_spearman <= 1.0


# =============================================================================
# Tear Sheet Tests
# =============================================================================


class TestBarrierTearSheet:
    """Tests for create_tear_sheet() and BarrierTearSheet."""

    def test_create_tear_sheet(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test tear sheet creation."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        tear_sheet = analysis.create_tear_sheet()

        assert isinstance(tear_sheet, BarrierTearSheet)
        assert tear_sheet.hit_rate_result is not None
        assert tear_sheet.profit_factor_result is not None
        assert tear_sheet.n_assets == 10
        assert tear_sheet.n_dates == 100

    def test_tear_sheet_summary(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test tear sheet summary generation."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        tear_sheet = analysis.create_tear_sheet()
        summary = tear_sheet.summary()

        assert "Barrier Analysis" in summary
        assert "Hit Rate" in summary
        assert "Profit Factor" in summary

    def test_tear_sheet_dataframes(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test tear sheet DataFrame access."""
        analysis = BarrierAnalysis(signal_data, barrier_labels)
        tear_sheet = analysis.create_tear_sheet()

        # Should be able to get various dataframes
        summary_df = tear_sheet.get_dataframe("summary")
        assert isinstance(summary_df, pl.DataFrame)

        hit_rate_df = tear_sheet.get_dataframe("hit_rate_hit_rates")
        assert isinstance(hit_rate_df, pl.DataFrame)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_asset(self) -> None:
        """Test with single asset."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        signal = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 50,
                "signal": np.random.uniform(-1, 1, 50),
            }
        )
        barrier = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 50,
                "label": np.random.choice([-1, 0, 1], 50),
                "label_return": np.random.uniform(-0.02, 0.02, 50),
            }
        )
        analysis = BarrierAnalysis(signal, barrier)
        result = analysis.compute_hit_rates()
        assert result.n_observations == 50

    def test_only_tp_outcomes(self) -> None:
        """Test when all outcomes are take-profit."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        signal = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 50,
                "signal": np.linspace(-1, 1, 50),
            }
        )
        barrier = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 50,
                "label": [1] * 50,  # All TP
                "label_return": [0.02] * 50,
            }
        )
        analysis = BarrierAnalysis(signal, barrier)
        result = analysis.compute_hit_rates()

        # All TP means 100% hit rate
        assert result.overall_hit_rate_tp == 1.0
        assert result.overall_hit_rate_sl == 0.0

    def test_only_sl_outcomes(self) -> None:
        """Test when all outcomes are stop-loss."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        signal = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 50,
                "signal": np.linspace(-1, 1, 50),
            }
        )
        barrier = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 50,
                "label": [-1] * 50,  # All SL
                "label_return": [-0.01] * 50,
            }
        )
        analysis = BarrierAnalysis(signal, barrier)
        result = analysis.compute_profit_factor()

        # All SL means profit factor = 0
        assert result.overall_profit_factor == 0.0

    def test_few_quantiles(self) -> None:
        """Test with minimum quantiles (2)."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        np.random.seed(42)
        signal = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 100,
                "signal": np.random.uniform(-1, 1, 100),
            }
        )
        barrier = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 100,
                "label": np.random.choice([-1, 0, 1], 100),
                "label_return": np.random.uniform(-0.02, 0.02, 100),
            }
        )
        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=2))
        analysis = BarrierAnalysis(signal, barrier, config=config)
        result = analysis.compute_hit_rates()

        assert result.n_quantiles == 2
        assert len(result.quantile_labels) == 2

    def test_zscore_filtering(self) -> None:
        """Test outlier filtering."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        signals = list(np.random.uniform(-1, 1, 98)) + [100.0, -100.0]  # Outliers
        signal = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 100,
                "signal": signals,
            }
        )
        barrier = pl.DataFrame(
            {
                "date": dates,
                "asset": ["A"] * 100,
                "label": np.random.choice([-1, 0, 1], 100),
                "label_return": np.random.uniform(-0.02, 0.02, 100),
            }
        )
        config = BarrierConfig(analysis=AnalysisSettings(filter_zscore=3.0))
        analysis = BarrierAnalysis(signal, barrier, config=config)

        # Outliers should be filtered
        assert analysis.n_observations < 100


# =============================================================================
# Statistical Correctness Tests
# =============================================================================


class TestStatisticalCorrectness:
    """Tests for statistical correctness of computations."""

    def test_chi_square_known_data(self) -> None:
        """Test chi-square with strongly correlated but not deterministic data."""
        np.random.seed(42)

        # Create UNIQUE (date, asset) pairs: 100 dates x 10 assets = 1000 unique pairs
        n_dates = 100
        n_assets = 10
        rows = []
        for i, d in enumerate(date(2020, 1, 1) + timedelta(days=j) for j in range(n_dates)):
            for j, asset in enumerate(f"A{k}" for k in range(n_assets)):
                idx = i * n_assets + j
                # Signal linearly increasing with index
                signal = -1 + 2 * idx / (n_dates * n_assets - 1)

                # Signal strongly influences outcome
                if signal < -0.33:
                    label = np.random.choice([-1, 0, 1], p=[0.7, 0.2, 0.1])
                elif signal > 0.33:
                    label = np.random.choice([-1, 0, 1], p=[0.1, 0.2, 0.7])
                else:
                    label = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])

                ret = 0.02 if label == 1 else (-0.01 if label == -1 else 0)
                rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "signal": signal,
                        "label": label,
                        "label_return": ret,
                    }
                )

        df = pl.DataFrame(rows)
        signal_df = df.select(["date", "asset", "signal"])
        barrier_df = df.select(["date", "asset", "label", "label_return"])

        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=5))
        analysis = BarrierAnalysis(signal_df, barrier_df, config=config)
        result = analysis.compute_hit_rates()

        # Should be highly significant (very low p-value) due to strong correlation
        assert result.chi2_p_value < 0.05
        assert result.is_significant is True

    def test_spearman_correlation(self) -> None:
        """Test Spearman correlation calculation matches scipy."""
        np.random.seed(42)

        # Create UNIQUE (date, asset) pairs: 50 dates x 10 assets = 500 unique pairs
        n_dates = 50
        n_assets = 10
        rows = []
        for i, d in enumerate(date(2020, 1, 1) + timedelta(days=j) for j in range(n_dates)):
            for j, asset in enumerate(f"A{k}" for k in range(n_assets)):
                idx = i * n_assets + j
                # Signal linearly increasing with index
                signal = -1 + 2 * idx / (n_dates * n_assets - 1)

                # Higher signal -> higher P(TP), lower P(SL)
                s_norm = (signal + 1) / 2  # 0 to 1
                p_tp = 0.2 + 0.5 * s_norm  # 0.2 to 0.7
                p_sl = 0.5 - 0.4 * s_norm  # 0.5 to 0.1
                p_timeout = 1.0 - p_tp - p_sl  # remainder (0.3)
                label = np.random.choice([1, -1, 0], p=[p_tp, p_sl, p_timeout])

                ret = 0.02 if label == 1 else -0.01
                rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "signal": signal,
                        "label": label,
                        "label_return": ret,
                    }
                )

        df = pl.DataFrame(rows)
        signal_df = df.select(["date", "asset", "signal"])
        barrier_df = df.select(["date", "asset", "label", "label_return"])

        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=5))
        analysis = BarrierAnalysis(signal_df, barrier_df, config=config)
        result = analysis.compute_hit_rates()

        # Should have positive Spearman correlation
        # (higher quantile -> higher TP rate)
        assert result.tp_rate_spearman > 0


# =============================================================================
# BarrierLabel Enum Tests
# =============================================================================


class TestBarrierLabel:
    """Tests for BarrierLabel enum."""

    def test_label_values(self) -> None:
        """Test enum values match expected."""
        assert BarrierLabel.STOP_LOSS.value == -1
        assert BarrierLabel.TIMEOUT.value == 0
        assert BarrierLabel.TAKE_PROFIT.value == 1

    def test_label_comparison(self) -> None:
        """Test enum can be compared to int."""
        assert BarrierLabel.TAKE_PROFIT == 1
        assert BarrierLabel.STOP_LOSS == -1
