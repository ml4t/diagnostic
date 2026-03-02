"""Tests for result schemas (TASK-010).

Validates that all result schemas:
1. Can be instantiated with valid data
2. Support JSON serialization round-trip
3. Provide DataFrame access methods
4. Validate and reject invalid data
5. Produce readable summaries
"""

import json
from datetime import datetime

import polars as pl
import pytest
from pydantic import ValidationError

from ml4t.diagnostic.results import (
    ACFResult,
    BaseResult,
    BayesianComparisonResult,
    CrossFeatureResult,
    DSRResult,
    FDRResult,
    FeatureDiagnosticsResultSchema,
    FeatureOutcomeResultSchema,
    ICAnalysisResult,
    MinTRLResult,
    PortfolioEvaluationResult,
    PortfolioMetrics,
    PSRResult,
    SharpeFrameworkResult,
    StationarityTestResult,
    ThresholdAnalysisResult,
)

# =============================================================================
# Base Result Tests
# =============================================================================


def test_base_result_metadata():
    """Test that BaseResult captures metadata correctly."""

    class TestResult(BaseResult):
        analysis_type: str = "test"

        def get_dataframe(self, metric=None):
            return pl.DataFrame({"value": [1]})

        def summary(self):
            return "Test summary"

    result = TestResult()

    # Check metadata
    assert result.analysis_type == "test"
    assert result.version == "2.0.0"
    assert result.created_at  # Should be ISO timestamp

    # Verify timestamp is valid ISO format
    datetime.fromisoformat(result.created_at)


def test_base_result_to_dict():
    """Test dict export."""

    class TestResult(BaseResult):
        analysis_type: str = "test"
        value: int = 42

        def get_dataframe(self, metric=None):
            return pl.DataFrame()

        def summary(self):
            return ""

    result = TestResult()
    data = result.to_dict()

    assert isinstance(data, dict)
    assert data["analysis_type"] == "test"
    assert data["value"] == 42
    assert "created_at" in data


def test_base_result_json_serialization():
    """Test JSON serialization."""

    class TestResult(BaseResult):
        analysis_type: str = "test"
        value: float = 3.14

        def get_dataframe(self, metric=None):
            return pl.DataFrame()

        def summary(self):
            return ""

    result = TestResult()
    json_str = result.to_json_string()

    # Parse back
    data = json.loads(json_str)
    assert data["analysis_type"] == "test"
    assert data["value"] == pytest.approx(3.14)


# =============================================================================
# Module A: Feature Diagnostics
# =============================================================================


def test_stationarity_test_result():
    """Test StationarityTestResult instantiation and methods."""
    result = StationarityTestResult(
        feature_name="returns",
        adf_statistic=-3.5,
        adf_pvalue=0.01,
        adf_is_stationary=True,
        kpss_statistic=0.3,
        kpss_pvalue=0.10,
        kpss_is_stationary=True,
    )

    # Check fields
    assert result.feature_name == "returns"
    assert result.adf_is_stationary is True

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert "feature" in df.columns
    assert df["adf_stationary"][0] is True

    # Test summary
    summary = result.summary()
    assert "returns" in summary
    assert "Stationary" in summary


def test_acf_result():
    """Test ACFResult instantiation and methods."""
    result = ACFResult(
        feature_name="returns",
        acf_values=[1.0, 0.5, 0.3, 0.1],
        pacf_values=[1.0, 0.5, 0.1, 0.05],
        significant_lags_acf=[1, 2],
        significant_lags_pacf=[1],
        ljung_box_statistic=25.5,
        ljung_box_pvalue=0.001,
    )

    assert result.feature_name == "returns"
    assert len(result.acf_values) == 4

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 4
    assert "acf" in df.columns
    assert "pacf" in df.columns

    # Test summary
    summary = result.summary()
    assert "ACF/PACF" in summary
    assert "Autocorrelation present" in summary


def test_feature_diagnostics_result():
    """Test FeatureDiagnosticsResult with nested results."""
    stat_test = StationarityTestResult(
        feature_name="returns",
        adf_statistic=-3.5,
        adf_pvalue=0.01,
        adf_is_stationary=True,
    )

    acf = ACFResult(
        feature_name="returns",
        acf_values=[1.0, 0.5, 0.3],
        pacf_values=[1.0, 0.5, 0.1],
    )

    result = FeatureDiagnosticsResultSchema(
        stationarity_tests=[stat_test],
        acf_results=[acf],
        volatility_clustering={"garch_detected": True},
        distribution_stats={"skewness": -0.2, "kurtosis": 3.5},
    )

    assert len(result.stationarity_tests) == 1
    assert len(result.acf_results) == 1

    # Test stationarity DataFrame
    df = result.get_stationarity_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test ACF DataFrame
    df = result.get_acf_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert "feature" in df.columns

    # Test summary
    summary = result.summary()
    assert "Feature Diagnostics" in summary


# =============================================================================
# Module B: Cross-Feature Analysis
# =============================================================================


def test_cross_feature_result():
    """Test CrossFeatureResult with correlation matrix."""
    result = CrossFeatureResult(
        correlation_matrix=[
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ],
        feature_names=["f1", "f2", "f3"],
        redundant_features=[("f1", "f2", 0.8)],
    )

    assert len(result.feature_names) == 3
    assert result.correlation_matrix[0][0] == 1.0

    # Test correlation DataFrame
    df = result.get_correlation_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 9  # 3x3 matrix in long format

    # Test redundancy DataFrame
    df = result.get_redundancy_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test summary
    summary = result.summary()
    assert "Cross-Feature" in summary
    assert "Redundant pairs" in summary


# =============================================================================
# Module C: Feature-Outcome Relationships
# =============================================================================


def test_ic_analysis_result():
    """Test ICAnalysisResult."""
    result = ICAnalysisResult(
        feature_name="momentum",
        ic_values=[0.05, 0.04, 0.03, 0.02],
        mean_ic=0.035,
        ic_std=0.01,
        ic_ir=3.5,
        pvalue=0.001,
        hac_adjusted_pvalue=0.005,
    )

    assert result.feature_name == "momentum"
    assert result.mean_ic == pytest.approx(0.035)

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 4
    assert "ic" in df.columns

    # Test summary
    summary = result.summary()
    assert "momentum" in summary
    assert "IC" in summary


def test_threshold_analysis_result():
    """Test ThresholdAnalysisResult."""
    result = ThresholdAnalysisResult(
        feature_name="rsi",
        optimal_threshold=70.0,
        precision=0.65,
        recall=0.55,
        f1_score=0.60,
        lift=1.8,
        coverage=0.15,
    )

    assert result.feature_name == "rsi"
    assert result.optimal_threshold == 70.0

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert "precision" in df.columns

    # Test summary
    summary = result.summary()
    assert "rsi" in summary
    assert "70.0" in summary


def test_feature_outcome_result():
    """Test FeatureOutcomeResult with IC and threshold analysis."""
    ic = ICAnalysisResult(
        feature_name="momentum",
        ic_values=[0.05, 0.04],
        mean_ic=0.045,
        ic_std=0.005,
        ic_ir=9.0,
        hac_adjusted_pvalue=0.001,
    )

    threshold = ThresholdAnalysisResult(
        feature_name="momentum",
        optimal_threshold=0.5,
        precision=0.70,
        recall=0.60,
        f1_score=0.65,
        lift=2.0,
        coverage=0.20,
    )

    result = FeatureOutcomeResultSchema(
        ic_results=[ic],
        threshold_results=[threshold],
        ml_importance={"momentum": 0.85, "volatility": 0.15},
    )

    assert len(result.ic_results) == 1
    assert len(result.threshold_results) == 1

    # Test IC DataFrame
    df = result.get_ic_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test threshold DataFrame
    df = result.get_threshold_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test summary
    summary = result.summary()
    assert "Feature-Outcome" in summary


# =============================================================================
# Module D: Portfolio Evaluation
# =============================================================================


def test_portfolio_metrics():
    """Test PortfolioMetrics."""
    result = PortfolioMetrics(
        total_return=0.50,
        annualized_return=0.15,
        annualized_volatility=0.20,
        sharpe_ratio=0.75,
        sortino_ratio=1.0,
        max_drawdown=-0.30,
        calmar_ratio=0.50,
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=-0.015,
        profit_factor=1.5,
        skewness=-0.3,
        kurtosis=2.5,
    )

    assert result.sharpe_ratio == pytest.approx(0.75)
    assert result.max_drawdown == pytest.approx(-0.30)

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert "sharpe_ratio" in df.columns

    # Test summary
    summary = result.summary()
    assert "Portfolio Metrics" in summary
    assert "Sharpe" in summary


def test_bayesian_comparison_result():
    """Test BayesianComparisonResult."""
    result = BayesianComparisonResult(
        strategy_a_name="Strategy A",
        strategy_b_name="Strategy B",
        prior_sharpe_mean=0.5,
        prior_sharpe_std=0.2,
        posterior_sharpe_mean=0.7,
        posterior_sharpe_std=0.15,
        probability_a_better=0.85,
        credible_interval_95=(0.4, 1.0),
    )

    assert result.strategy_a_name == "Strategy A"
    assert result.probability_a_better == pytest.approx(0.85)

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test summary
    summary = result.summary()
    assert "Bayesian" in summary
    assert "Strategy A" in summary


def test_portfolio_evaluation_result():
    """Test PortfolioEvaluationResult."""
    metrics = PortfolioMetrics(
        total_return=0.50,
        annualized_return=0.15,
        annualized_volatility=0.20,
        sharpe_ratio=0.75,
        sortino_ratio=1.0,
        max_drawdown=-0.30,
        calmar_ratio=0.50,
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=-0.015,
        profit_factor=1.5,
        skewness=-0.3,
        kurtosis=2.5,
    )

    result = PortfolioEvaluationResult(
        metrics=metrics,
        drawdown_analysis={
            "max_duration_days": 120,
            "avg_drawdown": -0.15,
            "num_drawdowns": 5,
        },
    )

    assert result.metrics.sharpe_ratio == pytest.approx(0.75)

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)

    # Test summary
    summary = result.summary()
    assert "Portfolio Evaluation" in summary


# =============================================================================
# Sharpe Framework
# =============================================================================


def test_psr_result():
    """Test PSRResult."""
    result = PSRResult(
        observed_sharpe=0.85,
        target_sharpe=0.50,
        psr_value=0.92,
        skewness=-0.2,
        kurtosis=3.0,
        n_observations=1000,
    )

    assert result.observed_sharpe == pytest.approx(0.85)
    assert result.psr_value == pytest.approx(0.92)

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test summary
    summary = result.summary()
    assert "PSR" in summary
    # PSR of 0.92 should show "Moderate confidence" (0.80-0.95 range)
    assert "confidence" in summary.lower()


def test_min_trl_result():
    """Test MinTRLResult."""
    result = MinTRLResult(
        observed_sharpe=0.75,
        target_sharpe=0.50,
        min_trl_days=500,
        actual_days=1000,
        is_sufficient=True,
        skewness=-0.1,
        kurtosis=2.5,
    )

    assert result.is_sufficient is True

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test summary
    summary = result.summary()
    assert "MinTRL" in summary
    assert "adequate" in summary


def test_dsr_result():
    """Test DSRResult."""
    result = DSRResult(
        observed_sharpe=0.85,
        dsr_value=0.65,
        adjusted_pvalue=0.03,
        is_significant=True,
        n_trials=10,
        variance_trials=0.05,
    )

    assert result.is_significant is True

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test summary
    summary = result.summary()
    assert "DSR" in summary
    assert "Significant" in summary


def test_fdr_result():
    """Test FDRResult."""
    result = FDRResult(
        observed_sharpe=0.85,
        null_sharpe=0.0,
        alternative_sharpe=0.50,
        prior_h0=0.70,
        ofdr=0.15,
        pfdr=0.25,
    )

    assert result.ofdr == pytest.approx(0.15)

    # Test DataFrame
    df = result.get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1

    # Test summary
    summary = result.summary()
    assert "FDR" in summary


def test_sharpe_framework_result():
    """Test SharpeFrameworkResult with all components."""
    psr = PSRResult(
        observed_sharpe=0.85,
        target_sharpe=0.50,
        psr_value=0.92,
        skewness=-0.2,
        kurtosis=3.0,
        n_observations=1000,
    )

    min_trl = MinTRLResult(
        observed_sharpe=0.85,
        target_sharpe=0.50,
        min_trl_days=500,
        actual_days=1000,
        is_sufficient=True,
        skewness=-0.2,
        kurtosis=3.0,
    )

    result = SharpeFrameworkResult(
        psr=psr,
        min_trl=min_trl,
    )

    assert result.psr is not None
    assert result.min_trl is not None

    # Test DataFrame with different names
    df = result.get_dataframe(name="psr")
    assert isinstance(df, pl.DataFrame)

    df = result.get_dataframe(name="min_trl")
    assert isinstance(df, pl.DataFrame)

    # Test summary
    summary = result.summary()
    assert "Enhanced Sharpe Framework" in summary


# =============================================================================
# JSON Serialization Round-Trip Tests
# =============================================================================


def test_json_round_trip_stationarity():
    """Test JSON serialization round-trip for StationarityTestResult."""
    original = StationarityTestResult(
        feature_name="returns",
        adf_statistic=-3.5,
        adf_pvalue=0.01,
        adf_is_stationary=True,
    )

    # Serialize
    json_str = original.to_json_string()

    # Deserialize
    data = json.loads(json_str)
    reconstructed = StationarityTestResult(**data)

    # Compare
    assert reconstructed.feature_name == original.feature_name
    assert reconstructed.adf_statistic == original.adf_statistic
    assert reconstructed.adf_is_stationary == original.adf_is_stationary


def test_json_round_trip_portfolio_metrics():
    """Test JSON serialization round-trip for PortfolioMetrics."""
    original = PortfolioMetrics(
        total_return=0.50,
        annualized_return=0.15,
        annualized_volatility=0.20,
        sharpe_ratio=0.75,
        sortino_ratio=1.0,
        max_drawdown=-0.30,
        calmar_ratio=0.50,
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=-0.015,
        profit_factor=1.5,
        skewness=-0.3,
        kurtosis=2.5,
    )

    # Serialize
    json_str = original.to_json_string()

    # Deserialize
    data = json.loads(json_str)
    reconstructed = PortfolioMetrics(**data)

    # Compare
    assert reconstructed.sharpe_ratio == pytest.approx(original.sharpe_ratio)
    assert reconstructed.max_drawdown == pytest.approx(original.max_drawdown)


# =============================================================================
# Validation Tests
# =============================================================================


def test_validation_psr_bounds():
    """Test that PSR value must be between 0 and 1."""
    with pytest.raises(ValidationError):
        PSRResult(
            observed_sharpe=0.5,
            target_sharpe=0.3,
            psr_value=1.5,  # Invalid: > 1
            skewness=0.0,
            kurtosis=0.0,
            n_observations=100,
        )


def test_validation_min_trl_positive():
    """Test that MinTRL days must be positive."""
    with pytest.raises(ValidationError):
        MinTRLResult(
            observed_sharpe=0.5,
            target_sharpe=0.3,
            min_trl_days=0,  # Invalid: must be > 0
            actual_days=100,
            is_sufficient=False,
            skewness=0.0,
            kurtosis=0.0,
        )


def test_validation_extra_fields_forbidden():
    """Test that extra fields are rejected (typo detection)."""
    with pytest.raises(ValidationError):
        StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
            typo_field=123,  # Invalid: extra field
        )


# =============================================================================
# Edge Cases
# =============================================================================


def test_empty_feature_diagnostics():
    """Test FeatureDiagnosticsResult with empty lists."""
    result = FeatureDiagnosticsResultSchema()

    df = result.get_stationarity_dataframe()
    assert len(df) == 0

    df = result.get_acf_dataframe()
    assert len(df) == 0

    summary = result.summary()
    assert "Feature Diagnostics" in summary


def test_none_optional_fields():
    """Test that optional fields can be None."""
    result = ACFResult(
        feature_name="returns",
        acf_values=[1.0, 0.5],
        pacf_values=[1.0, 0.5],
        ljung_box_statistic=None,  # Optional
        ljung_box_pvalue=None,  # Optional
    )

    assert result.ljung_box_statistic is None
    summary = result.summary()
    assert isinstance(summary, str)
