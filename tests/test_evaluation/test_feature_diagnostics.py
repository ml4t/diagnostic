"""Tests for feature_diagnostics module."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError as PydanticValidationError

from ml4t.diagnostic.config import (
    ACFSettings,
    DiagnosticConfig,
    DistributionSettings,
    StationaritySettings,
    VolatilitySettings,
)
from ml4t.diagnostic.errors import ValidationError
from ml4t.diagnostic.evaluation.feature_diagnostics import (
    FeatureDiagnostics,
    FeatureDiagnosticsResult,
)


class TestDiagnosticConfigForFeatureDiagnostics:
    """Tests for canonical feature diagnostics configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = DiagnosticConfig()

        assert config.stationarity.enabled is True
        assert config.acf.enabled is True
        assert config.volatility.enabled is True
        assert config.distribution.enabled is True
        assert config.distribution.alpha == 0.05
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = DiagnosticConfig(
            stationarity=StationaritySettings(enabled=False),
            acf=ACFSettings(enabled=True),
            volatility=VolatilitySettings(enabled=False),
            distribution=DistributionSettings(enabled=True, alpha=0.01),
            verbose=True,
        )

        assert config.stationarity.enabled is False
        assert config.acf.enabled is True
        assert config.volatility.enabled is False
        assert config.distribution.enabled is True
        assert config.distribution.alpha == 0.01
        assert config.verbose is True

    def test_invalid_alpha(self):
        """Test validation of alpha parameter."""
        with pytest.raises(PydanticValidationError):
            DistributionSettings(alpha=0.0)

        with pytest.raises(PydanticValidationError):
            DistributionSettings(alpha=1.0)

        with pytest.raises(PydanticValidationError):
            DistributionSettings(alpha=-0.05)

    def test_no_modules_enabled(self):
        """Test validation when no modules enabled."""
        with pytest.raises(ValidationError, match="At least one diagnostic module"):
            FeatureDiagnostics(
                DiagnosticConfig(
                    stationarity=StationaritySettings(enabled=False),
                    acf=ACFSettings(enabled=False),
                    volatility=VolatilitySettings(enabled=False),
                    distribution=DistributionSettings(enabled=False),
                )
            )

    def test_specific_test_selection(self):
        """Test selecting specific tests within modules."""
        config = DiagnosticConfig(
            stationarity=StationaritySettings(adf_enabled=True, kpss_enabled=True, pp_enabled=False),
            distribution=DistributionSettings(compute_tails=False),
        )

        assert config.stationarity.adf_enabled is True
        assert config.stationarity.kpss_enabled is True
        assert config.stationarity.pp_enabled is False
        assert config.distribution.compute_tails is False


class TestFeatureDiagnostics:
    """Tests for FeatureDiagnostics class."""

    @pytest.fixture
    def white_noise(self):
        """Generate white noise (ideal feature)."""
        np.random.seed(42)
        return np.random.randn(1000)

    @pytest.fixture
    def random_walk(self):
        """Generate random walk (non-stationary)."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(1000))

    @pytest.fixture
    def ar1_process(self):
        """Generate AR(1) process with autocorrelation."""
        np.random.seed(42)
        n = 1000
        phi = 0.7
        x = np.zeros(n)
        noise = np.random.randn(n)

        for i in range(1, n):
            x[i] = phi * x[i - 1] + noise[i]

        return x

    @pytest.fixture
    def garch_process(self):
        """Generate GARCH process with volatility clustering."""
        np.random.seed(42)
        n = 1000
        omega = 0.1
        alpha = 0.3
        beta = 0.6

        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for i in range(1, n):
            sigma2[i] = omega + alpha * returns[i - 1] ** 2 + beta * sigma2[i - 1]
            returns[i] = np.sqrt(sigma2[i]) * np.random.randn()

        return returns

    @pytest.fixture
    def heavy_tail_process(self):
        """Generate heavy-tailed process (Student-t)."""
        np.random.seed(42)
        return np.random.standard_t(df=3, size=1000)

    def test_init_default_config(self):
        """Test initialization with default config."""
        diagnostics = FeatureDiagnostics()
        assert diagnostics.config is not None
        assert diagnostics.config.stationarity.enabled is True

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = DiagnosticConfig(stationarity=StationaritySettings(significance_level=0.01))
        diagnostics = FeatureDiagnostics(config)
        assert diagnostics.config.stationarity.significance_level == 0.01

    def test_run_diagnostics_white_noise(self, white_noise):
        """Test diagnostics on white noise (ideal feature)."""
        diagnostics = FeatureDiagnostics()
        result = diagnostics.run_diagnostics(white_noise, name="white_noise")

        # Check result type
        assert isinstance(result, FeatureDiagnosticsResult)
        assert result.feature_name == "white_noise"
        assert result.n_obs == 1000

        # Check module results exist
        assert result.stationarity is not None
        assert result.autocorrelation is not None
        assert result.volatility is not None
        assert result.distribution is not None

        # Check expectations for white noise
        assert result.stationarity.consensus in [
            "strong_stationary",
            "likely_stationary",
        ]
        assert not result.volatility.has_volatility_clustering
        # White noise should have minimal autocorrelation
        assert len(result.autocorrelation.significant_acf_lags) < 50  # Allow some by chance

        # Check health score is good
        assert result.health_score > 0.6  # Should be quite healthy

        # Check summary works
        summary = result.summary()
        assert "white_noise" in summary
        assert "Health Score" in summary

    def test_run_diagnostics_random_walk(self, random_walk):
        """Test diagnostics on random walk (non-stationary)."""
        diagnostics = FeatureDiagnostics()
        result = diagnostics.run_diagnostics(random_walk, name="random_walk")

        # Should detect non-stationarity
        assert result.stationarity.consensus in [
            "strong_nonstationary",
            "likely_nonstationary",
        ]

        # Should flag non-stationarity
        assert "NON_STATIONARY" in result.flags

        # Should recommend transformation
        assert any("differenc" in rec.lower() for rec in result.recommendations)

    def test_run_diagnostics_ar1(self, ar1_process):
        """Test diagnostics on AR(1) process."""
        diagnostics = FeatureDiagnostics()
        result = diagnostics.run_diagnostics(ar1_process, name="ar1")

        # Should detect autocorrelation
        assert not result.autocorrelation.is_white_noise  # Has autocorrelation
        assert len(result.autocorrelation.significant_acf_lags) > 0

        # Should recommend AR modeling
        assert any("ar" in rec.lower() or "ma" in rec.lower() for rec in result.recommendations)

    def test_run_diagnostics_garch(self, garch_process):
        """Test diagnostics on GARCH process."""
        diagnostics = FeatureDiagnostics()
        result = diagnostics.run_diagnostics(garch_process, name="garch")

        # Should detect volatility clustering
        assert result.volatility.has_volatility_clustering
        assert "VOLATILITY_CLUSTERING" in result.flags

        # Should recommend GARCH modeling
        assert any(
            "garch" in rec.lower() or "volatility" in rec.lower() for rec in result.recommendations
        )

    def test_run_diagnostics_heavy_tails(self, heavy_tail_process):
        """Test diagnostics on heavy-tailed process."""
        diagnostics = FeatureDiagnostics()
        result = diagnostics.run_diagnostics(heavy_tail_process, name="heavy_tail")

        # Should detect non-normality
        assert not result.distribution.is_normal
        assert "NON_NORMAL" in result.flags

        # Should detect heavy tails (check flags or recommendations, not just distribution name)
        # The distribution recommendation logic is complex, but it should flag heavy tails
        has_heavy_tail_indication = "HEAVY_TAILS" in result.flags or any(
            "heavy tail" in rec.lower() or "student-t" in rec.lower()
            for rec in result.recommendations
        )
        assert has_heavy_tail_indication

    def test_run_diagnostics_series_input(self, white_noise):
        """Test with pandas Series input."""
        series = pd.Series(white_noise, name="test_series")

        diagnostics = FeatureDiagnostics()
        result = diagnostics.run_diagnostics(series, name="series_test")

        assert result.feature_name == "series_test"
        assert result.n_obs == 1000

    def test_run_diagnostics_polars_series_input(self, white_noise):
        """Test with Polars Series input."""
        series = pl.Series("test_series", white_noise)

        diagnostics = FeatureDiagnostics()
        result = diagnostics.run_diagnostics(series, name="series_test")

        assert result.feature_name == "series_test"
        assert result.n_obs == 1000

    def test_run_diagnostics_invalid_input(self):
        """Test error handling for invalid input."""
        diagnostics = FeatureDiagnostics()

        # Wrong type
        with pytest.raises(ValidationError, match="must be pd.Series or np.ndarray"):
            diagnostics.run_diagnostics([1, 2, 3], name="test")

        # Wrong dimension
        with pytest.raises(ValidationError, match="must be 1-dimensional"):
            diagnostics.run_diagnostics(np.random.randn(10, 10), name="test")

        # Empty data
        with pytest.raises(ValidationError, match="must not be empty"):
            diagnostics.run_diagnostics(np.array([]), name="test")

    def test_run_diagnostics_selective_modules(self, white_noise):
        """Test running only selected modules."""
        # Only stationarity
        config = DiagnosticConfig(
            stationarity=StationaritySettings(enabled=True),
            acf=ACFSettings(enabled=False),
            volatility=VolatilitySettings(enabled=False),
            distribution=DistributionSettings(enabled=False),
        )

        diagnostics = FeatureDiagnostics(config)
        result = diagnostics.run_diagnostics(white_noise, name="test")

        assert result.stationarity is not None
        assert result.autocorrelation is None
        assert result.volatility is None
        assert result.distribution is None

    def test_summary_df_creation(self, white_noise):
        """Test summary DataFrame creation."""
        diagnostics = FeatureDiagnostics()
        result = diagnostics.run_diagnostics(white_noise, name="test")

        # Check DataFrame exists and has expected columns
        assert isinstance(result.summary_df, pd.DataFrame)
        assert not result.summary_df.empty
        assert "Module" in result.summary_df.columns
        assert "Test" in result.summary_df.columns
        assert "Result" in result.summary_df.columns

        # Check all modules represented
        modules = result.summary_df["Module"].unique()
        assert "Stationarity" in modules
        assert "Autocorrelation" in modules
        assert "Volatility" in modules
        assert "Distribution" in modules

    def test_health_score_calculation(self, white_noise, random_walk):
        """Test health score calculation."""
        diagnostics = FeatureDiagnostics()

        # White noise should have high health score
        result_good = diagnostics.run_diagnostics(white_noise, name="good")
        assert result_good.health_score > 0.6

        # Random walk should have lower health score
        result_bad = diagnostics.run_diagnostics(random_walk, name="bad")
        assert result_bad.health_score < result_good.health_score

    def test_flags_identification(self, random_walk, garch_process):
        """Test warning flags identification."""
        diagnostics = FeatureDiagnostics()

        # Random walk should have NON_STATIONARY flag
        result_rw = diagnostics.run_diagnostics(random_walk, name="rw")
        assert "NON_STATIONARY" in result_rw.flags

        # GARCH should have VOLATILITY_CLUSTERING flag
        result_garch = diagnostics.run_diagnostics(garch_process, name="garch")
        assert "VOLATILITY_CLUSTERING" in result_garch.flags

    def test_recommendations_generation(self, white_noise, random_walk):
        """Test recommendations generation."""
        diagnostics = FeatureDiagnostics()

        # White noise should have recommendations (even if minimal - may have 1-2 spurious significant lags)
        result_good = diagnostics.run_diagnostics(white_noise, name="good")
        assert len(result_good.recommendations) > 0
        # White noise has high health score even if it has a few spurious autocorrelations
        assert result_good.health_score > 0.7

        # Random walk should recommend transformation
        result_bad = diagnostics.run_diagnostics(random_walk, name="bad")
        assert any(
            "differenc" in rec.lower() or "detrend" in rec.lower()
            for rec in result_bad.recommendations
        )


class TestBatchDiagnostics:
    """Tests for batch processing."""

    @pytest.fixture
    def multi_feature_df(self):
        """Create DataFrame with multiple features."""
        np.random.seed(42)
        n = 500

        return pd.DataFrame(
            {
                "white_noise": np.random.randn(n),
                "random_walk": np.cumsum(np.random.randn(n)),
                "ar1": self._generate_ar1(n, phi=0.6),
            }
        )

    @pytest.fixture
    def multi_feature_pl_df(self):
        """Create Polars DataFrame with multiple features."""
        np.random.seed(42)
        n = 500
        return pl.DataFrame(
            {
                "white_noise": np.random.randn(n),
                "random_walk": np.cumsum(np.random.randn(n)),
                "ar1": self._generate_ar1(n, phi=0.6),
            }
        )

    def _generate_ar1(self, n, phi):
        """Helper to generate AR(1) process."""
        x = np.zeros(n)
        noise = np.random.randn(n)

        for i in range(1, n):
            x[i] = phi * x[i - 1] + noise[i]

        return x

    def test_run_batch_diagnostics(self, multi_feature_df):
        """Test batch processing of multiple features."""
        diagnostics = FeatureDiagnostics()
        results = diagnostics.run_batch_diagnostics(multi_feature_df)

        # Check all features processed
        assert len(results) == 3
        assert "white_noise" in results
        assert "random_walk" in results
        assert "ar1" in results

        # Check each result is valid
        for name, result in results.items():
            assert isinstance(result, FeatureDiagnosticsResult)
            assert result.feature_name == name
            assert result.n_obs == 500

    def test_run_batch_selective_features(self, multi_feature_df):
        """Test batch processing with selective features."""
        diagnostics = FeatureDiagnostics()
        results = diagnostics.run_batch_diagnostics(
            multi_feature_df, feature_names=["white_noise", "ar1"]
        )

        # Check only selected features processed
        assert len(results) == 2
        assert "white_noise" in results
        assert "ar1" in results
        assert "random_walk" not in results

    def test_run_batch_invalid_feature_name(self, multi_feature_df):
        """Test batch processing with invalid feature name."""
        diagnostics = FeatureDiagnostics()

        # Should skip invalid feature names without error
        results = diagnostics.run_batch_diagnostics(
            multi_feature_df, feature_names=["white_noise", "nonexistent"]
        )

        # Only valid feature should be in results
        assert len(results) == 1
        assert "white_noise" in results

    def test_run_batch_invalid_input(self):
        """Test batch processing error handling."""
        diagnostics = FeatureDiagnostics()

        # Not a DataFrame
        with pytest.raises(ValidationError, match="must be pd.DataFrame"):
            diagnostics.run_batch_diagnostics(np.random.randn(100))

    def test_run_batch_diagnostics_polars(self, multi_feature_pl_df):
        """Test batch processing with Polars DataFrame input."""
        diagnostics = FeatureDiagnostics()
        results = diagnostics.run_batch_diagnostics(multi_feature_pl_df)

        assert len(results) == 3
        assert "white_noise" in results
        assert "random_walk" in results
        assert "ar1" in results

    def test_batch_health_comparison(self, multi_feature_df):
        """Test comparing health scores across features."""
        diagnostics = FeatureDiagnostics()
        results = diagnostics.run_batch_diagnostics(multi_feature_df)

        # White noise should be healthiest
        assert results["white_noise"].health_score > results["random_walk"].health_score
        assert results["white_noise"].health_score > results["ar1"].health_score


class TestFeatureDiagnosticsResult:
    """Tests for FeatureDiagnosticsResult class."""

    @pytest.fixture
    def result_with_all_modules(self):
        """Create result with all module results."""
        diagnostics = FeatureDiagnostics()
        np.random.seed(42)
        data = np.random.randn(500)
        return diagnostics.run_diagnostics(data, name="test_feature")

    def test_result_initialization(self, result_with_all_modules):
        """Test result object initialization."""
        result = result_with_all_modules

        assert result.feature_name == "test_feature"
        assert result.n_obs == 500
        assert result.health_score >= 0.0
        assert result.health_score <= 1.0
        assert isinstance(result.recommendations, list)
        assert isinstance(result.flags, list)
        assert isinstance(result.summary_df, pd.DataFrame)

    def test_summary_method(self, result_with_all_modules):
        """Test summary() method."""
        result = result_with_all_modules
        summary = result.summary()

        assert isinstance(summary, str)
        assert "test_feature" in summary
        assert "Health Score" in summary
        assert "Recommendations" in summary

    def test_result_with_partial_modules(self):
        """Test result with only some modules run."""
        config = DiagnosticConfig(
            stationarity=StationaritySettings(enabled=True),
            acf=ACFSettings(enabled=False),
            volatility=VolatilitySettings(enabled=False),
            distribution=DistributionSettings(enabled=True),
        )

        diagnostics = FeatureDiagnostics(config)
        np.random.seed(42)
        data = np.random.randn(500)
        result = diagnostics.run_diagnostics(data, name="partial")

        assert result.stationarity is not None
        assert result.autocorrelation is None
        assert result.volatility is None
        assert result.distribution is not None

        # Should still generate valid summary
        summary = result.summary()
        assert "partial" in summary


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_very_short_series(self):
        """Test with very short time series."""
        diagnostics = FeatureDiagnostics()
        short_data = np.random.randn(20)  # Minimum for most tests

        # Should complete without error (though results may be unreliable)
        result = diagnostics.run_diagnostics(short_data, name="short")
        assert result.n_obs == 20

    def test_constant_series(self):
        """Test with constant series."""
        diagnostics = FeatureDiagnostics()
        constant = np.ones(100)

        # Should handle gracefully (may skip some tests)
        result = diagnostics.run_diagnostics(constant, name="constant")
        assert result is not None

    def test_series_with_nans(self):
        """Test with series containing NaNs."""
        diagnostics = FeatureDiagnostics()
        data_with_nan = np.random.randn(100)
        data_with_nan[50] = np.nan

        # statsmodels will handle NaN removal, should not raise
        result = diagnostics.run_diagnostics(data_with_nan, name="with_nan")
        assert result is not None

    def test_series_with_infs(self):
        """Test with series containing infinities."""
        diagnostics = FeatureDiagnostics()
        data_with_inf = np.random.randn(100)
        data_with_inf[50] = np.inf

        # Some tests may fail, but should not crash
        result = diagnostics.run_diagnostics(data_with_inf, name="with_inf")
        assert result is not None

    def test_zero_variance_series(self):
        """Test with zero-variance series."""
        diagnostics = FeatureDiagnostics()
        zero_var = np.full(100, 5.0)

        # Should handle gracefully
        result = diagnostics.run_diagnostics(zero_var, name="zero_var")
        assert result is not None


class TestVerboseMode:
    """Tests for verbose logging."""

    def test_verbose_logging(self, caplog):
        """Test that verbose mode produces log output."""
        import logging

        caplog.set_level(logging.INFO)

        config = DiagnosticConfig(verbose=True)
        diagnostics = FeatureDiagnostics(config)

        np.random.seed(42)
        data = np.random.randn(100)

        diagnostics.run_diagnostics(data, name="test")

        # Check that log messages were produced
        # Note: May need to adjust based on actual logger configuration
        assert len(caplog.records) > 0
