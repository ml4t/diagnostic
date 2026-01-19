"""Tests for ValidatedCrossValidation combining CPCV with DSR."""

from __future__ import annotations

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.validated_cv import (
    ModelProtocol,
    ValidatedCrossValidation,
    ValidatedCrossValidationConfig,
    ValidationFoldResult,
    ValidationResult,
    validated_cross_val_score,
)


class SimpleModel:
    """Simple model for testing."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleModel:
        """Fit model (just store mean for predictions)."""
        self.mean_ = np.mean(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stored mean."""
        return np.full(len(X), self.mean_)


class SignModel:
    """Model that predicts sign of mean."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> SignModel:
        """Fit model."""
        self.coef_ = np.mean(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict sign."""
        return np.sign(X[:, 0])


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    # Add some signal
    y = 0.3 * X[:, 0] + 0.1 * X[:, 1] + np.random.randn(n_samples) * 0.5
    times = np.arange(n_samples)

    return X, y, times


@pytest.fixture
def positive_sharpe_data():
    """Data that will produce positive Sharpe ratios."""
    np.random.seed(123)
    n_samples = 600
    n_features = 3

    X = np.random.randn(n_samples, n_features)
    # Strong signal that will produce positive Sharpe
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n_samples) * 0.3
    times = np.arange(n_samples)

    return X, y, times


class TestValidatedCrossValidationConfig:
    """Tests for configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ValidatedCrossValidationConfig()
        assert config.n_groups == 10
        assert config.n_test_groups == 2
        assert config.embargo_pct == 0.01
        assert config.label_horizon == 0
        assert config.sharpe_star == 0.0
        assert config.significance_level == 0.95
        assert config.annualization_factor == 252.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ValidatedCrossValidationConfig(
            n_groups=5,
            n_test_groups=1,
            embargo_pct=0.02,
            label_horizon=5,
            sharpe_star=0.5,
            significance_level=0.99,
        )
        assert config.n_groups == 5
        assert config.n_test_groups == 1
        assert config.embargo_pct == 0.02
        assert config.label_horizon == 5
        assert config.sharpe_star == 0.5
        assert config.significance_level == 0.99

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            ValidatedCrossValidationConfig(n_groups=1)  # Must be >= 2

        with pytest.raises(ValueError):
            ValidatedCrossValidationConfig(embargo_pct=-0.1)

        with pytest.raises(ValueError):
            ValidatedCrossValidationConfig(significance_level=0.3)  # Must be >= 0.5


class TestValidationFoldResult:
    """Tests for fold result dataclass."""

    def test_fold_result_creation(self):
        """Test creating fold result."""
        returns = np.array([0.01, -0.02, 0.015, 0.005])
        result = ValidationFoldResult(
            fold_idx=0,
            train_size=100,
            test_size=20,
            sharpe_ratio=0.5,
            returns=returns,
        )
        assert result.fold_idx == 0
        assert result.train_size == 100
        assert result.test_size == 20
        assert result.sharpe_ratio == 0.5
        assert len(result.returns) == 4
        assert result.predictions is None

    def test_fold_result_with_predictions(self):
        """Test fold result with predictions."""
        returns = np.array([0.01, -0.02])
        predictions = np.array([1.0, -1.0])
        result = ValidationFoldResult(
            fold_idx=1,
            train_size=50,
            test_size=10,
            sharpe_ratio=0.3,
            returns=returns,
            predictions=predictions,
        )
        assert result.predictions is not None
        assert len(result.predictions) == 2


class TestValidationResult:
    """Tests for validation result dataclass."""

    def test_empty_result(self):
        """Test empty result."""
        result = ValidationResult()
        assert result.n_folds == 0
        assert result.mean_sharpe == 0.0
        assert not result.is_significant

    def test_result_with_folds(self):
        """Test result with fold data."""
        fold_results = [
            ValidationFoldResult(0, 100, 20, 0.5, np.array([0.01])),
            ValidationFoldResult(1, 100, 20, 0.6, np.array([0.02])),
            ValidationFoldResult(2, 100, 20, 0.4, np.array([0.015])),
        ]
        result = ValidationResult(
            fold_results=fold_results,
            n_folds=3,
            mean_sharpe=0.5,
            std_sharpe=0.1,
            dsr=0.85,
            dsr_zscore=1.5,
            expected_max_sharpe=0.4,
            is_significant=False,
            significance_level=0.95,
            interpretation=["Test interpretation"],
        )
        assert result.n_folds == 3
        assert result.mean_sharpe == 0.5
        assert result.dsr == 0.85
        assert not result.is_significant

    def test_summary(self):
        """Test summary generation."""
        result = ValidationResult(
            n_folds=5,
            mean_sharpe=0.6,
            std_sharpe=0.15,
            dsr=0.92,
            dsr_zscore=1.8,
            expected_max_sharpe=0.45,
            is_significant=False,
            significance_level=0.95,
            interpretation=["Strategy shows promise"],
        )
        summary = result.summary()
        assert "Folds completed: 5" in summary
        assert "Mean Sharpe" in summary
        assert "DSR" in summary
        assert "0.92" in summary
        assert "NO" in summary  # Not significant

    def test_summary_significant(self):
        """Test summary when significant."""
        result = ValidationResult(
            n_folds=10,
            mean_sharpe=1.2,
            std_sharpe=0.2,
            dsr=0.98,
            dsr_zscore=2.5,
            expected_max_sharpe=0.8,
            is_significant=True,
            significance_level=0.95,
            interpretation=["Excellent strategy"],
        )
        summary = result.summary()
        assert "YES" in summary

    def test_to_dict(self):
        """Test dictionary export."""
        fold_results = [
            ValidationFoldResult(0, 100, 20, 0.5, np.array([0.01])),
            ValidationFoldResult(1, 100, 20, 0.7, np.array([0.02])),
        ]
        result = ValidationResult(
            fold_results=fold_results,
            n_folds=2,
            mean_sharpe=0.6,
            std_sharpe=0.1,
            dsr=0.85,
            dsr_zscore=1.5,
            expected_max_sharpe=0.4,
            is_significant=False,
            significance_level=0.95,
            interpretation=["Test"],
        )
        d = result.to_dict()
        assert d["n_folds"] == 2
        assert d["mean_sharpe"] == 0.6
        assert d["dsr"] == 0.85
        assert d["fold_sharpes"] == [0.5, 0.7]


class TestValidatedCrossValidation:
    """Tests for ValidatedCrossValidation class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        vcv = ValidatedCrossValidation()
        assert vcv.config.n_groups == 10
        assert vcv._cv is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ValidatedCrossValidationConfig(n_groups=5, embargo_pct=0.02)
        vcv = ValidatedCrossValidation(config)
        assert vcv.config.n_groups == 5
        assert vcv.config.embargo_pct == 0.02

    def test_fit_evaluate_basic(self, sample_data):
        """Test basic fit_evaluate workflow."""
        X, y, times = sample_data
        config = ValidatedCrossValidationConfig(n_groups=5, n_test_groups=1)
        vcv = ValidatedCrossValidation(config)
        model = SimpleModel()

        result = vcv.fit_evaluate(X, y, model, times=times)

        assert isinstance(result, ValidationResult)
        assert result.n_folds > 0
        assert len(result.fold_results) > 0
        assert result.dsr >= 0
        assert result.dsr <= 1

    def test_fit_evaluate_with_returns_fn(self, sample_data):
        """Test fit_evaluate with custom returns function."""
        X, y, times = sample_data
        config = ValidatedCrossValidationConfig(n_groups=5, n_test_groups=1)
        vcv = ValidatedCrossValidation(config)
        model = SignModel()

        def compute_returns(y_true, y_pred):
            """Custom returns computation."""
            positions = np.sign(y_pred)
            return positions * y_true * 0.01

        result = vcv.fit_evaluate(X, y, model, times=times, returns_fn=compute_returns)

        assert isinstance(result, ValidationResult)
        assert result.n_folds > 0

    def test_fit_evaluate_polars_input(self, sample_data):
        """Test fit_evaluate with Polars input."""
        import polars as pl

        X, y, times = sample_data
        X_pl = pl.DataFrame(X, schema=[f"f{i}" for i in range(X.shape[1])])
        y_pl = pl.Series("target", y)
        times_pl = pl.Series("time", times)

        config = ValidatedCrossValidationConfig(n_groups=5, n_test_groups=1)
        vcv = ValidatedCrossValidation(config)
        model = SimpleModel()

        result = vcv.fit_evaluate(X_pl, y_pl, model, times=times_pl)

        assert isinstance(result, ValidationResult)
        assert result.n_folds > 0

    def test_fit_evaluate_without_times(self, sample_data):
        """Test fit_evaluate without times array."""
        X, y, _ = sample_data
        config = ValidatedCrossValidationConfig(n_groups=5, n_test_groups=1)
        vcv = ValidatedCrossValidation(config)
        model = SimpleModel()

        result = vcv.fit_evaluate(X, y, model)

        assert isinstance(result, ValidationResult)
        assert result.n_folds > 0

    def test_evaluate_sharpes(self):
        """Test evaluation of pre-computed Sharpe ratios."""
        vcv = ValidatedCrossValidation()
        sharpes = [0.5, 0.6, 0.4, 0.7, 0.55, 0.45]

        result = vcv.evaluate_sharpes(sharpes)

        assert result.n_folds == 6
        assert result.mean_sharpe == pytest.approx(np.mean(sharpes), rel=1e-6)
        assert result.std_sharpe == pytest.approx(np.std(sharpes, ddof=1), rel=1e-6)
        assert 0 <= result.dsr <= 1

    def test_evaluate_sharpes_single(self):
        """Test evaluation with single Sharpe ratio."""
        vcv = ValidatedCrossValidation()
        sharpes = [1.0]

        result = vcv.evaluate_sharpes(sharpes)

        assert result.n_folds == 1
        assert result.std_sharpe == 0.0  # No variance with single sample

    def test_evaluate_sharpes_negative(self):
        """Test evaluation with negative Sharpe ratios."""
        vcv = ValidatedCrossValidation()
        sharpes = [-0.2, -0.1, -0.3, -0.15]

        result = vcv.evaluate_sharpes(sharpes)

        assert result.n_folds == 4
        assert result.mean_sharpe < 0
        assert not result.is_significant

    def test_evaluate_sharpes_high_sharpe(self):
        """Test evaluation with high Sharpe ratios."""
        config = ValidatedCrossValidationConfig(significance_level=0.90)
        vcv = ValidatedCrossValidation(config)
        sharpes = [2.0, 2.5, 1.8, 2.2, 2.1]

        result = vcv.evaluate_sharpes(sharpes)

        assert result.n_folds == 5
        assert result.mean_sharpe > 1.5
        assert result.dsr > 0.5  # Should have good DSR

    def test_compute_sharpe(self):
        """Test internal Sharpe computation."""
        vcv = ValidatedCrossValidation()
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])

        sharpe = vcv._compute_sharpe(returns)

        # Should be annualized
        expected_daily = np.mean(returns) / np.std(returns, ddof=1)
        expected_annual = expected_daily * np.sqrt(252)
        assert sharpe == pytest.approx(expected_annual, rel=1e-6)

    def test_compute_sharpe_edge_cases(self):
        """Test Sharpe computation edge cases."""
        vcv = ValidatedCrossValidation()

        # Empty returns
        assert vcv._compute_sharpe(np.array([])) == 0.0

        # Single return
        assert vcv._compute_sharpe(np.array([0.01])) == 0.0

        # Zero variance
        assert vcv._compute_sharpe(np.array([0.01, 0.01, 0.01])) == 0.0

    def test_interpretation_significant(self):
        """Test interpretation for significant result."""
        config = ValidatedCrossValidationConfig(significance_level=0.95)
        vcv = ValidatedCrossValidation(config)

        interp = vcv._generate_interpretation(
            mean_sharpe=0.8,
            max_sharpe=1.2,
            expected_max=0.5,
            dsr=0.98,
            is_significant=True,
        )

        assert any("statistically significant" in i for i in interp)
        assert any("paper trading" in i.lower() for i in interp)

    def test_interpretation_not_significant(self):
        """Test interpretation for non-significant result."""
        vcv = ValidatedCrossValidation()

        interp = vcv._generate_interpretation(
            mean_sharpe=0.2,
            max_sharpe=0.5,
            expected_max=0.6,
            dsr=0.6,
            is_significant=False,
        )

        assert any("NOT significant" in i for i in interp)
        assert any("overfit" in i.lower() for i in interp)

    def test_interpretation_overfitting_warning(self):
        """Test overfitting warning in interpretation."""
        vcv = ValidatedCrossValidation()

        interp = vcv._generate_interpretation(
            mean_sharpe=0.5,
            max_sharpe=1.5,
            expected_max=0.4,
            dsr=0.7,
            is_significant=False,
        )

        assert any("overfitting" in i.lower() for i in interp)


class TestValidatedCrossValScore:
    """Tests for the convenience function."""

    def test_basic_usage(self, sample_data):
        """Test basic usage of convenience function."""
        X, y, times = sample_data
        model = SimpleModel()

        result = validated_cross_val_score(model, X, y, times=times, n_groups=5)

        assert isinstance(result, ValidationResult)
        assert result.n_folds > 0

    def test_with_custom_params(self, sample_data):
        """Test with custom parameters."""
        X, y, times = sample_data
        model = SimpleModel()

        result = validated_cross_val_score(model, X, y, times=times, n_groups=8, embargo_pct=0.02)

        assert isinstance(result, ValidationResult)
        assert result.n_folds > 0


class TestModelProtocol:
    """Tests for ModelProtocol."""

    def test_simple_model_conforms(self):
        """Test that SimpleModel conforms to protocol."""
        model = SimpleModel()
        assert isinstance(model, ModelProtocol)

    def test_sign_model_conforms(self):
        """Test that SignModel conforms to protocol."""
        model = SignModel()
        assert isinstance(model, ModelProtocol)

    def test_sklearn_model_conforms(self):
        """Test that sklearn models conform to protocol."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=10)
            assert isinstance(model, ModelProtocol)
        except ImportError:
            pytest.skip("sklearn not available")


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, positive_sharpe_data):
        """Test complete workflow from data to interpretation."""
        X, y, times = positive_sharpe_data
        config = ValidatedCrossValidationConfig(
            n_groups=6,
            n_test_groups=1,
            embargo_pct=0.01,
            significance_level=0.90,
        )
        vcv = ValidatedCrossValidation(config)
        model = SignModel()

        result = vcv.fit_evaluate(X, y, model, times=times)

        # Check all components present
        assert result.n_folds > 0
        assert len(result.fold_results) > 0
        assert result.mean_sharpe != 0
        assert result.dsr >= 0
        assert result.dsr <= 1
        assert len(result.interpretation) > 0

        # Check summary works
        summary = result.summary()
        assert "Validated Cross-Validation Results" in summary

        # Check to_dict works
        d = result.to_dict()
        assert "fold_sharpes" in d
        assert len(d["fold_sharpes"]) == result.n_folds

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same seed."""
        X, y, times = sample_data
        config = ValidatedCrossValidationConfig(n_groups=5, n_test_groups=1)

        # Run twice with same model
        vcv1 = ValidatedCrossValidation(config)
        vcv2 = ValidatedCrossValidation(config)

        model1 = SimpleModel()
        model2 = SimpleModel()

        result1 = vcv1.fit_evaluate(X, y, model1, times=times)
        result2 = vcv2.fit_evaluate(X, y, model2, times=times)

        # Should get same results (deterministic CV splits)
        assert result1.n_folds == result2.n_folds
        assert result1.mean_sharpe == pytest.approx(result2.mean_sharpe, rel=1e-6)
