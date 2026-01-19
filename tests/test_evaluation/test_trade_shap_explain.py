"""Tests for Trade-SHAP explanation logic.

Tests cover TradeShapExplainer initialization, single trade explanation,
batch explanation, and various missing_value_strategy modes.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.trade_shap.explain import TradeShapExplainer
from ml4t.diagnostic.evaluation.trade_shap.models import (
    TradeExplainFailure,
    TradeShapExplanation,
)


class MockTradeMetrics:
    """Mock TradeMetrics for testing without full dependency."""

    def __init__(self, timestamp: datetime, symbol: str = "TEST") -> None:
        self.timestamp = timestamp
        self.symbol = symbol
        self.entry_price = 100.0
        self.exit_price = 95.0
        self.pnl = -5.0
        self.duration = timedelta(hours=1)
        self.direction = "long"


@pytest.fixture
def sample_features() -> pl.DataFrame:
    """Create sample features DataFrame with timestamps."""
    n_samples = 100
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "feature_a": np.random.randn(n_samples),
            "feature_b": np.random.randn(n_samples),
            "feature_c": np.random.randn(n_samples),
        }
    )


@pytest.fixture
def sample_shap_values() -> np.ndarray:
    """Create SHAP values matching sample_features."""
    n_samples = 100
    n_features = 3
    return np.random.randn(n_samples, n_features)


@pytest.fixture
def feature_names() -> list[str]:
    """Feature names matching sample data."""
    return ["feature_a", "feature_b", "feature_c"]


class TestTradeShapExplainerInit:
    """Tests for TradeShapExplainer initialization."""

    def test_valid_inputs_creates_explainer(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Valid inputs should create explainer correctly."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
        )

        assert explainer.features_df is sample_features
        assert explainer.feature_names == feature_names
        assert explainer.top_n_features is None
        assert explainer.alignment_mode == "entry"
        assert explainer.missing_value_strategy == "skip"

    def test_shap_rows_mismatch_raises_value_error(self, sample_features, feature_names):
        """SHAP rows != features_df rows should raise ValueError."""
        wrong_shap = np.random.randn(50, 3)  # 50 rows, but features has 100

        with pytest.raises(ValueError, match="SHAP values rows.*!= features_df rows"):
            TradeShapExplainer(
                features_df=sample_features,
                shap_values=wrong_shap,
                feature_names=feature_names,
            )

    def test_shap_columns_mismatch_raises_value_error(self, sample_features, feature_names):
        """SHAP columns != feature_names should raise ValueError."""
        wrong_shap = np.random.randn(100, 5)  # 5 columns, but only 3 features

        with pytest.raises(ValueError, match="SHAP values columns.*!= feature_names"):
            TradeShapExplainer(
                features_df=sample_features,
                shap_values=wrong_shap,
                feature_names=feature_names,
            )

    def test_tolerance_parameter_stored(self, sample_features, sample_shap_values, feature_names):
        """Tolerance parameter should be passed to aligner."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            tolerance_seconds=120.0,
            alignment_mode="nearest",
        )

        assert explainer.aligner.tolerance_seconds == 120.0

    def test_entry_mode_ignores_tolerance(self, sample_features, sample_shap_values, feature_names):
        """Alignment mode 'entry' should use tolerance=0 regardless of parameter."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            tolerance_seconds=120.0,
            alignment_mode="entry",  # This mode ignores tolerance
        )

        assert explainer.aligner.tolerance_seconds == 0.0

    def test_top_n_features_stored(self, sample_features, sample_shap_values, feature_names):
        """top_n_features parameter should be stored."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            top_n_features=2,
        )

        assert explainer.top_n_features == 2


class TestExplainSingle:
    """Tests for explain() method on single trade."""

    def test_exact_match_returns_explanation(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Exact timestamp match should return TradeShapExplanation."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
        )

        # Trade at row 10's timestamp (exact match)
        trade_time = datetime(2024, 1, 1, 19, 0, 0)  # base + 10 hours
        trade = MockTradeMetrics(timestamp=trade_time, symbol="BTC")

        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        assert result.trade_id == f"BTC_{trade_time.isoformat()}"
        assert result.timestamp == trade_time
        assert len(result.top_features) == 3  # All features
        assert len(result.feature_values) == 3
        assert result.shap_vector.shape == (3,)

    def test_no_match_skip_returns_failure(
        self, sample_features, sample_shap_values, feature_names
    ):
        """No match with strategy='skip' should return TradeExplainFailure."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            missing_value_strategy="skip",
        )

        # Trade at non-matching timestamp
        trade_time = datetime(2024, 6, 1, 12, 0, 0)  # Way outside range
        trade = MockTradeMetrics(timestamp=trade_time)

        result = explainer.explain(trade)

        assert isinstance(result, TradeExplainFailure)
        assert result.reason == "alignment_missing"
        assert "distance_seconds" in result.details

    def test_no_match_error_raises_value_error(
        self, sample_features, sample_shap_values, feature_names
    ):
        """No match with strategy='error' should raise ValueError."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            missing_value_strategy="error",
        )

        trade_time = datetime(2024, 6, 1, 12, 0, 0)
        trade = MockTradeMetrics(timestamp=trade_time)

        with pytest.raises(ValueError, match="Cannot align SHAP values"):
            explainer.explain(trade)

    def test_no_match_zero_returns_zero_vector(
        self, sample_features, sample_shap_values, feature_names
    ):
        """No match with strategy='zero' should return zero SHAP vector."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            missing_value_strategy="zero",
        )

        trade_time = datetime(2024, 6, 1, 12, 0, 0)
        trade = MockTradeMetrics(timestamp=trade_time)

        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        np.testing.assert_array_equal(result.shap_vector, np.zeros(3))
        assert all(v == 0.0 for v in result.feature_values.values())

    def test_nearest_match_within_tolerance(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Nearest match within tolerance should return explanation."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            tolerance_seconds=1800.0,  # 30 minutes
            alignment_mode="nearest",
        )

        # Trade 15 minutes after row 10's timestamp
        trade_time = datetime(2024, 1, 1, 19, 15, 0)  # base + 10h 15min
        trade = MockTradeMetrics(timestamp=trade_time)

        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)

    def test_top_n_features_limits_output(self, sample_features, sample_shap_values, feature_names):
        """top_n_features should limit top_features output."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            top_n_features=2,
        )

        trade_time = datetime(2024, 1, 1, 19, 0, 0)
        trade = MockTradeMetrics(timestamp=trade_time)

        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        assert len(result.top_features) == 2  # Limited to 2

    def test_top_features_sorted_by_abs_value(self, sample_features, feature_names):
        """Top features should be sorted by absolute SHAP value."""
        # Create SHAP values with known ordering
        shap = np.zeros((100, 3))
        shap[10, :] = [0.1, -0.5, 0.3]  # feature_b has highest |SHAP|

        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=shap,
            feature_names=feature_names,
        )

        trade_time = datetime(2024, 1, 1, 19, 0, 0)  # Row 10
        trade = MockTradeMetrics(timestamp=trade_time)

        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        # feature_b (|-0.5|=0.5) > feature_c (|0.3|) > feature_a (|0.1|)
        assert result.top_features[0][0] == "feature_b"
        assert result.top_features[1][0] == "feature_c"
        assert result.top_features[2][0] == "feature_a"

    def test_feature_values_extracted_correctly(self, sample_features, feature_names):
        """Feature values should be extracted from the correct row."""
        shap = np.random.randn(100, 3)

        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=shap,
            feature_names=feature_names,
        )

        trade_time = datetime(2024, 1, 1, 19, 0, 0)  # Row 10
        trade = MockTradeMetrics(timestamp=trade_time)

        result = explainer.explain(trade)

        # Verify feature values match row 10 of DataFrame
        expected = sample_features.row(10, named=True)
        for name in feature_names:
            assert result.feature_values[name] == pytest.approx(expected[name])


class TestExplainMany:
    """Tests for explain_many() method."""

    def test_empty_trades_returns_empty_tuple(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Empty trades list should return ([], [])."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
        )

        explanations, failures = explainer.explain_many([])

        assert explanations == []
        assert failures == []

    def test_all_successful_returns_explanations_only(
        self, sample_features, sample_shap_values, feature_names
    ):
        """All matching trades should return only explanations."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
        )

        # Create trades that match existing timestamps
        trades = [
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 9 + i, 0, 0), symbol=f"SYM{i}")
            for i in range(5)
        ]

        explanations, failures = explainer.explain_many(trades)

        assert len(explanations) == 5
        assert len(failures) == 0
        assert all(isinstance(e, TradeShapExplanation) for e in explanations)

    def test_all_failures_returns_failures_only(
        self, sample_features, sample_shap_values, feature_names
    ):
        """All non-matching trades should return only failures."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            missing_value_strategy="skip",
        )

        # Create trades with non-matching timestamps
        trades = [
            MockTradeMetrics(timestamp=datetime(2025, 1, 1, i, 0, 0), symbol=f"SYM{i}")
            for i in range(3)
        ]

        explanations, failures = explainer.explain_many(trades)

        assert len(explanations) == 0
        assert len(failures) == 3
        assert all(isinstance(f, TradeExplainFailure) for f in failures)

    def test_mixed_results_separates_correctly(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Mixed matching/non-matching trades should be separated correctly."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            missing_value_strategy="skip",
        )

        trades = [
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 0, 0)),  # Match
            MockTradeMetrics(timestamp=datetime(2025, 6, 1, 0, 0, 0)),  # No match
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 15, 0, 0)),  # Match
            MockTradeMetrics(timestamp=datetime(2025, 6, 2, 0, 0, 0)),  # No match
        ]

        explanations, failures = explainer.explain_many(trades)

        assert len(explanations) == 2
        assert len(failures) == 2


class TestEdgeCases:
    """Additional edge case tests."""

    def test_single_feature(self):
        """Single feature should work correctly."""
        n_samples = 50
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]

        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "only_feature": np.random.randn(n_samples),
            }
        )
        shap_values = np.random.randn(n_samples, 1)

        explainer = TradeShapExplainer(
            features_df=features_df,
            shap_values=shap_values,
            feature_names=["only_feature"],
        )

        trade = MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 0, 0))
        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        assert len(result.top_features) == 1
        assert result.top_features[0][0] == "only_feature"

    def test_many_features(self):
        """Many features should work correctly."""
        n_samples = 50
        n_features = 100
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]

        feature_names = [f"feat_{i}" for i in range(n_features)]
        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                **{name: np.random.randn(n_samples) for name in feature_names},
            }
        )
        shap_values = np.random.randn(n_samples, n_features)

        explainer = TradeShapExplainer(
            features_df=features_df,
            shap_values=shap_values,
            feature_names=feature_names,
        )

        trade = MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 0, 0))
        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        assert len(result.top_features) == n_features
        assert len(result.feature_values) == n_features

    def test_single_row_dataframe(self):
        """Single row DataFrame should work."""
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 12, 0, 0)],
                "feat_a": [1.0],
                "feat_b": [2.0],
            }
        )
        shap_values = np.array([[0.5, -0.3]])

        explainer = TradeShapExplainer(
            features_df=features_df,
            shap_values=shap_values,
            feature_names=["feat_a", "feat_b"],
        )

        trade = MockTradeMetrics(timestamp=datetime(2024, 1, 1, 12, 0, 0))
        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        assert result.shap_vector[0] == pytest.approx(0.5)
        assert result.shap_vector[1] == pytest.approx(-0.3)

    def test_boundary_timestamp_first_row(self, sample_features, sample_shap_values, feature_names):
        """First row timestamp should match correctly."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
        )

        # First row timestamp
        trade = MockTradeMetrics(timestamp=datetime(2024, 1, 1, 9, 0, 0))
        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)

    def test_boundary_timestamp_last_row(self, sample_features, sample_shap_values, feature_names):
        """Last row timestamp should match correctly."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
        )

        # Last row timestamp: base + 99 hours
        trade = MockTradeMetrics(timestamp=datetime(2024, 1, 5, 12, 0, 0))
        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)

    def test_trade_id_format(self, sample_features, sample_shap_values, feature_names):
        """Trade ID should be formatted as symbol_timestamp."""
        explainer = TradeShapExplainer(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
        )

        trade_time = datetime(2024, 1, 1, 10, 0, 0)
        trade = MockTradeMetrics(timestamp=trade_time, symbol="AAPL")
        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        assert result.trade_id == f"AAPL_{trade_time.isoformat()}"

    def test_zero_shap_values(self):
        """Zero SHAP values should be handled correctly."""
        n_samples = 10
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]

        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "feat": np.random.randn(n_samples),
            }
        )
        shap_values = np.zeros((n_samples, 1))

        explainer = TradeShapExplainer(
            features_df=features_df,
            shap_values=shap_values,
            feature_names=["feat"],
        )

        trade = MockTradeMetrics(timestamp=datetime(2024, 1, 1, 5, 0, 0))
        result = explainer.explain(trade)

        assert isinstance(result, TradeShapExplanation)
        assert result.shap_vector[0] == 0.0
        assert result.top_features[0][1] == 0.0
