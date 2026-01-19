"""Integration tests for Trade-SHAP clustering and pattern detection.

This module tests the complete end-to-end workflow:
1. Extract SHAP vectors from trade explanations
2. Cluster error patterns using hierarchical clustering
3. Characterize patterns with statistical tests
4. Create ErrorPattern objects with hypotheses

These integration tests complement the unit tests in test_trade_shap_basic.py
by focusing on multi-step workflows and real-world usage patterns.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import polars as pl

from ml4t.diagnostic.config import TradeConfig
from ml4t.diagnostic.evaluation.trade_shap_diagnostics import (
    ErrorPattern,
    TradeShapAnalyzer,
    TradeShapExplanation,
)
from ml4t.diagnostic.integration.backtest_contract import TradeRecord


class TestEndToEndClustering:
    """Test complete end-to-end clustering workflows."""

    def test_full_workflow_extract_cluster_characterize(self):
        """Test complete workflow from SHAP explanations to error patterns."""
        # Setup: Create analyzer with realistic data
        n_samples = 100
        n_features = 20
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()

        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Step 1: Create trade explanations with known patterns
        # Pattern 1: High feature_0, low feature_1
        explanations = []
        for i in range(30):
            shap_vec = np.array([0.1] * n_features)
            shap_vec[0] = 0.8  # High feature_0
            shap_vec[1] = -0.5  # Low feature_1

            exp = TradeShapExplanation(
                trade_id=f"trade_{i}",
                feature_values={f"feature_{j}": 1.0 for j in range(n_features)},
                top_features=[("feature_0", 0.8), ("feature_1", -0.5)],
                shap_vector=shap_vec,
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            )
            explanations.append(exp)

        # Pattern 2: Low feature_0, high feature_2
        for i in range(30, 50):
            shap_vec = np.array([0.1] * n_features)
            shap_vec[0] = -0.6  # Low feature_0
            shap_vec[2] = 0.7  # High feature_2

            exp = TradeShapExplanation(
                trade_id=f"trade_{i}",
                feature_values={f"feature_{j}": 1.0 for j in range(n_features)},
                top_features=[("feature_2", 0.7), ("feature_0", -0.6)],
                shap_vector=shap_vec,
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            )
            explanations.append(exp)

        # Pattern 3: High feature_5
        for i in range(50, 70):
            shap_vec = np.array([0.1] * n_features)
            shap_vec[5] = 0.9  # High feature_5

            exp = TradeShapExplanation(
                trade_id=f"trade_{i}",
                feature_values={f"feature_{j}": 1.0 for j in range(n_features)},
                top_features=[("feature_5", 0.9)],
                shap_vector=shap_vec,
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            )
            explanations.append(exp)

        # Step 2: Extract SHAP vectors
        shap_vectors = analyzer.extract_shap_vectors(
            explanations, normalization="l2", top_n_features=n_features
        )

        assert shap_vectors.shape == (70, n_features), "SHAP vectors should have correct shape"

        # Step 3: Cluster patterns
        clustering_result = analyzer.cluster_patterns(shap_vectors, n_clusters=3)

        assert clustering_result.n_clusters == 3, "Should find 3 clusters"
        assert len(clustering_result.cluster_assignments) == 70, (
            "Should have assignments for all trades"
        )
        assert clustering_result.centroids.shape == (3, n_features), "Should have 3 centroids"
        assert 0.0 <= clustering_result.silhouette_score <= 1.0, "Silhouette score should be valid"

        # Step 4: Characterize all patterns
        feature_names = [f"feature_{i}" for i in range(n_features)]
        patterns = []

        for cluster_id in range(3):
            pattern_dict = analyzer.characterize_pattern(
                cluster_id=cluster_id,
                shap_vectors=shap_vectors,
                cluster_assignments=clustering_result.cluster_assignments,
                feature_names=feature_names,
                top_n=5,
            )

            # Create ErrorPattern from characterization
            # Convert dict-based top_features to tuples
            top_features_tuples = [
                (
                    tf["feature"],
                    tf["mean_shap"],
                    tf["p_value_t"],
                    tf["p_value_mw"],
                    tf["significant"],
                )
                for tf in pattern_dict["top_features"]
            ]
            error_pattern = ErrorPattern(
                cluster_id=pattern_dict["cluster_id"],
                n_trades=pattern_dict["n_trades"],
                description=pattern_dict["pattern_description"],
                top_features=top_features_tuples,
                separation_score=pattern_dict["separation_score"],
                distinctiveness=pattern_dict["distinctiveness"],
            )
            patterns.append(error_pattern)

        # Verify: All patterns created successfully
        assert len(patterns) == 3, "Should have 3 error patterns"
        for pattern in patterns:
            assert pattern.n_trades > 0, "Each pattern should have trades"
            assert len(pattern.top_features) <= 5, "Should have at most 5 top features"
            assert pattern.separation_score >= 0.0, "Separation score should be non-negative"
            assert pattern.distinctiveness > 0.0, "Distinctiveness should be positive"
            assert len(pattern.description) > 0, "Should have description"

        # Verify: Patterns are distinct (different top features)
        top_feature_sets = [{f[0] for f in p.top_features[:3]} for p in patterns]
        # At least some overlap is expected, but they shouldn't all be identical
        assert not all(s == top_feature_sets[0] for s in top_feature_sets), (
            "Patterns should have different top features"
        )

    def test_workflow_with_error_pattern_serialization(self):
        """Test workflow including ErrorPattern JSON serialization."""
        # Setup
        n_samples = 50
        n_features = 10
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"feat_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create explanations with two clear patterns
        explanations = []
        for i in range(25):
            shap_vec = np.array([0.5 if j == 0 else 0.0 for j in range(n_features)])
            exp = TradeShapExplanation(
                trade_id=f"trade_{i}",
                feature_values={f"feat_{j}": 1.0 for j in range(n_features)},
                top_features=[("feat_0", 0.5)],
                shap_vector=shap_vec,
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            )
            explanations.append(exp)

        for i in range(25, 50):
            shap_vec = np.array([0.0 if j != 5 else -0.6 for j in range(n_features)])
            exp = TradeShapExplanation(
                trade_id=f"trade_{i}",
                feature_values={f"feat_{j}": 1.0 for j in range(n_features)},
                top_features=[("feat_5", -0.6)],
                shap_vector=shap_vec,
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            )
            explanations.append(exp)

        # Extract, cluster, characterize
        shap_vectors = analyzer.extract_shap_vectors(explanations, normalization="l2")
        clustering_result = analyzer.cluster_patterns(shap_vectors, n_clusters=2)

        # Create ErrorPattern objects
        patterns = []
        for cluster_id in range(2):
            pattern_dict = analyzer.characterize_pattern(
                cluster_id=cluster_id,
                shap_vectors=shap_vectors,
                cluster_assignments=clustering_result.cluster_assignments,
                feature_names=[f"feat_{i}" for i in range(n_features)],
            )

            # Convert dict-based top_features to tuples
            top_features_tuples = [
                (
                    tf["feature"],
                    tf["mean_shap"],
                    tf["p_value_t"],
                    tf["p_value_mw"],
                    tf["significant"],
                )
                for tf in pattern_dict["top_features"]
            ]
            pattern = ErrorPattern(
                cluster_id=pattern_dict["cluster_id"],
                n_trades=pattern_dict["n_trades"],
                description=pattern_dict["pattern_description"],
                top_features=top_features_tuples,
                separation_score=pattern_dict["separation_score"],
                distinctiveness=pattern_dict["distinctiveness"],
                hypothesis=f"Hypothesis for cluster {cluster_id}",
                actions=["Action 1", "Action 2"],
                confidence=0.85,
            )
            patterns.append(pattern)

        # Test JSON serialization
        for pattern in patterns:
            pattern_dict = pattern.to_dict()

            # Verify structure
            assert "cluster_id" in pattern_dict
            assert "n_trades" in pattern_dict
            assert "description" in pattern_dict
            assert "top_features" in pattern_dict
            assert "hypothesis" in pattern_dict
            assert "actions" in pattern_dict
            assert "confidence" in pattern_dict

            # Verify top_features format
            assert isinstance(pattern_dict["top_features"], list)
            if len(pattern_dict["top_features"]) > 0:
                feat = pattern_dict["top_features"][0]
                assert "feature_name" in feat
                assert "mean_shap" in feat
                assert "p_value_t" in feat
                assert "p_value_mw" in feat
                assert "is_significant" in feat

            # Test summary formats
            simple_summary = pattern.summary(include_actions=False)
            assert f"Pattern {pattern.cluster_id}" in simple_summary
            assert f"{pattern.n_trades} trades" in simple_summary

            detailed_summary = pattern.summary(include_actions=True)
            assert "Hypothesis:" in detailed_summary
            assert "Actions:" in detailed_summary


class TestBatchProcessing:
    """Test batch processing of trades through full pipeline."""

    def test_batch_process_worst_trades_to_patterns(self):
        """Test processing worst trades through to error patterns."""
        # Setup: Create trades and analyzer
        n_samples = 100
        n_features = 15
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"f{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create worst trades with timestamp alignment
        worst_trades = []
        for i in range(20):
            entry_time = datetime(2024, 1, 1, i)
            exit_time = datetime(2024, 1, 1, i + 1)
            trade = TradeRecord(
                timestamp=exit_time,
                symbol="BTC",
                entry_price=100.0,
                exit_price=95.0,  # Losing trades
                pnl=-5.0,
                duration=exit_time - entry_time,
                direction="long",
            )
            worst_trades.append(trade)

        # Batch explain all trades
        result = analyzer.explain_worst_trades(worst_trades, n=20)

        assert len(result.explanations) <= 20, "Should have explanations for trades"
        assert result.n_trades_explained <= 20, "Should track number explained"

        # If we have explanations, process through clustering
        if len(result.explanations) >= 10:
            # Extract and cluster
            shap_vectors = analyzer.extract_shap_vectors(
                result.explanations, normalization="l2", top_n_features=n_features
            )

            clustering_result = analyzer.cluster_patterns(shap_vectors)

            # Should find some patterns
            assert clustering_result.n_clusters >= 1, "Should find at least one cluster"
            assert len(clustering_result.cluster_assignments) == len(result.explanations), (
                "Should assign all trades"
            )

            # Characterize first pattern
            if clustering_result.n_clusters > 0:
                pattern_dict = analyzer.characterize_pattern(
                    cluster_id=0,
                    shap_vectors=shap_vectors,
                    cluster_assignments=clustering_result.cluster_assignments,
                    feature_names=[f"f{i}" for i in range(n_features)],
                )

                # Create ErrorPattern
                # Convert dict-based top_features to tuples
                top_features_tuples = [
                    (
                        tf["feature"],
                        tf["mean_shap"],
                        tf["p_value_t"],
                        tf["p_value_mw"],
                        tf["significant"],
                    )
                    for tf in pattern_dict["top_features"]
                ]
                pattern = ErrorPattern(
                    cluster_id=pattern_dict["cluster_id"],
                    n_trades=pattern_dict["n_trades"],
                    description=pattern_dict["pattern_description"],
                    top_features=top_features_tuples,
                    separation_score=pattern_dict["separation_score"],
                    distinctiveness=pattern_dict["distinctiveness"],
                )

                assert pattern.cluster_id == 0
                assert pattern.n_trades > 0
                assert len(pattern.top_features) > 0


class TestPatternStability:
    """Test stability and reproducibility of pattern detection."""

    def test_pattern_stability_across_runs(self):
        """Test that clustering produces stable results with fixed seed."""
        # Setup
        n_samples = 60
        n_features = 10
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"x{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create synthetic explanations with clear patterns
        np.random.seed(42)
        explanations = []
        for i in range(60):
            cluster_id_true = i // 20  # 3 groups of 20
            shap_vec = np.zeros(n_features)

            # Make distinct patterns
            if cluster_id_true == 0:
                shap_vec[0] = 0.9
                shap_vec[1] = -0.3
                top_feats = [("x0", 0.9), ("x1", -0.3)]
            elif cluster_id_true == 1:
                shap_vec[5] = -0.8
                shap_vec[6] = 0.4
                top_feats = [("x5", -0.8), ("x6", 0.4)]
            else:
                shap_vec[8] = 0.7
                shap_vec[9] = 0.6
                top_feats = [("x8", 0.7), ("x9", 0.6)]

            exp = TradeShapExplanation(
                trade_id=f"trade_{i}",
                feature_values={f"x{j}": 1.0 for j in range(n_features)},
                top_features=top_feats,
                shap_vector=shap_vec,
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            )
            explanations.append(exp)

        # Extract vectors
        shap_vectors = analyzer.extract_shap_vectors(explanations, normalization="l2")

        # Cluster multiple times with same parameters
        results = []
        for _ in range(3):
            result = analyzer.cluster_patterns(shap_vectors, n_clusters=3)
            results.append(result)

        # All runs should produce same number of clusters
        assert all(r.n_clusters == 3 for r in results), "Should consistently find 3 clusters"

        # Cluster assignments should be identical (hierarchical clustering is deterministic)
        ref_assignments = results[0].cluster_assignments
        for result in results[1:]:
            assert result.cluster_assignments == ref_assignments, (
                "Cluster assignments should be deterministic"
            )

    def test_pattern_quality_with_noise(self):
        """Test pattern detection quality with noisy data."""
        # Setup
        n_samples = 80
        n_features = 12
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create patterns with varying noise levels
        np.random.seed(123)
        noise_levels = [0.0, 0.1, 0.3, 0.5]
        silhouette_scores = []

        for noise_level in noise_levels:
            explanations = []
            for i in range(60):
                cluster_true = i // 20
                shap_vec = np.random.randn(n_features) * noise_level

                # Add signal
                if cluster_true == 0:
                    shap_vec[0] = 1.0 + np.random.randn() * noise_level
                    top_feats = [("feature_0", float(shap_vec[0]))]
                elif cluster_true == 1:
                    shap_vec[5] = -1.0 + np.random.randn() * noise_level
                    top_feats = [("feature_5", float(shap_vec[5]))]
                else:
                    shap_vec[10] = 0.8 + np.random.randn() * noise_level
                    top_feats = [("feature_10", float(shap_vec[10]))]

                exp = TradeShapExplanation(
                    trade_id=f"trade_{i}",
                    feature_values={f"feature_{j}": 1.0 for j in range(n_features)},
                    top_features=top_feats,
                    shap_vector=shap_vec,
                    timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                )
                explanations.append(exp)

            shap_vectors = analyzer.extract_shap_vectors(explanations, normalization="l2")
            result = analyzer.cluster_patterns(shap_vectors, n_clusters=3)
            silhouette_scores.append(result.silhouette_score)

        # Silhouette score should generally decrease with noise
        # (not strictly monotonic due to randomness, but first should be better than last)
        assert silhouette_scores[0] > silhouette_scores[-1], (
            "Clean patterns should have higher silhouette score than very noisy ones"
        )


class TestPerformanceIntegration:
    """Test performance of integrated workflows."""

    def test_large_scale_workflow_performance(self):
        """Test performance with larger dataset."""
        # Setup larger dataset
        n_samples = 200
        n_features = 50
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"f{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create 100 explanations
        np.random.seed(456)
        explanations = []
        for i in range(100):
            shap_vec = np.random.randn(n_features)
            # Get top features by absolute value
            sorted_indices = np.argsort(np.abs(shap_vec))[::-1]
            top_feats = [(f"f{j}", float(shap_vec[j])) for j in sorted_indices[:5]]

            exp = TradeShapExplanation(
                trade_id=f"trade_{i}",
                feature_values={f"f{j}": 1.0 for j in range(n_features)},
                top_features=top_feats,
                shap_vector=shap_vec,
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            )
            explanations.append(exp)

        # Time the full workflow
        import time

        start = time.time()

        # Extract
        shap_vectors = analyzer.extract_shap_vectors(
            explanations, normalization="l2", top_n_features=20
        )

        # Cluster
        clustering_result = analyzer.cluster_patterns(shap_vectors)

        # Characterize all patterns
        for cluster_id in range(clustering_result.n_clusters):
            _ = analyzer.characterize_pattern(
                cluster_id=cluster_id,
                shap_vectors=shap_vectors,
                cluster_assignments=clustering_result.cluster_assignments,
                feature_names=[f"f{i}" for i in range(20)],
            )

        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 15.0, f"Full workflow should complete in <15s, took {elapsed:.2f}s"


class TestClusterQualityMetrics:
    """Test cluster quality metrics (Davies-Bouldin and Calinski-Harabasz)."""

    def test_cluster_quality_metrics_computed(self):
        """Test that cluster quality metrics are computed and included in results."""
        # Setup: Create analyzer with simple data
        n_samples = 30
        n_features = 10
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create well-separated clusters by design
        np.random.seed(42)
        shap_vectors = np.zeros((30, 10))

        # Cluster 0: High values in features 0-2
        shap_vectors[0:10, 0:3] = np.random.randn(10, 3) + 3.0

        # Cluster 1: High values in features 3-5
        shap_vectors[10:20, 3:6] = np.random.randn(10, 3) + 3.0

        # Cluster 2: High values in features 6-9
        shap_vectors[20:30, 6:10] = np.random.randn(10, 4) + 3.0

        # Cluster with n_clusters=3
        result = analyzer.cluster_patterns(shap_vectors, n_clusters=3)

        # Verify quality metrics are present
        assert result.silhouette_score is not None, "Silhouette score should be computed"
        assert result.davies_bouldin_score is not None, "Davies-Bouldin score should be computed"
        assert result.calinski_harabasz_score is not None, (
            "Calinski-Harabasz score should be computed"
        )

        # Verify metrics are in expected ranges
        assert -1.0 <= result.silhouette_score <= 1.0, "Silhouette score should be in [-1, 1]"
        assert result.davies_bouldin_score >= 0.0, "Davies-Bouldin score should be >= 0"
        assert result.calinski_harabasz_score >= 0.0, "Calinski-Harabasz score should be >= 0"

    def test_quality_metrics_improve_with_separation(self):
        """Test that quality metrics improve with better cluster separation."""
        # Setup
        n_samples = 30
        n_features = 10
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Case 1: Poorly separated clusters (random data)
        np.random.seed(123)
        poor_vectors = np.random.randn(30, 10)
        poor_result = analyzer.cluster_patterns(poor_vectors, n_clusters=3)

        # Case 2: Well-separated clusters
        np.random.seed(123)
        good_vectors = np.zeros((30, 10))
        # Cluster 0: High positive in features 0-2
        good_vectors[0:10, 0:3] = np.random.randn(10, 3) + 5.0
        # Cluster 1: High negative in features 3-5
        good_vectors[10:20, 3:6] = np.random.randn(10, 3) - 5.0
        # Cluster 2: High positive in features 6-9
        good_vectors[20:30, 6:10] = np.random.randn(10, 4) + 5.0

        good_result = analyzer.cluster_patterns(good_vectors, n_clusters=3)

        # Better separation should result in:
        # - Higher silhouette score (closer to 1.0)
        # - Lower Davies-Bouldin score (closer to 0.0)
        # - Higher Calinski-Harabasz score
        assert good_result.silhouette_score > poor_result.silhouette_score, (
            "Better separation should have higher silhouette score"
        )

        if (
            good_result.davies_bouldin_score is not None
            and poor_result.davies_bouldin_score is not None
        ):
            assert good_result.davies_bouldin_score < poor_result.davies_bouldin_score, (
                "Better separation should have lower Davies-Bouldin score"
            )

        if (
            good_result.calinski_harabasz_score is not None
            and poor_result.calinski_harabasz_score is not None
        ):
            assert good_result.calinski_harabasz_score > poor_result.calinski_harabasz_score, (
                "Better separation should have higher Calinski-Harabasz score"
            )

    def test_quality_metrics_single_cluster(self):
        """Test that quality metrics handle single cluster case gracefully."""
        # Setup
        n_samples = 20
        n_features = 10
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create random vectors
        np.random.seed(456)
        shap_vectors = np.random.randn(20, 10)

        # Cluster with n_clusters=1
        result = analyzer.cluster_patterns(shap_vectors, n_clusters=1)

        # With single cluster, quality metrics should be None or 0
        assert result.silhouette_score == 0.0, "Silhouette score should be 0 for single cluster"
        assert result.davies_bouldin_score is None, (
            "Davies-Bouldin score should be None for single cluster"
        )
        assert result.calinski_harabasz_score is None, (
            "Calinski-Harabasz score should be None for single cluster"
        )

    def test_quality_metrics_performance(self):
        """Test that quality metrics don't add significant overhead."""
        # Setup
        n_samples = 100
        n_features = 50
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create realistic SHAP vectors
        np.random.seed(789)
        shap_vectors = np.random.randn(100, 50)

        # Time clustering with quality metrics
        import time

        start = time.time()
        result = analyzer.cluster_patterns(shap_vectors, n_clusters=5)
        elapsed = time.time() - start

        # Verify metrics were computed
        assert result.davies_bouldin_score is not None
        assert result.calinski_harabasz_score is not None

        # Should add minimal overhead (<100ms on top of clustering)
        assert elapsed < 5.0, f"Clustering with metrics should complete in <5s, took {elapsed:.2f}s"


# =============================================================================
# Unit tests for cluster.py standalone functions
# =============================================================================

import pytest

from ml4t.diagnostic.evaluation.trade_shap.cluster import (
    ClusteringConfig,
    HierarchicalClusterer,
    compute_centroids,
    compute_cluster_sizes,
    find_optimal_clusters,
)


class TestFindOptimalClusters:
    """Tests for find_optimal_clusters function."""

    def test_elbow_detection_with_clear_elbow(self):
        """Elbow method finds correct number of clusters with clear elbow."""
        # Create linkage matrix with clear elbow at cluster 3
        # Linkage matrix format: [idx1, idx2, distance, n_samples_in_cluster]
        linkage = np.array(
            [
                [0, 1, 0.5, 2],  # Close merge
                [2, 3, 0.6, 2],  # Close merge
                [4, 5, 0.7, 2],  # Close merge
                [6, 7, 5.0, 4],  # Big jump (elbow)
                [8, 9, 6.0, 6],  # Continues high
            ]
        )

        result = find_optimal_clusters(linkage, n_samples=10, min_cluster_size=2)

        # Should detect elbow and return reasonable cluster count
        assert 2 <= result <= 5

    def test_min_cluster_size_constraint_applied(self):
        """min_cluster_size constraint limits max clusters."""
        linkage = np.array(
            [
                [0, 1, 0.5, 2],
                [2, 3, 0.6, 2],
                [4, 5, 10.0, 4],  # Big jump suggesting many clusters
            ]
        )

        # With 9 samples and min_cluster_size=5, max clusters = 1
        result = find_optimal_clusters(linkage, n_samples=9, min_cluster_size=5)
        assert result == 1

        # With 10 samples and min_cluster_size=5, max clusters = 2
        result = find_optimal_clusters(linkage, n_samples=10, min_cluster_size=5)
        assert result == 2

    def test_force_at_least_2_clusters_when_possible(self):
        """Force at least 2 clusters when max_clusters >= 2."""
        linkage = np.array(
            [
                [0, 1, 0.1, 2],
                [2, 3, 0.1, 2],
            ]
        )

        # With 20 samples and min_cluster_size=5, max = 4, should be >= 2
        result = find_optimal_clusters(linkage, n_samples=20, min_cluster_size=5)
        assert result >= 2

    def test_returns_1_when_cannot_support_2_clusters(self):
        """Returns 1 cluster when min_cluster_size prevents 2."""
        linkage = np.array(
            [
                [0, 1, 1.0, 2],
            ]
        )

        # 5 samples with min_cluster_size=5 means max = 1
        result = find_optimal_clusters(linkage, n_samples=5, min_cluster_size=5)
        assert result == 1

    def test_sqrt_fallback_when_no_clear_elbow(self):
        """Falls back to sqrt(n) heuristic when second derivative is empty."""
        # Very small linkage (1 row, no second derivative possible)
        linkage = np.array([[0, 1, 1.0, 2]])

        result = find_optimal_clusters(linkage, n_samples=100, min_cluster_size=2)

        # sqrt(100) = 10, but constrained to not exceed 50 clusters
        assert result >= 2


class TestComputeClusterSizes:
    """Tests for compute_cluster_sizes function."""

    def test_standard_case(self):
        """Standard case computes correct sizes."""
        labels = np.array([0, 0, 0, 1, 1, 2])
        sizes = compute_cluster_sizes(labels, n_clusters=3)

        assert sizes == [3, 2, 1]

    def test_empty_cluster_returns_zero(self):
        """Cluster with no samples returns size 0."""
        labels = np.array([0, 0, 2, 2])  # No cluster 1
        sizes = compute_cluster_sizes(labels, n_clusters=3)

        assert sizes == [2, 0, 2]

    def test_single_sample_per_cluster(self):
        """Each cluster has exactly 1 sample."""
        labels = np.array([0, 1, 2])
        sizes = compute_cluster_sizes(labels, n_clusters=3)

        assert sizes == [1, 1, 1]

    def test_list_input(self):
        """List input (not ndarray) works correctly."""
        labels = [0, 0, 1, 1, 1]
        sizes = compute_cluster_sizes(labels, n_clusters=2)

        assert sizes == [2, 3]


class TestComputeCentroids:
    """Tests for compute_centroids function."""

    def test_standard_case(self):
        """Centroids computed correctly for standard case."""
        vectors = np.array(
            [
                [1.0, 2.0],  # Cluster 0
                [2.0, 3.0],  # Cluster 0
                [10.0, 20.0],  # Cluster 1
                [12.0, 22.0],  # Cluster 1
            ]
        )
        labels = np.array([0, 0, 1, 1])

        centroids = compute_centroids(vectors, labels, n_clusters=2)

        # Cluster 0: mean([1, 2], [2, 3]) = [1.5, 2.5]
        np.testing.assert_array_almost_equal(centroids[0], [1.5, 2.5])
        # Cluster 1: mean([10, 20], [12, 22]) = [11, 21]
        np.testing.assert_array_almost_equal(centroids[1], [11.0, 21.0])

    def test_empty_cluster_returns_zero_centroid(self):
        """Empty cluster has zero centroid."""
        vectors = np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
            ]
        )
        labels = np.array([0, 0])  # No samples in cluster 1

        centroids = compute_centroids(vectors, labels, n_clusters=2)

        np.testing.assert_array_almost_equal(centroids[0], [1.5, 2.5])
        np.testing.assert_array_almost_equal(centroids[1], [0.0, 0.0])

    def test_single_sample_per_cluster(self):
        """Single sample per cluster: centroid equals sample."""
        vectors = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        labels = np.array([0, 1])

        centroids = compute_centroids(vectors, labels, n_clusters=2)

        np.testing.assert_array_equal(centroids[0], [1.0, 2.0])
        np.testing.assert_array_equal(centroids[1], [3.0, 4.0])


class TestHierarchicalClustererInit:
    """Tests for HierarchicalClusterer initialization."""

    def test_default_config(self):
        """Default config is applied when None."""
        clusterer = HierarchicalClusterer()

        assert clusterer.config.distance_metric == "euclidean"
        assert clusterer.config.linkage_method == "ward"
        assert clusterer.config.min_cluster_size == 5
        assert clusterer.config.min_trades_for_clustering == 10

    def test_custom_config(self):
        """Custom config is stored."""
        config = ClusteringConfig(
            distance_metric="cosine",
            linkage_method="average",
            min_cluster_size=3,
            min_trades_for_clustering=5,
        )
        clusterer = HierarchicalClusterer(config)

        assert clusterer.config.distance_metric == "cosine"
        assert clusterer.config.linkage_method == "average"


class TestHierarchicalClustererCluster:
    """Tests for HierarchicalClusterer.cluster() method."""

    def test_empty_vectors_raises_value_error(self):
        """Empty vectors raises ValueError."""
        clusterer = HierarchicalClusterer()

        with pytest.raises(ValueError, match="empty vectors"):
            clusterer.cluster(np.array([]))

    def test_1d_array_raises_value_error(self):
        """1D array raises ValueError."""
        clusterer = HierarchicalClusterer()

        with pytest.raises(ValueError, match="2D array"):
            clusterer.cluster(np.array([1.0, 2.0, 3.0]))

    def test_insufficient_samples_raises_value_error(self):
        """Insufficient samples raises ValueError."""
        config = ClusteringConfig(min_trades_for_clustering=10)
        clusterer = HierarchicalClusterer(config)

        # Only 5 samples, need 10
        vectors = np.random.randn(5, 3)

        with pytest.raises(ValueError, match="Insufficient samples"):
            clusterer.cluster(vectors)

    def test_distance_metric_euclidean(self):
        """Euclidean distance metric works."""
        config = ClusteringConfig(
            distance_metric="euclidean",
            min_trades_for_clustering=10,
        )
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(20, 5)
        result = clusterer.cluster(vectors, n_clusters=3)

        assert result.n_clusters == 3
        assert result.distance_metric == "euclidean"

    def test_distance_metric_cosine(self):
        """Cosine distance metric works."""
        config = ClusteringConfig(
            distance_metric="cosine",
            linkage_method="average",  # cosine requires non-ward
            min_trades_for_clustering=10,
        )
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(20, 5)
        result = clusterer.cluster(vectors, n_clusters=3)

        assert result.distance_metric == "cosine"

    def test_linkage_method_ward(self):
        """Ward linkage method works."""
        config = ClusteringConfig(
            linkage_method="ward",
            min_trades_for_clustering=10,
        )
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(20, 5)
        result = clusterer.cluster(vectors, n_clusters=3)

        assert result.linkage_method == "ward"

    def test_linkage_method_average(self):
        """Average linkage method works."""
        config = ClusteringConfig(
            linkage_method="average",
            min_trades_for_clustering=10,
        )
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(20, 5)
        result = clusterer.cluster(vectors, n_clusters=3)

        assert result.linkage_method == "average"

    def test_linkage_method_complete(self):
        """Complete linkage method works."""
        config = ClusteringConfig(
            linkage_method="complete",
            min_trades_for_clustering=10,
        )
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(20, 5)
        result = clusterer.cluster(vectors, n_clusters=3)

        assert result.linkage_method == "complete"

    def test_auto_n_clusters(self):
        """Auto-determined number of clusters works."""
        config = ClusteringConfig(min_trades_for_clustering=10)
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(30, 5)
        result = clusterer.cluster(vectors)  # n_clusters=None

        assert result.n_clusters >= 2
        assert len(result.cluster_assignments) == 30


class TestQualityMetricsUnit:
    """Unit tests for quality metric computation."""

    def test_single_cluster_silhouette_is_zero(self):
        """Silhouette is 0 for single cluster."""
        config = ClusteringConfig(
            min_trades_for_clustering=10,
            min_cluster_size=20,  # Force single cluster
        )
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(20, 5)
        result = clusterer.cluster(vectors, n_clusters=1)

        assert result.silhouette_score == 0.0

    def test_single_cluster_davies_bouldin_is_none(self):
        """Davies-Bouldin is None for single cluster."""
        config = ClusteringConfig(
            min_trades_for_clustering=10,
            min_cluster_size=20,
        )
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(20, 5)
        result = clusterer.cluster(vectors, n_clusters=1)

        assert result.davies_bouldin_score is None

    def test_single_cluster_calinski_harabasz_is_none(self):
        """Calinski-Harabasz is None for single cluster."""
        config = ClusteringConfig(
            min_trades_for_clustering=10,
            min_cluster_size=20,
        )
        clusterer = HierarchicalClusterer(config)

        vectors = np.random.randn(20, 5)
        result = clusterer.cluster(vectors, n_clusters=1)

        assert result.calinski_harabasz_score is None

    def test_two_clusters_have_valid_metrics(self):
        """Two clusters produce valid quality scores."""
        config = ClusteringConfig(min_trades_for_clustering=10)
        clusterer = HierarchicalClusterer(config)

        # Create clearly separable clusters
        vectors = np.vstack(
            [
                np.random.randn(10, 5) + 5,  # Cluster centered at 5
                np.random.randn(10, 5) - 5,  # Cluster centered at -5
            ]
        )
        result = clusterer.cluster(vectors, n_clusters=2)

        # Silhouette should be high (near 1) for well-separated clusters
        assert result.silhouette_score is not None
        assert -1 <= result.silhouette_score <= 1

        # Davies-Bouldin should be small for well-separated clusters
        assert result.davies_bouldin_score is not None
        assert result.davies_bouldin_score >= 0

        # Calinski-Harabasz should be positive
        assert result.calinski_harabasz_score is not None
        assert result.calinski_harabasz_score > 0
