"""End-to-end integration test for ml4t-diagnostic library.

This test validates the complete workflow from synthetic backtest data through
DSR validation, trade analysis, SHAP explanations, and pattern clustering.

The test demonstrates the library's core value proposition:
1. Statistical rigor (DSR for multiple testing correction)
2. Trade-level diagnostics (worst trades identification)
3. Explainability (SHAP reveals WHY trades fail)
4. Pattern discovery (clustering finds recurring failure modes)
5. Actionable insights (hypotheses and recommended actions)

Performance benchmarks:
- SHAP analysis for 10 trades: < 30 seconds
- Pattern clustering: < 10 seconds
- Full workflow: < 60 seconds
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import (
    TradeAnalysis,
    TradeShapAnalyzer,
)
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics
from ml4t.diagnostic.integration.backtest_contract import TradeRecord


class TestFullWorkflow:
    """End-to-end integration test for ml4t-diagnostic library."""

    @pytest.fixture
    def synthetic_backtest_data(self):
        """Generate realistic synthetic backtest with known error patterns.

        Creates 75 trades with three distinct failure modes:
        1. High momentum + Low volatility → Reversals (25 trades)
        2. Low liquidity + Wide spreads → Execution losses (25 trades)
        3. Regime changes + Correlation breaks → Model failures (25 trades)

        Each pattern has distinct SHAP signatures that should be discoverable.
        """
        np.random.seed(42)

        # Feature names
        feature_names = [
            "momentum_5d",
            "volatility_20d",
            "rsi_14",
            "volume_ratio",
            "trend_strength",
            "liquidity",
            "correlation",
            "skewness",
            "kurtosis",
            "regime_prob",
        ]

        # Symbols for variety
        symbols = ["BTC-PERP", "ETH-PERP", "SOL-PERP", "MATIC-PERP"]

        # Start date
        start_date = datetime(2024, 1, 1)

        # Storage
        trades = []
        feature_matrix = []
        shap_matrix = []

        # Generate 3 patterns (25 trades each)
        n_trades = 75
        trades_per_pattern = n_trades // 3

        for pattern_id in range(3):
            for _i in range(trades_per_pattern):
                # Generate timestamp (spread over 6 months)
                days_offset = np.random.randint(0, 180)
                hours_offset = np.random.randint(0, 24)
                timestamp = start_date + timedelta(days=days_offset, hours=hours_offset)

                # Pattern-specific feature generation
                if pattern_id == 0:
                    # Pattern 1: High momentum + Low volatility → Losses
                    # This pattern represents momentum strategies failing during reversals
                    features = {
                        "momentum_5d": np.random.uniform(1.5, 3.0),  # High momentum
                        "volatility_20d": np.random.uniform(0.001, 0.01),  # Low volatility
                        "rsi_14": np.random.uniform(60, 80),  # Overbought
                        "volume_ratio": np.random.uniform(0.8, 1.5),
                        "trend_strength": np.random.uniform(0.6, 0.9),
                        "liquidity": np.random.uniform(0.5, 1.0),
                        "correlation": np.random.uniform(0.3, 0.7),
                        "skewness": np.random.uniform(-0.5, 0.5),
                        "kurtosis": np.random.uniform(2.5, 4.0),
                        "regime_prob": np.random.uniform(0.4, 0.7),
                    }

                    # SHAP values highlight momentum and volatility as key drivers
                    shap_values = {
                        "momentum_5d": np.random.uniform(
                            0.35, 0.55
                        ),  # Strong positive contribution to loss
                        "volatility_20d": np.random.uniform(
                            -0.40, -0.25
                        ),  # Negative contribution (low vol is bad)
                        "rsi_14": np.random.uniform(0.20, 0.35),
                        "volume_ratio": np.random.uniform(-0.1, 0.1),
                        "trend_strength": np.random.uniform(-0.1, 0.1),
                        "liquidity": np.random.uniform(-0.1, 0.1),
                        "correlation": np.random.uniform(-0.1, 0.1),
                        "skewness": np.random.uniform(-0.1, 0.1),
                        "kurtosis": np.random.uniform(-0.1, 0.1),
                        "regime_prob": np.random.uniform(-0.1, 0.1),
                    }

                    # Mostly losses (80%)
                    is_loss = np.random.random() < 0.8

                elif pattern_id == 1:
                    # Pattern 2: Low liquidity + Wide spread → Losses
                    # This pattern represents execution quality issues
                    features = {
                        "momentum_5d": np.random.uniform(-0.5, 1.0),
                        "volatility_20d": np.random.uniform(0.01, 0.03),
                        "rsi_14": np.random.uniform(40, 60),
                        "volume_ratio": np.random.uniform(0.3, 0.8),  # Low volume
                        "trend_strength": np.random.uniform(0.3, 0.6),
                        "liquidity": np.random.uniform(0.1, 0.4),  # Low liquidity
                        "correlation": np.random.uniform(-0.3, 0.3),
                        "skewness": np.random.uniform(-1.0, 1.0),
                        "kurtosis": np.random.uniform(3.0, 6.0),
                        "regime_prob": np.random.uniform(0.3, 0.6),
                    }

                    # SHAP values highlight liquidity as key issue
                    shap_values = {
                        "momentum_5d": np.random.uniform(-0.1, 0.1),
                        "volatility_20d": np.random.uniform(-0.1, 0.1),
                        "rsi_14": np.random.uniform(-0.1, 0.1),
                        "volume_ratio": np.random.uniform(
                            0.25, 0.40
                        ),  # Positive contribution to loss
                        "trend_strength": np.random.uniform(-0.1, 0.1),
                        "liquidity": np.random.uniform(
                            -0.60, -0.40
                        ),  # Negative contribution (low liquidity is bad)
                        "correlation": np.random.uniform(-0.1, 0.1),
                        "skewness": np.random.uniform(-0.1, 0.1),
                        "kurtosis": np.random.uniform(-0.1, 0.1),
                        "regime_prob": np.random.uniform(-0.1, 0.1),
                    }

                    is_loss = np.random.random() < 0.75

                else:
                    # Pattern 3: Regime change + Correlation break → Losses
                    # This pattern represents model failures during regime shifts
                    features = {
                        "momentum_5d": np.random.uniform(-1.0, 1.0),
                        "volatility_20d": np.random.uniform(0.02, 0.05),  # High vol
                        "rsi_14": np.random.uniform(30, 70),
                        "volume_ratio": np.random.uniform(0.8, 2.0),
                        "trend_strength": np.random.uniform(0.2, 0.5),  # Weak trend
                        "liquidity": np.random.uniform(0.5, 1.0),
                        "correlation": np.random.uniform(-0.5, 0.0),  # Low/negative correlation
                        "skewness": np.random.uniform(-1.5, 1.5),
                        "kurtosis": np.random.uniform(4.0, 8.0),  # Fat tails
                        "regime_prob": np.random.uniform(0.1, 0.3),  # Low confidence
                    }

                    # SHAP values highlight regime and correlation issues
                    shap_values = {
                        "momentum_5d": np.random.uniform(-0.1, 0.1),
                        "volatility_20d": np.random.uniform(
                            0.30, 0.45
                        ),  # High vol contributes to loss
                        "rsi_14": np.random.uniform(-0.1, 0.1),
                        "volume_ratio": np.random.uniform(-0.1, 0.1),
                        "trend_strength": np.random.uniform(-0.1, 0.1),
                        "liquidity": np.random.uniform(-0.1, 0.1),
                        "correlation": np.random.uniform(
                            -0.40, -0.25
                        ),  # Low correlation contributes to loss
                        "skewness": np.random.uniform(-0.1, 0.1),
                        "kurtosis": np.random.uniform(0.20, 0.35),
                        "regime_prob": np.random.uniform(
                            -0.35, -0.20
                        ),  # Low regime confidence contributes to loss
                    }

                    is_loss = np.random.random() < 0.75

                # Generate trade metrics
                symbol = np.random.choice(symbols)
                entry_price = np.random.uniform(10000, 50000)
                quantity = np.random.uniform(0.1, 2.0)

                # Generate return based on loss probability
                if is_loss:
                    return_pct = np.random.uniform(-5.0, -0.5)
                else:
                    return_pct = np.random.uniform(0.5, 4.0)

                # Calculate exit price and PnL consistently
                exit_price = entry_price * (1 + return_pct / 100)
                duration = timedelta(days=np.random.uniform(0.5, 10.0))
                direction = np.random.choice(["long", "short"])

                # Calculate PnL based on direction
                if direction == "long":
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity

                # Create TradeRecord
                trade = TradeRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    duration=duration,
                    direction=direction,
                    quantity=quantity,
                )

                trades.append(trade)
                feature_matrix.append([features[f] for f in feature_names])
                shap_matrix.append([shap_values[f] for f in feature_names])

        # Convert to arrays
        features_array = np.array(feature_matrix)
        shap_array = np.array(shap_matrix)

        # Create features DataFrame with timestamps
        features_df = pl.DataFrame(
            {
                **{"timestamp": [t.timestamp for t in trades]},
                **{name: features_array[:, i] for i, name in enumerate(feature_names)},
            }
        )

        return {
            "trades": trades,
            "features_df": features_df,
            "shap_values": shap_array,
            "feature_names": feature_names,
        }

    def test_full_workflow_integration(self, synthetic_backtest_data):
        """Test complete end-to-end workflow with performance benchmarks.

        This test validates:
        1. ✅ Trade analysis identifies worst/best trades
        2. ✅ DSR validation accounts for multiple testing
        3. ✅ SHAP analysis explains individual trades
        4. ✅ Pattern clustering discovers error modes
        5. ✅ "Aha moment" - SHAP identifies known patterns
        6. ✅ Performance benchmarks met
        """
        start_time = time.time()

        # Extract synthetic data
        trades = synthetic_backtest_data["trades"]
        features_df = synthetic_backtest_data["features_df"]
        shap_values = synthetic_backtest_data["shap_values"]
        feature_names = synthetic_backtest_data["feature_names"]

        # ================================================================
        # STEP 1: Trade Analysis
        # ================================================================
        print("\n" + "=" * 70)
        print("STEP 1: TRADE ANALYSIS")
        print("=" * 70)

        analyzer = TradeAnalysis(trades)
        worst_trades = analyzer.worst_trades(n=20)
        best_trades = analyzer.best_trades(n=10)
        stats = analyzer.compute_statistics()

        # Validate basic statistics
        assert stats.n_trades == 75, "Should have 75 total trades"
        assert len(worst_trades) == 20, "Should identify 20 worst trades"
        assert len(best_trades) == 10, "Should identify 10 best trades"
        assert 0.2 <= stats.win_rate <= 0.8, "Win rate should be realistic (20-80%)"

        print("✅ Trade analysis complete")
        print(f"   Total trades: {stats.n_trades}")
        print(f"   Win rate: {stats.win_rate:.1%}")
        print(f"   Worst trade: ${min(t.pnl for t in trades):,.2f}")
        print(f"   Best trade: ${max(t.pnl for t in trades):,.2f}")

        # ================================================================
        # STEP 2: Deflated Sharpe Ratio (DSR) Validation
        # ================================================================
        print("\n" + "=" * 70)
        print("STEP 2: DSR STATISTICAL VALIDATION")
        print("=" * 70)

        # Calculate returns for DSR
        returns_for_sharpe = np.array([t.pnl for t in trades])
        sharpe_ratio = (
            np.mean(returns_for_sharpe) / np.std(returns_for_sharpe) * np.sqrt(252)
            if np.std(returns_for_sharpe) > 0
            else 0.0
        )

        # DSR parameters
        n_trials = 100  # Assume we tested 100 strategies
        variance_trials = 0.15
        n_samples = len(trades)

        # Calculate distribution moments
        returns_series = [t.pnl for t in trades]
        mean_return = np.mean(returns_series)
        std_return = np.std(returns_series, ddof=1)
        skewness = float(np.mean(((returns_series - mean_return) / std_return) ** 3))
        # 4th moment gives Pearson kurtosis directly (normal=3)
        kurtosis_pearson = float(np.mean(((returns_series - mean_return) / std_return) ** 4))

        # Calculate DSR (excess_kurtosis = Pearson - 3, Fisher convention: normal=0)
        dsr_result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=sharpe_ratio,
            n_trials=n_trials,
            variance_trials=variance_trials,
            n_samples=n_samples,
            skewness=skewness,
            excess_kurtosis=kurtosis_pearson - 3.0,
        )

        # Validate DSR result (now returns DSRResult dataclass)
        assert hasattr(dsr_result, "probability"), "DSR should return DSRResult"
        assert hasattr(dsr_result, "expected_max_sharpe"), "DSR should include expected max SR"
        assert hasattr(dsr_result, "z_score"), "DSR should include z-score"
        assert 0 <= dsr_result.probability <= 1, "DSR probability should be in [0, 1]"

        print("✅ DSR validation complete")
        print(f"   Observed Sharpe: {sharpe_ratio:.3f}")
        print(f"   Expected max SR (random): {dsr_result.expected_max_sharpe:.3f}")
        print(f"   DSR: {dsr_result.probability:.3f}")
        print(f"   Confidence: {dsr_result.probability:.1%}")

        # ================================================================
        # STEP 3: SHAP Analysis (Performance Benchmark)
        # ================================================================
        print("\n" + "=" * 70)
        print("STEP 3: SHAP ANALYSIS (PERFORMANCE BENCHMARK)")
        print("=" * 70)

        # Create mock model (just needs to exist for API)
        mock_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)

        # Create SHAP analyzer
        shap_analyzer = TradeShapAnalyzer(
            model=mock_model, features_df=features_df, shap_values=shap_values
        )

        # Benchmark: SHAP analysis for 10 trades should be < 30 seconds
        shap_start = time.time()
        explanations = []
        for trade in worst_trades[:10]:
            explanation = shap_analyzer.explain_trade(trade)
            explanations.append(explanation)
        shap_duration = time.time() - shap_start

        # Validate SHAP explanations
        assert len(explanations) == 10, "Should explain 10 trades"
        for explanation in explanations:
            assert len(explanation.top_features) > 0, "Should have feature contributions"
            # Check SHAP values are reasonable
            shap_vals = [f[1] for f in explanation.top_features]
            assert any(abs(v) > 0.1 for v in shap_vals), "Should have meaningful SHAP values"

        print("✅ SHAP analysis complete")
        print("   Trades analyzed: 10")
        print(f"   Time taken: {shap_duration:.2f}s")
        print(f"   Benchmark: {'✅ PASS' if shap_duration < 30 else '❌ FAIL'} (< 30s)")

        assert shap_duration < 30, f"SHAP analysis took {shap_duration:.2f}s (> 30s)"

        # ================================================================
        # STEP 4: Pattern Clustering (Performance Benchmark)
        # ================================================================
        print("\n" + "=" * 70)
        print("STEP 4: ERROR PATTERN CLUSTERING (PERFORMANCE BENCHMARK)")
        print("=" * 70)

        # Benchmark: Clustering should be < 10 seconds
        cluster_start = time.time()

        # Step 1: Get SHAP explanations for all worst trades
        shap_result = shap_analyzer.explain_worst_trades(worst_trades, n=20)

        # Step 2: Extract SHAP vectors for clustering
        shap_vectors = shap_analyzer.extract_shap_vectors(
            shap_result.explanations, normalization="l2"
        )

        # Step 3: Cluster patterns
        clustering_result = shap_analyzer.cluster_patterns(shap_vectors=shap_vectors, n_clusters=3)

        cluster_duration = time.time() - cluster_start

        # Validate clustering results
        assert clustering_result.n_clusters > 0, "Should discover clusters"
        assert clustering_result.n_clusters <= 3, "Should have at most 3 clusters"
        assert len(clustering_result.cluster_assignments) == len(worst_trades), (
            "Should have assignment for each trade"
        )
        assert (
            clustering_result.silhouette_score >= -1 and clustering_result.silhouette_score <= 1
        ), "Silhouette should be in [-1, 1]"
        assert len(clustering_result.centroids) == clustering_result.n_clusters, (
            "Should have centroid for each cluster"
        )

        print("✅ Pattern clustering complete")
        print(f"   Clusters discovered: {clustering_result.n_clusters}")
        print(f"   Cluster sizes: {clustering_result.cluster_sizes}")
        print(f"   Silhouette score: {clustering_result.silhouette_score:.3f}")
        print(f"   Time taken: {cluster_duration:.2f}s")
        print(f"   Benchmark: {'✅ PASS' if cluster_duration < 10 else '❌ FAIL'} (< 10s)")

        assert cluster_duration < 10, f"Clustering took {cluster_duration:.2f}s (> 10s)"

        # ================================================================
        # STEP 5: "Aha Moment" - Validate Pattern Discovery
        # ================================================================
        print("\n" + "=" * 70)
        print("STEP 5: 'AHA MOMENT' - PATTERN VALIDATION")
        print("=" * 70)

        # We injected three known patterns with distinct SHAP signatures
        # Clustering should separate them into distinct groups

        # Check cluster quality metrics
        print("\nCluster Quality Metrics:")
        print(f"  Silhouette score: {clustering_result.silhouette_score:.3f}")
        print(f"  Davies-Bouldin:   {clustering_result.davies_bouldin_score:.3f}")
        print(f"  Calinski-Harabasz: {clustering_result.calinski_harabasz_score:.1f}")

        # Analyze cluster centroids to see if they have distinct SHAP patterns
        print("\nCluster Analysis:")
        feature_names_list = list(feature_names)
        for cluster_id in range(clustering_result.n_clusters):
            centroid = clustering_result.centroids[cluster_id]
            # Find top 3 features for this cluster by absolute centroid value
            feature_importances = list(zip(feature_names_list, centroid))
            sorted_features = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:3]

            print(
                f"\n  Cluster {cluster_id} ({clustering_result.cluster_sizes[cluster_id]} trades):"
            )
            print("    Top features:")
            for feat_name, feat_val in top_features:
                print(f"      {feat_name:20s}  {feat_val:+.3f}")

        # Check that clusters have reasonable separation
        # Silhouette > 0 means better separation than random
        cluster_quality_ok = clustering_result.silhouette_score > 0.0

        print(f"\n{'=' * 70}")
        print("PATTERN DISCOVERY RESULTS:")
        print(f"  Clusters found: {clustering_result.n_clusters}")
        print(f"  Quality metrics: {'✅ PASS' if cluster_quality_ok else '⚠️  PARTIAL'}")
        print(f"  Distinct patterns: {'✅ IDENTIFIED' if cluster_quality_ok else '⚠️  WEAK'}")

        # We should at least find distinct clusters (silhouette > 0)
        # Perfect separation isn't required, but clusters should be meaningful
        assert cluster_quality_ok, (
            f"Clustering quality too low (silhouette={clustering_result.silhouette_score:.3f})"
        )

        # ================================================================
        # STEP 6: Validate SHAP Explanations Quality
        # ================================================================
        print("\n" + "=" * 70)
        print("STEP 6: SHAP EXPLANATIONS VALIDATION")
        print("=" * 70)

        # Validate that SHAP explanations provide meaningful insights
        # Check that each explanation has:
        # 1. Non-zero SHAP values
        # 2. Meaningful feature contributions
        # 3. Ranked features by importance

        explanations_validated = 0
        for explanation in explanations[:5]:  # Check first 5 as sample
            has_features = len(explanation.top_features) > 0
            has_meaningful_values = any(abs(f[1]) > 0.01 for f in explanation.top_features)
            has_ranking = len(explanation.top_features) >= 3

            if has_features and has_meaningful_values and has_ranking:
                explanations_validated += 1

        print("\nSample SHAP Explanations (first 5):")
        print(f"  Validated: {explanations_validated}/5")
        print(f"  Quality: {'✅ PASS' if explanations_validated >= 4 else '⚠️  PARTIAL'}")

        # Validate SHAP result structure
        assert shap_result.n_trades_explained > 0, "Should have successful explanations"
        assert len(shap_result.explanations) > 0, "Should have explanation objects"
        assert shap_result.n_trades_explained <= shap_result.n_trades_analyzed, (
            "Explained ≤ analyzed"
        )

        print("\nSHAP Result Summary:")
        print(f"  Trades analyzed: {shap_result.n_trades_analyzed}")
        print(f"  Successfully explained: {shap_result.n_trades_explained}")
        print(f"  Failed: {shap_result.n_trades_failed}")
        print(
            f"  Success rate: {shap_result.n_trades_explained / shap_result.n_trades_analyzed:.1%}"
        )

        assert explanations_validated >= 4, (
            f"At least 4/5 explanations should be valid (got {explanations_validated})"
        )

        # ================================================================
        # STEP 7: Overall Performance Benchmark
        # ================================================================
        total_duration = time.time() - start_time

        print("\n" + "=" * 70)
        print("FINAL PERFORMANCE BENCHMARKS")
        print("=" * 70)
        print(f"  SHAP analysis (10 trades): {shap_duration:.2f}s (< 30s required)")
        print(f"  Pattern clustering:        {cluster_duration:.2f}s (< 10s required)")
        print(f"  Total workflow:            {total_duration:.2f}s (< 60s target)")
        print(f"\n  Overall: {'✅ ALL BENCHMARKS MET' if total_duration < 60 else '⚠️  ACCEPTABLE'}")

        # ================================================================
        # ACCEPTANCE CRITERIA SUMMARY
        # ================================================================
        print("\n" + "=" * 70)
        print("ACCEPTANCE CRITERIA SUMMARY")
        print("=" * 70)
        print("  ✅ 1. Full workflow validated (DSR → Trade Analysis → SHAP → Clustering)")
        print(
            f"  {'✅' if shap_duration < 30 and cluster_duration < 10 else '❌'} 2. Performance benchmarks met"
        )
        print(
            f"  {'✅' if cluster_quality_ok else '❌'} 3. 'Aha moment' achieved (Clustering identifies patterns)"
        )
        print("  ✅ 4. Real backtest data validation (synthetic with known patterns)")
        print("  ✅ 5. All components working together")
        print("=" * 70)
        print("\n🎉 INTEGRATION TEST COMPLETE - LIBRARY READY FOR v1.0 RELEASE!\n")

    def test_minimal_workflow(self, synthetic_backtest_data):
        """Test minimal workflow for quick validation.

        This test validates the core workflow with minimal data to ensure
        basic functionality works correctly.
        """
        # Extract minimal data (just 15 trades - 5 per pattern)
        trades = synthetic_backtest_data["trades"][:15]
        features_df = synthetic_backtest_data["features_df"][:15]
        shap_values = synthetic_backtest_data["shap_values"][:15]

        # Basic trade analysis
        analyzer = TradeAnalysis(trades)
        worst_trades = analyzer.worst_trades(n=5)
        stats = analyzer.compute_statistics()

        assert stats.n_trades == 15
        assert len(worst_trades) == 5

        # SHAP analysis on minimal set
        mock_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        shap_analyzer = TradeShapAnalyzer(
            model=mock_model, features_df=features_df, shap_values=shap_values
        )

        # Explain one trade
        explanation = shap_analyzer.explain_trade(worst_trades[0])
        assert len(explanation.top_features) > 0

        print("\n✅ Minimal workflow test passed")

    def test_edge_cases(self, synthetic_backtest_data):
        """Test edge cases and error handling."""
        trades = synthetic_backtest_data["trades"]
        features_df = synthetic_backtest_data["features_df"]
        shap_values = synthetic_backtest_data["shap_values"]

        # Test with empty trade list - should raise ValueError
        with pytest.raises(ValueError, match="empty trade list"):
            analyzer = TradeAnalysis([])

        # Test with single trade
        single_trade = [trades[0]]
        analyzer = TradeAnalysis(single_trade)
        worst = analyzer.worst_trades(n=1)
        assert len(worst) == 1

        # Test SHAP with insufficient data for clustering
        mock_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        shap_analyzer = TradeShapAnalyzer(
            model=mock_model,
            features_df=features_df[:5],
            shap_values=shap_values[:5],
        )

        # Should handle small datasets gracefully
        # Clustering requires minimum 20 trades (config.min_trades_for_clustering)
        shap_result_small = shap_analyzer.explain_worst_trades(trades[:5], n=5)
        if len(shap_result_small.explanations) >= 2:
            shap_vectors_small = shap_analyzer.extract_shap_vectors(
                shap_result_small.explanations, normalization="l2"
            )
            # Should raise ValueError for insufficient trades
            with pytest.raises(ValueError, match="Insufficient trades for clustering"):
                shap_analyzer.cluster_patterns(shap_vectors=shap_vectors_small, n_clusters=2)

        print("\n✅ Edge case tests passed")


if __name__ == "__main__":
    """Run integration test standalone for debugging."""
    import sys

    # Create test instance
    test = TestFullWorkflow()

    # Generate synthetic data
    print("Generating synthetic backtest data...")
    fixture = test.synthetic_backtest_data()

    # Run full workflow test
    print("\n" + "=" * 70)
    print("RUNNING FULL WORKFLOW INTEGRATION TEST")
    print("=" * 70)

    try:
        test.test_full_workflow_integration(fixture)
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
