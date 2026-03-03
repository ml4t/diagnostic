"""Integration tests for feature selection.

Tests that FeatureSelector works end-to-end with real diagnostic
analysis output, not just hand-crafted fixtures.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.drift import analyze_drift
from ml4t.diagnostic.selection import (
    FeatureSelector,
    SelectionReport,
    SelectionStep,
)
from ml4t.diagnostic.selection.types import (
    FeatureICResults,
    FeatureImportanceResults,
    FeatureOutcomeResult,
)

# ---------------------------------------------------------------------------
# Top-level import test
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    """Verify the selection module is accessible from ml4t.diagnostic."""

    def test_import_from_top_level(self):
        import ml4t.diagnostic as diag

        assert hasattr(diag, "FeatureSelector")
        assert hasattr(diag, "SelectionReport")

    def test_import_selection_submodule(self):
        from ml4t.diagnostic import selection

        assert hasattr(selection, "FeatureSelector")
        assert hasattr(selection, "FeatureICResults")
        assert hasattr(selection, "FeatureOutcomeResult")

    def test_import_types_directly(self):
        from ml4t.diagnostic.selection.types import (
            FeatureICResults,
            FeatureImportanceResults,
            FeatureOutcomeResult,
        )

        assert FeatureICResults is not None
        assert FeatureImportanceResults is not None
        assert FeatureOutcomeResult is not None


# ---------------------------------------------------------------------------
# Correlation matrix format tests
# ---------------------------------------------------------------------------


class TestCorrelationMatrixFormats:
    """Test FeatureSelector handles different correlation matrix formats."""

    @pytest.fixture
    def basic_outcome(self) -> FeatureOutcomeResult:
        """Minimal outcome results for correlation tests."""
        features = ["f1", "f2", "f3"]
        ic_results = {
            f: FeatureICResults(
                feature=f,
                ic_mean=0.05 - i * 0.01,
                ic_std=0.01,
                ic_ir=3.0,
                t_stat=5.0,
                p_value=0.01,
                ic_by_lag={1: 0.05 - i * 0.01},
                n_observations=100,
            )
            for i, f in enumerate(features)
        }
        return FeatureOutcomeResult(features=features, ic_results=ic_results)

    def test_with_feature_column(self, basic_outcome: FeatureOutcomeResult):
        """Correlation matrix with explicit 'feature' index column."""
        corr = pl.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "f1": [1.0, 0.9, 0.1],
                "f2": [0.9, 1.0, 0.2],
                "f3": [0.1, 0.2, 1.0],
            }
        )
        selector = FeatureSelector(basic_outcome, corr)
        selector.filter_by_correlation(threshold=0.8)

        selected = selector.get_selected_features()
        # f1 and f2 are correlated > 0.8; f1 has higher IC so f2 is removed
        assert "f1" in selected
        assert "f2" not in selected
        assert "f3" in selected

    def test_without_feature_column(self, basic_outcome: FeatureOutcomeResult):
        """Correlation matrix without 'feature' column (column names are features)."""
        corr = pl.DataFrame(
            {
                "f1": [1.0, 0.9, 0.1],
                "f2": [0.9, 1.0, 0.2],
                "f3": [0.1, 0.2, 1.0],
            }
        )
        selector = FeatureSelector(basic_outcome, corr)
        selector.filter_by_correlation(threshold=0.8)

        selected = selector.get_selected_features()
        assert "f1" in selected
        assert "f2" not in selected
        assert "f3" in selected

    def test_features_missing_from_matrix(self, basic_outcome: FeatureOutcomeResult):
        """Features in selector but not in correlation matrix are kept."""
        # Only f1 and f2 in the matrix, f3 missing
        corr = pl.DataFrame(
            {
                "feature": ["f1", "f2"],
                "f1": [1.0, 0.3],
                "f2": [0.3, 1.0],
            }
        )
        selector = FeatureSelector(basic_outcome, corr)
        selector.filter_by_correlation(threshold=0.8)

        selected = selector.get_selected_features()
        # No pair exceeds 0.8, f3 stays because it's not in the matrix
        assert "f1" in selected
        assert "f2" in selected
        assert "f3" in selected

    def test_single_feature_in_matrix(self, basic_outcome: FeatureOutcomeResult):
        """Single feature in correlation matrix should not crash."""
        corr = pl.DataFrame({"feature": ["f1"], "f1": [1.0]})
        selector = FeatureSelector(basic_outcome, corr)
        selector.filter_by_correlation(threshold=0.8)

        assert len(selector.get_selected_features()) == 3


# ---------------------------------------------------------------------------
# Real drift analysis integration
# ---------------------------------------------------------------------------


class TestDriftIntegration:
    """Test FeatureSelector with real analyze_drift() output."""

    def test_with_real_drift_analysis(self):
        """Run actual drift detection and feed results to FeatureSelector."""
        rng = np.random.default_rng(42)

        # Reference data: 3 features, 500 samples
        reference = pl.DataFrame(
            {
                "stable_1": rng.normal(0, 1, 500),
                "stable_2": rng.normal(0, 1, 500),
                "drifted": rng.normal(0, 1, 500),
            }
        )

        # Test data: drifted feature has shifted mean
        test = pl.DataFrame(
            {
                "stable_1": rng.normal(0, 1, 500),
                "stable_2": rng.normal(0.1, 1, 500),
                "drifted": rng.normal(2.0, 1, 500),  # Large shift
            }
        )

        drift_result = analyze_drift(reference.to_pandas(), test.to_pandas(), methods=["psi"])

        # Build FeatureOutcomeResult with real drift
        features = ["stable_1", "stable_2", "drifted"]
        ic_results = {
            f: FeatureICResults(
                feature=f,
                ic_mean=0.05,
                ic_std=0.01,
                ic_ir=5.0,
                t_stat=10.0,
                p_value=0.001,
                ic_by_lag={1: 0.05},
                n_observations=100,
            )
            for f in features
        }

        outcome = FeatureOutcomeResult(
            features=features,
            ic_results=ic_results,
            drift_results=drift_result,
        )

        selector = FeatureSelector(outcome)
        selector.filter_by_drift(method="psi")

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        # "drifted" feature should be removed (PSI red alert)
        assert "drifted" in removed
        # stable features should remain
        assert "stable_1" in selected


# ---------------------------------------------------------------------------
# Polars correlation computation integration
# ---------------------------------------------------------------------------


class TestPolarsCorrelationIntegration:
    """Test with Polars-computed correlation matrices."""

    def test_with_polars_corr(self):
        """Use an actual Polars .corr() output as input."""
        rng = np.random.default_rng(99)

        # Create features where f1 and f2 are highly correlated
        base = rng.normal(0, 1, 200)
        data = pl.DataFrame(
            {
                "f1": base + rng.normal(0, 0.1, 200),
                "f2": base + rng.normal(0, 0.1, 200),  # ~same as f1
                "f3": rng.normal(0, 1, 200),  # independent
            }
        )

        # Polars .corr() returns a DataFrame with column names as features
        corr_matrix = data.corr()

        # Build outcome
        features = ["f1", "f2", "f3"]
        ic_results = {
            "f1": FeatureICResults(
                feature="f1",
                ic_mean=0.08,
                ic_std=0.01,
                ic_ir=8.0,
                t_stat=10.0,
                p_value=0.001,
                ic_by_lag={1: 0.08},
                n_observations=100,
            ),
            "f2": FeatureICResults(
                feature="f2",
                ic_mean=0.03,
                ic_std=0.01,
                ic_ir=3.0,
                t_stat=5.0,
                p_value=0.01,
                ic_by_lag={1: 0.03},
                n_observations=100,
            ),
            "f3": FeatureICResults(
                feature="f3",
                ic_mean=0.05,
                ic_std=0.01,
                ic_ir=5.0,
                t_stat=7.0,
                p_value=0.005,
                ic_by_lag={1: 0.05},
                n_observations=100,
            ),
        }
        outcome = FeatureOutcomeResult(features=features, ic_results=ic_results)

        selector = FeatureSelector(outcome, corr_matrix)
        selector.filter_by_correlation(threshold=0.8)

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        # f1 (IC=0.08) and f2 (IC=0.03) are correlated; f2 should be removed
        assert "f1" in selected
        assert "f2" in removed
        assert "f3" in selected


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """End-to-end pipeline with multiple filter stages."""

    @pytest.fixture
    def realistic_outcome(self) -> tuple[FeatureOutcomeResult, pl.DataFrame]:
        """Build a realistic 10-feature outcome with correlation matrix."""
        rng = np.random.default_rng(123)
        n_features = 10
        names = [f"feat_{i:02d}" for i in range(n_features)]

        # IC: first 6 strong, last 4 weak
        ic_results = {}
        for i, name in enumerate(names):
            ic_val = 0.06 - i * 0.006  # 0.06 down to 0.006
            ic_results[name] = FeatureICResults(
                feature=name,
                ic_mean=ic_val,
                ic_std=0.01,
                ic_ir=ic_val / 0.01,
                t_stat=ic_val / 0.01 * 2,
                p_value=0.001 if ic_val > 0.02 else 0.2,
                ic_by_lag={1: ic_val},
                n_observations=200,
            )

        # Importance: monotonically decreasing
        importance_results = {}
        for i, name in enumerate(names):
            imp = 0.3 - i * 0.025
            importance_results[name] = FeatureImportanceResults(
                feature=name,
                mdi_importance=imp,
                permutation_importance=imp * 0.9,
                permutation_std=0.01,
                shap_mean=imp * 0.8,
                shap_std=0.01,
                rank_mdi=i + 1,
                rank_permutation=i + 1,
            )

        # Drift: last 2 features drifted
        from ml4t.diagnostic.evaluation.drift import (
            DriftSummaryResult,
            FeatureDriftResult,
            PSIResult,
        )

        drift_features = []
        for i, name in enumerate(names):
            is_drifted = i >= 8
            drift_features.append(
                FeatureDriftResult(
                    feature=name,
                    psi_result=PSIResult(
                        psi=0.3 if is_drifted else 0.05,
                        bin_psi=np.array([0.06 if is_drifted else 0.01] * 5),
                        bin_edges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        reference_counts=np.array([20] * 5),
                        test_counts=np.array([20] * 5),
                        reference_percents=np.array([0.2] * 5),
                        test_percents=np.array([0.2] * 5),
                        n_bins=5,
                        is_categorical=False,
                        alert_level="red" if is_drifted else "green",
                        interpretation="drift" if is_drifted else "stable",
                    ),
                    drifted=is_drifted,
                    n_methods_run=1,
                    n_methods_detected=1 if is_drifted else 0,
                    drift_probability=1.0 if is_drifted else 0.0,
                )
            )

        drift_results = DriftSummaryResult(
            feature_results=drift_features,
            n_features=n_features,
            n_features_drifted=2,
            drifted_features=["feat_08", "feat_09"],
            overall_drifted=True,
        )

        outcome = FeatureOutcomeResult(
            features=names,
            ic_results=ic_results,
            importance_results=importance_results,
            drift_results=drift_results,
        )

        # Correlation matrix: feat_01 and feat_02 correlated at 0.92
        base = rng.normal(0, 1, 300)
        cols = {}
        for i, name in enumerate(names):
            if i == 2:
                cols[name] = base + rng.normal(0, 0.3, 300)  # correlated with feat_01
            elif i == 1:
                cols[name] = base + rng.normal(0, 0.2, 300)  # correlated with feat_02
            else:
                cols[name] = rng.normal(0, 1, 300)
        corr = pl.DataFrame(cols).corr()

        return outcome, corr

    def test_full_pipeline_filters_correctly(
        self, realistic_outcome: tuple[FeatureOutcomeResult, pl.DataFrame]
    ):
        """Full 4-stage pipeline removes features progressively."""
        outcome, corr = realistic_outcome

        selector = FeatureSelector(outcome, corr)
        selector.run_pipeline(
            [
                ("drift", {"threshold": 0.2, "method": "psi"}),
                ("ic", {"threshold": 0.02}),
                ("correlation", {"threshold": 0.8, "keep_strategy": "higher_ic"}),
                ("importance", {"threshold": 0.05, "method": "mdi"}),
            ]
        )

        report = selector.get_selection_report()
        selected = selector.get_selected_features()

        # Verify pipeline progression
        assert len(report.steps) == 4
        assert report.steps[0].step_name == "Drift Filtering"
        assert report.steps[1].step_name == "IC Filtering"
        assert report.steps[2].step_name == "Correlation Filtering"
        assert report.steps[3].step_name == "Importance Filtering (MDI)"

        # Drift should remove feat_08, feat_09
        assert "feat_08" not in selected
        assert "feat_09" not in selected

        # IC filter removes weak features (ic < 0.02)
        # feat_07 (ic=0.018), should be removed
        assert "feat_07" not in selected

        # Remaining features should have decent IC and importance
        for f in selected:
            assert outcome.ic_results[f].ic_mean >= 0.02
            assert outcome.importance_results[f].mdi_importance >= 0.05

        # Total removal rate should be significant
        assert report.removal_rate > 30

    def test_pipeline_report_summary_complete(
        self, realistic_outcome: tuple[FeatureOutcomeResult, pl.DataFrame]
    ):
        """Report summary contains all expected sections."""
        outcome, corr = realistic_outcome

        selector = FeatureSelector(outcome, corr)
        selector.run_pipeline(
            [
                ("drift", {"threshold": 0.2}),
                ("ic", {"threshold": 0.02}),
            ]
        )

        report = selector.get_selection_report()
        summary = report.summary()

        assert "Feature Selection Report" in summary
        assert "Initial Features: 10" in summary
        assert "Drift Filtering" in summary
        assert "IC Filtering" in summary
        assert "Final Selected Features" in summary

    def test_reset_restores_all_features(
        self, realistic_outcome: tuple[FeatureOutcomeResult, pl.DataFrame]
    ):
        """Reset after pipeline restores initial state completely."""
        outcome, corr = realistic_outcome

        selector = FeatureSelector(outcome, corr)
        selector.run_pipeline(
            [
                ("drift", {"threshold": 0.2}),
                ("ic", {"threshold": 0.02}),
            ]
        )

        removed_count = len(selector.get_removed_features())
        assert removed_count > 0

        selector.reset()

        assert len(selector.get_selected_features()) == 10
        assert len(selector.get_removed_features()) == 0
        assert len(selector.selection_steps) == 0

    def test_chaining_equivalent_to_pipeline(
        self, realistic_outcome: tuple[FeatureOutcomeResult, pl.DataFrame]
    ):
        """Method chaining produces same result as run_pipeline."""
        outcome, corr = realistic_outcome

        # Pipeline
        s1 = FeatureSelector(outcome, corr)
        s1.run_pipeline(
            [
                ("ic", {"threshold": 0.02}),
                ("importance", {"threshold": 0.1, "method": "mdi"}),
            ]
        )

        # Chaining
        s2 = FeatureSelector(outcome, corr)
        s2.filter_by_ic(threshold=0.02).filter_by_importance(threshold=0.1, method="mdi")

        assert s1.get_selected_features() == s2.get_selected_features()
        assert s1.get_removed_features() == s2.get_removed_features()
