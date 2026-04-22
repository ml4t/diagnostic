"""Tests for the canonical public metrics namespace."""

from __future__ import annotations


def test_metrics_package_exports_core_metric_groups():
    """The short metrics namespace should expose the common public helpers."""
    from ml4t.diagnostic import metrics

    expected = {
        "hit_rate",
        "compute_forward_returns",
        "pooled_ic",
        "cross_sectional_ic",
        "cross_sectional_ic_series",
        "compute_ic_hac_stats",
        "compute_conditional_ic",
        "compute_monotonicity",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "compute_permutation_importance",
        "compute_mdi_importance",
        "compute_mda_importance",
        "compute_shap_importance",
        "analyze_ml_importance",
        "compute_h_statistic",
        "compute_shap_interactions",
        "analyze_interactions",
        "compute_fold_percentiles",
    }

    assert expected.issubset(set(metrics.__all__))
    for name in expected:
        assert hasattr(metrics, name)


def test_metrics_submodules_are_importable():
    """The canonical metric families should be importable by short paths."""
    from ml4t.diagnostic.metrics.basic import hit_rate
    from ml4t.diagnostic.metrics.conditional import compute_conditional_ic
    from ml4t.diagnostic.metrics.ic import cross_sectional_ic
    from ml4t.diagnostic.metrics.ic_inference import compute_ic_hac_stats
    from ml4t.diagnostic.metrics.importance import analyze_ml_importance
    from ml4t.diagnostic.metrics.interactions import analyze_interactions
    from ml4t.diagnostic.metrics.monotonicity import compute_monotonicity
    from ml4t.diagnostic.metrics.risk_adjusted import sharpe_ratio

    assert callable(hit_rate)
    assert callable(compute_conditional_ic)
    assert callable(cross_sectional_ic)
    assert callable(compute_ic_hac_stats)
    assert callable(analyze_ml_importance)
    assert callable(analyze_interactions)
    assert callable(compute_monotonicity)
    assert callable(sharpe_ratio)
