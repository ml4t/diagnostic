"""Behavioral tests for feature-importance visualization extraction."""

from __future__ import annotations

import numpy as np

from ml4t.diagnostic.visualization.data_extraction import extract_importance_viz_data


def test_pfi_repeat_values_and_rank_stability_follow_feature_repeat_axes() -> None:
    """PFI's feature-by-repeat matrix becomes repeat records and per-feature ranks."""
    pfi = {
        "feature_names": ["signal", "support", "noise"],
        "importances_mean": [0.57, 0.53, 0.40],
        "importances_std": [0.33, 0.25, 0.36],
        "importances_raw": np.array(
            [
                [0.9, 0.1, 0.7],
                [0.2, 0.8, 0.6],
                [0.1, 0.2, 0.9],
            ]
        ),
        "n_repeats": 3,
        "scoring": "r2",
    }
    results = {
        "consensus_ranking": ["signal", "support", "noise"],
        "method_results": {"pfi": pfi},
        "methods_run": ["pfi"],
    }

    extracted = extract_importance_viz_data(results)

    assert extracted["per_method"]["pfi"]["raw_values"] == [
        {"signal": 0.9, "support": 0.2, "noise": 0.1},
        {"signal": 0.1, "support": 0.8, "noise": 0.2},
        {"signal": 0.7, "support": 0.6, "noise": 0.9},
    ]
    assert extracted["uncertainty"]["rank_stability"] == {
        "signal": [1, 3, 2],
        "support": [2, 1, 3],
        "noise": [3, 2, 1],
    }
