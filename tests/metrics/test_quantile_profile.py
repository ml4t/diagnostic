"""Tests for cross-sectional quantile profiles."""

from __future__ import annotations

import math

import polars as pl
import pytest


def _time_trending_panel() -> pl.DataFrame:
    rows = []
    for date_index in range(5):
        for symbol_index in range(10):
            rows.append(
                {
                    "symbol": f"S{symbol_index:02d}",
                    "timestamp": date_index,
                    "feature": date_index * 100.0 + symbol_index,
                    "label": float(date_index),
                }
            )
    return pl.DataFrame(rows)


def test_per_date_default_does_not_manufacture_pooled_time_trend() -> None:
    """Default bucketing preserves cross-sectional meaning across dates."""
    from ml4t.diagnostic import quantile_profile

    panel = _time_trending_panel()

    per_date = quantile_profile(
        panel,
        feature="feature",
        label="label",
        n_quantiles=5,
        keys=("symbol", "timestamp"),
        min_per_bucket=1,
    )
    pooled = quantile_profile(
        panel,
        feature="feature",
        label="label",
        n_quantiles=5,
        by=None,
        keys=("symbol", "timestamp"),
        min_per_bucket=1,
    )

    assert per_date.is_pooled is False
    assert per_date.counts == {1: 10, 2: 10, 3: 10, 4: 10, 5: 10}
    assert per_date.means == {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0}
    assert math.isnan(per_date.monotonicity)

    assert pooled.is_pooled is True
    assert pooled.counts == {1: 10, 2: 10, 3: 10, 4: 10, 5: 10}
    assert pooled.means == {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}
    assert pooled.monotonicity == pytest.approx(1.0)


def test_sparse_dates_are_excluded_before_per_date_bucketing() -> None:
    """A date that cannot fill every bucket does not affect the profile."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = pl.DataFrame(
        {
            "symbol": [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(5)],
            "timestamp": [1] * 4 + [2] * 5,
            "feature": [100.0, 101.0, 102.0, 103.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "label": [100.0, 100.0, 100.0, 100.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    result = quantile_profile(
        panel,
        feature="feature",
        label="label",
        n_quantiles=5,
        keys=("symbol", "timestamp"),
        min_per_bucket=1,
    )

    assert result.counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert result.means == {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}
    assert result.monotonicity == pytest.approx(1.0)


def test_low_bucket_counts_are_reported_but_not_scored() -> None:
    """Counts below the requested minimum suppress monotonicity only."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = pl.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(5)],
            "timestamp": [1] * 5,
            "feature": list(range(5)),
            "label": list(range(5)),
        }
    )

    result = quantile_profile(
        panel,
        feature="feature",
        label="label",
        n_quantiles=5,
        keys=("symbol", "timestamp"),
        min_per_bucket=2,
    )

    assert result.counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert result.means == {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}
    assert math.isnan(result.monotonicity)


def test_invalid_rows_are_removed_before_group_size_filtering() -> None:
    """Null feature-label pairs cannot make an undersized date appear usable."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = pl.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(5)],
            "timestamp": [1] * 5,
            "feature": [0.0, 1.0, 2.0, 3.0, None],
            "label": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="No groups contain at least 5 valid observations"):
        quantile_profile(
            panel,
            feature="feature",
            label="label",
            n_quantiles=5,
            keys=("symbol", "timestamp"),
            min_per_bucket=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_quantiles": 1}, "n_quantiles must be at least 2"),
        ({"min_per_bucket": 0}, "min_per_bucket must be at least 1"),
        ({"feature": "missing"}, "Missing columns"),
        ({"by": "missing"}, "Missing columns"),
    ],
)
def test_validates_configuration_and_required_columns(
    kwargs: dict[str, object], message: str
) -> None:
    """Invalid requests fail before profile computation."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = _time_trending_panel()
    call_kwargs = {
        "feature": "feature",
        "label": "label",
        "n_quantiles": 5,
        "keys": ("symbol", "timestamp"),
        "min_per_bucket": 1,
    }
    call_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        quantile_profile(panel, **call_kwargs)


def test_rejects_duplicate_panel_keys() -> None:
    """Duplicate entity-time observations are rejected when keys are supplied."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = pl.concat([_time_trending_panel(), _time_trending_panel().head(1)])

    with pytest.raises(ValueError, match="keys must uniquely identify rows"):
        quantile_profile(
            panel,
            feature="feature",
            label="label",
            keys=("symbol", "timestamp"),
            min_per_bucket=1,
        )


def test_public_import_paths_export_profile_helper_and_result() -> None:
    """The helper is available through all canonical public namespaces."""
    from ml4t.diagnostic import QuantileProfile, quantile_profile
    from ml4t.diagnostic.api import QuantileProfile as ApiQuantileProfile
    from ml4t.diagnostic.api import quantile_profile as api_quantile_profile
    from ml4t.diagnostic.metrics import QuantileProfile as MetricsQuantileProfile
    from ml4t.diagnostic.metrics import quantile_profile as metrics_quantile_profile

    assert ApiQuantileProfile is QuantileProfile
    assert MetricsQuantileProfile is QuantileProfile
    assert api_quantile_profile is quantile_profile
    assert metrics_quantile_profile is quantile_profile
