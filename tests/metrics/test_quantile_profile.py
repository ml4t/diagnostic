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


def test_rejects_null_panel_keys() -> None:
    """A declared panel key must identify every row."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = _time_trending_panel().with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("symbol")).alias("symbol")
    )

    with pytest.raises(ValueError, match="keys cannot contain null"):
        quantile_profile(
            panel,
            feature="feature",
            label="label",
            keys=("symbol", "timestamp"),
        )


def test_rejects_non_numeric_feature() -> None:
    """Bucket inputs must have numeric data types."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = _time_trending_panel().with_columns(pl.col("feature").cast(pl.String))

    with pytest.raises(ValueError, match="'feature' must be numeric"):
        quantile_profile(panel, feature="feature", label="label")


def test_rejects_panel_without_finite_pairs() -> None:
    """Filtering every pair is an explicit input error."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = pl.DataFrame(
        {"timestamp": [1, 1], "feature": [float("nan"), None], "label": [1.0, 2.0]}
    )

    with pytest.raises(ValueError, match="No finite feature-label pairs"):
        quantile_profile(panel, feature="feature", label="label", n_quantiles=2)


@pytest.mark.parametrize("keys", ["", ("symbol", "symbol")])
def test_rejects_invalid_key_names(keys: object) -> None:
    """Key declarations must contain unique non-empty names."""
    from ml4t.diagnostic.metrics import quantile_profile

    with pytest.raises(ValueError, match="keys must"):
        quantile_profile(
            _time_trending_panel(),
            feature="feature",
            label="label",
            keys=keys,  # type: ignore[arg-type]
        )


def test_rejects_non_dataframe_panel() -> None:
    """The public contract requires a Polars DataFrame."""
    from ml4t.diagnostic.metrics import quantile_profile

    with pytest.raises(TypeError, match="Polars DataFrame"):
        quantile_profile([], feature="feature", label="label")  # type: ignore[arg-type]


def test_non_finite_and_null_group_rows_are_excluded() -> None:
    """Only complete finite feature-label pairs enter grouped profiles."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = pl.DataFrame(
        {
            "timestamp": [1, 1, 1, None, 1],
            "feature": [0.0, 1.0, 2.0, 3.0, float("inf")],
            "label": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    profile = quantile_profile(
        panel,
        feature="feature",
        label="label",
        n_quantiles=3,
        min_per_bucket=1,
    )

    assert sum(profile.counts.values()) == 3


def test_tied_features_are_permutation_invariant() -> None:
    """Equal feature values cannot manufacture an order-dependent profile."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = pl.DataFrame(
        {
            "timestamp": [1] * 5,
            "feature": [0.0] * 5,
            "label": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    forward = quantile_profile(
        panel, feature="feature", label="label", n_quantiles=5, min_per_bucket=1
    )
    reverse = quantile_profile(
        panel.reverse(), feature="feature", label="label", n_quantiles=5, min_per_bucket=1
    )

    assert forward.counts == reverse.counts
    assert forward.means[3] == reverse.means[3] == 2.0
    assert all(math.isnan(forward.means[bucket]) for bucket in (1, 2, 4, 5))
    assert all(math.isnan(reverse.means[bucket]) for bucket in (1, 2, 4, 5))
    assert math.isnan(forward.monotonicity)


def test_pooled_profile_rejects_fewer_rows_than_quantiles() -> None:
    """Pooled and grouped modes enforce the same minimum profile size."""
    from ml4t.diagnostic.metrics import quantile_profile

    panel = pl.DataFrame({"feature": [1.0, 2.0], "label": [10.0, 20.0]})

    with pytest.raises(ValueError, match="at least 5 valid observations"):
        quantile_profile(
            panel,
            feature="feature",
            label="label",
            n_quantiles=5,
            by=None,
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
