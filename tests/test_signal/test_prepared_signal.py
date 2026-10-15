"""Prepared-panel workflows must preserve supplied sample and bucket identities."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import spearmanr

from ml4t.diagnostic.signal import compute_spread, extract_signal_ic_series, quantize_factor


def supplied_panel():
    index = pd.date_range("2025-01-02", periods=3, tz="America/New_York")
    frame = pd.DataFrame(
        {
            "date": index.repeat(12),
            "asset": [f"asset-{i:02}" for i in range(12)] * 3,
            "factor": np.tile(np.arange(12, dtype=float), 3),
            "quantile": np.tile(np.repeat([3, 2, 1], 4), 3),
            "1D_fwd_return": np.random.default_rng(120).normal(0.001, 0.01, 36),
            "5D_fwd_return": np.random.default_rng(121).normal(0.002, 0.02, 36),
            "group": ["a", "b"] * 18,
        }
    )
    return pl.from_pandas(frame)


def test_supplied_buckets_and_all_shared_rows_reach_existing_calculations_without_prices():
    from ml4t.diagnostic import analyze_prepared_signal

    panel = supplied_panel()
    result = analyze_prepared_signal(panel, periods=(5, 1), quantiles=3, min_assets=2)
    assert result.prepared_data.equals(panel)
    assert result.periods == (5, 1)
    assert result.n_assets == 12
    assert result.n_dates == 3
    assert result.count_by_quantile == {1: 12, 2: 12, 3: 12}
    assert result.preparation_metadata["input_rows"] == 36
    assert result.preparation_metadata["excluded_rows"] == 0
    assert result.turnover is None
    for period in (5, 1):
        for bucket in (1, 2, 3):
            expected = panel.filter(pl.col("quantile") == bucket)[f"{period}D_fwd_return"].mean()
            assert result.quantile_returns[f"{period}D"][bucket] == pytest.approx(expected)
        expected_ic = [
            spearmanr(part["factor"].to_numpy(), part[f"{period}D_fwd_return"].to_numpy()).statistic
            for part in panel.partition_by("date", maintain_order=True)
        ]
        np.testing.assert_allclose(result.ic_series[f"{period}D"], expected_ic)


def test_ordinary_alphalens_multiindex_and_column_names_adapt_without_requantizing():
    from ml4t.diagnostic import analyze_prepared_signal

    panel = supplied_panel()
    legacy = (
        panel.to_pandas()
        .rename(
            columns={"quantile": "factor_quantile", "1D_fwd_return": "1D", "5D_fwd_return": "5D"}
        )
        .set_index(["date", "asset"])
    )
    result = analyze_prepared_signal(legacy, periods=(1, 5), quantiles=3, min_assets=2)
    assert result.prepared_data.equals(panel)
    assert legacy.index.names == ["date", "asset"]
    assert "factor_quantile" in legacy.columns


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate",
        "missing_factor",
        "missing_return",
        "infinite",
        "fractional_bucket",
        "bucket_range",
        "missing_asset",
        "missing_horizon",
        "ambiguous_bucket",
    ],
)
def test_invalid_prepared_pipeline_inputs_fail_before_output_changes(problem, tmp_path):
    from ml4t.diagnostic import analyze_prepared_signal

    panel = supplied_panel()
    if problem == "duplicate":
        panel = pl.concat([panel, panel.head(1)])
    elif problem == "missing_factor":
        panel = panel.with_columns(pl.col("factor").shift(-panel.height))
    elif problem == "missing_return":
        panel = panel.with_columns(pl.col("1D_fwd_return").shift(-panel.height))
    elif problem == "infinite":
        panel = panel.with_columns((pl.col("1D_fwd_return") / 0).alias("1D_fwd_return"))
    elif problem == "fractional_bucket":
        panel = panel.with_columns(pl.col("quantile") / 2)
    elif problem == "bucket_range":
        panel = panel.with_columns(pl.col("quantile") + 1)
    elif problem == "missing_asset":
        panel = panel.with_columns(pl.col("asset").replace_strict({}, default=None))
    elif problem == "missing_horizon":
        panel = panel.drop("1D_fwd_return")
    else:
        panel = panel.with_columns(pl.col("quantile").alias("factor_quantile"))
    before = panel.clone()
    output = tmp_path / "existing.json"
    output.write_text("existing result")
    with pytest.raises(
        (ValueError, TypeError), match="prepared|factor|return|quantile|asset|duplicate"
    ):
        analyze_prepared_signal(panel, periods=(1, 5), quantiles=3).to_json(str(output))
    assert panel.equals(before)
    assert output.read_text() == "existing result"


def test_prepared_snapshot_and_json_round_trip_preserve_nanoseconds_and_caller_ownership(tmp_path):
    from ml4t.diagnostic import SignalResult, analyze_prepared_signal

    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-02T00:00:00.000000001Z"),
            pd.Timestamp("2025-01-02T00:00:00.000000002Z"),
        ]
    )
    panel = supplied_panel().head(24).with_columns(pl.from_pandas(index.repeat(12)).alias("date"))
    result = analyze_prepared_signal(panel, periods=(1, 5), quantiles=3, min_assets=2)
    saved = panel.clone()
    panel.replace_column(panel.get_column_index("factor"), pl.Series("factor", [99.0] * 24))
    exposed = result.prepared_data
    exposed.replace_column(exposed.get_column_index("factor"), pl.Series("factor", [77.0] * 24))
    assert result.prepared_data.equals(saved)
    output = tmp_path / "result.json"
    result.to_json(str(output))
    restored = SignalResult.from_json(str(output))
    assert restored.prepared_data.equals(saved)
    assert restored.preparation_metadata == result.preparation_metadata
    assert restored.periods == (1, 5)
    assert result.ic_dates["1D"] == [str(stamp) for stamp in index]


def test_existing_public_ic_extraction_cannot_collapse_distinct_nanosecond_keys():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-02T00:00:00.000000001Z"),
            pd.Timestamp("2025-01-02T00:00:00.000000002Z"),
        ]
    )
    panel = supplied_panel().head(24).with_columns(pl.from_pandas(index.repeat(12)).alias("date"))
    dates, _ = extract_signal_ic_series(panel, 1, min_obs=2)
    assert [pd.Timestamp(stamp).value for stamp in dates] == list(index.asi8)


def test_single_bucket_pipeline_has_no_available_long_short_spread():
    panel = quantize_factor(supplied_panel(), n_quantiles=1)
    spread = compute_spread(panel, 1, n_quantiles=1)
    assert all(np.isnan(value) for value in spread.values())


def test_numpy_horizon_grid_can_serialize_without_numpy_integer_metadata_errors(tmp_path):
    from ml4t.diagnostic import SignalResult, analyze_prepared_signal

    result = analyze_prepared_signal(
        supplied_panel(), periods=tuple(np.array([5, 1])), quantiles=np.int64(3)
    )
    output = tmp_path / "result.json"
    result.to_json(str(output))
    assert SignalResult.from_json(str(output)).periods == (5, 1)


def test_single_supplied_bucket_remains_useful_without_a_fabricated_long_short_book():
    from ml4t.diagnostic import analyze_prepared_signal

    panel = supplied_panel().with_columns(pl.lit(1).alias("quantile"))
    result = analyze_prepared_signal(panel, periods=(1,), quantiles=1)
    assert result.prepared_data.height == 36
    assert np.isfinite(result.quantile_returns["1D"][1])
    assert np.isnan(result.spread["1D"])
    assert np.isnan(result.spread_p_value["1D"])
