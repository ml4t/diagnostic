"""Forecasts must train before the live boundary and retain each sampled observation."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _analysis(values, *, benchmark=None):
    dates = pd.date_range("2025-01-13", periods=len(values), tz="UTC")
    return PortfolioAnalysis.from_frames(
        pd.Series(values, index=dates),
        benchmark=None if benchmark is None else pd.Series(benchmark, index=dates),
        calendar=None,
    ), dates


def test_constant_training_has_exact_forward_products_and_actual_live_overlay():
    analysis, dates = _analysis([0.1, 0.1, 0.9, -0.8])
    result = analysis.compute_wealth_cones(
        live_start=dates[2],
        horizon=3,
        n_samples=4,
        seed=42,
        initial_capital=1000,
    )
    np.testing.assert_allclose(result.paths["wealth"], [1100, 1210, 1331] * 4)
    assert result.training["return"].to_list() == [0.1, 0.1]
    np.testing.assert_allclose(result.live["wealth"], [1900, 380])
    assert result.live["date"].to_list() == dates[2:].to_list()
    assert result.metadata["horizon_unit"] == "observations"
    assert result.metadata["calendar"] is None


@pytest.mark.parametrize("method", ["iid", "stationary"])
def test_all_post_live_values_can_change_without_changing_training_draws_or_bounds(method):
    first, dates = _analysis([0.0, 0.1, -0.01, 0.02, 0.03])
    changed, _ = _analysis([0.0, 0.1, 0.8, -0.7, 0.6])
    options = {"live_start": dates[2], "horizon": 9, "n_samples": 16, "seed": 123, "method": method}
    if method == "stationary":
        options["block_size"] = 3
    before, after = first.compute_wealth_cones(**options), changed.compute_wealth_cones(**options)
    assert before.training.equals(after.training)
    assert before.paths.equals(after.paths)
    assert before.bands.equals(after.bands)
    assert not before.live.equals(after.live)


def test_total_loss_does_not_resurrect_or_turn_into_nan():
    analysis, dates = _analysis([-1.0, -1.0, 0.1, 0.2])
    result = analysis.compute_wealth_cones(live_start=dates[2], horizon=3, n_samples=4)
    assert result.paths["wealth"].to_list() == [0.0] * 12


def test_paired_benchmark_uses_the_same_training_rows_without_destroying_covariance():
    analysis, dates = _analysis([0.01, 0.02, 0.03, 0.4], benchmark=[0.02, 0.04, 0.06, -0.8])
    result = analysis.compute_wealth_cones(live_start=dates[3], horizon=8, n_samples=3)
    np.testing.assert_allclose(
        result.paths["sampled_benchmark"], 2 * result.paths["sampled_return"]
    )
    for frame in result.paths.partition_by("sample", maintain_order=True):
        expected = 1.0
        for row in frame.iter_rows(named=True):
            expected *= 1 + [0.02, 0.04, 0.06][row["training_row"]]
            assert row["benchmark_wealth"] == pytest.approx(expected)


def test_stationary_paths_preserve_chronological_blocks_and_allow_longer_forecasts():
    analysis, dates = _analysis([0.01, 0.02, 0.03, 0.04, 0.05, 0.9])
    result = analysis.compute_wealth_cones(
        live_start=dates[5],
        horizon=20,
        n_samples=2,
        method="stationary",
        block_size=1e9,
    )
    for frame in result.paths.partition_by("sample", maintain_order=True):
        rows = frame["training_row"].to_numpy()
        np.testing.assert_array_equal(rows[1:], (rows[:-1] + 1) % 5)


def test_quantiles_and_population_deviation_are_distinct_defined_band_rows():
    analysis, dates = _analysis([-0.1, 0.1, 0.9])
    result = analysis.compute_wealth_cones(
        live_start=dates[2],
        horizon=1,
        n_samples=20,
        seed=7,
        quantiles=(0.0, 1.0),
        cone_std=(1.0,),
        initial_capital=100,
    )
    values = result.paths["wealth"].to_numpy()
    rows = result.bands.filter(pl.col("step") == 1)
    assert rows.filter((pl.col("definition") == "quantile") & (pl.col("level") == 0))["wealth"][
        0
    ] == pytest.approx(90)
    assert rows.filter((pl.col("definition") == "quantile") & (pl.col("level") == 1))["wealth"][
        0
    ] == pytest.approx(110)
    actual = rows.filter((pl.col("definition") == "mean_std") & (pl.col("level") == -1))["wealth"][
        0
    ]
    assert actual == pytest.approx(values.mean() - values.std(ddof=0))


def test_forecast_results_own_training_paths_bands_and_exact_clock_json():
    analysis, dates = _analysis([0.01, 0.02, -0.03])
    result = analysis.compute_wealth_cones(live_start=dates[2], horizon=3, n_samples=2)
    before = result.paths
    result.paths.replace_column(3, pl.Series("sampled_return", [999.0] * 6))
    result.metadata["seed"] = 99
    assert result.paths.equals(before)
    restored = type(result).from_json(result.to_json())
    for name in ("training", "live", "paths", "bands"):
        assert getattr(restored, name).equals(getattr(result, name))
    assert restored.metadata == result.metadata


@pytest.mark.parametrize(
    "options, message",
    [
        ({"horizon": 0}, "horizon"),
        ({"n_samples": True}, "n_samples"),
        ({"method": "stationary", "block_size": 0}, "block_size"),
        ({"method": "iid", "block_size": 2}, "block_size"),
        ({"initial_capital": 0}, "initial_capital"),
        ({"quantiles": (0.5, 0.5)}, "quantiles"),
        ({"cone_std": (-1,)}, "cone_std"),
        ({"horizon_unit": "days"}, "horizon_unit"),
    ],
)
def test_invalid_forecast_choices_fail_before_producing_a_result(options, message):
    analysis, dates = _analysis([0.01, 0.02, 0.03])
    kwargs = {"live_start": dates[2], "horizon": 3, "n_samples": 2}
    kwargs.update(options)
    with pytest.raises(ValueError, match=message):
        analysis.compute_wealth_cones(**kwargs)


def test_empty_training_is_not_replaced_by_future_observations():
    analysis, dates = _analysis([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="pre-live"):
        analysis.compute_wealth_cones(live_start=dates[0], horizon=3, n_samples=2)


def test_none_live_boundary_cannot_silently_use_the_whole_sample_for_forecasting():
    analysis, _ = _analysis([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="live_start"):
        analysis.compute_wealth_cones(live_start=None, horizon=3, n_samples=2)
