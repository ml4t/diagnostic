"""Prevent IC decay from reporting input rows instead of paired horizon samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_ic_decay


def _workflow(scenario: str) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, int]]:
    dates = pd.bdate_range("2025-01-02", periods=4, tz="UTC")
    prices = pl.DataFrame(
        {
            "date": [date for date in dates for _ in range(2)],
            "asset": ["A", "B"] * 4,
            "close": [256.0, 512.0, 288.0, 640.0, 270.0, 720.0, 337.5, 674.0],
        }
    )
    predictions = prices.select("date", "asset", pl.col("close").alias("prediction"))
    expected = {1: 6, 2: 4, 3: 2}
    if scenario == "causal_warmup":
        predictions = prices.with_columns(
            pl.col("close").pct_change().over("asset").alias("prediction")
        ).select("date", "asset", "prediction")
        expected = {1: 4, 2: 2, 3: 0}
    elif scenario == "sparse_query":
        predictions = predictions.filter(pl.col("date") != dates[1])
        expected = {1: 4, 2: 2, 3: 2}
    elif scenario == "observed_price_gap":
        prices = prices.filter(~((pl.col("date") == dates[1]) & (pl.col("asset") == "B")))
        expected = {1: 5, 2: 3, 3: 1}
    elif scenario == "missing_endpoint":
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
        marks = prices.to_pandas().set_index(["date", "asset"])
        marks = marks.drop(index=(dates[1], "B")).reindex(index)
        prices = pl.from_pandas(marks.reset_index())
        expected = {1: 4, 2: 3, 3: 2}
    elif scenario == "one_pair":
        predictions = predictions.filter(
            ((pl.col("date") == dates[0]) & (pl.col("asset") == "A"))
            | ((pl.col("date") == dates[3]) & (pl.col("asset") == "B"))
        )
        expected = {1: 1, 2: 1, 3: 1}
    elif scenario == "constant":
        predictions = predictions.with_columns(pl.lit(1.0).alias("prediction"))
    elif scenario == "no_overlap":
        predictions = predictions.with_columns(pl.col("date").dt.offset_by("30d"))
        expected = {1: 0, 2: 0, 3: 0}
    elif scenario == "all_warmup":
        predictions = prices.with_columns(
            pl.col("close").rolling_mean(window_size=5).over("asset").alias("prediction")
        ).select("date", "asset", "prediction")
        expected = {1: 0, 2: 0, 3: 0}
    elif scenario == "empty_query":
        predictions = predictions.head(0)
        expected = {1: 0, 2: 0, 3: 0}
    return predictions, prices, expected


@pytest.mark.parametrize(
    "scenario",
    [
        "complete",
        "causal_warmup",
        "sparse_query",
        "observed_price_gap",
        "missing_endpoint",
        "one_pair",
        "constant",
        "no_overlap",
        "all_warmup",
        "empty_query",
    ],
)
@pytest.mark.parametrize("pandas_input", [False, True])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_decay_counts_the_observations_actually_used_at_each_horizon(
    scenario, pandas_input, method
):
    predictions, prices, expected = _workflow(scenario)
    result = compute_ic_decay(
        predictions.to_pandas() if pandas_input else predictions,
        prices.to_pandas() if pandas_input else prices,
        horizons=[1, 2, 3],
        group_col="asset",
        method=method,
        estimate_half_life=False,
    )

    assert result["n_observations"] == expected
    for horizon, count in expected.items():
        if count < 2 or scenario == "constant":
            assert np.isnan(result["ic_by_horizon"][horizon])
