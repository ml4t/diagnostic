"""Factor regression must retain dated samples and compound the fitted intercept."""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import ml4t.diagnostic.signal as signal


def _price_panel(nanoseconds=False):
    dates = pd.bdate_range("2024-09-09", periods=9, tz="UTC")
    if nanoseconds:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, unit="ns") for i in range(9)])
    universe = np.array([-0.03, 0.01, 0.04, -0.02, 0.005, 0.02, -0.01, 0.015])
    prices = pd.DataFrame(
        {
            "A": 100 * np.r_[1, np.cumprod(1 + 0.001 + 0.6 * universe)],
            "B": 100 * np.r_[1, np.cumprod(1 - 0.001 + 1.4 * universe)],
        },
        index=dates,
    )
    marks = (
        prices.stack(future_stack=True).rename("price").rename_axis(["date", "asset"]).reset_index()
    )
    factor = marks[marks.date < dates[-1]][["date", "asset"]].copy()
    factor["factor"] = (factor.asset == "A").astype(float)
    frame = signal.prepare_data(factor, marks, periods=(1,), quantiles=2, filter_zscore=None)
    return frame.rename({"1D_fwd_return": "outcome"}), dates[:-1], universe


def _analyze(frame, **options):
    return signal.compute_factor_alpha_beta(
        frame, return_columns=["outcome"], annualization_factors={"outcome": 252}, **options
    )


def test_price_derived_factor_regression_compounds_intercept_not_generic_linear_annual_alpha():
    frame, dates, universe = _price_panel()
    result = _analyze(frame, demeaned=False)
    row = result.statistics.row(0, named=True)
    assert row["beta"] == pytest.approx(0.6, abs=1e-12)
    assert row["alpha_per_period"] == pytest.approx(0.001, abs=1e-12)
    assert row["annual_alpha"] == pytest.approx(np.expm1(np.log1p(0.001) * 252), abs=1e-12)
    assert row["annual_alpha"] != pytest.approx(0.252, abs=1e-3)
    assert row["n_obs"] == 8 and row["status"] == "defined"
    observations = result.observations
    assert observations["date"].to_pandas().tolist() == dates.tolist()
    np.testing.assert_allclose(observations["universe_return"], universe, atol=1e-14)
    np.testing.assert_allclose(observations["portfolio_return"], 0.001 + 0.6 * universe, atol=1e-14)
    assert observations["n_universe"].to_list() == [2] * 8
    assert observations["n_portfolio"].to_list() == [2] * 8
    assert observations["paired"].to_list() == [True] * 8
    np.testing.assert_allclose(observations["residual"], 0, atol=1e-14)


@pytest.mark.parametrize("representation", ["series", "pandas", "polars"])
def test_supplied_returns_join_dates_instead_of_truncating_or_mutating_the_caller(representation):
    frame, dates, universe = _price_panel()
    provided = pd.Series(-0.002 - 0.3 * universe, index=dates, name="caller_name").iloc[::-1]
    provided.iloc[2] = np.nan
    provided = provided.drop(dates[0])
    if representation != "series":
        provided = provided.rename("outcome").to_frame()
        if representation == "polars":
            provided = pl.from_pandas(provided.rename_axis("date").reset_index())
    before = provided.clone() if isinstance(provided, pl.DataFrame) else provided.copy(deep=True)
    result = _analyze(frame, returns=provided, group_adjust=True)
    row = result.statistics.row(0, named=True)
    assert row["n_dates"] == 7 and row["n_obs"] == 6
    assert row["beta"] == pytest.approx(-0.3, abs=1e-12)
    assert row["alpha_per_period"] == pytest.approx(-0.002, abs=1e-12)
    assert row["annual_alpha"] == pytest.approx(np.expm1(np.log1p(-0.002) * 252), abs=1e-12)
    assert result.metadata["weight_options"] is None
    assert result.observations["n_portfolio"].to_list().count(0) == 1
    if isinstance(provided, pl.DataFrame):
        assert_frame_equal(provided, before)
    elif isinstance(provided, pd.Series):
        pd.testing.assert_series_equal(provided, before)
    else:
        pd.testing.assert_frame_equal(provided, before)


def test_panel_and_strict_json_round_trip_preserve_owned_nanosecond_samples():
    frame, _, _ = _price_panel(nanoseconds=True)
    panel = signal.prepare_signal_panel(
        frame.select("date", "asset", "factor"),
        frame.select("date", "asset", "outcome"),
        quantiles=2,
        max_loss=0,
    )
    before = panel.data
    result = panel.factor_alpha_beta(annualization_factors={"outcome": 252}, demeaned=False)
    restored = signal.FactorAlphaBetaResult.from_json(result.to_json())
    assert_frame_equal(restored.observations, result.observations)
    assert_frame_equal(restored.statistics, result.statistics)
    assert restored.observations["date"].n_unique() == 8
    assert restored.observations.schema["date"] == pl.Datetime("ns", "UTC")
    json.loads(result.to_json(), parse_constant=lambda token: pytest.fail(token))
    copy = result.statistics
    copy.replace_column(copy.columns.index("beta"), pl.Series("beta", [-99.0]))
    metadata = result.metadata
    metadata["weight_options"]["demeaned"] = True
    assert result.statistics["beta"][0] == pytest.approx(0.6, abs=1e-12)
    assert result.metadata["weight_options"]["demeaned"] is False
    assert_frame_equal(panel.data, before)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"other": 252},
        {"outcome": 0},
        {"outcome": True},
        {"outcome": float("nan")},
        {"outcome": float("inf")},
    ],
)
def test_annualization_requires_explicit_positive_finite_scale_for_each_selected_outcome(value):
    frame, _, _ = _price_panel()
    with pytest.raises((TypeError, ValueError), match="annualization_factors"):
        signal.compute_factor_alpha_beta(
            frame, return_columns=["outcome"], annualization_factors=value
        )


@pytest.mark.parametrize("constant", [0.0, 0.01])
def test_constant_benchmark_is_unidentified_not_an_invented_zero_beta(constant):
    frame, dates, _ = _price_panel()
    frame = frame.with_columns(pl.lit(constant).alias("outcome"))
    supplied = pd.Series(np.arange(8) / 100, index=dates)
    result = _analyze(frame, returns=supplied)
    row = result.statistics.row(0, named=True)
    assert row["status"] == "constant_benchmark" and row["n_obs"] == 8
    assert row["alpha_per_period"] is None and row["beta"] is None
    assert row["annual_alpha"] is None


def test_missing_horizon_tails_stay_null_and_do_not_become_zero_portfolio_dates():
    frame, dates, _ = _price_panel()
    frame = frame.with_columns(
        pl.when(pl.col("date") == dates[-1])
        .then(None)
        .otherwise(pl.col("outcome"))
        .alias("outcome")
    )
    result = _analyze(frame, demeaned=False)
    last = result.observations.tail(1).row(0, named=True)
    assert last["portfolio_return"] is None and last["universe_return"] is None
    assert last["n_portfolio"] == last["n_universe"] == 0 and last["paired"] is False
    assert result.statistics["n_obs"].to_list() == [7]
    assert result.statistics["beta"][0] == pytest.approx(0.6, abs=1e-12)
