"""Factor regression must retain dated samples and compound the fitted intercept."""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import ml4t.diagnostic.signal as signal


def _price_panel(nanoseconds=False, periods=(1,)):
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
    frame = signal.prepare_data(factor, marks, periods=periods, quantiles=2, filter_zscore=None)
    return (
        frame.rename({f"{p}D_fwd_return": "outcome" if p == 1 else "second" for p in periods}),
        dates[:-1],
        universe,
    )


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


def test_supplied_regression_does_not_require_or_validate_an_unused_factor_score():
    frame, dates, universe = _price_panel()
    supplied = pd.Series(0.001 + 0.6 * universe, index=dates)
    result = _analyze(frame.drop("factor"), returns=supplied, factor_col=None)
    assert result.statistics["beta"][0] == pytest.approx(0.6, abs=1e-12)
    assert result.observations["n_universe"].to_list() == [2] * 8
    assert result.observations["paired"].to_list() == [True] * 8
    with pytest.raises(ValueError, match="factor_col is required"):
        _analyze(frame.drop("factor"), factor_col=None)


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


def test_two_price_derived_horizons_retain_requested_order_and_separate_joint_samples():
    frame, _, _ = _price_panel(periods=(1, 2))
    result = signal.compute_factor_alpha_beta(
        frame,
        return_columns=["second", "outcome"],
        annualization_factors={"outcome": 252, "second": 126},
        demeaned=False,
    )
    assert result.statistics["return_column"].to_list() == ["second", "outcome"]
    assert result.statistics["n_obs"].to_list() == [7, 8]
    assert result.statistics["annualization_factor"].to_list() == [126, 252]
    sample = frame.to_pandas()
    for column in ["second", "outcome"]:
        wide = sample.pivot(index="date", columns="asset", values=column).dropna()
        x = wide.mean(axis=1).to_numpy()
        y = wide.A.to_numpy()
        expected_beta = np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean())
        expected_intercept = y.mean() - expected_beta * x.mean()
        row = result.statistics.filter(pl.col("return_column") == column).row(0, named=True)
        assert row["beta"] == pytest.approx(expected_beta, abs=1e-12)
        assert row["alpha_per_period"] == pytest.approx(expected_intercept, abs=1e-12)
        assert row["annual_alpha"] == pytest.approx(
            np.expm1(np.log1p(expected_intercept) * row["annualization_factor"]), abs=1e-12
        )


@pytest.mark.parametrize("n", [0, 1])
def test_insufficient_joint_sample_reports_counts_and_nulls_instead_of_fitting(n):
    frame, dates, _ = _price_panel()
    returns = pd.Series([0.01] * n, index=dates[:n], dtype=float)
    result = _analyze(frame, returns=returns)
    row = result.statistics.row(0, named=True)
    assert row["n_dates"] == row["n_obs"] == n
    assert row["status"] == ("no_joint_observations" if n == 0 else "insufficient_observations")
    assert row["alpha_per_period"] is row["beta"] is row["annual_alpha"] is None
    assert result.observations.height == n
    restored = signal.FactorAlphaBetaResult.from_json(result.to_json())
    assert_frame_equal(restored.statistics, result.statistics)


def test_supplied_dates_outside_universe_are_visible_but_never_enter_the_regression():
    frame, dates, universe = _price_panel()
    supplied = pd.Series(0.001 + 0.6 * universe, index=dates)
    extra = dates[-1] + pd.Timedelta(days=1)
    supplied.loc[extra] = 100
    result = _analyze(frame, returns=supplied)
    row = result.observations.tail(1).row(0, named=True)
    assert row["n_universe"] == 0 and row["universe_return"] is None
    assert row["portfolio_return"] == 100 and row["paired"] is False
    assert row["fitted_return"] is None and row["residual"] is None
    assert result.statistics["n_dates"][0] == 9 and result.statistics["n_obs"][0] == 8
    assert result.statistics["beta"][0] == pytest.approx(0.6, abs=1e-12)


def test_zero_signal_is_zero_exposure_with_zero_intercept_against_a_varying_universe():
    frame, _, _ = _price_panel()
    result = _analyze(frame.with_columns(pl.lit(0.0).alias("factor")))
    assert result.observations["portfolio_return"].to_list() == [0.0] * 8
    row = result.statistics.row(0, named=True)
    assert row["status"] == "defined" and row["n_obs"] == 8
    assert row["alpha_per_period"] == row["beta"] == row["annual_alpha"] == 0


@pytest.mark.parametrize(
    "value,status,annual",
    [
        (-1.0, "defined", -1.0),
        (-1.1, "negative_gross_intercept", None),
        (100.0, "annualization_overflow", None),
        (1e308, "annualization_overflow", None),
    ],
)
def test_compound_domain_and_overflow_do_not_erase_defined_period_coefficients(
    value, status, annual
):
    frame, dates, _ = _price_panel()
    result = _analyze(frame, returns=pd.Series(value, index=dates))
    row = result.statistics.row(0, named=True)
    assert row["status"] == "defined" and row["alpha_per_period"] == value and row["beta"] == 0
    assert row["annualization_status"] == status and row["annual_alpha"] == annual
    json.loads(result.to_json(), parse_constant=lambda token: pytest.fail(token))


@pytest.mark.parametrize("change", ["duplicate", "missing", "timezone"])
def test_ambiguous_provided_date_keys_are_rejected_before_regression(change):
    frame, dates, _ = _price_panel()
    index = dates.copy()
    if change == "duplicate":
        index = pd.DatetimeIndex([dates[0]] * 8)
    elif change == "missing":
        index = pd.DatetimeIndex([pd.NaT, *dates[1:]])
    else:
        index = dates.tz_convert("America/New_York")
    with pytest.raises(ValueError, match="duplicate|nonmissing|timezones"):
        _analyze(frame, returns=pd.Series(0.01, index=index))


def test_tiny_supplied_intercept_retains_compound_alpha_instead_of_rounding_gross_to_one():
    frame, dates, _ = _price_panel()
    result = _analyze(frame, returns=pd.Series(1e-18, index=dates))
    row = result.statistics.row(0, named=True)
    assert row["alpha_per_period"] == pytest.approx(1e-18, abs=1e-32)
    assert row["annual_alpha"] == pytest.approx(2.52e-16, abs=1e-30)
    assert row["annual_alpha"] > 0


def test_custom_physical_names_do_not_change_the_public_sample_or_regression():
    frame, _, _ = _price_panel()
    renamed = frame.rename({"date": "instant", "asset": "security", "factor": "score"})
    expected = _analyze(frame, demeaned=False)
    actual = _analyze(
        renamed, date_col="instant", asset_col="security", factor_col="score", demeaned=False
    )
    assert_frame_equal(actual.statistics, expected.statistics)
    assert_frame_equal(actual.observations, expected.observations)


@pytest.mark.parametrize("provided_beta", [None, 1e-308], ids=["factor-portfolio", "tiny-exposure"])
def test_finite_price_derived_universe_mean_does_not_overflow_and_discard_a_valid_date(
    provided_beta,
):
    dates = pd.bdate_range("2024-09-09", periods=4, tz="UTC")
    prices = pd.DataFrame(
        {"A": [1.0, 1e308, 5e307, 7.5e307], "B": [1.0, 1e308, 5e307, 7.5e307]}, index=dates
    )
    marks = (
        prices.stack(future_stack=True).rename("price").rename_axis(["date", "asset"]).reset_index()
    )
    factor = marks[marks.date < dates[-1]][["date", "asset"]].assign(factor=1.0)
    frame = signal.prepare_data(factor, marks, periods=(1,), quantiles=2, filter_zscore=None)
    result = signal.compute_factor_alpha_beta(
        frame,
        return_columns=["1D_fwd_return"],
        annualization_factors={"1D_fwd_return": 252},
        demeaned=False,
        returns=None
        if provided_beta is None
        else prices.A.pct_change(fill_method=None).shift(-1).iloc[:-1] * provided_beta,
    )
    row = result.statistics.row(0, named=True)
    assert row["n_obs"] == row["n_dates"] == 3
    np.testing.assert_allclose(
        result.observations["universe_return"], [1e308, -0.5, 0.5], rtol=1e-14
    )
    assert result.observations["paired"].to_list() == [True] * 3
    assert result.observations["n_universe"].to_list() == [2] * 3
    assert result.observations["n_portfolio"].to_list() == (
        [2] * 3 if provided_beta is None else [1] * 3
    )
    assert row["beta"] == (1 if provided_beta is None else provided_beta)
    assert row["alpha_per_period"] == row["annual_alpha"] == 0
    assert row["fit_coordinate_policy"] == "unit_max_abs"


def test_extended_finite_annualization_scale_must_also_be_representable_in_result_float64():
    frame, _, _ = _price_panel()
    with np.errstate(over="ignore"):
        scale = np.longdouble("1e400")
    with pytest.raises(ValueError, match="annualization_factors"):
        signal.compute_factor_alpha_beta(
            frame, return_columns=["outcome"], annualization_factors={"outcome": scale}
        )
