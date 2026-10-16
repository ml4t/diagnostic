"""Dated persistence ranks each complete scored cross-section before pairing assets."""

import itertools
import math

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import compute_autocorrelation, prepare_signal_panel


def _inputs(profile="departures", nanoseconds=False):
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    if nanoseconds:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, unit="ns") for i in range(5)])
    prices = pd.DataFrame(
        {
            "A": [128.0, 144.0, 180.0, 202.5, 253.125],
            "B": [128.0, 160.0, 140.0, 175.0, 153.125],
            "C": [128.0, 192.0, 288.0, 432.0, 648.0],
            "D": [128.0, 256.0, 288.0, 576.0, 648.0],
        },
        index=dates,
    )
    if profile in {"mixed", "ties"}:
        prices["E"] = [128.0, 320.0, 128.0, 320.0, 128.0]
        prices["F"] = [128.0, 384.0, 128.0, 384.0, 128.0]
    if profile == "ties":
        prices.loc[dates[1], ["C", "D"]] = 160.0
    elif profile == "swap":
        prices.loc[dates[3], ["B", "C"]] = [210.0, 360.0]
    elif profile == "reverse":
        prices.loc[dates[3]] = prices.loc[dates[2]] * [0.875, 0.75, 0.5, 0.25]
    elif profile == "constant":
        prices = pd.concat([prices["A"].rename(asset) for asset in "ABCD"], axis=1)
    elif profile == "singleton":
        prices = prices[["A"]]
    factor = (
        prices.pct_change(fill_method=None)
        .loc[dates[[1, 3]]]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    removed = {"departures": ["C"], "mixed": ["B", "C"], "ties": ["C", "D"]}
    if profile in removed:
        factor = factor.drop([(dates[3], asset) for asset in removed[profile]])
    marks = prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"])
    returns = (
        compute_forward_returns(marks.reset_index(), periods=[1, 4], group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_4"]]
        .rename(columns={"fwd_ret_1": "1D", "fwd_ret_4": "4D"})
    )
    panel = prepare_signal_panel(factor, returns[["1D"]], quantiles=1, max_loss=0)
    raw = factor.to_frame().join(returns)
    assert panel.data.height == len(factor)
    return panel, raw, dates


def _direct(data, **options):
    from ml4t.diagnostic.signal import compute_rank_autocorrelation

    return compute_rank_autocorrelation(data, **options)


@pytest.mark.parametrize(
    "profile,expected,n_previous,n_current,n_obs",
    [
        ("departures", 9 / math.sqrt(84), 4, 3, 3),
        ("mixed", 8 / math.sqrt(70), 6, 4, 4),
        ("ties", 17 / math.sqrt(295), 6, 4, 4),
        ("swap", 0.8, 4, 4, 4),
        ("reverse", -1.0, 4, 4, 4),
    ],
)
@pytest.mark.parametrize(
    "route,representation", list(itertools.product(["panel", "direct"], ["pandas", "polars"]))
)
def test_quote_ranks_are_computed_before_asset_intersection_and_average_ties(
    profile, expected, n_previous, n_current, n_obs, route, representation
):
    panel, raw, _ = _inputs(profile)
    before = panel.data
    data = raw if representation == "pandas" else pl.from_pandas(raw.reset_index())
    original = data.copy(deep=True) if representation == "pandas" else data.clone()
    result = panel.rank_autocorrelation() if route == "panel" else _direct(data)
    rows = result.to_dicts()
    assert len(rows) == 2
    first, last = rows
    assert first["status"] == "insufficient_history" and first["autocorrelation"] is None
    assert first["n_previous"] is None and first["n_obs"] == 0
    assert last["status"] == "defined"
    assert last["autocorrelation"] == pytest.approx(expected, abs=1e-14)
    assert (last["n_previous"], last["n_current"], last["n_obs"]) == (n_previous, n_current, n_obs)
    assert last["previous_universe_size"] == n_previous
    assert last["current_universe_size"] == n_current
    assert last["rank_scope"] == "date" and last["tie_method"] == "average"
    assert last["correlation_method"] == "pearson" and last["min_obs"] == 2
    assert last["clock_source"] == "observed" and last["lag_unit"] == "observations"
    assert panel.data.equals(before)
    if representation == "pandas":
        pd.testing.assert_frame_equal(data, original)
    else:
        assert data.equals(original)


def test_legacy_paired_spearman_definition_remains_distinct():
    panel, _, _ = _inputs()
    assert compute_autocorrelation(panel.data, lags=[1], min_obs=2) == [1.0]


@pytest.mark.parametrize(
    "nanoseconds,representation", list(itertools.product([False, True], ["pandas", "polars"]))
)
def test_supplied_missing_observation_is_not_replaced_by_next_available_date(
    nanoseconds, representation
):
    panel, _, dates = _inputs(nanoseconds=nanoseconds)
    grid = pd.DataFrame({"date": dates[[1, 2, 3]]})
    if representation == "polars":
        grid = pl.from_pandas(grid)
    original = grid.copy(deep=True) if representation == "pandas" else grid.clone()
    result = panel.rank_autocorrelation(lags=[2, 1, 5], date_grid=grid)
    assert result["lag"].to_list() == [2] * 3 + [1] * 3 + [5] * 3
    selected = result.filter(pl.col("lag") == 2).to_pandas()
    assert [pd.Timestamp(value).value for value in selected["date"]] == [
        day.value for day in dates[[1, 2, 3]]
    ]
    assert selected.iloc[-1]["autocorrelation"] == pytest.approx(9 / math.sqrt(84))
    missing = result.filter(pl.col("lag") == 1).to_dicts()
    assert all(row["autocorrelation"] is None for row in missing)
    assert missing[1]["n_current"] == missing[1]["current_universe_size"] == 0
    assert missing[2]["n_previous"] == missing[2]["previous_universe_size"] == 0
    assert missing[2]["status"] == "insufficient_observations"
    assert result["clock_source"].unique().to_list() == ["supplied"]
    if representation == "pandas":
        pd.testing.assert_frame_equal(grid, original)
    else:
        assert grid.equals(original)


@pytest.mark.parametrize(
    "profile,status,count",
    [("constant", "constant_ranks", 4), ("singleton", "insufficient_observations", 1)],
)
def test_constant_or_singleton_quote_cross_sections_are_undefined_not_zero(profile, status, count):
    panel, _, _ = _inputs(profile)
    row = panel.rank_autocorrelation().to_dicts()[-1]
    assert row["autocorrelation"] is None and row["status"] == status
    assert row["n_obs"] == count


def test_explicit_minimum_does_not_hide_actual_common_membership():
    panel, _, _ = _inputs()
    row = panel.rank_autocorrelation(min_obs=4).to_dicts()[-1]
    assert row["n_obs"] == 3 and row["status"] == "insufficient_observations"
    assert row["autocorrelation"] is None and row["min_obs"] == 4


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_partial_score_feed_drops_only_unrankable_scores_and_retains_universe(missing):
    _, _, dates = _inputs("swap")
    _, scored, _ = _inputs()
    keys = pd.MultiIndex.from_product(
        [dates[[1, 3]], list("ABCD")], names=["date", "asset"]
    ).to_frame(index=False)
    joined = keys.merge(scored[["factor"]].reset_index(), on=["date", "asset"], how="left")
    values = pl.from_pandas(joined, nan_to_null=missing is None)
    row = _direct(values).to_dicts()[-1]
    assert row["current_universe_size"] == 4 and row["n_current"] == row["n_obs"] == 3
    assert row["autocorrelation"] == pytest.approx(9 / math.sqrt(84))


@pytest.mark.parametrize(
    "renames",
    [
        {"factor": "asset_score", "asset": "score", "date": "asof"},
        {"factor": "score", "asset": "rank", "date": "__clock_key"},
    ],
)
def test_physical_selectors_do_not_collide_with_internal_rank_or_pair_aliases(renames):
    panel, _, dates = _inputs(nanoseconds=True)
    values = panel.data.rename(renames)
    row = _direct(
        values,
        factor_col=renames["factor"],
        date_col=renames["date"],
        asset_col=renames["asset"],
        date_grid=pl.from_pandas(pd.DataFrame({renames["date"]: dates[[1, 3]]})),
    ).to_dicts()[-1]
    assert row["autocorrelation"] == pytest.approx(9 / math.sqrt(84))


@pytest.mark.parametrize(
    "options",
    [
        {"lags": []},
        {"lags": [0]},
        {"lags": [True]},
        {"lags": [1, 1]},
        {"min_obs": 1},
        {"min_obs": True},
        {"min_obs": 2.5},
    ],
)
def test_invalid_lags_or_minimum_cannot_silently_change_the_sample(options):
    panel, _, _ = _inputs()
    with pytest.raises(ValueError):
        panel.rank_autocorrelation(**options)


@pytest.mark.parametrize("kind", ["duplicate", "omitted", "timezone"])
def test_duplicate_missing_or_incompatible_schedule_grid_is_rejected(kind):
    panel, _, dates = _inputs()
    clock = (
        dates[[1, 1, 3]]
        if kind == "duplicate"
        else dates[[1]]
        if kind == "omitted"
        else dates[[1, 3]].tz_localize(None)
    )
    with pytest.raises((ValueError, TypeError)):
        panel.rank_autocorrelation(date_grid=pd.DataFrame({"date": clock}))


def test_repeated_feed_batches_cannot_multiply_asset_pairs():
    panel, _, _ = _inputs()
    with pytest.raises(ValueError, match="unique|duplicate"):
        _direct(pl.concat([panel.data, panel.data]))


def test_zero_price_produced_infinite_score_is_ordered_before_pairing():
    dates = pd.bdate_range("2025-01-02", periods=3)
    prices = pd.DataFrame(
        {"A": [2.0, 2.0, 2.0], "B": [1.0, 2.0, 4.0], "C": [0.0, 2.0, 0.0]}, index=dates
    )
    factor = (
        prices.pct_change(fill_method=None)
        .iloc[1:]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(["date", "asset"])
    )
    assert math.isinf(factor.loc[(dates[1], "C")])
    row = _direct(factor.to_frame()).to_dicts()[-1]
    assert row["n_obs"] == 3 and row["autocorrelation"] == pytest.approx(-0.5)


@pytest.mark.parametrize("kind", ["missing_scores", "warmup"])
def test_unavailable_model_outputs_retain_typed_empty_or_unrankable_observations(kind):
    panel, raw, dates = _inputs()
    if kind == "missing_scores":
        keys = raw.reset_index()[["date", "asset"]]
        empty_scores = raw.iloc[:0][["factor"]].reset_index()
        data = keys.merge(empty_scores, on=["date", "asset"], how="left")
        result = _direct(data)
        assert result["n_current"].to_list() == [0, 0]
        assert result["current_universe_size"].to_list() == [4, 3]
        assert result["autocorrelation"].null_count() == 2
    else:
        warmup = raw[["factor"]].reset_index()
        warmup["factor"] = warmup.groupby("asset")["factor"].transform(
            lambda values: values.rolling(20).mean()
        )
        cleaned = prepare_signal_panel(warmup, raw[["1D"]], quantiles=1, max_loss=1)
        assert cleaned.data.is_empty()
        result = cleaned.rank_autocorrelation()
        assert result.is_empty() and result.schema["autocorrelation"] == pl.Float64
        grid = cleaned.rank_autocorrelation(date_grid=pd.DataFrame({"date": dates[[1, 3]]}))
        assert grid["n_current"].to_list() == [0, 0]
        assert grid["autocorrelation"].null_count() == 2
    assert panel.data.height == 7


@pytest.mark.parametrize(
    "timezone,origin", [("America/New_York", "1969-12-31"), (None, "2025-01-02")]
)
def test_precise_negative_epoch_and_declared_timezone_survive_rank_pairing(timezone, origin):
    panel, _, _ = _inputs(nanoseconds=True)
    values = panel.data
    new_dates = pd.DatetimeIndex(
        [pd.Timestamp(origin, tz=timezone) + pd.Timedelta(i * 200, unit="ns") for i in range(2)]
    )
    old_dates = values["date"].unique().sort()
    mapping = pl.DataFrame({"date": old_dates, "new_date": pl.from_pandas(pd.Series(new_dates))})
    values = values.join(mapping, on="date").drop("date").rename({"new_date": "date"})
    result = _direct(values).to_pandas()
    assert [pd.Timestamp(value).value for value in result["date"]] == [
        day.value for day in new_dates
    ]
    assert result.iloc[-1]["autocorrelation"] == pytest.approx(9 / math.sqrt(84))
