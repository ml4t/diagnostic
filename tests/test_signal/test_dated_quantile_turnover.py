"""Dated turnover must count entrants on a declared observation clock."""

import itertools

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import compute_turnover, prepare_signal_panel


def _inputs(nanoseconds=False):
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
    factor = (
        prices.pct_change(fill_method=None)
        .loc[dates[[1, 3]]]
        .stack(future_stack=True)
        .rename("factor")
    )
    factor.index = factor.index.set_names(["date", "asset"])
    factor = factor.drop((dates[3], "C"))
    marks = (
        prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    returns = (
        compute_forward_returns(marks, periods=1, group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1"]]
        .rename(columns={"fwd_ret_1": "1D"})
    )
    panel = prepare_signal_panel(factor, returns, quantiles=2, max_loss=0)
    expected = {
        (dates[1].value, "A"): 1,
        (dates[1].value, "B"): 1,
        (dates[1].value, "C"): 2,
        (dates[1].value, "D"): 2,
        (dates[3].value, "A"): 1,
        (dates[3].value, "B"): 1,
        (dates[3].value, "D"): 2,
    }
    assert {
        (pd.Timestamp(row["date"]).value, row["asset"]): row["quantile"]
        for row in panel.data.to_pandas().to_dict(orient="records")
    } == expected
    return panel, dates


def _call(panel, representation, route, **options):
    if route == "panel":
        return panel.quantile_turnover(**options)
    from ml4t.diagnostic.signal import compute_quantile_turnover

    data = panel.data
    value = data.to_pandas().set_index(["date", "asset"]) if representation == "pandas" else data
    before = value.copy(deep=True) if representation == "pandas" else value.clone()
    result = compute_quantile_turnover(value, **options)
    if representation == "pandas":
        pd.testing.assert_frame_equal(value, before)
    else:
        assert value.equals(before)
    return result


@pytest.mark.parametrize(
    "representation,route,supplied,nanoseconds",
    list(
        itertools.product(["pandas", "polars"], ["panel", "direct"], [False, True], [False, True])
    ),
)
def test_quote_derived_departures_do_not_become_entrants_or_close_clock_gaps(
    representation, route, supplied, nanoseconds
):
    panel, dates = _inputs(nanoseconds)
    clock = list(dates[[1, 2, 3]]) if supplied else list(dates[[1, 3]])
    grid = pd.DataFrame({"date": clock}) if supplied else None
    if supplied and representation == "polars":
        grid = pl.from_pandas(grid)
    before = panel.data
    result = _call(panel, representation, route, quantiles=[2, 1], lags=[2, 1], date_grid=grid)
    assert result.height == len(clock) * 4
    assert result.select("lag", "quantile").unique(maintain_order=True).rows() == [
        (2, 2),
        (2, 1),
        (1, 2),
        (1, 1),
    ]
    membership = {
        (dates[1].value, 1): {"A", "B"},
        (dates[1].value, 2): {"C", "D"},
        (dates[3].value, 1): {"A", "B"},
        (dates[3].value, 2): {"D"},
    }
    universe = {dates[1].value: 4, dates[3].value: 3}
    for row in result.to_pandas().to_dict(orient="records"):
        position = [day.value for day in clock].index(pd.Timestamp(row["date"]).value)
        current = membership.get((clock[position].value, row["quantile"]), set())
        previous = (
            membership.get((clock[position - row["lag"]].value, row["quantile"]), set())
            if position >= row["lag"]
            else None
        )
        assert row["n_current"] == len(current)
        assert (
            pd.isna(row["n_previous"]) if previous is None else row["n_previous"] == len(previous)
        )
        assert row["current_universe_size"] == universe.get(clock[position].value, 0)
        if previous is None:
            assert pd.isna(row["previous_universe_size"])
        else:
            assert row["previous_universe_size"] == universe.get(
                clock[position - row["lag"]].value, 0
            )
        if current and previous:
            assert row["turnover"] == 0.0
            assert row["n_entered"] == 0 and row["n_overlap"] == len(current)
            assert row["status"] == "defined"
        else:
            assert (
                pd.isna(row["turnover"]) and pd.isna(row["n_entered"]) and pd.isna(row["n_overlap"])
            )
            expected_status = (
                "insufficient_history"
                if previous is None
                else "empty_current"
                if not current
                else "empty_previous"
            )
            assert row["status"] == expected_status
        assert row["turnover_method"] == "entrants_over_current"
        assert row["lag_unit"] == "observations"
        assert row["clock_source"] == ("supplied" if supplied else "observed")
    assert compute_turnover(panel.data, n_quantiles=2) == 0.25
    assert panel.data.equals(before)


def test_existing_scalar_larger_membership_definition_remains_available():
    panel, _ = _inputs()
    assert compute_turnover(panel.data, n_quantiles=2) == 0.25


@pytest.mark.parametrize("representation", ["pandas", "polars"])
def test_custom_names_noncontiguous_labels_and_explicit_grid_keep_membership(representation):
    from ml4t.diagnostic.signal import compute_quantile_turnover

    panel, dates = _inputs(nanoseconds=True)
    data = panel.data.with_columns((pl.col("quantile") * 2).alias("quantile")).rename(
        {"date": "timestamp", "asset": "instrument", "quantile": "bucket"}
    )
    grid = pl.from_pandas(pd.DataFrame({"timestamp": dates[[1, 2, 3]]}))
    value = (
        data.to_pandas().set_index(["timestamp", "instrument"])
        if representation == "pandas"
        else data
    )
    result = compute_quantile_turnover(
        value,
        date_col="timestamp",
        asset_col="instrument",
        quantile_col="bucket",
        quantiles=[4, 2],
        date_grid=grid,
    )
    expected = panel.quantile_turnover(
        quantiles=[2, 1], date_grid=grid.rename({"timestamp": "date"})
    )
    assert result.drop("quantile").equals(expected.drop("quantile"))
    assert result["quantile"].unique(maintain_order=True).to_list() == [4, 2]


@pytest.mark.parametrize("bad", [(), (0,), (-1,), (True,), (1.5,), (1, 1), "1"])
@pytest.mark.parametrize("route", ["panel", "direct"])
def test_invalid_lag_declarations_reject_without_changing_owned_sample(bad, route):
    panel, _ = _inputs()
    before = panel.data
    with pytest.raises(ValueError, match="lags"):
        _call(panel, "polars", route, lags=bad)
    assert panel.data.equals(before)


@pytest.mark.parametrize("bad", [[], [0], [True], [1.5], [1, 1]])
def test_invalid_bucket_declarations_reject_instead_of_coercing_labels(bad):
    panel, _ = _inputs()
    with pytest.raises(ValueError, match="quantiles"):
        panel.quantile_turnover(quantiles=bad)


def test_duplicate_feed_batches_and_incomplete_grids_do_not_silently_change_clock():
    from ml4t.diagnostic.signal import compute_quantile_turnover

    panel, dates = _inputs()
    batch = pl.concat([panel.data, panel.data])
    with pytest.raises(ValueError, match="duplicate"):
        compute_quantile_turnover(batch)
    grid = pl.from_pandas(pd.DataFrame({"date": dates[[1, 3]]}))
    with pytest.raises(ValueError, match="duplicate"):
        panel.quantile_turnover(date_grid=pl.concat([grid, grid]))
    with pytest.raises(ValueError, match="cover"):
        panel.quantile_turnover(date_grid=grid.head(1))


@pytest.mark.parametrize("supplied", [False, True])
def test_real_lagged_feature_warmup_retains_unavailable_not_zero_turnover(supplied):
    panel, dates = _inputs(nanoseconds=True)
    data = panel.data
    warmup = data.select("date", "asset", "factor").with_columns(
        pl.col("factor").shift(20).over("asset")
    )
    empty = prepare_signal_panel(
        warmup, data.select("date", "asset", "1D"), quantiles=2, max_loss=1
    )
    grid = pl.from_pandas(pd.DataFrame({"date": dates[[1, 3]]})) if supplied else None
    result = empty.quantile_turnover(quantiles=[1, 2], date_grid=grid)
    assert result.height == (4 if supplied else 0)
    assert result.schema["date"] == data.schema["date"]
    assert result.schema["turnover"] == pl.Float64
    assert result["turnover"].null_count() == result.height
    assert result["n_current"].to_list() == [0] * result.height


def test_missing_interchange_bucket_labels_keep_clock_and_unavailable_history():
    from ml4t.diagnostic.signal import compute_quantile_turnover

    panel, _ = _inputs()
    data = panel.data.with_columns(
        pl.when(pl.col("date") == pl.col("date").min())
        .then(None)
        .otherwise(pl.col("quantile"))
        .alias("quantile")
    )
    result = compute_quantile_turnover(data, quantiles=[2])
    assert result["n_current"].to_list() == [0, 1]
    assert result["status"].to_list() == ["insufficient_history", "empty_previous"]
    assert result["turnover"].null_count() == 2
