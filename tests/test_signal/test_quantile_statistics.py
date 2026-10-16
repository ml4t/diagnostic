"""Daily bucket uncertainty and equal-date means must not pool asset-date rows."""

import math

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def _panel(representation="multiindex", nanoseconds=False):
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    if nanoseconds:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, unit="ns") for i in range(5)])
    wide = pd.DataFrame(
        {
            "A": [128.0, 144.0, 180.0, 202.5, 253.125],
            "B": [128.0, 160.0, 140.0, 175.0, 153.125],
            "C": [128.0, 192.0, 288.0, 432.0, 648.0],
            "D": [128.0, 256.0, 288.0, 576.0, 648.0],
        },
        index=dates,
    )
    factor = wide.pct_change(fill_method=None).loc[dates[[1, 3]]].stack(future_stack=True)
    factor.index = factor.index.set_names(["date", "asset"])
    factor = factor.rename("factor").drop((dates[3], "D"))
    marks = (
        wide.stack(future_stack=True).rename("close").rename_axis(["date", "asset"]).reset_index()
    )
    forward = compute_forward_returns(marks, periods=[1, 2], group_col="asset")
    forward = forward.set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]].rename(
        columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"}
    )
    groups = pd.Series(
        [
            "L" if asset in (["A", "C"] if day == dates[3] else ["A", "B"]) else "H"
            for day, asset in factor.index
        ],
        index=factor.index,
        name="group",
    )
    if representation != "multiindex":
        factor, forward, groups = factor.reset_index(), forward.reset_index(), groups.reset_index()
        if representation == "polars":
            factor, forward, groups = (
                pl.from_pandas(factor),
                pl.from_pandas(forward),
                pl.from_pandas(groups),
            )
    panel = prepare_signal_panel(
        factor,
        forward[["1D"]]
        if representation == "multiindex"
        else forward.drop(columns="2D")
        if representation == "pandas"
        else forward.drop("2D"),
        groups=groups,
        quantiles=2,
        max_loss=0,
    )
    return panel, dates


def _expected(by_date, by_group, demeaned, group_adjust):
    rows = [
        (0, "L", 1, 0.25),
        (0, "L", 1, -0.125),
        (0, "H", 2, 0.5),
        (0, "H", 2, 0.125),
        (1, "L", 1, 0.25),
        (1, "H", 1, -0.125),
        (1, "L", 2, 0.5),
    ]
    adjusted = []
    for day, group, bucket, value in rows:
        peers = [r[3] for r in rows if r[0] == day and (not group_adjust or r[1] == group)]
        adjusted.append(
            (
                day,
                group,
                bucket,
                value - math.fsum(peers) / len(peers) if group_adjust or demeaned else value,
            )
        )
    daily = {}
    for day, group, bucket, value in adjusted:
        daily.setdefault((day, group if by_group else None, bucket), []).append(value)
    if by_date:
        return daily
    summary = {}
    for (_, group, bucket), values in daily.items():
        summary.setdefault((None, group, bucket), []).append(math.fsum(values) / len(values))
    return summary


@pytest.mark.parametrize("representation", ["multiindex", "pandas", "polars"])
@pytest.mark.parametrize("by_date", [False, True])
@pytest.mark.parametrize("by_group", [False, True])
@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("group_adjust", [False, True])
def test_quantile_statistics_preserve_daily_samples_and_equal_date_uncertainty(
    representation, by_date, by_group, demeaned, group_adjust
):
    panel, dates = _panel(representation)
    before = panel.data
    result = panel.quantile_statistics(
        by_date=by_date, by_group=by_group, demeaned=demeaned, group_adjust=group_adjust
    )
    expected = _expected(by_date, by_group, demeaned, group_adjust)
    observed = {}
    for row in result.iter_rows(named=True):
        day = (
            [dates[1].value, dates[3].value].index(pd.Timestamp(row["date"]).value)
            if by_date
            else None
        )
        key = (day, row.get("group"), row["quantile"])
        values = expected[key]
        mean = math.fsum(values) / len(values)
        std = (
            math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
            if len(values) > 1
            else math.nan
        )
        assert row["mean"] == pytest.approx(mean, abs=1e-14)
        assert row["n_obs"] == len(values)
        assert row["sample_unit"] == ("asset" if by_date else "date")
        assert row["return_adjustment"] == (
            "date_group_mean" if group_adjust else "date_mean" if demeaned else "none"
        )
        assert row["return_column"] == "1D"
        if math.isnan(std):
            assert row["std"] is None and row["standard_error"] is None
        else:
            assert row["std"] == pytest.approx(std, abs=1e-14)
            assert row["standard_error"] == pytest.approx(std / math.sqrt(len(values)), abs=1e-14)
        observed[key] = row["mean"]
    assert set(observed) == set(expected)
    assert panel.data.equals(before)


def test_equal_date_upper_bucket_is_not_the_existing_pooled_mean():
    panel, _ = _panel()
    native = panel.analyze(min_assets=2)
    assert native.quantile_returns["1D"][2] == pytest.approx(0.375)
    result = panel.quantile_statistics(by_date=False, demeaned=False)
    upper = result.filter(pl.col("quantile") == 2).row(0, named=True)
    assert upper["mean"] == pytest.approx(0.40625)
    assert upper["standard_error"] == pytest.approx(0.09375)
    assert upper["n_obs"] == 2


@pytest.mark.parametrize("by_group", [False, True])
def test_daily_statistics_do_not_truncate_distinct_quote_nanoseconds_into_one_timestamp(by_group):
    panel, dates = _panel(nanoseconds=True)
    result = panel.quantile_statistics(by_date=True, by_group=by_group, demeaned=False)
    assert result["date"].dtype == pl.Datetime("ns", "UTC")
    assert sorted(result["date"].dt.timestamp("ns").unique().to_list()) == [
        dates[1].value,
        dates[3].value,
    ]
    assert result["n_obs"].sum() == 7


def test_actual_feature_warmup_can_produce_an_empty_owned_quantile_sample():
    panel, _ = _panel()
    original = panel.data
    warmup = original.select("date", "asset", pl.col("factor").rolling_mean(20))
    empty = prepare_signal_panel(
        warmup, original.select("date", "asset", "1D"), groups={}, quantiles=2, max_loss=1
    )
    for by_date in [False, True]:
        for by_group in [False, True]:
            result = empty.quantile_statistics(by_date=by_date, by_group=by_group)
            assert result.is_empty()
            assert result.schema["mean"] == pl.Float64


@pytest.mark.parametrize("by_date", [False, True])
def test_partial_forward_tail_has_zero_asset_or_date_count_not_an_invented_mean(by_date):
    from ml4t.diagnostic.signal import compute_quantile_statistics

    panel, dates = _panel()
    raw = panel.data.with_columns(
        pl.when(pl.col("date") == dates[3]).then(None).otherwise(pl.col("1D")).alias("next_quote")
    )
    before = raw.clone()
    result = compute_quantile_statistics(
        raw, return_columns=["next_quote", "1D"], by_date=by_date, demeaned=False
    )
    missing = result.filter(pl.col("return_column") == "next_quote")
    if by_date:
        tail = missing.filter(pl.col("date") == dates[3])
        assert tail["n_obs"].to_list() == [0, 0]
        assert tail["mean"].null_count() == 2
    else:
        assert missing["n_obs"].to_list() == [1, 1]
        assert missing["standard_error"].null_count() == 2
    assert raw.equals(before)


def test_custom_input_names_cannot_shadow_canonical_quantile_result_dimensions():
    from ml4t.diagnostic.signal import compute_quantile_statistics

    panel, _ = _panel()
    source = panel.data.rename(
        {
            "date": "mean",
            "factor": "standard_error",
            "asset": "return_column",
            "quantile": "n_obs",
            "group": "std",
            "1D": "next_quote",
        }
    )
    before = source.clone()
    result = compute_quantile_statistics(
        source,
        return_columns=["next_quote"],
        date_col="mean",
        factor_col="standard_error",
        asset_col="return_column",
        quantile_col="n_obs",
        group_col="std",
        by_date=True,
        by_group=True,
        group_adjust=True,
    )
    assert result["n_obs"].sum() == 7
    assert set(result["group"].to_list()) == {"L", "H"}
    assert result["quantile"].n_unique() == 2
    assert source.equals(before)


@pytest.mark.parametrize("option", ["by_date", "by_group", "demeaned", "group_adjust"])
def test_string_boolean_never_silently_changes_the_quantile_adjustment_policy(option):
    panel, _ = _panel()
    with pytest.raises(ValueError, match=option):
        panel.quantile_statistics(**{option: "false"})
