"""Daily bucket uncertainty and equal-date means must not pool asset-date rows."""

import math

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def _panel(representation="multiindex"):
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
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
