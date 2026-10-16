"""Group adjustment and group dimensions must use the same owned observation pairs."""

import math

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def _inputs(nanosecond=False):
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    if nanosecond:
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
    factor = (
        wide.pct_change(fill_method=None)
        .loc[[dates[1], dates[3]]]
        .stack(future_stack=True)
        .rename("factor")
    )
    factor.index = factor.index.set_names(["date", "asset"])
    marks = (
        wide.stack(future_stack=True)
        .rename("close")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )
    forward = (
        compute_forward_returns(marks, periods=[1, 2], group_col="asset")
        .set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]]
        .rename(columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"})
    )
    groups = pd.Series(
        [
            "L" if asset in (["A", "C"] if day == dates[3] else ["A", "B"]) else "H"
            for day, asset in factor.index
        ],
        index=factor.index,
        name="group",
    )
    return factor, forward, groups, dates


@pytest.mark.parametrize("representation", ["multiindex", "pandas", "polars"])
@pytest.mark.parametrize("dated", [False, True])
@pytest.mark.parametrize("labeled", [False, True])
@pytest.mark.parametrize("adjusted", [False, True])
@pytest.mark.parametrize("by_group", [False, True])
def test_grouped_ic_preserves_independent_date_group_pairs_not_grouped_bucket_aliases(
    representation, dated, labeled, adjusted, by_group
):
    factor, forward, groups, dates = _inputs()
    forward = forward[["1D"]]
    if not dated:
        groups = {"A": "L", "B": "L", "C": "H", "D": "H"}
    if representation != "multiindex":
        factor, forward = factor.reset_index(), forward.reset_index()
        if dated:
            groups = groups.reset_index()
        if representation == "polars":
            factor, forward = pl.from_pandas(factor), pl.from_pandas(forward)
            if dated:
                groups = pl.from_pandas(groups)
    labels = {"L": "sector L", "H": "sector H"} if labeled else None
    panel = prepare_signal_panel(
        factor, forward, quantiles=2, groups=groups, group_labels=labels, max_loss=0.0
    )
    observed = panel.ic_by_date(group_adjust=adjusted, by_group=by_group, min_obs=2)
    assert observed["return_column"].to_list() == ["1D"] * (4 if by_group else 2)
    assert (
        observed["n_obs"].to_list() == [2] * 4
        if by_group
        else observed["n_obs"].to_list() == [4, 4]
    )
    assert observed["status"].to_list() == ["defined"] * len(observed)
    assert observed["ic_method"].to_list() == ["spearman"] * len(observed)
    assert observed["return_adjustment"].to_list() == [
        "date_group_mean" if adjusted else "none"
    ] * len(observed)
    assert observed["min_obs"].to_list() == [2] * len(observed)
    day_values = observed["date"].dt.timestamp("ns").to_list()
    if by_group:
        expected = {(dates[1].value, labels.get(g, g) if labels else g): -1.0 for g in ["H", "L"]}
        expected.update(
            {
                (dates[3].value, labels.get(g, g) if labels else g): 1.0 if dated else -1.0
                for g in ["H", "L"]
            }
        )
        actual = dict(
            zip(
                zip(day_values, observed["group"].to_list(), strict=True),
                observed["ic"].to_list(),
                strict=True,
            )
        )
    else:
        assert "group" not in observed.columns
        expected = {
            dates[1].value: -1 / math.sqrt(5) if adjusted else 0.0,
            dates[3].value: (2 / math.sqrt(5) if dated else -1 / math.sqrt(5)) if adjusted else 0.0,
        }
        actual = dict(zip(day_values, observed["ic"].to_list(), strict=True))
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=0.0, abs=1e-14)


def test_arbitrary_return_selector_preserves_submicrosecond_keys_without_claiming_days():
    factor, forward, groups, dates = _inputs(nanosecond=True)
    panel = prepare_signal_panel(
        factor,
        forward[["1D"]].rename(columns={"1D": "next_quote"}),
        groups=groups,
        quantiles=2,
        max_loss=0.0,
    )
    result = panel.ic_by_date(group_adjust=True, min_obs=2)
    assert result["date"].dtype == pl.Datetime("ns", "UTC")
    assert result["date"].cast(pl.Int64).to_list() == [dates[1].value, dates[3].value]
    assert result["return_column"].to_list() == ["next_quote", "next_quote"]
    assert result["ic"].to_list() == pytest.approx([-1 / math.sqrt(5), 2 / math.sqrt(5)], abs=1e-14)
