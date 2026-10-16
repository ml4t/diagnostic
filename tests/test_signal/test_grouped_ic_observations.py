"""Group adjustment and group dimensions must use the same owned observation pairs."""

import math

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def _inputs(nanosecond=False, zero_quote=False, constant_outcome=False):
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
    if zero_quote:
        wide.loc[dates[2], "A"] = 0.0
    if constant_outcome:
        wide.loc[dates[2]] = wide.loc[dates[1]] * 1.125
        wide.loc[dates[4]] = wide.loc[dates[3]] * 1.125
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


@pytest.mark.parametrize("by_group", [False, True])
def test_explicit_grid_holes_have_zero_pairs_and_null_ic_not_fabricated_observations(by_group):
    factor, forward, groups, dates = _inputs()
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0.0)
    requested = pd.date_range(dates[1], dates[3], freq="D")
    grid = (
        pd.DataFrame({"date": requested})
        if not by_group
        else pd.MultiIndex.from_product([requested, ["H", "L"]], names=["date", "group"]).to_frame(
            index=False
        )
    )
    grid["date"] = grid["date"].astype("datetime64[ns, UTC]")
    before = grid.copy(deep=True)
    observed = panel.ic_by_date(by_group=by_group, group_adjust=True, date_grid=grid)
    assert len(observed) == len(grid)
    assert observed["date"].dtype == pl.Datetime("ns", "UTC")
    holes = observed.filter(
        ~pl.col("date").dt.timestamp("ns").is_in([dates[1].value, dates[3].value])
    )
    assert holes.height == 3 * (2 if by_group else 1)
    assert holes["n_obs"].to_list() == [0] * holes.height
    assert holes["ic"].null_count() == holes.height
    assert holes["status"].to_list() == ["no_pairs"] * holes.height
    valid = observed.filter(pl.col("status") == "defined")
    assert valid["n_obs"].to_list() == ([2] * 4 if by_group else [4, 4])
    pd.testing.assert_frame_equal(grid, before)


def test_actual_all_warmup_panel_can_expose_an_explicit_empty_group_grid():
    factor, forward, _, dates = _inputs()
    warmup = factor.groupby(level="asset").transform(lambda values: values.rolling(20).mean())
    panel = prepare_signal_panel(warmup, forward[["1D"]], groups={}, quantiles=2, max_loss=1.0)
    assert panel.ic_by_date(by_group=True).height == 0
    grid = pd.DataFrame({"date": [dates[1]] * 2, "group": ["H", "L"]})
    result = panel.ic_by_date(by_group=True, date_grid=grid)
    assert result["group"].to_list() == ["H", "L"]
    assert result["ic"].null_count() == 2
    assert result["n_obs"].to_list() == [0, 0] and result["status"].to_list() == ["no_pairs"] * 2


def test_categorical_feed_groups_match_explicit_string_grid_labels_without_changing_codes():
    factor, forward, groups, dates = _inputs()
    panel = prepare_signal_panel(
        factor, forward[["1D"]], groups=groups.astype("category"), quantiles=2, max_loss=0.0
    )
    grid = pd.MultiIndex.from_product(
        [dates[[1, 3]], ["H", "L"]], names=["date", "group"]
    ).to_frame(index=False)
    result = panel.ic_by_date(by_group=True, date_grid=grid)
    assert result["group"].to_list() == ["H", "L", "H", "L"]
    assert result["ic"].to_list() == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    assert result["n_obs"].to_list() == [2] * 4


def test_group_ic_does_not_expose_owned_data_or_depend_on_later_caller_mutations():
    from polars.testing import assert_frame_equal

    factor, forward, groups, _ = _inputs()
    passed_forward = forward[["1D"]].copy(deep=True)
    panel = prepare_signal_panel(factor, passed_forward, groups=groups, quantiles=2, max_loss=0.0)
    original = panel.ic_by_date(group_adjust=True, by_group=True)
    accessed = panel.ic_by_date(group_adjust=True, by_group=True)
    accessed.replace_column(accessed.get_column_index("ic"), accessed["ic"] * 0.0)
    factor.iloc[:] = 0.0
    passed_forward.iloc[:, :] = -1.0
    groups.iloc[:] = "changed"
    assert_frame_equal(panel.ic_by_date(group_adjust=True, by_group=True), original)


def test_per_horizon_raw_input_counts_missing_future_tails_without_dropping_undefined_dates():
    from ml4t.diagnostic.signal import compute_signal_ic

    factor, forward, _, dates = _inputs()
    frame = factor.to_frame().join(forward)
    before = frame.copy(deep=True)
    observed = compute_signal_ic(frame, return_columns=["2D", "1D"], min_obs=2)
    assert observed["return_column"].to_list() == ["2D", "2D", "1D", "1D"]
    assert observed["n_obs"].to_list() == [4, 0, 4, 4]
    assert observed["ic"].to_list()[0] == pytest.approx(7 / math.sqrt(90), abs=1e-14)
    assert observed["ic"].to_list()[1] is None
    assert observed["status"].to_list() == ["defined", "no_pairs", "defined", "defined"]
    assert observed["date"].dt.timestamp("ns").to_list() == [dates[1].value, dates[3].value] * 2
    pd.testing.assert_frame_equal(frame, before)


def test_zero_quote_nonfinite_feature_is_not_used_in_another_assets_group_mean():
    from ml4t.diagnostic.signal import compute_signal_ic

    factor, forward, groups, _ = _inputs(zero_quote=True)
    raw = factor.to_frame().join(forward).join(groups)
    before = raw.copy(deep=True)
    result = compute_signal_ic(raw, return_columns=["1D"], group_adjust=True)
    assert result["n_obs"].to_list() == [4, 3]
    assert result["ic"].to_list() == pytest.approx([0.2, 1.0], abs=1e-14)
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize("constant", ["factor", "outcome"])
def test_undefined_cross_section_retains_actual_pairs_and_the_specific_constant_reason(constant):
    factor, forward, groups, dates = _inputs(constant_outcome=constant == "outcome")
    if constant == "factor":
        factor.loc[dates[1]] = factor.loc[dates[1]].mean()
    panel = prepare_signal_panel(
        factor,
        forward[["1D"]],
        groups=groups,
        quantiles=2 if constant == "outcome" else 1,
        max_loss=0.0,
    )
    result = panel.ic_by_date()
    first = result.row(0, named=True)
    assert first["ic"] is None and first["n_obs"] == 4
    assert first["status"] == ("constant_factor" if constant == "factor" else "constant_outcome")


def test_group_minimum_uses_actual_pairs_not_the_full_cross_section_count():
    factor, forward, groups, _ = _inputs()
    result = prepare_signal_panel(
        factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0.0
    ).ic_by_date(by_group=True, min_obs=3)
    assert result["n_obs"].to_list() == [2] * 4
    assert result["ic"].null_count() == 4
    assert result["status"].to_list() == ["insufficient_pairs"] * 4


def test_dated_group_gap_keeps_singletons_counted_but_does_not_invent_a_correlation():
    factor, forward, groups, dates = _inputs()
    groups = groups.drop((dates[1], "A")).reindex(factor.index)
    panel = prepare_signal_panel(
        factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0.125
    )
    result = panel.ic_by_date(by_group=True, group_adjust=True, min_obs=1)
    singleton = result.filter((pl.col("date") == dates[1]) & (pl.col("group") == "L")).row(
        0, named=True
    )
    assert singleton["n_obs"] == 1 and singleton["ic"] is None
    assert singleton["status"] == "insufficient_pairs"
    global_result = panel.ic_by_date(group_adjust=True)
    assert global_result["n_obs"].to_list() == [3, 4]
    assert global_result["ic"].to_list()[0] == pytest.approx(-0.5, abs=1e-14)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"group_adjust": "false"}, "group_adjust"),
        ({"by_group": "false"}, "by_group"),
        ({"min_obs": 0}, "min_obs"),
        ({"min_obs": True}, "min_obs"),
        ({"min_obs": 1.5}, "min_obs"),
        ({"method": "kendall"}, "Unknown method"),
    ],
)
def test_invalid_ic_policy_is_rejected_on_an_otherwise_valid_grouped_panel(options, message):
    factor, forward, groups, _ = _inputs()
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0.0)
    with pytest.raises(ValueError, match=message):
        panel.ic_by_date(**options)


def test_grouped_grid_requires_explicit_group_keys_before_calculation():
    factor, forward, groups, dates = _inputs()
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0.0)
    with pytest.raises(ValueError, match="IC grid.*missing key"):
        panel.ic_by_date(by_group=True, date_grid=pd.DataFrame({"date": dates[[1, 3]]}))


@pytest.mark.parametrize("date_key", ["status", "ic_method"])
def test_legacy_custom_time_column_still_analyzes_when_its_name_matches_new_ic_metadata(date_key):
    factor, forward, _, _ = _inputs()
    factor = factor.rename("score")
    factor.index = factor.index.set_names([date_key, "symbol"])
    forward.index = forward.index.set_names([date_key, "symbol"])
    panel = prepare_signal_panel(
        factor,
        forward[["1D"]],
        factor_col="score",
        date_col=date_key,
        asset_col="symbol",
        quantiles=2,
        max_loss=0.0,
    )
    result = panel.analyze(min_assets=2)
    assert result.n_dates == 2 and result.n_assets == 4
    assert result.ic_series["1D"] == pytest.approx([0.0, 0.0], abs=1e-14)


def test_ic_tables_use_canonical_dimensions_without_losing_conflicting_input_names():
    factor, forward, _, dates = _inputs()
    factor = factor.rename("ic_method")
    factor.index = factor.index.set_names(["status", "return_column"])
    forward.index = forward.index.set_names(["status", "return_column"])
    panel = prepare_signal_panel(
        factor,
        forward[["1D"]],
        factor_col="ic_method",
        date_col="status",
        asset_col="return_column",
        quantiles=2,
        max_loss=0.0,
    )
    result = panel.ic_by_date()
    assert result["date"].dt.timestamp("ns").to_list() == [dates[1].value, dates[3].value]
    assert result["status"].to_list() == ["defined"] * 2
    assert result["ic_method"].to_list() == ["spearman"] * 2
    assert result["return_column"].to_list() == ["1D"] * 2
    assert result["n_obs"].to_list() == [4] * 2
    assert result["ic"].to_list() == pytest.approx([0.0, 0.0], abs=1e-14)


@pytest.mark.parametrize("by_group", [False, True])
@pytest.mark.parametrize("route", ["panel", "frame"])
def test_demeaned_outcomes_are_inspectable_and_shared_without_reassigning_rows_or_buckets(
    by_group, route
):
    factor, forward, groups, _ = _inputs()
    panel = prepare_signal_panel(factor, forward[["1D"]], groups=groups, quantiles=2, max_loss=0.0)
    original = panel.data
    if route == "panel":
        adjusted = panel.demean_returns(by_group=by_group)
    else:
        from ml4t.diagnostic.signal import demean_signal_returns

        supplied = original.to_pandas().set_index(["date", "asset"])
        before = supplied.copy(deep=True)
        adjusted = demean_signal_returns(supplied, return_columns=["1D"], by_group=by_group)
        pd.testing.assert_frame_equal(supplied, before)
    expected = (
        [0.1875, -0.1875, 0.1875, -0.1875, -0.125, -0.125, 0.125, 0.125]
        if by_group
        else [0.0625, -0.3125, 0.3125, -0.0625] * 2
    )
    assert adjusted["1D"].to_list() == expected
    assert adjusted.select("date", "asset", "factor", "quantile", "group").equals(
        original.select("date", "asset", "factor", "quantile", "group")
    )
    assert panel.data.equals(original)


def test_inspectable_group_demeaning_marks_the_actual_infinite_factor_pair_unavailable():
    from ml4t.diagnostic.signal import demean_signal_returns

    factor, forward, groups, dates = _inputs(zero_quote=True)
    raw = factor.to_frame().join(forward[["1D"]]).join(groups)
    before = raw.copy(deep=True)
    result = demean_signal_returns(raw, return_columns=["1D"], by_group=True)
    observed = result.to_pandas().set_index(["date", "asset"])
    assert len(observed) == 8
    assert math.isnan(observed.loc[(dates[3], "A"), "1D"])
    assert observed.loc[(dates[3], "B"), "1D"] == -0.125
    assert observed.loc[(dates[3], "C"), "1D"] == 0.0
    assert observed.loc[(dates[3], "D"), "1D"] == 0.125
    pd.testing.assert_frame_equal(raw, before)
