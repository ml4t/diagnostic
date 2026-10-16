"""Holding extrema must preserve signed values, stable IDs and defined NAV samples."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from test_portfolio_positions import _actual_prepared_inputs

from ml4t.diagnostic.evaluation import PortfolioAnalysis

WEIGHTS = ["max_long_weight", "median_long_weight", "median_short_weight", "min_short_weight"]


def _holdings():
    returns, positions, valuations, _ = _actual_prepared_inputs()
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations, complete_snapshots=True
    )
    return result, returns, positions, valuations


@pytest.mark.parametrize("top", [1, 10, 10**30])
@pytest.mark.parametrize("side", ["long", "short", "absolute"])
def test_top_holdings_use_actual_across_sample_signed_marks_not_final_row_or_nav_weights(top, side):
    result, _, _, _ = _holdings()
    expected = {
        "long": [("A", 400.0)],
        "short": [("B", -300.0)],
        "absolute": [("A", 400.0), ("B", 300.0)],
    }
    assert result.top_positions(top=top, side=side).rows() == expected[side][:top]


def test_actual_cash_only_observation_has_no_side_concentration_instead_of_zero_risk():
    result, returns, _, _ = _holdings()
    assert result.timeline["date"].to_list() == returns.index.to_list()
    assert result.timeline.select(WEIGHTS).row(0) == (None,) * 4
    np.testing.assert_allclose(result.timeline.select(WEIGHTS).row(-1), [0.4, 0.4, -0.3, -0.3])
    assert result.timeline["gross_hhi"][-1] == pytest.approx(25 / 49)


@pytest.mark.parametrize("nav", [None, 0.0, -1000.0])
def test_undefined_nav_keeps_raw_top_marks_but_never_fabricates_position_weight_extrema(nav):
    _, returns, positions, valuations = _holdings()
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations.with_columns(pl.lit(nav, dtype=pl.Float64).alias("equity")),
        complete_snapshots=True,
    )
    assert result.top_positions(side="short").rows() == [("B", -300.0)]
    assert result.timeline.select(WEIGHTS).null_count().row(0) == (3,) * 4


def test_signed_concentration_uses_reported_equity_not_reconstructed_cash_plus_holdings():
    _, returns, positions, valuations = _holdings()
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations.with_columns(pl.lit(200.0).alias("equity")), complete_snapshots=True
    )
    np.testing.assert_allclose(result.timeline.select(WEIGHTS).row(-1), [2.0, 2.0, -1.5, -1.5])


def test_top_holdings_capture_earlier_extrema_and_same_asset_changing_side():
    _, returns, positions, valuations = _holdings()
    positions.iloc[-1] = [-500.0, 600.0]
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations, complete_snapshots=True
    )
    assert result.top_positions(side="long").rows() == [("B", 600.0), ("A", 400.0)]
    assert result.top_positions(side="short").rows() == [("A", -500.0), ("B", -300.0)]
    assert result.top_positions().rows() == [("B", 600.0), ("A", 500.0)]
    np.testing.assert_allclose(result.timeline.select(WEIGHTS).row(-1), [0.6, 0.6, -0.5, -0.5])


def test_side_medians_exclude_cash_and_zero_assets_and_do_not_alias_the_largest_weight():
    _, returns, positions, valuations = _holdings()
    positions["C"] = [0.0, 100.0, 100.0]
    positions["D"] = [0.0, -100.0, -100.0]
    positions["E"] = 0.0
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations, complete_snapshots=True
    )
    np.testing.assert_allclose(result.timeline.select(WEIGHTS).row(-1), [0.4, 0.25, -0.2, -0.3])


def test_sparse_complete_holdings_keep_integer_ids_and_do_not_add_absent_zeros_to_side_medians():
    _, returns, positions, valuations = _holdings()
    long = positions.reset_index().melt(id_vars="date", var_name="asset", value_name="value")
    long["asset"] = long["asset"].map({"A": 1, "B": 2})
    long = long.loc[(long["asset"] == 1) | (long["value"] != 0)]
    result = PortfolioAnalysis(returns, positions=long).compute_holdings_analysis(
        valuations, complete_snapshots=True
    )
    assert result.top_positions().rows() == [(1, 400.0), (2, 300.0)]
    assert result.timeline.select(WEIGHTS).row(0) == (None,) * 4
    np.testing.assert_allclose(result.timeline.select(WEIGHTS).row(-1), [0.4, 0.4, -0.3, -0.3])


def test_cash_only_portfolio_has_empty_typed_top_table_and_missing_signed_concentration():
    _, returns, _, valuations = _holdings()
    result = PortfolioAnalysis(
        returns, positions=pd.DataFrame({"cash": [1000.0] * 3}, index=returns.index)
    ).compute_holdings_analysis(
        valuations.with_columns(pl.lit(1000.0).alias("cash")), complete_snapshots=True
    )
    for side in ["long", "short", "absolute"]:
        top = result.top_positions(side=side)
        assert top.is_empty() and top.schema == {"asset": pl.Utf8, "value": pl.Float64}
    assert result.timeline.select(WEIGHTS).null_count().row(0) == (3,) * 4


def test_top_absolute_includes_explicit_zero_asset_but_excludes_account_cash():
    _, returns, positions, valuations = _holdings()
    positions["C"] = 0.0
    positions["cash"] = valuations["cash"].to_numpy()
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations, complete_snapshots=True
    )
    assert result.top_positions().rows() == [("A", 400.0), ("B", 300.0), ("C", 0.0)]
    assert result.top_positions(side="long").rows() == [("A", 400.0)]


def test_tied_top_extrema_sort_by_stable_asset_id_not_input_column_or_hash_group_order():
    _, returns, positions, valuations = _holdings()
    positions["B"] = positions["A"]
    positions = positions[["B", "A"]]
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations, complete_snapshots=True
    )
    assert result.top_positions(top=1).rows() == [("A", 400.0)]
    np.testing.assert_allclose(result.timeline.select(WEIGHTS[:2]).row(-1), [0.4, 0.4])
    assert result.timeline.select(WEIGHTS[2:]).row(-1) == (None, None)


def test_zero_top_returns_typed_empty_without_changing_result_or_later_calls():
    result, _, _, _ = _holdings()
    empty = result.top_positions(top=0)
    assert empty.is_empty() and empty.schema == {"asset": pl.Utf8, "value": pl.Float64}
    assert result.top_positions().rows() == [("A", 400.0), ("B", 300.0)]


@pytest.mark.parametrize("top", [-1, 1.5, True, None, "1"])
def test_invalid_top_limit_cannot_silently_truncate_or_treat_boolean_as_one(top):
    result, _, _, _ = _holdings()
    result.top_positions()
    with pytest.raises((TypeError, ValueError), match="top"):
        result.top_positions(top=top)


@pytest.mark.parametrize("side", ["", "net", None])
def test_unknown_top_side_cannot_silently_fall_back_to_absolute(side):
    result, _, _, _ = _holdings()
    result.top_positions()
    with pytest.raises(ValueError, match="side"):
        result.top_positions(side=side)


def test_top_tables_own_values_and_do_not_mutate_holdings_or_subsequent_queries():
    result, _, _, _ = _holdings()
    before = result.positions.clone()
    top = result.top_positions()
    top.replace_column(1, pl.Series("value", [99.0, 99.0]))
    assert result.positions.equals(before)
    assert result.top_positions().rows() == [("A", 400.0), ("B", 300.0)]
