"""Supplied accounts keep authoritative returns, holdings, cash and fills separate."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _frames():
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    returns = pd.Series([0.01, -0.02], index=dates, name="net_return")
    positions = pd.DataFrame(
        {"A": [600.0, 500.0], "B": [-100.0, -50.0], "cash": [510.0, 539.8]}, index=dates
    )
    valuations = pd.DataFrame({"date": dates, "equity": [1010.0, 989.8]})
    fills = pd.DataFrame(
        {
            "timestamp": dates,
            "asset": ["A", "B"],
            "quantity": [0.5, -0.25],
            "price": [100.0, 200.0],
            "commission": [None, 0.25],
        }
    )
    return returns, positions, valuations, fills


def test_from_frames_preserves_supplied_net_returns_and_unknown_fees_in_owned_inputs():
    returns, positions, valuations, fills = _frames()
    original = [value.copy(deep=True) for value in (returns, positions, valuations, fills)]
    analysis = PortfolioAnalysis.from_frames(
        returns,
        positions=positions,
        valuations=valuations,
        transactions=fills,
        complete_snapshots=True,
        calendar=None,
        periods_per_year=252,
    )
    np.testing.assert_array_equal(analysis.returns, [0.01, -0.02])
    inputs = analysis.input_data
    assert inputs.transactions["commission"].null_count() == 1
    assert inputs.metadata["calendar"] is None
    assert inputs.metadata["returns_kind"] == "net"
    assert inputs.metadata["complete_snapshots"] is True
    assert inputs.transactions["quantity"].to_list() == [0.5, -0.25]
    holdings = analysis.compute_holdings_analysis()
    assert holdings.timeline["cash"].to_list() == [510.0, 539.8]
    assert holdings.timeline["gross_exposure"].to_list() == [700.0, 550.0]
    assert holdings.timeline["equity"].to_list() == [1010.0, 989.8]
    assert holdings.timeline["gross_exposure_fraction"].to_list() == pytest.approx(
        [700 / 1010, 550 / 989.8]
    )
    for actual, expected in zip((returns, positions, valuations, fills), original, strict=True):
        if isinstance(actual, pd.Series):
            pd.testing.assert_series_equal(actual, expected)
        else:
            pd.testing.assert_frame_equal(actual, expected)
    view = inputs.transactions
    view.replace_column(view.columns.index("quantity"), pl.Series("quantity", [999.0, 999.0]))
    assert analysis.input_data.transactions["quantity"].to_list() == [0.5, -0.25]


def test_missing_nav_does_not_become_sum_of_cash_and_signed_positions():
    returns, positions, _, _ = _frames()
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions, complete_snapshots=True, calendar=None
    )
    holdings = analysis.compute_holdings_analysis()
    assert holdings.timeline["equity"].null_count() == 2
    assert holdings.timeline["gross_exposure"].to_list() == [700.0, 550.0]
    assert holdings.timeline["gross_exposure_fraction"].null_count() == 2


def test_pandas_all_unknown_execution_cost_columns_remain_unknown_not_invalid_or_zero():
    returns, _, _, fills = _frames()
    fills["commission"] = pd.Series([None, None], dtype="object")
    fills["slippage"] = pd.Series([None, None], dtype="object")
    analysis = PortfolioAnalysis.from_frames(returns, transactions=fills, calendar=None)
    for column in ("commission", "slippage"):
        assert analysis.input_data.transactions[column].null_count() == 2
        assert analysis.input_data.transactions[column].dtype == pl.Float64


def test_explicitly_incomplete_positions_are_not_forward_filled_or_liquidated():
    returns, positions, valuations, _ = _frames()
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions.iloc[:1], valuations=valuations, calendar=None
    )
    assert analysis.input_data.positions["date"].n_unique() == 1
    assert analysis.input_data.metadata["complete_snapshots"] is False
    with pytest.raises(ValueError, match="complete_snapshots"):
        analysis.compute_holdings_analysis()
    with pytest.raises(ValueError, match="dates|keys"):
        PortfolioAnalysis.from_frames(
            returns,
            positions=positions.iloc[:1],
            valuations=valuations,
            complete_snapshots=True,
            calendar=None,
        )


def test_benchmark_aligns_by_key_and_nanosecond_observation_keys_are_never_rounded():
    dates = pd.date_range("2025-07-07T10:00Z", periods=3, freq="1ns")
    returns = pd.Series([0.01, 0.02, 0.03], index=dates)
    benchmark = pd.Series([0.06, 0.04, 0.05], index=dates[[2, 0, 1]])
    analysis = PortfolioAnalysis.from_frames(returns, benchmark=benchmark, calendar=None)
    np.testing.assert_array_equal(analysis.benchmark, [0.04, 0.05, 0.06])
    assert analysis.dates.cast(pl.Int64).to_list() == dates.as_unit("ns").asi8.tolist()


def test_declared_aliases_and_calendar_null_survive_owned_input_serialization():
    from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioInputData

    returns = pl.DataFrame({"when": pl.Series([1, 2], dtype=pl.Date), "net": [0.01, 0.02]})
    positions = pl.DataFrame(
        {"when": pl.Series([1, 2], dtype=pl.Date), "sid": [2**63 - 1] * 2, "value": [100.0, 101.0]}
    )
    analysis = PortfolioAnalysis.from_frames(
        returns,
        return_col="net",
        positions=positions,
        complete_snapshots=True,
        feed_spec={
            "time_contract_version": 2,
            "timestamp_col": "when",
            "entity_col": "sid",
            "calendar": None,
        },
    )
    restored = PortfolioInputData.from_json(analysis.input_data.to_json())
    assert restored.positions["asset"].to_list() == [2**63 - 1] * 2
    assert restored.returns["date"].dtype == pl.Date
    assert restored.metadata["calendar_source"] == "feed_spec"
    assert restored.metadata["feed_spec"]["calendar"] is None


def test_all_cash_snapshots_retain_dates_without_a_fabricated_cash_asset():
    returns, _, valuations, _ = _frames()
    positions = pd.DataFrame({"cash": [1010.0, 989.8]}, index=returns.index)
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions, valuations=valuations, complete_snapshots=True
    )
    holdings = analysis.compute_holdings_analysis()
    assert holdings.positions.is_empty()
    assert holdings.timeline["gross_exposure"].to_list() == [0.0, 0.0]
    assert holdings.timeline["cash"].to_list() == [1010.0, 989.8]


def test_raw_integer_execution_quantities_do_not_lose_ticks_before_any_accounting():
    returns, _, _, _ = _frames()
    fills = pl.DataFrame(
        {
            "timestamp": returns.index,
            "asset": ["A", "A"],
            "quantity": [2**53 + 1, -(2**63)],
            "price": [100, 101],
            "side": ["buy", "sell"],
            "multiplier": [1, 50],
        }
    )
    analysis = PortfolioAnalysis.from_frames(returns, transactions=fills)
    assert analysis.input_data.transactions["quantity"].to_list() == [2**53 + 1, -(2**63)]
    assert analysis.input_data.transactions["quantity"].dtype == pl.Int64


def test_return_values_are_not_inferred_from_nav_across_an_unreported_cash_flow():
    returns, positions, valuations, _ = _frames()
    valuations.loc[0, "equity"] = 1110.0
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions, valuations=valuations, complete_snapshots=True
    )
    assert analysis.compute_summary_stats().total_return == pytest.approx(1.01 * 0.98 - 1)
    assert analysis.compute_holdings_analysis().timeline["equity"][0] == 1110.0


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_position",
        "unknown_position_mark",
        "string_position_dates",
        "nonfinite_nav",
        "missing_quantity",
        "unknown_multiplier",
        "conflicting_buy_sign",
        "cash_conflict",
    ],
)
def test_invalid_optional_accounts_fail_without_modifying_any_caller_frame(kind):
    returns, positions, valuations, fills = _frames()
    if kind == "duplicate_position":
        positions = pd.concat([positions, positions.iloc[:1]])
    elif kind == "unknown_position_mark":
        positions.iloc[0, 0] = np.nan
    elif kind == "string_position_dates":
        positions.index = positions.index.astype(str)
    elif kind == "nonfinite_nav":
        valuations.loc[0, "equity"] = np.inf
    elif kind == "missing_quantity":
        fills.loc[0, "quantity"] = np.nan
    elif kind == "unknown_multiplier":
        fills["multiplier"] = [np.nan, 10.0]
    elif kind == "conflicting_buy_sign":
        fills["side"] = ["buy", "buy"]
    else:
        valuations["cash"] = [0.0, 0.0]
    copies = [value.copy(deep=True) for value in (returns, positions, valuations, fills)]
    with pytest.raises((ValueError, TypeError)):
        PortfolioAnalysis.from_frames(
            returns, positions=positions, valuations=valuations, transactions=fills
        )
    for actual, expected in zip((returns, positions, valuations, fills), copies, strict=True):
        if isinstance(actual, pd.Series):
            pd.testing.assert_series_equal(actual, expected)
        else:
            pd.testing.assert_frame_equal(actual, expected)


def test_actual_asset_named_cash_is_distinct_from_reported_cash_in_long_inputs():
    returns, _, valuations, _ = _frames()
    holdings = pd.DataFrame(
        {"date": returns.index, "asset": ["cash", "cash"], "value": [100.0, 200.0]}
    )
    valuations["cash"] = [910.0, 789.8]
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=holdings, valuations=valuations, complete_snapshots=True
    )
    result = analysis.compute_holdings_analysis()
    assert result.timeline["gross_exposure"].to_list() == [100.0, 200.0]
    assert result.timeline["cash"].to_list() == [910.0, 789.8]


def test_numeric_wide_asset_ids_are_not_stringified_and_initial_nav_is_retained():
    returns, _, _, _ = _frames()
    positions = pd.DataFrame(
        {2**63 - 1: [100.0, 200.0], 2**63 - 2: [50.0, 60.0]}, index=returns.index
    )
    valuations = pd.DataFrame(
        {
            "date": returns.index.insert(0, returns.index[0] - pd.Timedelta("1D")),
            "equity": [1000.0, 1010.0, 989.8],
        }
    )
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions, valuations=valuations, complete_snapshots=True
    )
    assert analysis.input_data.positions["asset"].unique().sort().to_list() == [
        2**63 - 2,
        2**63 - 1,
    ]
    assert analysis.input_data.valuations["equity"].to_list() == [1000.0, 1010.0, 989.8]
    assert analysis.compute_holdings_analysis().timeline["equity"].to_list() == [1010.0, 989.8]


@pytest.mark.parametrize("index_mode", ["columns", "matching_index_and_column"])
def test_pandas_wide_column_keys_keep_numeric_ids_and_do_not_turn_date_into_an_asset(index_mode):
    returns, _, valuations, _ = _frames()
    positions = pd.DataFrame({2**63 - 1: [100.0, 200.0], "date": returns.index})
    if index_mode == "matching_index_and_column":
        positions.index = returns.index
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions, valuations=valuations, complete_snapshots=True
    )
    assert analysis.input_data.positions["asset"].unique().to_list() == [2**63 - 1]
    assert analysis.compute_holdings_analysis().timeline["gross_exposure"].to_list() == [
        100.0,
        200.0,
    ]


def test_literal_asset_named_date_remains_an_asset_when_the_datetime_index_is_the_key():
    returns, _, valuations, _ = _frames()
    positions = pd.DataFrame({"date": [100.0, 200.0]}, index=returns.index)
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions, valuations=valuations, complete_snapshots=True
    )
    assert analysis.input_data.positions["asset"].to_list() == ["date", "date"]


def test_normalized_supplied_holdings_reach_the_existing_report(tmp_path):
    pytest.importorskip("plotly")
    returns, positions, valuations, _ = _frames()
    report = PortfolioAnalysis.from_frames(
        returns, positions=positions, valuations=valuations, complete_snapshots=True
    ).create_tear_sheet()
    assert report.availability["positions"]["status"] == "available"
    assert report.holdings.timeline["gross_exposure"].to_list() == [700.0, 550.0]
    output = tmp_path / "reported-holdings.html"
    report.save_html(output, include_plotlyjs=False)
    assert "Portfolio Tear Sheet" in output.read_text()


@pytest.mark.parametrize(
    "role", ["return", "benchmark", "position_asset", "auto_position_asset", "transaction_asset"]
)
def test_declared_missing_columns_never_fall_back_to_a_different_canonical_column(role):
    returns, _, _, fills = _frames()
    table = returns.rename("return").rename_axis("date").reset_index()
    with pytest.raises(ValueError, match="missing|require"):
        if role == "return":
            PortfolioAnalysis.from_frames(table, return_col="missing")
        elif role == "benchmark":
            PortfolioAnalysis.from_frames(returns, benchmark=table, benchmark_col="missing")
        elif role == "position_asset":
            positions = pd.DataFrame(
                {"date": returns.index, "asset": ["A", "A"], "value": [100.0, 200.0]}
            )
            PortfolioAnalysis.from_frames(
                returns, positions=positions, positions_format="long", entity_col="missing"
            )
        elif role == "auto_position_asset":
            positions = pd.DataFrame(
                {"date": returns.index, "asset": [1, 1], "value": [100.0, 200.0]}
            )
            PortfolioAnalysis.from_frames(returns, positions=positions, entity_col="missing")
        else:
            PortfolioAnalysis.from_frames(returns, transactions=fills, entity_col="missing")
