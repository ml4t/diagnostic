"""Real Zipline exports preserve account quantities without a legacy runtime dependency."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.integration.zipline import portfolio_analysis_from_zipline

FIXTURE = Path(__file__).parent / "data/zipline320_account.json"


def _performance():
    captured = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frame = pd.DataFrame(captured["records"])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date")))
    return frame


def test_actual_zipline320_capture_reaches_holdings_and_costs_without_a_legacy_engine():
    frame = _performance()
    before = deepcopy(frame)
    analysis = portfolio_analysis_from_zipline(frame, calendar="NYSE")
    inputs = analysis.input_data
    np.testing.assert_array_equal(analysis.returns, frame["returns"])
    assert inputs.positions["asset"].unique().to_list() == [7]
    assert inputs.positions["quantity"].to_list() == [0, 3, 3, 2]
    assert inputs.positions["mark"].to_list() == [None, 100, 110, 120]
    assert inputs.positions["multiplier"].to_list() == [None, 1, 1, 1]
    assert inputs.transactions["quantity"].to_list() == [3, -1]
    assert inputs.transactions["commission"].to_list() == [None, None]
    holdings = analysis.compute_holdings_analysis()
    assert holdings.timeline["gross_exposure"].to_list() == [0, 300, 330, 240]
    assert holdings.timeline["cash"].to_list() == [1000, 699.7, 699.7, 819.6]
    assert holdings.timeline["equity"].to_list() == [1000, 999.7, 1029.7, 1059.6]
    activity = analysis.compute_transaction_analysis(incremental_bps=[0, 10])
    assert activity.metrics["total_filled_notional"] == 420
    assert activity.timeline["turnover"].to_list() == pytest.approx([None, 0.3, 0, 120 / 1029.7])
    scenario = activity.scenarios.filter(pl.col("incremental_bps") == 10)
    assert scenario["incremental_cost_return"].to_list() == pytest.approx(
        [0, 0.0003, 0, 0.12 / 1029.7]
    )
    assert activity.metrics["total_implementation_cost"] is None
    pd.testing.assert_frame_equal(frame, before)


def test_familiar_zipline_tuple_retains_cash_only_dates_and_actual_units_without_mutation():
    from ml4t.diagnostic.compat.pyfolio.utils import extract_rets_pos_txn_from_zipline

    performance = _performance()
    before = deepcopy(performance)
    returns, positions, transactions = extract_rets_pos_txn_from_zipline(performance)
    np.testing.assert_array_equal(returns, performance["returns"])
    assert returns.index.equals(performance.index.normalize())
    assert positions[7].to_list() == [0, 300, 330, 240]
    assert positions["cash"].to_list() == [1000, 699.7, 699.7, 819.6]
    assert transactions["amount"].to_list() == [3, -1]
    assert transactions["txn_dollars"].to_list() == [-300, 120]
    assert transactions["commission"].isna().all()
    assert transactions["sid"].to_list() == [7, 7]
    assert transactions.index.equals(pd.DatetimeIndex(["2025-07-08T20:00Z", "2025-07-10T20:00Z"]))
    pd.testing.assert_frame_equal(performance, before)


def test_cash_only_zipline_tuple_is_an_empty_asset_book_not_an_absent_account():
    from ml4t.diagnostic.compat.pyfolio.utils import extract_rets_pos_txn_from_zipline

    performance = _performance()
    performance["positions"] = [[] for _ in range(len(performance))]
    performance["transactions"] = [[] for _ in range(len(performance))]
    _, positions, transactions = extract_rets_pos_txn_from_zipline(performance)
    assert list(positions.columns) == ["cash"]
    assert positions["cash"].to_list() == [1000, 699.7, 699.7, 819.6]
    assert transactions.empty


def test_raw_sid_zipline_tuple_uses_declared_cash_price_units_or_actual_point_multiplier():
    from ml4t.diagnostic.compat.pyfolio.utils import extract_rets_pos_txn_from_zipline

    performance = _performance()
    for name in ("positions", "transactions"):
        performance[name] = [
            [{**row, "sid": row["sid"]["sid"]} for row in records] for records in performance[name]
        ]
    _, ordinary, _ = extract_rets_pos_txn_from_zipline(performance)
    _, futures, _ = extract_rets_pos_txn_from_zipline(
        performance, asset_metadata={7: {"multiplier": 50}}
    )
    assert ordinary[7].to_list() == [0, 300, 330, 240]
    assert futures[7].to_list() == [0, 15000, 16500, 12000]
    pd.testing.assert_series_equal(ordinary["cash"], futures["cash"])


def test_zipline_tuple_matches_published_reference_on_shared_nonempty_equity_books():
    reference = pytest.importorskip("pyfolio.utils")
    from ml4t.diagnostic.compat.pyfolio.utils import extract_rets_pos_txn_from_zipline

    performance = _performance()
    performance.iat[0, performance.columns.get_loc("positions")] = [
        {"sid": 7, "amount": 0, "last_sale_price": 100.0}
    ]
    for name in ("positions", "transactions"):
        performance[name] = [
            [
                {**row, "sid": row["sid"]["sid"] if isinstance(row["sid"], dict) else row["sid"]}
                for row in records
            ]
            for records in performance[name]
        ]
    # JSON captured Timestamp instants as strings; restore the actual producer type.
    performance["transactions"] = [
        [{**row, "dt": pd.Timestamp(row["dt"])} for row in records]
        for records in performance["transactions"]
    ]
    expected = reference.extract_rets_pos_txn_from_zipline(deepcopy(performance))
    actual = extract_rets_pos_txn_from_zipline(performance)
    pd.testing.assert_series_equal(actual[0], expected[0], check_freq=False)
    pd.testing.assert_frame_equal(actual[1], expected[1], check_freq=False)
    pd.testing.assert_frame_equal(actual[2], expected[2], check_freq=False)


def test_actual_zipline320_capture_reaches_the_existing_portfolio_report(tmp_path):
    pytest.importorskip("plotly")
    analysis = portfolio_analysis_from_zipline(_performance(), calendar="NYSE")
    activity = analysis.compute_transaction_analysis()
    sheet = analysis.create_tear_sheet(transactions=activity)
    assert sheet.holdings is not None and sheet.transactions is not None
    sheet.save_html(tmp_path / "zipline-account.html", include_plotlyjs=True)


def test_source_calendar_omission_null_and_explicit_override_use_the_shared_contract():
    frame = _performance()
    inherited = portfolio_analysis_from_zipline(
        frame,
        feed_spec={
            "timestamp_col": "when",
            "entity_col": "sid",
            "calendar": None,
            "time_contract_version": 2,
        },
    )
    assert inherited.input_data.metadata["calendar"] is None
    assert inherited.input_data.metadata["calendar_source"] == "feed_spec"
    assert inherited.input_data.positions["asset"].unique().to_list() == [7]
    assert inherited.compute_transaction_analysis().timeline.height == 4
    override = portfolio_analysis_from_zipline(frame, calendar=None)
    assert override.input_data.metadata["calendar_source"] == "argument"
    assert portfolio_analysis_from_zipline(frame).input_data.metadata["calendar"] == "NYSE"


def test_unknown_optional_ledgers_disable_only_dependent_analysis_not_authoritative_returns():
    frame = _performance()[["returns"]]
    analysis = portfolio_analysis_from_zipline(frame, calendar=None)
    assert analysis.input_data.positions is None
    assert analysis.input_data.transactions is None
    assert analysis.input_data.valuations is None
    assert np.isfinite(analysis.compute_summary_stats().annual_return)


def test_same_symbol_different_sids_and_renamed_symbols_never_merge_instruments():
    frame = _performance()
    frame["transactions"] = [[] for _ in range(len(frame))]
    for index in range(len(frame)):
        frame.iat[index, frame.columns.get_loc("positions")] = [
            {
                "sid": {"sid": 7, "symbol": "NEW" if index else "OLD", "multiplier": 1},
                "amount": 2,
                "last_sale_price": 100,
            },
            {
                "sid": {"sid": 8, "symbol": "NEW", "multiplier": 1},
                "amount": -1,
                "last_sale_price": 100,
            },
        ]
    analysis = portfolio_analysis_from_zipline(frame, calendar=None)
    assert analysis.input_data.positions["asset"].unique().sort().to_list() == [7, 8]
    assert analysis.compute_holdings_analysis().timeline["gross_exposure"].to_list() == [300] * 4


def test_futures_marked_notional_is_distinct_from_reported_cash_and_nav():
    frame = _performance()
    for index in range(len(frame)):
        frame.iat[index, frame.columns.get_loc("positions")] = [
            {
                "sid": {"sid": 19, "symbol": "ESH25", "multiplier": 50},
                "amount": 1,
                "last_sale_price": 4000,
            }
        ]
    frame["portfolio_value"] = [100000.0] * 4
    frame["ending_cash"] = [100000.0] * 4
    frame["transactions"] = [[] for _ in range(len(frame))]
    analysis = portfolio_analysis_from_zipline(frame, calendar=None)
    holdings = analysis.compute_holdings_analysis()
    assert holdings.timeline["gross_exposure"].to_list() == [200000] * 4
    assert holdings.timeline["cash"].to_list() == [100000] * 4
    assert holdings.timeline["equity"].to_list() == [100000] * 4


@pytest.mark.parametrize("bad", [None, {}, "missing"])
def test_missing_nested_snapshots_are_not_treated_as_complete_empty_books(bad):
    frame = _performance()
    frame.iat[0, frame.columns.get_loc("positions")] = bad
    before = deepcopy(frame)
    with pytest.raises(ValueError, match="nested record lists"):
        portfolio_analysis_from_zipline(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_integer_sids_require_declared_multipliers_instead_of_guessing_asset_class():
    frame = _performance()
    for name in ("positions", "transactions"):
        frame[name] = frame[name].map(
            lambda rows: [{**record, "sid": record["sid"]["sid"]} for record in rows]
        )
    with pytest.raises(ValueError, match="multipliers"):
        portfolio_analysis_from_zipline(frame)
    analysis = portfolio_analysis_from_zipline(
        frame, asset_metadata={7: {"symbol": "A", "multiplier": 1}}
    )
    assert analysis.input_data.transactions["quantity"].to_list() == [3, -1]


def test_source_physical_symbol_alias_never_overwrites_stable_ids_with_display_labels():
    analysis = portfolio_analysis_from_zipline(
        _performance(), feed_spec={"timestamp_col": "when", "entity_col": "symbol"}
    )
    assert analysis.input_data.transactions["asset"].to_list() == [7, 7]
    assert analysis.input_data.transactions["symbol"].to_list() == ["A", "A"]
    assert analysis.input_data.metadata["feed_spec"]["entity_col"] == "symbol"


def test_importing_zipline_adapter_and_consuming_capture_never_imports_execution_or_legacy_engines():
    code = f"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('zipline', 'pyfolio', 'empyrical', 'alphalens') or fullname.startswith('ml4t.backtest'):
            raise RuntimeError('forbidden optional engine import: ' + fullname)
sys.meta_path.insert(0, Block())
import json, pandas as pd
from ml4t.diagnostic.integration.zipline import portfolio_analysis_from_zipline
frame = pd.DataFrame(json.load(open({str(FIXTURE)!r}))['records'])
frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop('date')))
result = portfolio_analysis_from_zipline(frame, calendar=None)
assert result.input_data.transactions.height == 2
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
