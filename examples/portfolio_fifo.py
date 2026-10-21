"""Match actual-format fills from a declared flat book, without a legacy engine."""

from __future__ import annotations

import pandas as pd

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def main() -> None:
    dates = pd.date_range("2025-07-07", periods=3, tz="UTC")
    # The supplied synthetic net returns are independent of lot attribution.
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, -0.02, 0.03], index=dates),
        transactions=pd.DataFrame(
            {
                "timestamp": dates,
                "asset": [7, 7, 7],
                "quantity": [10.0, 5.0, -12.0],
                "price": [100.0, 110.0, 120.0],
                "commission": [10.0, 10.0, 12.0],
            }
        ),
        calendar=None,
        transaction_timestamp_col="timestamp",
    )
    result = analysis.compute_round_trips(initial_state="flat")
    assert result.metrics["realized_gross_pnl"] == 220.0
    assert result.metrics["realized_net_pnl"] == 194.0
    assert result.open_lots["quantity"].to_list() == [3.0]
    assert result.open_lots["entry_price"].to_list() == [110.0]
    assert result.open_lots["remaining_commission"].to_list() == [6.0]
    assert analysis.returns.tolist() == [0.01, -0.02, 0.03]
    print(result.segments)
    print(result.open_lots)
    print("FIFO gross220/net194; remaining3@110 with6entry fees, not a fabricated close")
    completed = result.completed_trades()
    statistics = result.to_trade_analysis().compute_statistics()
    assert completed.height == statistics.n_trades == 1
    assert completed["entry_notional"].to_list() == [1220.0]
    assert statistics.total_pnl == 194.0
    posterior = result.win_probability()
    assert posterior["n_observations"] == 1
    assert posterior["mean"] == 2 / 3
    print("One closing fill, not two independent trades; net lot return194/1220, Beta(2,1)")
    contributions = result.portfolio_contributions(pd.Series([1000.0], index=dates[-1:]))
    assert contributions["gross_nav_contribution"].to_list() == [0.22]
    assert contributions["net_nav_contribution"].to_list() == [0.194]
    print(
        "Supplied beginning NAV1000: .22gross/.194net portfolio contribution, not220/1220lot return"
    )
    marked = analysis.compute_round_trips(
        initial_state="flat",
        marks=pd.DataFrame(
            {"timestamp": pd.to_datetime(["2025-07-09T21:00Z"]), "asset": [7], "price": [130.0]}
        ),
        as_of="2025-07-09T22:00Z",
        synthetic_close=True,
    )
    assert marked.metrics["realized_gross_pnl"] == 220.0
    assert marked.metrics["unrealized_gross_pnl"] == 60.0
    assert marked.metrics["marked_net_pnl"] == 248.0
    assert marked.open_lots["quantity"].to_list() == [3.0]
    assert marked.synthetic_segments["is_synthetic"].to_list() == [True]
    print("Quote130 known1hour before as_of:60unrealized,54net; synthetic row is not a fill")
    carried = analysis.compute_round_trips(
        initial_state="supplied",
        opening_as_of="2025-07-08T00:00Z",
        opening_lots=pd.DataFrame(
            {
                "asset": [7],
                "quantity": [10.0],
                "entry_timestamp": pd.to_datetime(["2025-07-07T00:00Z"]),
                "entry_price": [100.0],
                "remaining_commission": [10.0],
            }
        ),
    )
    assert carried.metrics["realized_gross_pnl"] == 220.0
    assert carried.metrics["realized_net_pnl"] == 194.0
    assert carried.metadata["excluded_before_opening"] == 1
    print("Supplied complete opening inventory excludes its prior fill from replay")
    intraday_fills = analysis.input_data.transactions.to_pandas()
    intraday_fills["timestamp"] = pd.to_datetime(
        ["2025-07-07T12:00Z", "2025-07-07T13:00Z", "2025-07-08T12:00Z"]
    )
    grouping_case = PortfolioAnalysis.from_frames(
        analysis.input_data.returns, transactions=intraday_fills, calendar=None
    )
    actual = grouping_case.compute_round_trips(initial_state="flat")
    grouped = grouping_case.compute_round_trips(initial_state="flat", matching="consecutive_vwap")
    assert actual.metrics["realized_gross_pnl"] == 220.0
    assert grouped.metrics["realized_gross_pnl"] == 200.0
    assert grouped.metrics["realized_net_pnl"] == 172.0
    assert grouped.completed_trades().height == 1
    assert grouped.matching_groups["n_source_executions"].to_list() == [2, 1]
    print("Explicit8hconsecutive grouping gives200/172realized, not actual FIFO220/194")


if __name__ == "__main__":
    main()
