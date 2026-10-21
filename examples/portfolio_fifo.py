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


if __name__ == "__main__":
    main()
