"""Run familiar transaction calculations without requiring Pyfolio at runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml4t.diagnostic.compat.pyfolio import txn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Compare with Pyfolio0.10.0")
    parser.add_argument(
        "--report", type=Path, help="Render all six transaction plot workflows (viz extra)"
    )
    args = parser.parse_args()
    dates = pd.date_range("2025-07-07", periods=3, tz="America/New_York")
    returns = pd.Series([0.01, -0.02, 0.03], index=dates)
    positions = pd.DataFrame(
        {"A": [200.0, 400.0, 600.0], "cash": [800.0, 600.0, 400.0]}, index=dates
    )
    transactions = pd.DataFrame(
        {"amount": [1.0, -1.0, 2.0], "price": [100.0, 100.0, 100.0]},
        index=pd.to_datetime(
            ["2025-07-07T15:00Z", "2025-07-07T16:00Z", "2025-07-09T15:00Z"]
        ).tz_convert("America/New_York"),
    )
    volume = txn.get_txn_vol(transactions)
    turnover = txn.get_turnover(positions, transactions)
    adjusted = txn.adjust_returns_for_slippage(returns, positions, transactions, 10)
    np.testing.assert_allclose(volume["txn_shares"], [2.0, 2.0])
    np.testing.assert_allclose(volume["txn_volume"], [200.0, 200.0])
    np.testing.assert_allclose(turnover, [2.0, 0.0, 0.4])
    np.testing.assert_allclose(adjusted, [0.0098, -0.02, 0.0298])
    if args.compare:
        from importlib.metadata import version

        import pyfolio.txn as reference

        if version("pyfolio-reloaded") != "0.10.0":
            raise ValueError("The reference recipe requires pyfolio-reloaded==0.10.0")
        pd.testing.assert_frame_equal(volume, reference.get_txn_vol(transactions))
        pd.testing.assert_series_equal(turnover, reference.get_turnover(positions, transactions))
        pd.testing.assert_series_equal(
            adjusted, reference.adjust_returns_for_slippage(returns, positions, transactions, 10)
        )
        print("Published Pyfolio0.10.0 matches on the identical daily input sample")
    print("Cash volumes200/200; unit volumes2/2; legacy AGB turnover2/0/.4")
    print("Extra10bps gives net returns .0098/-.02/.0298; no observed fee subtraction")
    if args.report:
        from ml4t.diagnostic.compat.pyfolio import plotting
        from ml4t.diagnostic.evaluation import PortfolioAnalysis
        from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet

        figures = {
            "Daily Turnover": plotting.plot_turnover(returns, transactions, positions),
            "Daily Volume": plotting.plot_daily_volume(returns, transactions),
            "Turnover Distribution": plotting.plot_daily_turnover_hist(transactions, positions),
            "Filtered Execution Timing": plotting.plot_txn_time_hist(transactions),
            "Slippage Wealth Paths": plotting.plot_slippage_sweep(returns, positions, transactions),
            "Slippage Annual Sensitivity": plotting.plot_slippage_sensitivity(
                returns, positions, transactions
            ),
        }
        analysis = PortfolioAnalysis.from_frames(returns, calendar=None, periods_per_year=252)
        sheet = PortfolioTearSheet(
            metrics=analysis.compute_summary_stats(),
            figures=figures,
            availability={
                "legacy_transaction_conventions": {
                    "status": "available",
                    "reason": "Observed daily inputs; zero-seeded AGB, closing position value, NY09:30-16:00 timing filter and daily252 annualization. No intraday position reconstruction or full legacy sheet is implied.",
                }
            },
        )
        sheet.save_html(args.report, include_plotlyjs=True)
        print(f"Wrote six prepared transaction plots to {args.report}")


if __name__ == "__main__":
    main()
