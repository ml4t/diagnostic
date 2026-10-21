"""Consume an actual opt-in Backtest bar account through Diagnostic's owned boundary.

The default consumes a bundled real capture without Backtest. --run-engine
requires the unreleased source pin in the migration guide, not published0.1.12.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
import polars as pl

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioInputData


def run_engine() -> PortfolioAnalysis:
    from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
    from ml4t.backtest.config import CommissionType, SlippageType

    from ml4t.diagnostic.integration import portfolio_analysis_from_result

    class BuyOnce(Strategy):
        def __init__(self):
            self.ordered = False

        def on_data(self, timestamp, data, context, broker):
            if not self.ordered:
                broker.submit_order("A", 2)
                self.ordered = True

    timestamps = [datetime(2025, 7, 7, 14, minute, tzinfo=UTC) for minute in range(3)]
    prices = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": ["A"] * 3,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0, 100.0, 110.0]),
            "volume": [10000.0] * 3,
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        BuyOnce(),
        BacktestConfig(
            initial_cash=1000,
            calendar="NYSE",
            commission_type=CommissionType.NONE,
            slippage_type=SlippageType.NONE,
        ),
        holdings_sampling="bar",
    ).run()
    # One-minute NYSE observations, using the explicit252*390 annualization convention.
    return portfolio_analysis_from_result(result, calendar=None, periods_per_year=252 * 390)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--run-engine", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument(
        "--input", type=Path, default=Path(__file__).parent / "data/backtest_bar_account.json"
    )
    args = parser.parse_args()
    if args.run_engine:
        analysis = run_engine()
        producer_versions = {
            name: version(name) for name in ("ml4t-backtest", "ml4t-specs", "pandas", "polars")
        }
        if args.capture:
            direct_url = json.loads(
                distribution("ml4t-backtest").read_text("direct_url.json") or "{}"
            )
            args.capture.write_text(
                json.dumps(
                    {
                        "producer_revision": direct_url.get("vcs_info", {}).get("commit_id"),
                        "producer_versions": producer_versions,
                        "input_data": json.loads(analysis.input_data.to_json()),
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
    else:
        if args.capture:
            parser.error("--capture requires --run-engine and the documented producer")
        captured = json.loads(args.input.read_text())
        data = PortfolioInputData.from_json(json.dumps(captured["input_data"]))
        analysis = PortfolioAnalysis.from_frames(
            data.returns,
            return_col="return",
            positions=data.positions,
            transactions=data.transactions,
            valuations=data.valuations,
            complete_snapshots=True,
            calendar=None,
            periods_per_year=data.metadata["periods_per_year"],
        )
        producer_versions = captured["producer_versions"]
    np.testing.assert_allclose(analysis.returns, [0.0, 0.0, 0.02])
    holdings = analysis.compute_holdings_analysis()
    np.testing.assert_allclose(holdings.timeline["gross_exposure"], [0.0, 200.0, 220.0])
    np.testing.assert_allclose(holdings.timeline["cash"], [1000.0, 800.0, 800.0])
    np.testing.assert_allclose(holdings.timeline["equity"], [1000.0, 1000.0, 1020.0])
    activity = analysis.compute_transaction_analysis(incremental_bps=[0, 10])
    np.testing.assert_allclose(activity.timeline["filled_notional"], [0.0, 200.0, 0.0])
    np.testing.assert_allclose(
        activity.scenarios.filter(pl.col("incremental_bps") == 10)["incremental_cost_return"],
        [0.0, 0.0002, 0.0],
    )
    print("Captured producer", producer_versions)
    print(holdings.timeline)
    print(activity.timeline)
    print("Actual Backtest complete bar capture: owned returns, holdings, cash/NAV and fills")
    if args.report:
        analysis.create_tear_sheet(transactions=activity).save_html(
            args.report, include_plotlyjs=True
        )


if __name__ == "__main__":
    main()
