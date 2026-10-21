"""Run a local Zipline bundle or analyze its portable captured account, without market downloads.

Use --capture PATH with optional Zipline3.2 installed to regenerate the producer.
Use --input PATH without Zipline to consume the same nested records. --report PATH
renders the prepared actual-fill workflow with Diagnostic's visualization extra.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from ml4t.diagnostic.integration.zipline import portfolio_analysis_from_zipline


def _portable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "sid"):
        return {
            "sid": int(value.sid),
            "symbol": value.symbol,
            "multiplier": float(value.price_multiplier),
        }
    return value.item() if hasattr(value, "item") else value


def capture_zipline() -> tuple[pd.DataFrame, dict[str, str]]:
    from zipline import run_algorithm
    from zipline.api import order, set_commission, set_slippage, sid
    from zipline.assets.synthetic import make_simple_equity_info
    from zipline.data import bundles
    from zipline.finance.commission import PerShare
    from zipline.finance.slippage import FixedSlippage
    from zipline.utils.calendar_utils import get_calendar

    dates = pd.to_datetime(["2025-07-07", "2025-07-08", "2025-07-09", "2025-07-10"])
    name = "diagnostic-supplied-account-example"

    def ingest(environ, assets, minute, daily, adjustments, calendar, *args):
        assets.write(equities=make_simple_equity_info([7], dates[0], dates[-1], symbols=["A"]))
        bars = pd.DataFrame(
            {field: [100.0, 100.0, 110.0, 120.0] for field in ("open", "high", "low", "close")},
            index=dates,
        )
        bars["volume"] = 1000000
        daily.write([(7, bars)])
        adjustments.write()

    def initialize(context):
        context.observations = 0
        set_commission(us_equities=PerShare(cost=0.1, min_trade_cost=0))
        set_slippage(us_equities=FixedSlippage(spread=0))

    def handle_data(context, data):
        context.observations += 1
        if context.observations == 1:
            order(sid(7), 3)
        elif context.observations == 3:
            order(sid(7), -1)

    with tempfile.TemporaryDirectory(prefix="diagnostic-zipline-account-") as scratch:
        environ = {"ZIPLINE_ROOT": scratch}
        bundles.register(name, ingest, start_session=dates[0], end_session=dates[-1])
        try:
            bundles.ingest(name, environ=environ)
            performance = run_algorithm(
                start=dates[0],
                end=dates[-1],
                initialize=initialize,
                handle_data=handle_data,
                capital_base=1000,
                bundle=name,
                environ=environ,
                default_extension=False,
                trading_calendar=get_calendar("XNYS"),
                benchmark_returns=pd.Series(0.0, index=dates),
            )
        finally:
            bundles.unregister(name)
    versions = {
        name: version(name)
        for name in ("zipline-reloaded", "pandas", "numpy", "exchange-calendars")
    }
    return performance, versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--capture", type=Path)
    choice.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.input:
        captured = json.loads(args.input.read_text())
        performance = pd.DataFrame(captured["records"])
        performance.index = pd.DatetimeIndex(pd.to_datetime(performance.pop("date")))
    else:
        performance, versions = capture_zipline()
        columns = [
            "returns",
            "portfolio_value",
            "starting_cash",
            "ending_cash",
            "positions",
            "transactions",
        ]
        captured = {
            "format_version": 1,
            "producer_versions": versions,
            "producer": "published run_algorithm, local deterministic daily bundle, no downloads",
            "records": [
                {"date": timestamp.isoformat(), **_portable(row.to_dict())}
                for timestamp, row in performance[columns].iterrows()
            ],
        }
        args.capture.write_text(json.dumps(captured, indent=2, allow_nan=False) + "\n")
    # The --capture path consumes the actual in-memory result including Asset objects.
    analysis = portfolio_analysis_from_zipline(performance, calendar="NYSE")
    assert analysis.input_data.transactions["quantity"].to_list() == [3, -1]
    assert analysis.input_data.positions["asset"].unique().to_list() == [7]
    activity = analysis.compute_transaction_analysis(incremental_bps=[0, 10])
    print(json.dumps(captured.get("producer_versions", {}), sort_keys=True))
    print(analysis.compute_summary_stats().summary())
    print(analysis.compute_holdings_analysis().timeline)
    print(activity.timeline)
    assert activity.metrics["total_filled_notional"] == 420
    assert activity.metrics["total_implementation_cost"] is None
    print("Actual Zipline account: stable sid7, 420 traded, unknown fill commissions retained")
    if args.report:
        analysis.create_tear_sheet(transactions=activity).save_html(
            args.report, include_plotlyjs=True
        )


if __name__ == "__main__":
    main()
