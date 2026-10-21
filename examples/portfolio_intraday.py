"""Analyze complete observed intraday snapshots without manufacturing return observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis import analyze_intraday_exposure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Save the combined report (Plotly required)")
    args = parser.parse_args()
    clock = pd.to_datetime(["2025-07-07T13:30Z", "2025-07-07T14:00Z", "2025-07-07T20:00Z"])
    positions = pd.DataFrame(
        {"A": [400.0, 600.0, 600.0], "B": [-300.0] * 3, "cash": [900.0, 699.0, 699.0]}, index=clock
    )
    nav = pd.DataFrame({"equity": [1000.0, 999.0, 999.0]}, index=clock)
    observed = analyze_intraday_exposure(positions, nav, complete_snapshots=True, calendar=None)
    assert observed.timeline["gross_exposure"].to_list() == [700.0, 900.0, 900.0]
    assert observed.timeline["cash"].to_list() == [900.0, 699.0, 699.0]
    assert observed.metrics["max_gross_exposure"] == 900 / 999
    unknown_nav = analyze_intraday_exposure(positions, complete_snapshots=True, calendar=None)
    assert unknown_nav.timeline["gross_exposure"].to_list() == [700.0, 900.0, 900.0]
    assert unknown_nav.timeline["equity"].null_count() == 3
    assert unknown_nav.coverage["n_normalized_observations"] == 0
    print(observed.timeline)
    print("Three observed snapshots, no return-series input; supplied cash900/699/699 is unchanged")
    print(
        "Peak900/999 describes this sample, not a continuous-time maximum; unknown NAV stays unknown"
    )
    if args.report:
        from ml4t.diagnostic.visualization.portfolio import create_round_trip_dashboard

        # One separately supplied daily net return and two actual same-instant fills.
        # The complete carried quantities do not declare unknown original lot costs.
        analysis = PortfolioAnalysis.from_frames(
            pd.Series([-0.001], index=clock[-1:].normalize()),
            calendar=None,
            periods_per_year=252,
            transactions=pd.DataFrame(
                {
                    "timestamp": [clock[1]] * 2,
                    "asset": ["A"] * 2,
                    "quantity": [1.0] * 2,
                    "price": [100.0] * 2,
                    "commission": [0.5] * 2,
                }
            ),
        )
        trips = analysis.compute_round_trips(
            initial_state="supplied",
            opening_as_of=clock[0],
            opening_lots=pd.DataFrame({"asset": ["A", "B"], "quantity": [4.0, -3.0]}),
        )
        report = create_round_trip_dashboard(analysis, trips, intraday=observed)
        assert report.intraday.timeline.height == 3
        assert report.intraday.timeline["gross_exposure"].max() == 900
        assert report.round_trips.completed_trades().is_empty()
        assert report.round_trips.open_lots.filter(pl.col("asset") == "A")["quantity"].sum() == 6
        report.save_html(args.report, include_plotlyjs=True)
        print(
            f"Saved {args.report}: one daily return, three intraday observations, no invented closes"
        )


if __name__ == "__main__":
    main()
