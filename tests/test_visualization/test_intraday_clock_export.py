"""Real reconstructed known-at quote instants must retain nanoseconds in report exports."""

import pandas as pd
import polars as pl
import pytest

pytest.importorskip("plotly")

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import create_round_trip_dashboard


def test_actual_reconstructed_secondary_quote_clock_does_not_lose_nanoseconds_in_report_export():
    quote = pd.Timestamp("2025-07-07T13:30:00.000000007Z")
    fill = pd.Timestamp("2025-07-07T14:00:00.000000009Z")
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.0], index=[quote.normalize()]),
        calendar=None,
        transactions=pd.DataFrame(
            {
                "timestamp": [fill],
                "asset": ["A"],
                "quantity": [1.0],
                "price": [100.0],
                "commission": [0.0],
            }
        ),
    )
    trips = analysis.compute_round_trips(
        initial_state="flat",
        as_of="2025-07-07T20:00Z",
        marks=pd.DataFrame({"timestamp": [quote], "asset": ["A"], "price": [100.0]}),
    )
    intraday = trips.reconstruct_intraday_exposure(
        opening_cash=1000.0, cash_model="cash_price", opening_as_of="2025-07-07T13:30Z"
    )
    assert intraday.positions["mark_timestamp"].cast(pl.Int64)[0] == quote.value
    report = create_round_trip_dashboard(analysis, trips, intraday=intraday)
    row = report.to_dict()["intraday"]["positions"][0]
    assert row["mark_timestamp"] == quote.isoformat()
    assert pd.Timestamp(row["date"]).value == fill.value
    assert row["mark_age_ns"] == 1_800_000_000_002
    assert report.intraday.positions["mark_timestamp"].cast(pl.Int64)[0] == quote.value
