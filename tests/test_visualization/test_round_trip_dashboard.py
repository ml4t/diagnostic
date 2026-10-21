"""Actual supplied fills must reach tables, every report panel and exact owned export."""

import json
from html.parser import HTMLParser

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import create_round_trip_dashboard


class _ReportHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = 0
        self.plots = 0
        self.headings = []
        self.text = []
        self._heading = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables += 1
        if tag == "div" and dict(attrs).get("class") == "plot-container":
            self.plots += 1
        if tag == "h3":
            self._heading = True

    def handle_endtag(self, tag):
        if tag == "h3":
            self._heading = False

    def handle_data(self, data):
        self.text.append(data)
        if self._heading:
            self.headings.append(data)


def _account():
    dates = pd.date_range("2025-07-07", periods=12, tz="UTC")
    fills = []
    for index, cash in enumerate([20, -10, 0, 30, -20, 10]):
        side = 1 if index < 3 else -1
        for stamp, quantity, price in (
            (dates[index * 2], side, 100),
            (dates[index * 2 + 1], -side, 100 + side * cash),
        ):
            fills.append(
                {
                    "timestamp": stamp,
                    "asset": "A" if index % 2 == 0 else "B",
                    "quantity": quantity,
                    "price": price,
                    "commission": 1,
                }
            )
    return PortfolioAnalysis.from_frames(
        pd.Series([0.01, -0.02] * 6, index=dates),
        transactions=pd.DataFrame(fills),
        calendar=None,
    )


def test_actual_account_fills_reach_all_round_trip_panels_tables_and_beta_sample(tmp_path):
    analysis = _account()
    result = analysis.compute_round_trips(initial_state="flat")
    report = create_round_trip_dashboard(
        analysis, result, sector_mappings={"A": "Tech", "B": "Finance"}
    )
    assert set(report.figures).issuperset(
        {
            "Round-trip Lifetimes",
            "Round-trip Win Probability",
            "Round-trip Holding Duration",
            "Round-trip Cash PnL",
            "Round-trip Returns",
            "Round-trip Asset Attribution",
            "Round-trip Sector Attribution",
        }
    )
    assert report.round_trip_statistics is not None
    assert set(report.round_trip_statistics.tables) == {
        "pnl",
        "summary",
        "duration",
        "returns",
        "symbols",
    }
    assert list(report.figures["Round-trip Cash PnL"].data[0].x) == [18, -12, -2, 28, -22, 8]
    assert list(report.figures["Round-trip Holding Duration"].data[0].x) == [1] * 6
    assert list(report.figures["Round-trip Returns"].data[0].x) == [
        0.18,
        -0.12,
        -0.02,
        0.28,
        -0.22,
        0.08,
    ]
    posterior = report.figures["Round-trip Win Probability"].layout.meta["posterior"]
    assert posterior["posterior_alpha"] == posterior["posterior_beta"] == 4
    assert posterior["mean"] == 0.5
    assert posterior["n_observations"] == 6
    assert result.executions.equals(analysis.input_data.transactions)
    output = tmp_path / "round-trips.html"
    report.save_html(output, include_plotlyjs=True)
    page = _ReportHTML()
    page.feed(output.read_text())
    assert "Round-trip statistics" in page.text
    assert page.tables == 6
    assert page.headings == [
        "Pnl",
        "Summary",
        "Duration",
        "Returns",
        "Symbols",
        "Coverage",
    ]
    assert page.plots == len(report.figures)
    exported = report.to_dict()["round_trips"]
    assert exported["statistics"]["metadata"]["pnl"] == "net"
    assert exported["statistics"]["metadata"]["return_basis"] == "lot"
    assert len(exported["statistics"]["observations"]) == 6
    json.dumps(exported, allow_nan=False)


def test_signed_cash_attribution_can_be_negative_or_exceed_one_and_unmapped_assets_remain():
    result = _account().compute_round_trips(initial_state="flat")
    table = result.profit_attribution(sector_mappings={"A": "Tech"})
    assets = table.filter(pl.col("scope") == "asset").sort("asset")
    assert assets["pnl"].to_list() == [-6.0, 24.0]
    assert assets["pnl_fraction"].to_list() == pytest.approx([-1 / 3, 4 / 3])
    sectors = table.filter(pl.col("scope") == "sector").sort("sector")
    assert sectors["sector"].to_list() == ["No Sector Mapping", "Tech"]
    assert sectors["pnl"].to_list() == [24.0, -6.0]
    assert result.profit_attribution(pnl="gross").sort("asset")["pnl"].to_list() == [0.0, 30.0]


def test_missing_nav_denominators_omit_only_return_histogram_not_cash_or_lifetimes():
    analysis = _account()
    result = analysis.compute_round_trips(initial_state="flat")
    nav = pd.Series([1000.0], index=pd.date_range("2025-01-01", periods=1, tz="UTC"))
    report = create_round_trip_dashboard(analysis, result, return_basis="nav", portfolio_value=nav)
    assert not report.figures["Round-trip Returns"].data
    assert sum(report.figures["Round-trip Cash PnL"].data[0].x) == 18
    assert len(report.figures["Round-trip Lifetimes"].data) == 6
    assert (
        report.round_trip_statistics.coverage.filter(pl.col("scope") == "all")[
            "n_known_returns"
        ].item()
        == 0
    )


def _open_book(*, unknown_time=False, unknown_basis=False):
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, -0.01], index=dates),
        calendar=None,
        transactions=pd.DataFrame(
            {
                "timestamp": dates[-1:],
                "asset": ["A"],
                "quantity": [-1],
                "price": [120],
                "commission": [0],
            }
        ),
    )
    result = analysis.compute_round_trips(
        initial_state="supplied",
        opening_as_of=dates[0],
        opening_lots=pd.DataFrame(
            {
                "asset": ["A"],
                "quantity": [1],
                "remaining_commission": [0],
                "entry_price": [None if unknown_basis else 100],
                "entry_timestamp": pd.to_datetime([None if unknown_time else dates[0]]),
            }
        ),
    )
    return analysis, result


def test_unknown_carried_entry_time_omits_only_lifeline_and_duration_not_known_cash():
    analysis, result = _open_book(unknown_time=True)
    report = create_round_trip_dashboard(analysis, result)
    assert not report.figures["Round-trip Lifetimes"].data
    assert not report.figures["Round-trip Holding Duration"].data
    assert list(report.figures["Round-trip Cash PnL"].data[0].x) == [20]
    assert (
        report.figures["Round-trip Win Probability"].layout.meta["posterior"]["n_observations"] == 1
    )
    assert report.availability["round_trip_lifetimes"]["status"] == "unavailable"


def test_unknown_opening_cost_cannot_be_displayed_as_zero_pnl_or_dropped_silently():
    analysis, result = _open_book(unknown_basis=True)
    with pytest.raises(ValueError, match="net_pnl"):
        create_round_trip_dashboard(analysis, result)
    with pytest.raises(ValueError, match="unknown gross PnL"):
        result.profit_attribution(pnl="gross")


def test_synthetic_terminal_closes_are_opt_in_in_every_report_sample_and_export():
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, -0.01], index=dates),
        calendar=None,
        transactions=pd.DataFrame(
            {
                "timestamp": dates[:1],
                "asset": ["A"],
                "quantity": [1],
                "price": [100],
                "commission": [1],
            }
        ),
    )
    result = analysis.compute_round_trips(
        initial_state="flat",
        marks=pd.DataFrame({"timestamp": dates[-1:], "asset": ["A"], "price": [110]}),
        as_of=dates[-1],
        synthetic_close=True,
    )
    actual = create_round_trip_dashboard(analysis, result)
    assert actual.availability["round_trips"]["status"] == "empty"
    assert not actual.figures["Round-trip Cash PnL"].data
    assert (
        actual.figures["Round-trip Win Probability"].layout.meta["posterior"]["sample_state"]
        == "prior_only"
    )
    hypothetical = create_round_trip_dashboard(analysis, result, include_synthetic=True)
    assert list(hypothetical.figures["Round-trip Cash PnL"].data[0].x) == [9]
    assert hypothetical.figures["Round-trip Cash PnL"].data[0].name == "Synthetic"
    assert hypothetical.round_trip_statistics.observations["is_synthetic"].to_list() == [True]
    assert hypothetical.round_trips.open_lots.height == 1
    assert hypothetical.round_trips.executions.height == 1
    assert hypothetical.round_trips.metrics["realized_gross_pnl"] == 0


def test_report_owns_prepared_frames_and_exact_export_without_mutating_analysis_or_inputs():
    analysis = _account()
    result = analysis.compute_round_trips(initial_state="flat")
    before = result.segments.clone()
    report = create_round_trip_dashboard(analysis, result)
    result.metadata["changed"] = True
    result.segments.replace_column(
        result.segments.columns.index("gross_pnl"),
        pl.Series("gross_pnl", [999.0] * result.segments.height),
    )
    assert "changed" not in report.round_trips.metadata
    assert report.round_trips.segments.equals(before)
    exported = report.to_dict()["round_trips"]
    exported["metadata"]["changed"] = True
    assert "changed" not in report.to_dict()["round_trips"]["metadata"]
    assert analysis.input_data.transactions.equals(report.round_trips.executions)


def test_report_refuses_unrelated_prepared_execution_ledger():
    analysis = _account()
    _, other = _open_book()
    with pytest.raises(ValueError, match="exact actual transactions"):
        create_round_trip_dashboard(analysis, other)


def test_zero_total_cash_attribution_stays_undefined_instead_of_inventing_zero_shares():
    analysis, result = _open_book()
    fills = analysis.input_data.transactions.with_columns(pl.lit(100).alias("price"))
    zero = PortfolioAnalysis.from_frames(
        pd.Series([0.01, -0.01], index=pd.date_range("2025-07-07", periods=2, tz="UTC")),
        transactions=fills,
        calendar=None,
    )
    result = zero.compute_round_trips(
        initial_state="supplied",
        opening_lots=result.opening_lots,
        opening_as_of=pd.Timestamp("2025-07-07", tz="UTC"),
    )
    table = result.profit_attribution()
    assert table["pnl"].to_list() == [0.0]
    assert table["pnl_fraction"].to_list() == [None]


def test_real_reference_sector_complete_case_drop_does_not_remove_known_native_cash():
    reference = pytest.importorskip("pyfolio.round_trips")
    # A complete flat-book actual ledger produces a known cash trade with no NAV row.
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    fills = pd.DataFrame(
        {"symbol": ["A", "A"], "amount": [1, -1], "price": [100, 120]}, index=dates
    )
    unknown_nav = pd.Series([1000.0], index=dates[:1] - pd.Timedelta("1D"))
    observed = reference.extract_round_trips(fills, portfolio_value=unknown_nav)
    assert observed["pnl"].tolist() == [20]
    assert observed["returns"].isna().all()
    assert reference.apply_sector_mappings_to_round_trips(observed, {"A": "Tech"}).empty
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, -0.01], index=dates),
        calendar=None,
        transactions=fills.rename(columns={"symbol": "asset", "amount": "quantity"})
        .reset_index(names="timestamp")
        .assign(commission=0),
    )
    result = analysis.compute_round_trips(initial_state="flat")
    own = result.profit_attribution(pnl="gross", sector_mappings={"A": "Tech"})
    assert own.filter(pl.col("scope") == "sector")["pnl"].to_list() == [20.0]
    assert result.portfolio_contributions(unknown_nav)["gross_nav_contribution"].null_count() == 1


def test_real_reference_all_win_beta_failure_is_reconciled_with_finite_beta61_prior_sample():
    reference = pytest.importorskip("pyfolio.round_trips")
    plotting = pytest.importorskip("pyfolio.plotting")
    from matplotlib import pyplot as plt

    dates = pd.date_range("2025-07-07", periods=10, tz="UTC")
    fills = pd.DataFrame(
        {"symbol": ["A"] * 10, "amount": [1, -1] * 5, "price": [100, 110] * 5}, index=dates
    )
    observed = reference.extract_round_trips(fills)
    assert observed["pnl"].tolist() == [10] * 5
    try:
        with pytest.raises(ValueError, match="NaN|Inf"):
            plotting.plot_prob_profit_trade(observed)
    finally:
        plt.close("all")
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01] * 10, index=dates),
        calendar=None,
        transactions=fills.rename(columns={"symbol": "asset", "amount": "quantity"})
        .reset_index(names="timestamp")
        .assign(commission=0),
    )
    report = create_round_trip_dashboard(
        analysis, analysis.compute_round_trips(initial_state="flat")
    )
    posterior = report.figures["Round-trip Win Probability"].layout.meta["posterior"]
    assert posterior["posterior_alpha"] == 6
    assert posterior["posterior_beta"] == 1
    assert posterior["mean"] == 6 / 7
    assert all(pd.notna(value) for value in report.figures["Round-trip Win Probability"].data[0].y)


def test_exact_ns_instants_and_large_integer_asset_ids_survive_lifelines_and_owned_export():
    account = _account().input_data
    fills = account.transactions.with_columns(
        (pl.col("timestamp").cast(pl.Datetime("ns", "UTC")) + pl.duration(nanoseconds=7)).alias(
            "timestamp"
        ),
        pl.when(pl.col("asset") == "A")
        .then(pl.lit(2**53 + 1))
        .otherwise(pl.lit(2**53 + 2))
        .alias("asset"),
    )
    expected = pd.Timestamp("2025-07-07T00:00:00.000000007Z").value
    assert fills["timestamp"].cast(pl.Int64)[0] == expected
    analysis = PortfolioAnalysis.from_frames(account.returns, transactions=fills, calendar=None)
    report = create_round_trip_dashboard(
        analysis, analysis.compute_round_trips(initial_state="flat")
    )
    first = report.figures["Round-trip Lifetimes"].data[0]
    assert pd.Timestamp(first.customdata[0]).value == expected
    exported = report.to_dict()["round_trips"]
    assert pd.Timestamp(exported["executions"][0]["timestamp"]).value == expected
    assert exported["executions"][0]["asset"] == 2**53 + 1
    assert (
        pd.Timedelta(exported["statistics"]["observations"][0]["duration"]).value
        == pd.Timedelta("1D").value
    )
