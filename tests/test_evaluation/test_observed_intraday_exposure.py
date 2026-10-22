"""Real observed marked snapshots need no fabricated return or end-of-day clock."""

from datetime import date

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.evaluation.portfolio_analysis import analyze_intraday_exposure


def _snapshots():
    dates = pd.to_datetime(["2025-07-07T13:30Z", "2025-07-07T14:00Z", "2025-07-07T20:00Z"])
    # Actual supplied states around two same-instant1unit buys at100, .5cash fee each.
    positions = pd.DataFrame(
        {"A": [400.0, 600.0, 600.0], "B": [-300.0] * 3, "cash": [900.0, 699.0, 699.0]}, index=dates
    )
    nav = pd.DataFrame({"equity": [1000.0, 999.0, 999.0]}, index=dates)
    return positions, nav


def test_observed_intraday_own_clock_retains_actual_signed_values_cash_costs_and_no_return_input():
    positions, nav = _snapshots()
    before = [positions.copy(deep=True), nav.copy(deep=True)]
    result = analyze_intraday_exposure(positions, nav, complete_snapshots=True, calendar=None)
    assert result.snapshot_kind == "intraday"
    assert result.timeline["gross_exposure"].to_list() == [700.0, 900.0, 900.0]
    assert result.timeline["net_exposure"].to_list() == [100.0, 300.0, 300.0]
    assert result.timeline["short_exposure"].to_list() == [300.0] * 3
    assert result.timeline["cash"].to_list() == [900.0, 699.0, 699.0]
    assert result.timeline["gross_exposure_fraction"].to_list() == pytest.approx(
        [0.7, 900 / 999, 900 / 999]
    )
    assert result.timeline["gross_hhi"].to_list() == pytest.approx([25 / 49, 5 / 9, 5 / 9])
    assert result.metrics["max_gross_exposure"] == pytest.approx(900 / 999)
    assert result.coverage["n_observations"] == result.coverage["n_normalized_observations"] == 3
    assert result.metadata["method"] == "observed_complete_snapshots"
    assert (
        result.metadata["peak_semantics"]
        == "maximum over supplied observations, not a continuous-time maximum"
    )
    pd.testing.assert_frame_equal(positions, before[0])
    pd.testing.assert_frame_equal(nav, before[1])


def test_available_cash_plus_positions_never_invents_unreported_nav_or_normalized_exposure():
    positions, _ = _snapshots()
    result = analyze_intraday_exposure(positions, complete_snapshots=True, calendar=None)
    assert result.timeline["gross_exposure"].to_list() == [700.0, 900.0, 900.0]
    assert result.timeline["cash"].to_list() == [900.0, 699.0, 699.0]
    assert result.timeline["equity"].null_count() == result.timeline.height
    assert result.timeline["gross_exposure_fraction"].null_count() == result.timeline.height
    assert result.coverage["n_normalized_observations"] == 0
    assert result.metrics["max_gross_exposure"] is None


def test_declared_and_explicit_none_disable_nyse_default_without_relabeling_overnight_instants():
    dates = pd.to_datetime(["2025-07-06T23:30Z", "2025-07-07T00:30Z"])
    positions = pd.DataFrame({"A": [100.0, 100.0]}, index=dates)
    nav = pd.DataFrame({"equity": [1000.0, 1000.0]}, index=dates)
    metadata = FeedSpec(time_contract_version=2, calendar=None)
    inherited = analyze_intraday_exposure(
        positions, nav, complete_snapshots=True, feed_spec=metadata
    )
    disabled = analyze_intraday_exposure(positions, nav, complete_snapshots=True, calendar=None)
    for result in (inherited, disabled):
        assert result.metadata["calendar"] is None
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(result.timeline["date"].to_pandas()), dates, check_names=False
        )
        assert "session" not in result.timeline.columns
    assert inherited.metadata["calendar_source"] == "feed_spec"
    assert disabled.metadata["calendar_source"] == "argument"
    with pytest.raises(ValueError, match="sessions"):
        analyze_intraday_exposure(positions, nav, complete_snapshots=True)


def test_cme_sunday_evening_and_monday_utc_share_exchange_session_without_clock_compression():
    dates = pd.to_datetime(["2025-07-06T23:30:00.000000007Z", "2025-07-07T00:30:00.000000009Z"])
    positions = pd.DataFrame({"A": [100.0, 200.0]}, index=dates)
    nav = pd.DataFrame({"equity": [1000.0, 1000.0]}, index=dates)
    result = analyze_intraday_exposure(
        positions, nav, complete_snapshots=True, calendar="CME_Equity"
    )
    assert result.timeline["session"].to_list() == [date(2025, 7, 7)] * 2
    assert result.timeline["date"].cast(pl.Int64).to_list() == dates.as_unit("ns").asi8.tolist()
    assert result.timeline["gross_exposure"].to_list() == [100.0, 200.0]
    disabled = analyze_intraday_exposure(
        positions,
        nav,
        complete_snapshots=True,
        feed_spec=FeedSpec(time_contract_version=2, calendar="NYSE"),
        calendar=None,
    )
    assert disabled.metadata["calendar"] is None


def test_right_bar_close_rejects_open_anchor_but_default_snapshot_clock_retains_both_boundaries():
    positions, nav = _snapshots()
    feed = FeedSpec(time_contract_version=2, timestamp_semantics="bar_close", bar_label="right")
    # The opening account anchor is not a completed right-labeled bar.
    with pytest.raises(ValueError, match="sessions"):
        analyze_intraday_exposure(positions, nav, complete_snapshots=True, feed_spec=feed)
    result = analyze_intraday_exposure(
        positions.iloc[1:], nav.iloc[1:], complete_snapshots=True, feed_spec=feed
    )
    assert result.metadata["calendar"] == "NYSE"
    assert result.timeline["session"].to_list() == [date(2025, 7, 7)] * 2
    observed = analyze_intraday_exposure(positions, nav, complete_snapshots=True)
    assert observed.metadata["calendar"] == "NYSE"
    assert observed.metadata["calendar_source"] == "default"
    assert observed.timeline["session"].to_list() == [date(2025, 7, 7)] * 3


def test_naive_event_instants_require_declared_timezone_for_schedule_validation_not_guessed_utc():
    dates = pd.to_datetime(["2025-07-07T09:30", "2025-07-07T10:00"])
    positions = pd.DataFrame({"A": [100.0, 100.0]}, index=dates)
    with pytest.raises(ValueError, match="timezone"):
        analyze_intraday_exposure(positions, complete_snapshots=True)
    result = analyze_intraday_exposure(
        positions,
        complete_snapshots=True,
        feed_spec=FeedSpec(time_contract_version=2, timezone="America/New_York"),
    )
    assert result.timeline["date"].dtype.time_zone is None
    assert result.timeline["session"].to_list() == [date(2025, 7, 7)] * 2


def test_date_or_session_labels_cannot_be_claimed_as_observed_intraday_instants():
    labels = pl.DataFrame({"date": [date(2025, 7, 7)], "A": [100.0]})
    with pytest.raises(ValueError, match="instants"):
        analyze_intraday_exposure(labels, complete_snapshots=True, calendar=None)
    midnight = pd.DataFrame({"A": [100.0]}, index=pd.date_range("2025-07-07", periods=1, tz="UTC"))
    with pytest.raises(ValueError, match="session labels"):
        analyze_intraday_exposure(
            midnight,
            complete_snapshots=True,
            calendar=None,
            feed_spec=FeedSpec(time_contract_version=2, timestamp_semantics="session_label"),
        )


def test_incomplete_or_unknown_marked_snapshots_do_not_become_observed_zero_exposure():
    positions, nav = _snapshots()
    with pytest.raises(ValueError, match="complete_snapshots"):
        analyze_intraday_exposure(positions, nav, complete_snapshots=False, calendar=None)
    positions.iloc[1, positions.columns.get_loc("A")] = float("nan")
    before = positions.copy(deep=True)
    with pytest.raises(ValueError, match="value"):
        analyze_intraday_exposure(positions, nav, complete_snapshots=True, calendar=None)
    pd.testing.assert_frame_equal(positions, before)


def test_actual_all_cash_intraday_snapshots_are_zero_exposure_not_missing_assets_or_unknown_nav():
    dates = pd.date_range("2025-07-07T14:00Z", periods=2, freq="1h")
    positions = pd.DataFrame({"cash": [1000.0, 1000.0]}, index=dates)
    nav = pd.DataFrame({"equity": [1000.0, 1000.0]}, index=dates)
    result = analyze_intraday_exposure(positions, nav, complete_snapshots=True, calendar=None)
    assert result.positions.is_empty()
    assert result.timeline["gross_exposure_fraction"].to_list() == [0.0, 0.0]
    assert result.timeline["open_positions"].to_list() == [0, 0]
    assert result.timeline["gross_hhi"].null_count() == 2


def test_feed_column_mapping_large_integer_ids_and_ns_keys_survive_native_observed_route():
    dates = pl.Series(
        "at", [1751896800000000007, 1751896800000000009], dtype=pl.Datetime("ns", "UTC")
    )
    positions = pl.DataFrame({"at": dates, "sid": [2**53 + 1] * 2, "value": [100.0, 200.0]})
    nav = pl.DataFrame({"at": dates, "equity": [1000.0, 1000.0]})
    feed = FeedSpec(time_contract_version=2, timestamp_col="at", entity_col="sid", calendar=None)
    result = analyze_intraday_exposure(positions, nav, complete_snapshots=True, feed_spec=feed)
    assert result.positions["asset"].to_list() == [2**53 + 1] * 2
    assert result.timeline["date"].cast(pl.Int64).to_list() == dates.cast(pl.Int64).to_list()
    result.metadata["feed_spec"]["calendar"] = "CME_Equity"
    again = analyze_intraday_exposure(positions, nav, complete_snapshots=True, feed_spec=feed)
    assert again.metadata["feed_spec"]["calendar"] is None


def test_duplicate_holdings_or_conflicting_observed_cash_raise_without_partial_results():
    positions, nav = _snapshots()
    duplicates = pd.concat([positions, positions.iloc[:1]])
    with pytest.raises(ValueError, match="unique|duplicate"):
        analyze_intraday_exposure(duplicates, nav, complete_snapshots=True, calendar=None)
    nav["cash"] = [900.0, 999.0, 699.0]
    with pytest.raises(ValueError, match="cash"):
        analyze_intraday_exposure(positions, nav, complete_snapshots=True, calendar=None)
