"""Familiar reductions and display use actual extracted samples, not complete-case cash masks."""

from copy import deepcopy

import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import round_trips as rt


def _observations():
    transactions = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "amount": [1.0, -1.0, 1.0, -1.0],
            "price": [100.0, 110.0, 100.0, 120.0],
        },
        index=pd.to_datetime(
            ["2025-07-07T14:00Z", "2025-07-08T14:00Z", "2025-07-07T15:00Z", "2025-07-09T14:00Z"]
        ),
    )
    # The supplied beginning NAV really omits the second closing date.
    nav = pd.Series([1000.0], index=pd.to_datetime(["2025-07-08T00:00Z"]))
    return rt.extract_round_trips(transactions, nav)


@pytest.mark.parametrize(
    "column,metrics,expected",
    [
        ("pnl", {"sum": "sum", "mean": "mean"}, [30.0, 15.0]),
        ("pnl", ["sum", "mean"], [30.0, 15.0]),
        ("returns", {"known": "count", "mean": "mean"}, [1.0, 0.01]),
        ("duration", {"mean": "mean"}, [pd.Timedelta(hours=35, minutes=30)]),
        ("pnl", {"square_sum": lambda values: (values * values).sum()}, [500.0]),
    ],
)
def test_grouped_callbacks_keep_their_metric_specific_known_sample(column, metrics, expected):
    source = _observations()
    before = deepcopy(source)
    output = rt.agg_all_long_short(source, column, metrics)
    assert output["All trades"].to_list() == expected
    assert output["Long trades"].to_list() == expected
    assert output.columns.to_list() == ["All trades", "Long trades"]
    pd.testing.assert_frame_equal(source, before)


def test_sector_projection_explicitly_matches_legacy_complete_case_without_changing_original_cash():
    source = _observations()
    before = deepcopy(source)
    assert source["pnl"].to_list() == [10.0, 20.0]
    assert source["returns"].isna().to_list() == [False, True]
    projected = rt.apply_sector_mappings_to_round_trips(
        source, pd.Series({"A": "Tech", "B": "Rates"})
    )
    assert projected["symbol"].to_list() == ["Tech"]
    assert projected["pnl"].sum() == 10.0
    meta = projected.attrs["diagnostic_sector_projection"]
    assert meta["n_input_rows"] == 2 and meta["n_output_rows"] == 1
    assert meta["excluded_source_index"] == source.index[1:].to_list()
    unmapped = rt.apply_sector_mappings_to_round_trips(source, {})
    assert unmapped["symbol"].to_list() == ["No Sector Mapping"]
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("hide", [False, True])
def test_printed_tables_show_account_currency_and_nav_percent_with_hide_position_option(
    capsys, hide
):
    assert rt.print_round_trip_stats(_observations(), hide_pos=hide) is None
    text = capsys.readouterr().out
    for heading in ("Summary stats", "PnL stats", "Duration stats", "Return stats"):
        assert heading in text
    assert ("Symbol stats" in text) == (not hide)
    assert "30.00" in text and "1.00%" in text
    assert "supplied account currency" in text and "$" not in text
    assert "Unavailable" in text


def test_empty_grouped_sample_has_no_invented_all_trades_or_direction_counts():
    output = rt.agg_all_long_short(_observations().iloc[:0], "pnl", {"count": "count"})
    assert output.index.to_list() == ["count"] and not len(output.columns)


def test_actual_short_closure_is_its_own_group_not_a_signed_cash_filter():
    transactions = pd.DataFrame(
        {"symbol": ["S", "S"], "amount": [-1.0, 1.0], "price": [100.0, 95.0]},
        index=pd.to_datetime(["2025-07-07T14:00Z", "2025-07-08T14:00Z"]),
    )
    short = rt.extract_round_trips(transactions)
    source = pd.concat([_observations(), short], ignore_index=True)
    result = rt.agg_all_long_short(source, "pnl", [("sum", "sum")])
    assert result.columns.to_list() == ["All trades", "Short trades", "Long trades"]
    assert result.loc["sum"].to_list() == [35.0, 5.0, 30.0]


@pytest.mark.parametrize(
    "column,metrics",
    [("pnl", [("sum", "sum"), ("mean", "mean")]), ("duration", [("mean", "mean")])],
)
def test_actual_published_grouping_and_sector_helpers_match_on_shared_native_produced_observations(
    column, metrics
):
    reference = pytest.importorskip("pyfolio.round_trips")
    source = _observations()
    pd.testing.assert_frame_equal(
        rt.agg_all_long_short(source, column, metrics),
        reference.agg_all_long_short(source, column, metrics),
    )
    pd.testing.assert_frame_equal(
        rt.apply_sector_mappings_to_round_trips(source, {"A": "Tech"}),
        reference.apply_sector_mappings_to_round_trips(source, {"A": "Tech"}),
    )


@pytest.mark.parametrize("hide", [False, True])
def test_actual_published_print_route_receives_the_same_native_tables_and_percent_scale(
    monkeypatch, capsys, hide
):
    reference = pytest.importorskip("pyfolio.round_trips")
    displayed = []
    # Intercept presentation only: reference gen_round_trip_stats still executes.
    monkeypatch.setattr(
        reference,
        "print_table",
        lambda frame, **options: displayed.append((options["name"], frame.copy(deep=True))),
    )
    source = _observations()
    reference.print_round_trip_stats(source, hide_pos=hide)
    native = rt.gen_round_trip_stats(source)
    keys = ["summary", "pnl", "duration", "returns"] + ([] if hide else ["symbols"])
    assert len(displayed) == len(keys)
    for key, (_, frame) in zip(keys, displayed, strict=True):
        expected = native[key] * 100 if key in ("returns", "symbols") else native[key]
        pd.testing.assert_frame_equal(frame, expected)
    rt.print_round_trip_stats(source, hide_pos=hide)
    text = capsys.readouterr().out
    assert all(heading in text for heading, _ in displayed)


def test_actual_published_dictionary_aggregation_error_is_not_copied_into_modern_mapping_input():
    reference = pytest.importorskip("pyfolio.round_trips")
    source = _observations()
    with pytest.raises(pd.errors.SpecificationError, match="nested renamer"):
        reference.agg_all_long_short(source, "pnl", {"sum": "sum"})
    result = rt.agg_all_long_short(source, "pnl", {"sum": "sum"})
    assert result.loc["sum", "All trades"] == 30.0
