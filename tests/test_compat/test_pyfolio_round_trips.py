"""Familiar round-trip frames must preserve the native sample, labels and denominator."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import pyfolio as pf


def _fills(amounts=(10.0, 5.0, -12.0), prices=(100.0, 110.0, 120.0)):
    return pd.DataFrame(
        {"symbol": ["A"] * 3, "amount": amounts, "price": prices},
        index=pd.to_datetime(["2025-07-07T12:00Z", "2025-07-07T13:00Z", "2025-07-08T12:00Z"]),
    )


def test_familiar_extract_uses_consecutive_groups_and_does_not_alias_lot_return_to_nav():
    source = _fills()
    source.index.name = "execution"
    before = source.copy(deep=True)
    result = pf.round_trips.extract_round_trips(source)
    assert result.columns.to_list() == [
        "pnl",
        "open_dt",
        "close_dt",
        "long",
        "rt_returns",
        "symbol",
        "duration",
    ]
    assert result["pnl"].to_list() == [200.0]
    assert result["rt_returns"].to_list() == [200 / 1240]
    assert result["symbol"].to_list() == ["A"]
    assert result["long"].to_list() == [True]
    assert result.index.equals(pd.RangeIndex(1))
    pd.testing.assert_frame_equal(source, before)
    nav = pd.Series([1000.0], index=pd.DatetimeIndex(["2025-07-08T00:00Z"]))
    assert pf.round_trips.extract_round_trips(source, portfolio_value=nav)["returns"].to_list() == [
        0.2
    ]


def test_legacy_extraction_gross_profile_ignores_unused_cost_fields_instead_of_subtracting_twice():
    source = _fills()
    source["commission"] = ["unused", None, 1e9]
    source["slippage"] = ["unused"] * 3
    assert pf.round_trips.extract_round_trips(source)["pnl"].to_list() == [200.0]


def test_original_symbol_sorted_row_order_and_nav_assignment_survive_interleaved_assets():
    source = pd.DataFrame(
        {"symbol": ["B", "A", "B", "A"], "amount": [1, 1, -1, -1], "price": [100, 100, 150, 120]},
        index=pd.date_range("2025-07-07", periods=4, tz="UTC"),
    )
    nav = pd.Series(
        [1000.0, 2000.0], index=pd.DatetimeIndex(["2025-07-09T00:00Z", "2025-07-10T00:00Z"])
    )
    result = pf.round_trips.extract_round_trips(source, portfolio_value=nav)
    assert result["symbol"].to_list() == ["A", "B"]
    assert result["pnl"].to_list() == [20.0, 50.0]
    assert result["returns"].to_list() == [0.01, 0.05]
    assert pf.round_trips._groupby_consecutive(source)["symbol"].to_list() == ["A", "A", "B", "B"]


def test_integer_and_string_display_ids_remain_distinct_instead_of_row_float_upcast():
    source = pd.DataFrame(
        {
            "symbol": [7, "7", 7, "7"],
            "amount": [1, 1, -1, -1],
            "price": [100.0, 100.0, 110.0, 120.0],
        },
        index=pd.date_range("2025-07-07", periods=4, tz="UTC"),
    )
    result = pf.round_trips.extract_round_trips(source)
    assert result["symbol"].to_list() == [7, "7"]
    assert result["pnl"].to_list() == [10.0, 20.0]


@dataclass(frozen=True, order=True)
class AssetLabel:
    sid: int
    symbol: str


def test_supplied_asset_objects_need_no_execution_package_import_or_string_identity_change():
    asset = AssetLabel(7, "A")
    source = _fills()
    source["symbol"] = [asset] * 3
    result = pf.round_trips.extract_round_trips(source)
    assert result["symbol"].to_list() == [asset]
    assert result["pnl"].to_list() == [200.0]


def test_fractional_and_negative_price_reference_defects_do_not_remove_real_cash_pnl():
    fractional = pf.round_trips.extract_round_trips(_fills((0.1, 0.2, -0.3)))
    assert fractional["pnl"].to_list() == [4.0]
    source = _fills((1.0, 0.0, -1.0), (-100.0, -100.0, -90.0))
    assert pf.round_trips.extract_round_trips(source)["pnl"].to_list() == [10.0]


def test_initial_sale_is_the_explicit_legacy_flat_book_short_profile():
    result = pf.round_trips.extract_round_trips(_fills((-1.0, 0.0, 1.0), (100.0, 100.0, 90.0)))
    assert result["pnl"].to_list() == [10.0]
    assert result["long"].to_list() == [False]
    assert result["rt_returns"].to_list() == [0.1]


def test_empty_or_only_open_execution_book_returns_typed_no_completed_observations():
    source = _fills().iloc[:0]
    empty = pf.round_trips.extract_round_trips(source)
    assert empty.empty
    assert pd.api.types.is_float_dtype(empty["pnl"].dtype)
    assert pd.api.types.is_timedelta64_dtype(empty["duration"].dtype)
    assert pf.round_trips._groupby_consecutive(source).empty
    assert pf.round_trips.extract_round_trips(_fills().iloc[:1]).empty


def test_missing_nav_does_not_forward_fill_or_substitute_a_lot_return():
    nav = pd.Series([1000.0], index=pd.DatetimeIndex(["2025-07-07T00:00Z"]))
    result = pf.round_trips.extract_round_trips(_fills(), portfolio_value=nav)
    assert result["returns"].isna().all()
    assert result["rt_returns"].to_list() == [200 / 1240]


def test_custom_consecutive_gap_and_all_original_ns_keys_survive_dataframe_projection():
    source = _fills()
    source.index = pd.to_datetime(
        [
            "2025-07-07T12:00:00.000000001Z",
            "2025-07-07T13:00:00.000000002Z",
            "2025-07-08T12:00:00.000000003Z",
        ]
    )
    separated = pf.round_trips._groupby_consecutive(source, max_delta=pd.Timedelta("1h"))
    assert separated["amount"].to_list() == [10.0, 5.0, -12.0]
    grouped = pf.round_trips.extract_round_trips(source)
    assert grouped["open_dt"].iloc[0].value == source.index[0].value
    assert grouped["close_dt"].iloc[0].value == source.index[-1].value
    assert grouped["duration"].iloc[0].value == source.index[-1].value - source.index[0].value


def test_invalid_fill_or_duplicate_nav_rejects_without_modifying_source_buffers():
    source = _fills()
    source.iloc[0, source.columns.get_loc("price")] = float("nan")
    before = source.copy(deep=True)
    with pytest.raises(ValueError, match="price"):
        pf.round_trips.extract_round_trips(source)
    pd.testing.assert_frame_equal(source, before)
    nav = pd.Series([1000.0, 2000.0], index=pd.DatetimeIndex(["2025-07-08T00:00Z"] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        pf.round_trips.extract_round_trips(_fills(), portfolio_value=nav)


def test_real_published_pyfolio_frames_match_labels_keys_durations_membership_and_financial_sample():
    reference = pytest.importorskip("pyfolio.round_trips")
    source = _fills()
    nav = pd.Series([1000.0], index=pd.DatetimeIndex(["2025-07-08T00:00Z"]))
    expected = reference.extract_round_trips(source, portfolio_value=nav)
    actual = pf.round_trips.extract_round_trips(source, portfolio_value=nav)
    pd.testing.assert_frame_equal(
        actual.drop(columns=["pnl", "rt_returns", "returns"]),
        expected.drop(columns=["pnl", "rt_returns", "returns"]),
    )
    np.testing.assert_allclose(
        actual[["pnl", "rt_returns", "returns"]],
        expected[["pnl", "rt_returns", "returns"]],
        rtol=2e-15,
        atol=1e-12,
    )
    assert actual["pnl"].to_list() == [200.0]
    pd.testing.assert_frame_equal(
        pf.round_trips._groupby_consecutive(source), reference._groupby_consecutive(source)
    )


def test_classic_ieee_nav_and_zero_entry_notional_ratios_are_distinct_from_native_positive_nav():
    zero = pd.Series([0.0], index=pd.DatetimeIndex(["2025-07-08T00:00Z"]))
    result = pf.round_trips.extract_round_trips(_fills(), portfolio_value=zero)
    assert np.isposinf(result["returns"].iloc[0])
    negative = -pd.Series([1000.0], index=zero.index)
    assert pf.round_trips.extract_round_trips(_fills(), portfolio_value=negative)[
        "returns"
    ].to_list() == [-0.2]
    assert np.isposinf(
        pf.round_trips.extract_round_trips(_fills((1.0, 0.0, -1.0), (0.0, 0.0, 10.0)))[
            "rt_returns"
        ].iloc[0]
    )


def test_group_helper_retains_declared_symbol_categorical_dtype_without_extra_empty_assets():
    source = _fills()
    source["symbol"] = pd.Categorical(source["symbol"], categories=["unused", "A"], ordered=True)
    groups = pf.round_trips._groupby_consecutive(source)
    assert groups["symbol"].dtype == source["symbol"].dtype
    assert groups["symbol"].to_list() == ["A", "A"]


def test_real_reference_ieee_zero_and_negative_nav_outputs_are_not_silently_changed():
    reference = pytest.importorskip("pyfolio.round_trips")
    for value in (0.0, -1000.0):
        nav = pd.Series([value], index=pd.DatetimeIndex(["2025-07-08T00:00Z"]))
        actual = pf.round_trips.extract_round_trips(_fills(), portfolio_value=nav)
        expected = reference.extract_round_trips(_fills(), portfolio_value=nav)
        np.testing.assert_allclose(actual["returns"], expected["returns"], rtol=2e-15, atol=1e-12)
    zero_basis = _fills((1.0, 0.0, -1.0), (0.0, 0.0, 10.0))
    actual = pf.round_trips.extract_round_trips(zero_basis)
    expected = reference.extract_round_trips(zero_basis)
    assert np.isposinf(actual["rt_returns"].iloc[0])
    assert np.isposinf(expected["rt_returns"].iloc[0])
