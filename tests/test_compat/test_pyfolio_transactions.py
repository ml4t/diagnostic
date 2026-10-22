"""Legacy cash/share workflows retain labels and explicit financial definitions."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import txn


def _inputs():
    dates = pd.date_range("2025-07-07", periods=3, tz="America/New_York")
    positions = pd.DataFrame(
        {"A": [200.0, 400.0, 600.0], "cash": [800.0, 600.0, 400.0]}, index=dates
    )
    transactions = pd.DataFrame(
        {"amount": [1.0, -1.0, 2.0], "price": [100.0, 100.0, 100.0], "symbol": ["A"] * 3},
        index=pd.to_datetime(
            ["2025-07-07T15:00", "2025-07-07T16:00", "2025-07-09T15:00"]
        ).tz_localize("America/New_York"),
    )
    return pd.Series([0.01, -0.02, 0.03], index=dates), positions, transactions


def test_daily_volume_preserves_source_zone_and_absolute_units_without_mutation():
    _, _, transactions = _inputs()
    before = deepcopy(transactions)
    result = txn.get_txn_vol(transactions)
    expected = pd.DataFrame(
        {"txn_volume": [200.0, 200.0], "txn_shares": [2.0, 2.0]},
        index=transactions.index[[0, 2]].normalize(),
    )
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(transactions, before)


@pytest.mark.parametrize(
    "denominator, expected", [("AGB", [2.0, 0.0, 0.4]), ("portfolio_value", [0.2, 0.0, 0.2])]
)
def test_turnover_uses_explicit_legacy_zero_seed_or_closing_nav(denominator, expected):
    _, positions, transactions = _inputs()
    np.testing.assert_allclose(txn.get_turnover(positions, transactions, denominator), expected)


def test_extra_bps_uses_closing_position_value_without_subtracting_observed_fees_twice():
    returns, positions, transactions = _inputs()
    transactions["commission"] = [10.0, 10.0, 10.0]
    before = returns.copy(deep=True)
    result = txn.adjust_returns_for_slippage(returns, positions, transactions, 10)
    np.testing.assert_allclose(result, [0.0098, -0.02, 0.0298])
    pd.testing.assert_series_equal(returns, before)


def test_zero_bps_preserves_values_and_reference_output_frequency_without_mutating_input():
    returns, positions, transactions = _inputs()
    before = returns.copy(deep=True)
    expected = returns.copy(deep=True)
    expected.index.freq = None
    pd.testing.assert_series_equal(
        txn.adjust_returns_for_slippage(returns, positions, transactions, 0), expected
    )
    pd.testing.assert_series_equal(returns, before)


def test_transaction_mapping_preserves_missing_commission_and_original_nested_record():
    source = {
        "sid": {"sid": 7, "symbol": "A"},
        "amount": -2.0,
        "price": 100.0,
        "commission": None,
        "order_id": "o1",
        "dt": pd.Timestamp("2025-07-07T15:00Z"),
    }
    before = deepcopy(source)
    mapped = txn.map_transaction(pd.Series(source))
    assert mapped == {
        **{name: value for name, value in source.items() if name != "sid"},
        "sid": 7,
        "symbol": "A",
    }
    assert source == before


def test_nested_transactions_keep_nanoseconds_timezone_and_same_instant_execution_order():
    instant = pd.Timestamp("2025-07-07T15:00:00.000000001", tz="America/New_York")
    records = [
        {
            "sid": {"sid": 7, "symbol": "A"},
            "amount": amount,
            "price": price,
            "commission": None,
            "order_id": identifier,
            "dt": instant,
        }
        for amount, price, identifier in [(3.0, 100.0, "buy"), (-1.0, 120.0, "partial")]
    ]
    source = pd.Series(
        [[], records], index=pd.date_range("2025-07-06", periods=2, tz="America/New_York")
    )
    before = deepcopy(source)
    result = txn.make_transaction_frame(source)
    assert result["order_id"].to_list() == ["buy", "partial"]
    assert result["txn_dollars"].to_list() == [-300.0, 120.0]
    assert str(result.index.tz) == str(instant.tz)
    assert result.index[0].utcoffset() == instant.utcoffset()
    assert result.index.as_unit("ns").asi8.tolist() == [instant.value] * 2
    pd.testing.assert_series_equal(source, before)


def test_empty_nested_ledger_returns_typed_empty_records_not_a_fabricated_execution():
    result = txn.make_transaction_frame(
        pd.Series([[]], index=pd.date_range("2025-07-07", periods=1, tz="UTC"))
    )
    assert result.empty and "txn_dollars" in result
    assert str(result.index.tz) == "UTC"


def test_volume_missing_price_does_not_remove_known_units_from_a_different_metric():
    dates = pd.to_datetime(["2025-07-07T15:00Z", "2025-07-07T16:00Z", "2025-07-08T15:00Z"])
    result = txn.get_txn_vol(
        pd.DataFrame({"amount": [1.0, np.nan, 2.0], "price": [np.nan, 100.0, 5.0]}, index=dates)
    )
    assert result["txn_shares"].to_list() == [1.0, 2.0]
    assert result["txn_volume"].to_list() == [0.0, 10.0]


def test_absolute_integer_units_do_not_overflow_for_the_valid_signed64_bit_minimum():
    frame = pd.DataFrame(
        {"amount": [-(2**63), 1], "price": [1, 1]},
        index=pd.date_range("2025-07-07T15:00Z", periods=2, freq="1h"),
    )
    result = txn.get_txn_vol(frame)
    assert int(result["txn_shares"].iloc[0]) == 2**63 + 1
    assert int(result["txn_volume"].iloc[0]) == 2**63 + 1


def test_legacy_ieee_division_is_explicit_and_does_not_relabel_insolvent_nav_as_positive():
    dates = pd.date_range("2025-07-07", periods=3, tz="UTC")
    positions = pd.DataFrame({"A": [0.0, -100.0, 0.0], "cash": [0.0, 0.0, 0.0]}, index=dates)
    fills = pd.DataFrame({"amount": [1.0, 1.0, 1.0], "price": [100.0, 100.0, 100.0]}, index=dates)
    result = txn.get_turnover(positions, fills, "portfolio_value")
    assert result.iloc[0] == np.inf and result.iloc[1] == -1 and result.iloc[2] == np.inf


def test_supported_profiles_match_actual_published_pyfolio_on_the_same_inputs():
    from importlib.metadata import version

    reference = pytest.importorskip("pyfolio.txn")
    if version("pyfolio-reloaded") != "0.10.0":
        pytest.skip("corrected reference profile is pinned to0.10.0")
    returns, positions, transactions = _inputs()
    pd.testing.assert_frame_equal(
        txn.get_txn_vol(transactions), reference.get_txn_vol(transactions)
    )
    for denominator in ("AGB", "portfolio_value"):
        pd.testing.assert_series_equal(
            txn.get_turnover(positions, transactions, denominator),
            reference.get_turnover(positions, transactions, denominator),
        )
    for bps in (0, 10, 20):
        pd.testing.assert_series_equal(
            txn.adjust_returns_for_slippage(returns, positions, transactions, bps),
            reference.adjust_returns_for_slippage(returns, positions, transactions, bps),
        )


@pytest.mark.parametrize("missing_clock", [False, True])
def test_reference_missingness_and_signed_prices_keep_metric_specific_samples(missing_clock):
    reference = pytest.importorskip("pyfolio.txn")
    returns, positions, transactions = _inputs()
    positions.loc[positions.index[1], "A"] = np.nan
    transactions.loc[transactions.index[0], "price"] = -100.0
    transactions.loc[transactions.index[1], "amount"] = np.nan
    returns.iloc[1] = np.nan
    if missing_clock:
        returns = returns.iloc[1:]
        positions = positions.iloc[:-1]
    pd.testing.assert_frame_equal(
        txn.get_txn_vol(transactions), reference.get_txn_vol(transactions)
    )
    for denominator in ("AGB", "portfolio_value"):
        pd.testing.assert_series_equal(
            txn.get_turnover(positions, transactions, denominator),
            reference.get_turnover(positions, transactions, denominator),
        )
    for bps in (0, 10, -10):
        pd.testing.assert_series_equal(
            txn.adjust_returns_for_slippage(returns, positions, transactions, bps),
            reference.adjust_returns_for_slippage(returns, positions, transactions, bps),
        )
