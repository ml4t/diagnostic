"""Familiar account projections must retain each definition and original identity."""

import importlib
import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest


def _adapter():
    return importlib.import_module("ml4t.diagnostic.compat.pyfolio.pos")


def _positions():
    return pd.DataFrame(
        {
            203: [100.0, 0, -20, np.nan, 10],
            101: [-40.0, 0, 10, 5, np.nan],
            "cash": [140.0, 0, -40, np.nan, 90],
        },
        index=pd.date_range("2025-02-03", periods=5, tz="UTC"),
    ).rename_axis(columns="sid")


def test_public_positions_use_signed_nav_and_per_metric_missingness_without_mutation():
    pos = _adapter()
    source = _positions()
    before = source.copy(deep=True)
    allocations = pos.get_percent_alloc(source)
    np.testing.assert_allclose(allocations.iloc[0], [0.5, -0.2, 0.7])
    np.testing.assert_allclose(allocations.iloc[2], [0.4, -0.2, 0.8])
    assert allocations.iloc[1].isna().all()
    assert allocations.iloc[3, 1] == 1
    exposure = pos.get_long_short_pos(source)
    np.testing.assert_allclose(exposure.iloc[0], [0.5, -0.2, 0.3])
    np.testing.assert_allclose(exposure.iloc[2], [-0.2, 0.4, 0.2])
    assert exposure.iloc[3].isna().all(), "Missing cash cannot be replaced by zero NAV cash"
    summary = pos.get_max_median_position_concentration(source)
    np.testing.assert_allclose(summary.iloc[0], [0.5, 0.5, -0.2, -0.2])
    np.testing.assert_allclose(summary.iloc[2], [0.4, 0.4, -0.2, -0.2])
    pd.testing.assert_frame_equal(source, before)


def test_top_positions_preserve_original_asset_identity_input_ties_and_all_missing_asset():
    positions = pd.DataFrame({203: [100.0, -20], 101: [100.0, -20], 7: np.nan, "cash": 10})
    actual = _adapter().get_top_long_short_abs(positions, top=3)
    assert actual[0].index.tolist() == [203, 101]
    assert actual[1].index.tolist() == [203, 101]
    assert actual[2].index.tolist() == [203, 101, 7]
    assert np.isnan(actual[2].iloc[-1])
    assert all(value.empty for value in _adapter().get_top_long_short_abs(positions, top=0))


def test_sector_projection_warns_for_unknown_without_reallocating_its_marked_value():
    source = _positions()
    with pytest.warns(UserWarning, match="101"):
        actual = _adapter().get_sector_exposures(source, {203: "Technology"})
    np.testing.assert_allclose(actual["Technology"], [100, 0, -20, 0, 10])
    pd.testing.assert_series_equal(actual["cash"], source["cash"])
    assert actual.columns.tolist() == ["Technology", "cash"]


def test_reserved_cash_sector_is_rejected_instead_of_silently_overwriting_asset_exposure():
    source = pd.DataFrame({"A": [60.0], "cash": [40.0]})
    with pytest.raises(ValueError, match="cash.*sector|sector.*cash"):
        _adapter().get_sector_exposures(source, {"A": "cash"})


def test_extract_positions_uses_actual_marks_and_multipliers_without_return_reconstruction():
    from dataclasses import dataclass

    @dataclass(frozen=True, order=True)
    class Asset:
        sid: int
        price_multiplier: float

    dates = pd.date_range("2025-02-03", periods=2, tz="UTC")
    future = Asset(5, 50.0)
    positions = pd.DataFrame(
        {"sid": [future], "amount": [0.5], "last_sale_price": [20.0]},
        index=dates[:1].rename(None),
    )
    cash = pd.Series([100.0, 200.0], index=dates, name="account cash")
    before, cash_before = positions.copy(deep=True), cash.copy(deep=True)
    actual = _adapter().extract_pos(positions, cash)
    assert actual.index.equals(dates)
    np.testing.assert_allclose(actual[future], [500.0, 0])
    np.testing.assert_allclose(actual["cash"], [100.0, 200.0])
    assert actual.columns.name == "sid"
    pd.testing.assert_frame_equal(positions, before)
    pd.testing.assert_series_equal(cash, cash_before)


@pytest.mark.parametrize(
    "method",
    [
        "get_percent_alloc",
        "get_long_short_pos",
        "get_top_long_short_abs",
        "get_max_median_position_concentration",
    ],
)
@pytest.mark.parametrize("sample", ["mixed", "all_cash", "empty", "all_missing"])
def test_actual_reference_position_projections_match_values_labels_and_missingness(method, sample):
    reference = pytest.importorskip("pyfolio.pos")
    assert version("pyfolio-reloaded") == "0.10.0"
    source = _positions()
    if sample == "all_cash":
        source = source[["cash"]]
    elif sample == "empty":
        source = source.iloc[:0]
    elif sample == "all_missing":
        source = source * np.nan
    expected = getattr(reference, method)(source)
    actual = getattr(_adapter(), method)(source)
    if isinstance(expected, tuple):
        for left, right in zip(actual, expected, strict=True):
            pd.testing.assert_series_equal(left, right, check_exact=False, rtol=1e-12)
    else:
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12)


def test_actual_reference_sector_difference_has_independent_cash_conservation_proof():
    reference = pytest.importorskip("pyfolio.pos")
    source = pd.DataFrame({"A": [60.0], "cash": [40.0]})
    expected = reference.get_sector_exposures(source, {"A": "cash"})
    assert expected.iloc[0].sum() == 40.0 != 100.0
    with pytest.raises(ValueError, match="cash.*sector|sector.*cash"):
        _adapter().get_sector_exposures(source, {"A": "cash"})


@pytest.mark.parametrize("mapping_kind", ["dict", "series", "named_series"])
def test_actual_reference_sector_projection_keeps_grouping_and_original_missing_cash(mapping_kind):
    reference = pytest.importorskip("pyfolio.pos")
    source = _positions()
    mapping = {203: "Technology", 101: "Technology"}
    if mapping_kind != "dict":
        mapping = pd.Series(mapping, name="sector" if mapping_kind == "named_series" else None)
    pd.testing.assert_frame_equal(
        _adapter().get_sector_exposures(source, mapping),
        reference.get_sector_exposures(source, mapping),
    )


def test_actual_reference_extract_loses_an_all_cash_date_but_adapter_retains_supplied_account_clock():
    reference = pytest.importorskip("pyfolio.pos")
    dates = pd.date_range("2025-02-03", periods=2, tz="UTC")
    source = pd.DataFrame(
        {"sid": [101], "amount": [2.0], "last_sale_price": [10.0]}, index=dates[:1]
    )
    cash = pd.Series([80.0, 100.0], index=dates)
    expected = reference.extract_pos(source, cash.copy(deep=True))
    actual = _adapter().extract_pos(source, cash)
    assert expected.index.tolist() == dates[:1].tolist()
    assert actual.index.tolist() == dates.tolist()
    np.testing.assert_allclose(actual[101], [20, 0])
    np.testing.assert_allclose(actual["cash"], [80, 100])
    np.testing.assert_allclose(actual.sum(axis="columns"), [100, 100])


def test_actual_reference_repeated_position_records_retain_the_explicit_pivot_mean_convention():
    reference = pytest.importorskip("pyfolio.pos")
    dates = pd.date_range("2025-02-03", periods=2, tz="UTC").rename("index")
    source = pd.DataFrame(
        {"sid": [101, 101, 203], "amount": [2.0, 4, -1], "last_sale_price": [10.0, 10, 20]},
        index=dates.take([0, 0, 1]),
    )
    cash = pd.Series([100.0, 100.0], index=dates)
    expected = reference.extract_pos(source, cash.copy(deep=True))
    actual = _adapter().extract_pos(source, cash)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.loc[dates[0], 101] == 30


def test_empty_complete_position_records_preserve_supplied_all_cash_observations():
    dates = pd.date_range("2025-02-03", periods=2, tz="UTC")
    source = pd.DataFrame(
        {
            "sid": pd.Series([], dtype=int),
            "amount": pd.Series([], dtype=float),
            "last_sale_price": pd.Series([], dtype=float),
        },
        index=dates[:0],
    )
    cash = pd.Series([100.0, 110.0], index=dates)
    actual = _adapter().extract_pos(source, cash)
    assert actual.columns.tolist() == ["cash"]
    expected = pd.Series(
        [100.0, 110.0], index=pd.DatetimeIndex(["2025-02-03", "2025-02-04"], tz="UTC"), name="cash"
    )
    pd.testing.assert_series_equal(actual["cash"], expected)
    assert cash.index.freq == pd.offsets.Day()


def test_extract_unknown_cash_is_not_reported_as_an_observed_zero_balance():
    dates = pd.date_range("2025-02-03", periods=1, tz="UTC")
    source = pd.DataFrame({"sid": [101], "amount": [2.0], "last_sale_price": [10.0]}, index=dates)
    cash = pd.Series([np.nan], index=dates)
    actual = _adapter().extract_pos(source, cash)
    assert actual[101].iloc[0] == 20.0
    assert np.isnan(actual["cash"].iloc[0])


def test_actual_reference_extract_zero_fills_unknown_cash_but_adapter_retains_unknown():
    reference = pytest.importorskip("pyfolio.pos")
    dates = pd.date_range("2025-02-03", periods=1, tz="UTC")
    source = pd.DataFrame({"sid": [101], "amount": [2.0], "last_sale_price": [10.0]}, index=dates)
    cash = pd.Series([np.nan], index=dates)
    assert reference.extract_pos(source, cash.copy(deep=True))["cash"].iloc[0] == 0.0
    assert np.isnan(_adapter().extract_pos(source, cash)["cash"].iloc[0])


@pytest.mark.parametrize("top, error", [(True, TypeError), (1.5, TypeError), (-1, ValueError)])
def test_top_positions_reject_an_invalid_count_instead_of_changing_rank_membership(top, error):
    with pytest.raises(error, match="nonnegative integer"):
        _adapter().get_top_long_short_abs(_positions(), top=top)


def test_mandatory_position_reference_is_real_and_executes_an_actual_public_projection():
    required = os.environ.get("ML4T_REQUIRED_PYFOLIO_REFERENCE")
    if required is None:
        pytest.skip("Reference lane requires the installed oracle explicitly")
    import pyfolio.pos as reference

    assert required == version("pyfolio-reloaded") == "0.10.0"
    assert reference.get_percent_alloc(_positions()).iloc[0, 0] == 0.5
