"""Legacy portfolio journeys compare actual holdings and daily cash, not correlation."""

import itertools
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.alphalens import performance


def _factor_data():
    dates = pd.DatetimeIndex(["2025-07-04T10:00Z", "2025-07-04T14:00Z", "2025-07-07T10:00Z"])
    data = pd.DataFrame(
        {
            "factor": [-3.0, -1.0, 1.0, 3.0] * 3,
            "factor_quantile": [1, 1, 2, 2] * 3,
            "group": ["tech", "energy", "tech", "energy"] * 3,
            "1D": [0.01, 0.02, -0.01, 0.04, -0.02, 0.02, 0.01, 0.03, 0.03, 0.01, -0.01, 0.02],
            "2D": [0.02, 0.04, 0.01, 0.05, 0.01, -0.01, 0.03, 0.02, 0.02, 0.03, -0.02, 0.04],
        },
        index=pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["date", "asset"]),
    )
    return data


def _reference():
    ref = pytest.importorskip("alphalens.performance")
    if version("alphalens-reloaded") != "0.5.0":
        pytest.skip("corrected Alphalens0.5.0 profile, not the frozen historical implementation")
    return ref


def test_public_legacy_portfolio_calls_keep_fraction_units_and_zero_first_return():
    data = _factor_data()
    before = data.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        returns, positions, benchmark = performance.create_pyfolio_input(data, "2D")
    # The final outcome is -.375*.02-.125*.03+.125*(-.02)+.375*.04.
    np.testing.assert_allclose(returns, [0.0, 0.0, 0.0, 0.00125])
    assert returns.name is None
    np.testing.assert_allclose(positions.iloc[0], [-0.375, -0.125, 0.125, 0.375, 1.0])
    np.testing.assert_allclose(positions.iloc[1], [0.0, 0.0, 0.0, 0.0, 1.0])
    assert benchmark is not None and benchmark.name == "benchmark"
    np.testing.assert_allclose(benchmark, [0.0, 0.0, 0.0, 0.0125])
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
@pytest.mark.parametrize("selection", ["all", "buckets", "group"])
def test_corrected_reference_weights_returns_paths_and_event_positions(flags, selection):
    ref = _reference()
    data = _factor_data()
    long_short, group_neutral, equal_weight = flags
    options = {
        "long_short": long_short,
        "group_neutral": group_neutral,
        "equal_weight": equal_weight,
    }
    if selection == "buckets":
        options["quantiles"] = [2]
    if selection == "group":
        options["groups"] = ["tech"]
    pd.testing.assert_series_equal(
        performance.factor_weights(data, *flags).sort_index(),
        ref.factor_weights(data, *flags).sort_index(),
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        performance.factor_returns(data, *flags), ref.factor_returns(data, *flags), check_freq=False
    )
    pd.testing.assert_frame_equal(
        performance.factor_returns(data, *flags, by_asset=True).sort_index(),
        ref.factor_returns(data, *flags, by_asset=True).sort_index(),
        check_freq=False,
    )
    pd.testing.assert_series_equal(
        performance.factor_cumulative_returns(data, "2D", **options),
        ref.factor_cumulative_returns(data, "2D", **options),
        check_freq=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = performance.factor_positions(data, "2D", **options)
        expected = ref.factor_positions(data, "2D", **options)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


@pytest.mark.parametrize("capital", [None, 1.0, 1000.0])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"long_short": False, "equal_weight": True},
        {"group_neutral": True, "quantiles": [2]},
        {"groups": ["energy"], "benchmark_period": "absent"},
    ],
)
def test_corrected_reference_full_daily_interchange_options(capital, options):
    ref = _reference()
    data = _factor_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = performance.create_pyfolio_input(data, "2D", capital=capital, **options)
        expected = ref.create_pyfolio_input(data, "2D", capital=capital, **options)
    pd.testing.assert_series_equal(actual[0], expected[0], check_freq=False)
    pd.testing.assert_frame_equal(actual[1], expected[1], check_freq=False)
    if expected[2] is None:
        assert actual[2] is None
    else:
        pd.testing.assert_series_equal(actual[2], expected[2], check_freq=False)


def test_common_expiry_reconciliation_uses_every_allocation_not_the_frozen_single_pop_bug():
    weights = pd.Series(
        [0.25, 0.75, 0.75, 0.25, 0.0, 1.0],
        index=pd.MultiIndex.from_product(
            [pd.to_datetime(["2025-07-05", "2025-07-06", "2025-07-07"]), ["A", "B"]],
            names=["date", "asset"],
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = performance.positions(weights, "1D", freq=pd.offsets.BusinessDay())
        frozen_profile = performance.positions(
            weights, "1D", freq=pd.offsets.BusinessDay(), reference_version="0.4.6"
        )
    np.testing.assert_allclose(
        actual.to_numpy(), [[0.25, 0.75], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0]]
    )
    pd.testing.assert_frame_equal(actual, frozen_profile)


def test_calendar_none_disables_legacy_business_day_expiry_and_preserves_opaque_ids():
    ids = pd.Index([1, "1", ("fund", 2)], tupleize_cols=False)
    clock = pd.DatetimeIndex(["2025-07-04T10:00Z"])
    weights = pd.Series(
        [1.0, 2.0, 1.0], index=pd.MultiIndex.from_product([clock, ids], names=["date", "asset"])
    )
    result = performance.positions(weights, "1D", calendar=None)
    assert result.index[-1] == pd.Timestamp("2025-07-05T10:00Z")
    assert set(result.columns) == set(ids)
    np.testing.assert_allclose(result.loc[clock[0]].reindex(ids), [0.25, 0.5, 0.25])
    with pytest.raises(ValueError, match="disables calendar offsets"):
        performance.positions(weights, "1D", freq=pd.offsets.BusinessDay(), calendar=None)


def test_benchmark_uses_original_universe_but_zero_factor_names_are_not_equal_weighted():
    clock = pd.date_range("2025-07-07", periods=2, tz="UTC")
    data = pd.DataFrame(
        {
            "factor": [1.0, 3.0, 0.0] * 2,
            "factor_quantile": [1, 2, 1] * 2,
            "1D": [0.1, 0.3, 99.0, 0.2, 0.4, 99.0],
        },
        index=pd.MultiIndex.from_product([clock, ["A", "B", "C"]], names=["date", "asset"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        returns, _, benchmark = performance.create_pyfolio_input(
            data, "1D", long_short=False, quantiles=[2]
        )
    np.testing.assert_allclose(returns, [0.0, 0.4])
    np.testing.assert_allclose(benchmark, [0.0, 0.3])


def test_frozen_reference_duplicate_expiry_events_are_reconciled_against_active_sets():
    ref = pytest.importorskip("alphalens.performance")
    if version("alphalens-reloaded") != "0.4.6":
        pytest.skip("the frozen0.4.6single-pop defect is a separate reference profile")
    weights = pd.Series(
        [0.25, 0.75, 0.75, 0.25, 0.0, 1.0],
        index=pd.MultiIndex.from_product(
            [pd.to_datetime(["2025-07-05", "2025-07-06", "2025-07-07"]), ["A", "B"]],
            names=["date", "asset"],
        ),
    )
    old = ref.positions(weights, "1D", freq=pd.offsets.BusinessDay())
    # Frozen0.4.6retains duplicate Monday keys from the coincident expiries,
    # processes entries repeatedly and also removes only one due allocation.
    assert not old.index.is_unique
    assert old.index.tolist().count(pd.Timestamp("2025-07-07")) > 1
    fixed = performance.positions(
        weights, "1D", freq=pd.offsets.BusinessDay(), reference_version="0.4.6"
    )
    np.testing.assert_allclose(fixed.iloc[2], [0.0, 1.0])
    np.testing.assert_allclose(fixed.iloc[3], [0.0, 0.0])


def test_selection_does_not_truncate_the_independent_full_universe_benchmark_clock():
    data = _factor_data()
    data["group"] = ["excluded"] * 8 + ["selected"] * 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        returns, _, benchmark = performance.create_pyfolio_input(data, "2D", groups=["selected"])
    assert len(returns) == 1
    assert returns.index[0] == pd.Timestamp("2025-07-07", tz="UTC")
    assert benchmark.index.tolist() == list(pd.date_range("2025-07-04", periods=4, tz="UTC"))
    np.testing.assert_allclose(benchmark, [0.0, 0.0, 0.0, 0.0125])
    assert benchmark.index.freq == pd.offsets.Day()


@pytest.mark.parametrize("clock", ["B", "1D", "1h", "1ns"])
def test_regular_reference_clock_and_padding_are_preserved_in_the_full_workflow(clock):
    ref = _reference()
    dates = pd.date_range("2025-07-03T10:00Z", periods=4, freq=clock)
    data = pd.concat([_factor_data().iloc[:4].reset_index(drop=True)] * 4, ignore_index=True)
    data.index = pd.MultiIndex.from_product([dates, list("ABCD")], names=["date", "asset"])
    data = data.iloc[[0, 1, 2, 3, 12, 13, 14, 15]]
    period = {"B": "2D", "1D": "2D", "1h": "2h", "1ns": "2ns"}[clock]
    data = data.rename(columns={"2D": period})
    if clock in ("1h", "1ns"):
        # Legacy inferred3h/3ns offsets are not accepted by its daily-offset
        # helper. Explicit elapsed mode is the working destination route.
        with pytest.raises(ValueError, match="freq must be"):
            ref.create_pyfolio_input(data, period)
        with pytest.raises(ValueError, match="freq must be"):
            performance.create_pyfolio_input(data, period)
        actual = performance.create_pyfolio_input(data, period, calendar=None)
        np.testing.assert_allclose(actual[0], [0.0])
        np.testing.assert_allclose(actual[1].iloc[0], [-0.375, -0.125, 0.125, 0.375, 1.0])
        np.testing.assert_allclose(actual[2], [0.0])
        pd.testing.assert_frame_equal(
            performance.positions(performance.factor_weights(data), period, freq=pd.offsets.Day()),
            ref.positions(ref.factor_weights(data), period, freq=pd.offsets.Day()),
        )
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = performance.create_pyfolio_input(data, period)
        expected = ref.create_pyfolio_input(data, period)
    pd.testing.assert_series_equal(actual[0], expected[0])
    pd.testing.assert_frame_equal(actual[1], expected[1])
    pd.testing.assert_series_equal(actual[2], expected[2])
