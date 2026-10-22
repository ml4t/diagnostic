"""Direct event adapters must retain actual clocks, universes and date weights."""

from importlib.metadata import version
from itertools import product

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.alphalens import performance


def _inputs():
    dates = pd.date_range("2025-02-03", periods=4, tz="UTC")
    values = pd.DataFrame(
        {
            "A": [1.0, 2.0, 4.0, 8.0],
            "B": [10.0, 20.0, 30.0, 40.0],
            "C": [100.0, 200.0, 300.0, 400.0],
        },
        index=dates,
    )
    index = pd.MultiIndex.from_tuples(
        [(dates[0], "A"), (dates[0], "C"), (dates[2], "A"), (dates[2], "B"), (dates[2], "C")],
        names=["date", "asset"],
    )
    events = pd.DataFrame(
        {"factor_quantile": [1, 2, 1, 1, 2], "group": ["left", "right", "left", "right", "right"]},
        index=index,
    )
    return events, values


def test_common_start_uses_declared_peer_assets_and_keeps_duplicate_event_columns():
    events, values = _inputs()
    selected = events.loc[[(values.index[2], "A"), (values.index[2], "A")]]
    peers = events.loc[[(values.index[2], "C")]]
    original, original_values = selected.copy(deep=True), values.copy(deep=True)
    result = performance.common_start_returns(
        selected, values, 1, 1, cumulative=True, demean_by=peers
    )
    expected = pd.DataFrame(
        [[-198.0, -198.0], [-296.0, -296.0], [-392.0, -392.0]], index=[-1, 0, 1], columns=["A", "A"]
    )
    expected.columns.name = "asset"
    pd.testing.assert_frame_equal(result, expected)
    mean = performance.common_start_returns(
        selected, values, 1, 1, cumulative=True, mean_by_date=True, demean_by=peers
    )
    pd.testing.assert_frame_equal(mean, expected.iloc[:, :1].set_axis(pd.Index([0]), axis=1))
    pd.testing.assert_frame_equal(selected, original)
    pd.testing.assert_frame_equal(values, original_values)


def test_average_event_paths_weight_dates_not_assets_and_do_not_rebase_supplied_levels():
    events, values = _inputs()
    result = performance.average_cumulative_return_by_quantile(
        events, values, periods_before=1, periods_after=1, demeaned=False
    )
    np.testing.assert_allclose(result.loc[(1, "mean")], [11.0, 9.0, 13.0])
    np.testing.assert_allclose(
        result.loc[(1, "std")], [np.nan, np.sqrt(128.0), np.sqrt(242.0)], equal_nan=True
    )
    np.testing.assert_allclose(result.loc[(2, "mean")], [200.0, 200.0, 300.0])
    assert result.columns.tolist() == [-1, 0, 1]
    assert result.index.names == ["factor_quantile", None]


@pytest.mark.parametrize("cumulative,mean_by_date,demean", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("sample", ["complete", "clipped_missing", "opaque_duplicates"])
def test_actual_reference_common_paths_preserve_membership_missingness_and_clipped_clock(
    cumulative, mean_by_date, demean, sample
):
    reference = pytest.importorskip("alphalens.performance")
    profile = version("alphalens-reloaded")
    events, values = _inputs()
    if not cumulative:
        values = values / 1000
    if sample == "clipped_missing":
        values.iloc[1, 0] = np.nan
        extra = events.iloc[:1].copy()
        extra.index = pd.MultiIndex.from_tuples(
            [(values.index[-1] + pd.Timedelta("1D"), "A")], names=events.index.names
        )
        events = pd.concat([events, extra])
    if sample == "opaque_duplicates":
        # Integer identifiers remain integers; duplicate events are not lost when demeaned.
        events = events.rename(index={"A": 101, "B": 203, "C": 907}, level="asset")
        values = values.rename(columns={"A": 101, "B": 203, "C": 907})
        events = pd.concat([events, events.iloc[[2]]]).sort_index()
    selected = events[events.factor_quantile == 1]
    options = {
        "cumulative": cumulative,
        "mean_by_date": mean_by_date,
        "demean_by": events if demean else None,
    }
    expected = reference.common_start_returns(selected, values, 5, 6, **options)
    actual = performance.common_start_returns(
        selected, values, 5, 6, reference_version=profile, **options
    )
    # The reference selects individual asset columns through a Python set, so its
    # column order is unspecified. Compare every labelled path, including repeats.
    pd.testing.assert_frame_equal(
        actual.sort_index(axis=1),
        expected.sort_index(axis=1),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("demeaned,group_adjust,by_group", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("sample", ["complete", "missing", "clipped", "duplicate"])
def test_actual_reference_average_paths_keep_quantile_group_date_sample_order(
    demeaned, group_adjust, by_group, sample
):
    reference = pytest.importorskip("alphalens.performance")
    profile = version("alphalens-reloaded")
    events, values = _inputs()
    if sample == "missing":
        values.iloc[1, 0] = np.nan
    if sample == "duplicate":
        events = pd.concat([events, events.iloc[[2]]])
    options = {
        "periods_before": 5 if sample == "clipped" else 1,
        "periods_after": 6 if sample == "clipped" else 1,
        "demeaned": demeaned,
        "group_adjust": group_adjust,
        "by_group": by_group,
    }
    expected = reference.average_cumulative_return_by_quantile(events, values, **options)
    actual = performance.average_cumulative_return_by_quantile(
        events, values, reference_version=profile, **options
    )
    pd.testing.assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_event_adapter_rejects_missing_clock_and_invalid_group_without_mutation():
    events, values = _inputs()
    unknown = events.copy()
    unknown.index = pd.MultiIndex.from_tuples(
        [(date + pd.Timedelta("20D"), asset) for date, asset in unknown.index],
        names=events.index.names,
    )
    with pytest.raises(ValueError, match="event.*clock"):
        performance.common_start_returns(unknown, values, 1, 1, cumulative=True)
    with pytest.raises(ValueError, match="group"):
        performance.average_cumulative_return_by_quantile(
            events.drop(columns="group"), values, group_adjust=True
        )
