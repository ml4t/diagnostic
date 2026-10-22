"""Original transaction plot analytics use explicit legacy samples and native kernels."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from ml4t.diagnostic.compat.pyfolio import plotting


def _inputs():
    dates = pd.date_range("2025-07-07", periods=3, tz="America/New_York")
    returns = pd.Series([0.01, -0.02, 0.03], index=dates)
    positions = pd.DataFrame(
        {"A": [200.0, 400.0, 600.0], "cash": [800.0, 600.0, 400.0]}, index=dates
    )
    transactions = pd.DataFrame(
        {"amount": [1.0, -1.0, 2.0], "price": [100.0, 100.0, 100.0]},
        index=pd.to_datetime(
            ["2025-07-07T15:00Z", "2025-07-07T16:00Z", "2025-07-09T15:00Z"]
        ).tz_convert("America/New_York"),
    )
    return returns, positions, transactions


def test_daily_volume_turnover_and_distribution_preserve_their_distinct_samples():
    returns, positions, transactions = _inputs()
    before = deepcopy(transactions)
    turnover = plotting.plot_turnover(returns, transactions, positions)
    np.testing.assert_allclose(turnover.data[0].y, [2.0, 0.0, 0.4])
    np.testing.assert_allclose(turnover.data[1].y, [0.8])
    assert turnover.layout.meta["daily_mean"] == pytest.approx(0.8)
    volume = plotting.plot_daily_volume(returns, transactions)
    np.testing.assert_allclose(volume.data[0].y, [2.0, 2.0])
    assert volume.layout.meta["active_date_mean"] == 2.0
    histogram = plotting.plot_daily_turnover_hist(
        transactions, positions, turnover_denom="portfolio_value"
    )
    np.testing.assert_allclose(histogram.data[0].x, [0.2, 0.0, 0.2])
    pd.testing.assert_frame_equal(transactions, before)


def test_slippage_wealth_paths_compound_exact_window_returns_not_annual_rates():
    returns, positions, transactions = _inputs()
    figure = plotting.plot_slippage_sweep(returns, positions, transactions, slippage_params=(0, 10))
    np.testing.assert_allclose(
        figure.data[0].y, [1.01, 1.01 * 0.98, 1.01 * 0.98 * 1.03], rtol=1e-14
    )
    np.testing.assert_allclose(
        figure.data[1].y, [1.0098, 1.0098 * 0.98, 1.0098 * 0.98 * 1.0298], rtol=1e-14
    )


def test_sensitivity_uses_the_declared_return_scale_and_all99_legacy_bps_values():
    returns, positions, transactions = _inputs()
    figure = plotting.plot_slippage_sensitivity(
        returns, positions, transactions, periods_per_year=3
    )
    assert list(figure.data[0].x) == list(range(1, 100))
    assert figure.layout.meta["periods_per_year"] == 3
    assert figure.data[0].y[0] == pytest.approx(1.00998 * 0.98 * 1.02998 - 1)
    assert figure.data[0].y[9] == pytest.approx(1.0098 * 0.98 * 1.0298 - 1)
    doubled = plotting.plot_slippage_sensitivity(
        returns, positions, transactions, periods_per_year=6
    )
    assert doubled.data[0].y[9] == pytest.approx((1.0098 * 0.98 * 1.0298) ** 2 - 1)
    with pytest.raises(ValueError, match="periods_per_year"):
        plotting.plot_slippage_sensitivity(returns, positions, transactions, periods_per_year=0)


def test_filtered_timing_uses_only_selected_notional_and_includes_the_closing_minute():
    timestamps = pd.to_datetime(
        [
            "2025-07-07T01:00Z",
            "2025-07-07T13:30Z",
            "2025-07-07T19:59Z",
            "2025-07-07T20:00:59Z",
            "2025-07-07T20:01Z",
        ],
        format="ISO8601",
    )
    transactions = pd.DataFrame(
        {"amount": [1.0, 2.0, -3.0, 4.0, 5.0], "price": [100.0] * 5}, index=timestamps
    )
    before = transactions.copy(deep=True)
    result = plotting.plot_txn_time_hist(transactions)
    values = dict(zip(result.data[0].x, result.data[0].y, strict=True))
    assert values["09:30"] == pytest.approx(200 / 900)
    assert values["15:55"] == pytest.approx(300 / 900)
    assert values["16:00"] == pytest.approx(400 / 900)
    assert result.layout.meta["included_fills"] == 3
    assert result.layout.meta["excluded_fills"] == 2
    assert sum(values.values()) == pytest.approx(1)
    pd.testing.assert_frame_equal(transactions, before)


def test_empty_selected_timing_notional_is_undefined_not_a_zero_distribution():
    frame = pd.DataFrame(
        {"amount": [1.0], "price": [100.0]}, index=pd.DatetimeIndex(["2025-07-07T01:00Z"])
    )
    result = plotting.plot_txn_time_hist(frame)
    assert result.layout.meta["included_fills"] == 0
    assert result.layout.meta["excluded_fills"] == 1
    assert all(value is None for value in result.data[0].y)


def test_matplotlib_axis_reuse_is_rejected_instead_of_returning_a_misleading_empty_axis():
    _, positions, transactions = _inputs()
    with pytest.raises(TypeError, match="Plotly figures"):
        plotting.plot_daily_turnover_hist(transactions, positions, ax=object())


def test_actual_published_pyfolio_plot_payloads_match_under_the_same_options():
    reference = pytest.importorskip("pyfolio.plotting")
    import matplotlib.pyplot as plt

    returns, positions, transactions = _inputs()
    comparisons = [
        (plotting.plot_daily_volume, reference.plot_daily_volume, (returns, transactions), {}),
        (plotting.plot_turnover, reference.plot_turnover, (returns, transactions, positions), {}),
        (
            plotting.plot_slippage_sweep,
            reference.plot_slippage_sweep,
            (returns, positions, transactions),
            {"slippage_params": (0, 10, 20)},
        ),
        (
            plotting.plot_slippage_sensitivity,
            reference.plot_slippage_sensitivity,
            (returns, positions, transactions),
            {},
        ),
    ]
    for actual_function, reference_function, args, kwargs in comparisons:
        figure, axis = plt.subplots()
        try:
            expected = reference_function(*args, ax=axis, **kwargs)
            actual = actual_function(*args, **kwargs)
            expected_traces = (
                3
                if actual_function == plotting.plot_slippage_sweep
                else 2
                if actual_function == plotting.plot_turnover
                else 1
            )
            assert len(actual.data) == expected_traces
            assert len(expected.lines) == expected_traces + (
                actual_function in (plotting.plot_daily_volume, plotting.plot_turnover)
            )
            for trace, line in zip(actual.data, expected.lines[:expected_traces], strict=True):
                np.testing.assert_allclose(trace.y, line.get_ydata(), rtol=1e-12, atol=1e-14)
        finally:
            plt.close(figure)
    for bin_minutes in (5, 15):
        figure, axis = plt.subplots()
        try:
            expected = reference.plot_txn_time_hist(transactions, bin_minutes=bin_minutes, ax=axis)
            actual = plotting.plot_txn_time_hist(transactions, bin_minutes=bin_minutes)
            np.testing.assert_allclose(
                actual.data[0].y,
                [patch.get_height() for patch in expected.patches],
                rtol=1e-12,
                atol=1e-14,
            )
        finally:
            plt.close(figure)
