"""Geometric capture must not be confused with native terminal-wealth capture."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import empyrical as ep


def test_monthly_capture_annualizes_gains_not_the_terminal_wealth_ratio():
    from ml4t.diagnostic.evaluation.portfolio_analysis import up_down_capture

    returns = np.array([0.04, 0.01, -0.01, -0.03])
    benchmark = np.array([0.02, 0.01, -0.02, -0.01])
    up = ((1.04 * 1.01) ** 6 - 1) / ((1.02 * 1.01) ** 6 - 1)
    down = ((0.99 * 0.97) ** 6 - 1) / ((0.98 * 0.99) ** 6 - 1)
    assert ep.up_capture(returns, benchmark, period="monthly") == pytest.approx(up)
    assert ep.down_capture(returns, benchmark, period="monthly") == pytest.approx(down)
    assert ep.up_down_capture(returns, benchmark, period="monthly") == pytest.approx(up / down)
    assert up_down_capture(returns, benchmark) == pytest.approx(
        [(1.04 * 1.01) / (1.02 * 1.01), (0.99 * 0.97) / (0.98 * 0.99)]
    )


def test_capture_full_stream_counts_missing_rows_but_sign_subsets_use_present_pairs():
    returns = np.array([0.02, np.nan, -0.01, 0.03])
    benchmark = np.array([0.01, 0.04, -0.02, 0.02])
    expected = ((1.02 * 0.99 * 1.03) ** 3 - 1) / ((1.01 * 1.04 * 0.98 * 1.02) ** 3 - 1)
    assert ep.capture(returns, benchmark, period="monthly") == pytest.approx(expected)
    up = ((1.02 * 1.03) ** 6 - 1) / ((1.01 * 1.02) ** 6 - 1)
    assert ep.up_capture(returns, benchmark, period="monthly") == pytest.approx(up)
    assert np.isnan(ep.down_capture(np.array([0.01, 0.02]), np.array([0.01, 0.03])))


@pytest.mark.parametrize("name", ["capture", "up_capture", "down_capture", "up_down_capture"])
@pytest.mark.parametrize("kind", ["array", "dated", "matrix", "matrix_vector"])
def test_published_capture_workflow_preserves_actual_per_function_shapes(name, kind):
    reference = pytest.importorskip("empyrical")
    benchmark = np.array([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
    returns = 0.0003 + 1.2 * benchmark
    returns[1] = np.nan
    if kind == "dated":
        dates = pd.date_range("2025-01-01", periods=len(returns), tz="UTC")
        returns, benchmark = pd.Series(returns, index=dates), pd.Series(benchmark, index=dates)
    elif kind in {"matrix", "matrix_vector"}:
        returns = np.column_stack([returns, 0.0002 - 0.4 * benchmark])
        if kind == "matrix":
            benchmark = np.column_stack([benchmark, benchmark])
    actual = getattr(ep, name)(returns, benchmark, period="monthly")
    expected = getattr(reference, name)(returns.copy(), benchmark.copy(), period="monthly")
    assert np.shape(actual) == np.shape(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_capture_does_not_align_labels_before_independent_full_stream_products():
    reference = pytest.importorskip("empyrical")
    dates = pd.date_range("2025-01-01", periods=4, tz="UTC")
    returns = pd.Series([0.02, 0.01, -0.01], index=dates[:3])
    benchmark = pd.Series([0.01, -0.02, 0.03], index=dates[1:])
    assert ep.capture(returns, benchmark, period="monthly") == pytest.approx(
        reference.capture(returns, benchmark, period="monthly")
    )
