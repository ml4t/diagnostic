"""Tail p-values must stay positive and finite for extreme test statistics.

A two-tailed p-value written as ``2 * (1 - dist.cdf(abs(stat)))`` returns exactly
``0.0`` once ``abs(stat)`` passes roughly 8.35, because ``cdf`` rounds to ``1.0``
long before the tail mass underflows: the subtraction cancels every remaining
significant digit. The survival function ``dist.sf`` evaluates the tail directly
and stays accurate to the smallest normal double.

These tests fail against the ``1 - cdf`` form and pass against ``sf``.
"""

import re
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from ml4t.diagnostic.evaluation.factor.validation import _ljung_box
from ml4t.diagnostic.metrics.ic_inference import (
    compute_ic_hac_stats,
    compute_ic_summary_stats,
)

# Mean 0.05, tiny dispersion: t is ~113 on 29 degrees of freedom, deep in the
# zone where `1 - cdf` cancels to exactly zero.
_EXTREME_IC = 0.05 + 0.004 * (np.linspace(-1.0, 1.0, 30) - np.linspace(-1.0, 1.0, 30).mean())


def test_scipy_baseline_confirms_the_cancellation():
    """The premise: `1 - cdf` is exactly zero where `sf` is not."""
    assert 2 * (1 - stats.t.cdf(8.94, df=4231)) == 0.0
    assert 2 * stats.t.sf(8.94, df=4231) > 0.0


def test_ic_summary_stats_pvalue_survives_extreme_t():
    result = compute_ic_summary_stats(_EXTREME_IC)

    assert abs(result["t_stat"]) > 8.35, "fixture must reach the underflow zone"
    assert result["p_value"] > 0.0, "p-value underflowed to exactly zero"
    assert np.isfinite(result["p_value"])


def test_ic_hac_stats_pvalue_survives_extreme_t():
    with pytest.warns(UserWarning, match="label_horizon"):
        result = compute_ic_hac_stats(_EXTREME_IC)

    assert abs(result["t_stat"]) > 8.35, "fixture must reach the underflow zone"
    assert result["p_value"] > 0.0, "p-value underflowed to exactly zero"
    assert np.isfinite(result["p_value"])


def test_ljung_box_pvalue_survives_extreme_q():
    """A perfectly alternating residual series drives Q far into the tail.

    Kept short: a longer series pushes Q past 1900, where the true tail mass is
    below the smallest normal double and even ``sf`` legitimately returns zero.
    """
    residuals = np.tile([1.0, -1.0], 10)

    q, p_value = _ljung_box(residuals, max_lags=5)

    assert q > 50.0, "fixture must reach the underflow zone"
    assert p_value > 0.0, "p-value underflowed to exactly zero"
    assert np.isfinite(p_value)


def test_no_tail_probability_is_written_as_one_minus_cdf():
    """Static guard so the cancellation cannot come back anywhere in the package.

    Covers the sites this change touched that have no cheap numeric fixture -
    the event-study, binary-metric, conditional-IC, regularized-factor and
    tearsheet p-values.
    """
    src = Path(__file__).resolve().parents[2] / "src"
    pattern = re.compile(r"1(\.0)?\s*-\s*[A-Za-z_][\w.]*\.cdf\(")

    offenders = [
        f"{path.relative_to(src)}:{lineno}: {line.strip()}"
        for path in sorted(src.rglob("*.py"))
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if pattern.search(line)
    ]

    assert not offenders, "use dist.sf(x), not 1 - dist.cdf(x):\n" + "\n".join(offenders)
