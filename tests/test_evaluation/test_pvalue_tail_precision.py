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
from ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio import (
    deflated_sharpe_ratio_from_statistics,
)
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


def test_deflated_sharpe_pvalue_survives_extreme_z():
    """The DSR p-value is the two-line spelling: `probability` then `1 - probability`.

    A Sharpe of 0.5 over ten years of daily returns puts z above 25, so
    `probability` is 1.0 to the last bit and the subtraction leaves nothing.
    """
    result = deflated_sharpe_ratio_from_statistics(observed_sharpe=0.5, n_samples=2520, n_trials=1)

    assert result.z_score > 8.35, "fixture must reach the underflow zone"
    assert result.probability == 1.0, "premise: probability has saturated"
    assert result.p_value > 0.0, "p-value underflowed to exactly zero"
    assert np.isfinite(result.p_value)


SRC = Path(__file__).resolve().parents[2] / "src"

# `1 - norm.cdf(x)` written out in one expression.
DIRECT = re.compile(r"1(\.0)?\s*-\s*[A-Za-z_][\w.]*\.cdf\(")
# `probability = norm.cdf(x)` - a name bound to a CDF value.
CDF_BINDING = re.compile(r"^\s*([A-Za-z_]\w*)\s*=.*[A-Za-z_][\w.]*\.cdf\(")


def _one_minus(name: str) -> re.Pattern[str]:
    return re.compile(rf"1(\.0)?\s*-\s*{re.escape(name)}\b")


def _offenders(path: Path) -> list[str]:
    """Both spellings of the cancellation, in one file.

    The two-line form is the one that hides: ``probability = norm.cdf(z)``
    followed by ``p_value = 1 - probability`` cancels exactly as hard as the
    inline expression, and reads as arithmetic rather than as a tail. Names
    bound to a CDF value are collected first, then every ``1 - <that name>`` is
    flagged wherever it appears in the same file.
    """
    lines = path.read_text().splitlines()
    # Comments are prose, and prose that names the forbidden pattern - including
    # the ones explaining why a nearby line uses sf - is not the pattern.
    code = [line.split("#", 1)[0] for line in lines]
    bound = {m.group(1) for line in code if (m := CDF_BINDING.match(line))}
    indirect = [_one_minus(name) for name in sorted(bound)]

    return [
        f"{path.relative_to(SRC)}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(code, start=1)
        if DIRECT.search(line) or any(p.search(line) for p in indirect)
    ]


def test_no_tail_probability_is_written_as_one_minus_cdf():
    """Static guard so the cancellation cannot come back anywhere in the package.

    Covers the sites with no cheap numeric fixture - the event-study,
    binary-metric, conditional-IC, regularized-factor, tearsheet and deflated
    Sharpe p-values.
    """
    offenders = [line for path in sorted(SRC.rglob("*.py")) for line in _offenders(path)]

    assert not offenders, "use dist.sf(x), not 1 - dist.cdf(x):\n" + "\n".join(offenders)
