"""Tail p-values must stay positive and finite for extreme test statistics.

A two-tailed p-value written as ``2 * (1 - dist.cdf(abs(stat)))`` returns exactly
``0.0`` once ``abs(stat)`` passes roughly 8.35, because ``cdf`` rounds to ``1.0``
long before the tail mass underflows: the subtraction cancels every remaining
significant digit. The survival function ``dist.sf`` evaluates the tail directly
and stays accurate to the smallest normal double.

These tests fail against the ``1 - cdf`` form and pass against ``sf``.
"""

import ast
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


def _is_cdf_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cdf"
    )


def _is_literal_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and not isinstance(node.value, bool) and node.value == 1


def _cdf_bound_names(tree: ast.AST) -> set[str]:
    """Names assigned an expression that calls ``.cdf`` anywhere inside it."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value, targets = node.value, [node.target]
        else:
            continue
        if any(_is_cdf_call(n) for n in ast.walk(value)):
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _offenders(path: Path, rel: str | None = None) -> list[str]:
    """Both spellings of the cancellation, in one file.

    The two-line form is the one that hides: ``probability = norm.cdf(z)``
    followed by ``p_value = 1 - probability`` cancels exactly as hard as the
    inline expression, and reads as arithmetic rather than as a tail.

    Parsed rather than grepped, for two reasons. Comments and docstrings are
    invisible to ``ast``, so prose naming the pattern - including a comment
    explaining why a nearby line uses ``sf`` - does not trip it. And an
    assignment spread over several lines is one node, so ``probability = float(``
    with the ``norm.cdf(...)`` on the next line is still a CDF binding.

    Known limit: a CDF value reached through a subscript or attribute rather
    than a bare name (``1 - result["psr"]``) is not tracked.
    """
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bound = _cdf_bound_names(tree)
    lines = source.splitlines()
    label = rel if rel is not None else str(path.relative_to(SRC))

    hits = {
        node.lineno: f"{label}:{node.lineno}: {lines[node.lineno - 1].strip()}"
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and _is_literal_one(node.left)
        and (
            _is_cdf_call(node.right)
            or (isinstance(node.right, ast.Name) and node.right.id in bound)
        )
    }
    return [hits[line] for line in sorted(hits)]


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("inline", "p = 2 * (1 - stats.t.cdf(abs(t), df=n))\n"),
        ("bound name", "probability = norm.cdf(z)\np_value = 1 - probability\n"),
        (
            "binding wrapped over lines",
            "probability = float(\n    stats.norm.cdf(z)\n)\np_value = float(1.0 - probability)\n",
        ),
        (
            "subtraction wrapped over lines",
            "probability = stats.norm.cdf(z)\np_value = (\n    1\n    - probability\n)\n",
        ),
    ],
)
def test_detector_finds_every_layout(tmp_path: Path, label: str, source: str) -> None:
    """Every way of writing it, including the ones no regex sees."""
    path = tmp_path / "sample.py"
    path.write_text(source)

    assert _offenders(path, "sample.py"), f"missed the {label} layout"


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("sf", "p_value = 2 * stats.t.sf(abs(t), df=n)\n"),
        (
            "a comment naming the pattern",
            "# sf, not 1 - norm.cdf(z), which cancels\np = norm.sf(z)\n",
        ),
        ("a CDF used as a probability", "psr = float(norm.cdf(z))\nis_significant = psr >= 0.95\n"),
        ("an unrelated subtraction", "share = 1 - weight\n"),
    ],
)
def test_detector_does_not_fire_on_correct_code(tmp_path: Path, label: str, source: str) -> None:
    path = tmp_path / "sample.py"
    path.write_text(source)

    assert not _offenders(path, "sample.py"), f"false positive on {label}"


def test_no_tail_probability_is_written_as_one_minus_cdf():
    """Static guard so the cancellation cannot come back anywhere in the package.

    Covers the sites with no cheap numeric fixture - the event-study,
    binary-metric, conditional-IC, regularized-factor, tearsheet and deflated
    Sharpe p-values.
    """
    offenders = [line for path in sorted(SRC.rglob("*.py")) for line in _offenders(path)]

    assert not offenders, "use dist.sf(x), not 1 - dist.cdf(x):\n" + "\n".join(offenders)
