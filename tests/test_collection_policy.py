"""Standard pytest selection must retain financial reference correctness checks."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


def test_standard_selection_runs_financial_checks_and_preserves_explicit_timing_opt_in(
    pytester: pytest.Pytester,
):
    project = Path(__file__).parents[1]
    pytester.makeconftest(
        f"""
import importlib.util

spec = importlib.util.spec_from_file_location("actual_collection_policy", {str(project / "tests/conftest.py")!r})
actual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actual)
pytest_configure = actual.pytest_configure
pytest_collection_modifyitems = actual.pytest_collection_modifyitems
"""
    )
    contribution = pytester.makepyfile(
        test_contribution="""
import numpy as np
import pytest
from ml4t.diagnostic.evaluation import PortfolioAnalysis

def test_financial_benchmark_pairing():
    reference = np.array([.01, -.02, .03, .015])
    result = PortfolioAnalysis(2 * reference, benchmark=reference).compute_summary_stats()
    assert result.beta == pytest.approx(2)

@pytest.mark.parametrize("frequency", ["daily_benchmark", "monthly_reference"])
def test_reference_annualization(frequency):
    reference = np.array([.01, -.02, .03, .015])
    scale = 252 if frequency == "daily_benchmark" else 12
    result = PortfolioAnalysis(2 * reference, periods_per_year=scale).compute_summary_stats()
    expected = 2 * np.std(reference, ddof=1) * np.sqrt(scale)
    assert result.annual_volatility == pytest.approx(expected)

@pytest.mark.slow
def test_explicit_timing_opt_in():
    pytest.fail("Explicit timing tests must not run under the standard selection")
"""
    )
    result = pytester.runpytest_subprocess(
        "-c",
        str(project / "pyproject.toml"),
        "--rootdir",
        str(pytester.path),
        "--confcutdir",
        str(pytester.path),
        "-n",
        "0",
        "-q",
        str(contribution),
        timeout=60,
    )
    result.assert_outcomes(passed=3, deselected=1)
