"""Real pytest outcomes must not turn a missing or failed lifecycle case into green CI."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("outcome", ["pass", "failure", "skip"])
def test_independent_qualification_requires_one_executed_passing_case(tmp_path, outcome):
    script = Path(__file__).resolve().parents[2] / "scripts/check_windows_pdf_lifecycle.py"
    spec = importlib.util.spec_from_file_location("pdf_qualification", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    child = tmp_path / "test_actual_workflow.py"
    bodies = {
        "pass": "assert timeseries.perf_stats(pd.Series([.1, -.1]))['Cumulative returns'] == pytest.approx(-.01)",
        "failure": "timeseries.perf_stats(pd.DataFrame({'first': [.1, -.1], 'second': [.2, -.2]}))",
        "skip": "pytest.skip('Required reference input is unavailable')",
    }
    child.write_text(
        "import pytest\nimport pandas as pd\n"
        "from ml4t.diagnostic.compat.pyfolio import timeseries\n"
        "def test_report_workflow():\n    " + bodies[outcome] + "\n",
        encoding="utf-8",
    )
    attempt = tmp_path / "attempt-01"
    attempt.mkdir()
    result = module._run_case(tmp_path, str(child) + "::test_report_workflow", attempt / "case-01")
    assert result["collected_cases"] == 1
    assert result["timed_out"] is False
    assert result["passed"] is (outcome == "pass")
    assert result["exit_code"] == (1 if outcome == "failure" else 0)
    assert (tmp_path / result["junit"]).is_file()
    if outcome == "failure":
        assert "perf_stats requires one return series" in (tmp_path / result["log"]).read_text()
