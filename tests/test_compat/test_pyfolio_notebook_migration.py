"""Original notebook analytical replays must run without a legacy analytical engine."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/migrate_pyfolio_notebooks.py"
FIXTURES = ROOT / "tests/fixtures/pyfolio_original"
STANDALONE = {
    "round_trip_tear_sheet_example.ipynb": 7,
    "sector_mappings_example.ipynb": 17,
    "slippage_example.ipynb": 40,
    "single_stock_example.ipynb": 14,
}


def _history(path, *, aware=False):
    dates = pd.bdate_range("2019-07-01", periods=160, tz="UTC" if aware else None)
    prices = 100 * np.cumprod(1 + 0.001 + np.sin(np.arange(160) / 7) * 0.01)
    pd.DataFrame({"Close": prices}, index=dates).to_parquet(path)
    return path


def _cli(original, output, *arguments, execute=False):
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--source-notebook",
            str(original),
            "--output",
            str(output),
            *map(str, arguments),
            *(["--execute"] if execute else []),
        ],
        cwd=output.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=240,
        env=environment,
    )
    return completed


def _require_execution():
    for name in ("nbformat", "nbclient", "ipykernel", "plotly"):
        if importlib.util.find_spec(name) is None:
            if os.environ.get("ML4T_REQUIRED_PYFOLIO_NOTEBOOKS"):
                pytest.fail(f"required original-notebook execution dependency missing: {name}")
            pytest.skip(f"optional original-notebook execution dependency missing: {name}")


@pytest.mark.parametrize("name", STANDALONE)
def test_cli_converts_actual_original_notebooks_without_executing_or_overwriting(name, tmp_path):
    original = FIXTURES / name
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    output = tmp_path / name
    arguments = (
        ["--history", _history(tmp_path / "history.parquet")]
        if name == "single_stock_example.ipynb"
        else []
    )
    completed = _cli(original, output, *arguments)
    assert completed.returncode == 0, completed.stderr
    notebook = json.loads(output.read_text(encoding="utf-8"))
    record = notebook["metadata"]["diagnostic_migration"]
    assert record["input_notebook_sha256"] == digest
    assert record["edits"]
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    saved = output.read_bytes()
    refused = _cli(original, output, *arguments)
    assert refused.returncode != 0
    assert "output must be a new file" in refused.stderr
    assert output.read_bytes() == saved


@pytest.mark.parametrize("name,expected_plots", STANDALONE.items())
def test_public_cli_executes_original_financial_cells_and_actual_notebook_payloads(
    name, expected_plots, tmp_path
):
    _require_execution()
    arguments = (
        ["--history", _history(tmp_path / "history.parquet")]
        if name == "single_stock_example.ipynb"
        else []
    )
    output = tmp_path / name
    completed = _cli(FIXTURES / name, output, *arguments, execute=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    notebook = json.loads(output.read_text(encoding="utf-8"))
    outputs = [item for cell in notebook["cells"] for item in cell.get("outputs", [])]
    assert not any(item["output_type"] == "error" for item in outputs)
    plots = [
        item["data"]["application/vnd.plotly.v1+json"]
        for item in outputs
        if "application/vnd.plotly.v1+json" in item.get("data", {})
    ]
    assert len(plots) == expected_plots
    identity = [
        json.loads(line)["diagnostic_runtime"]
        for item in outputs
        if item["output_type"] == "stream"
        for line in "".join(item["text"]).splitlines()
        if line.startswith('{"diagnostic_runtime":')
    ]
    assert len(identity) == 1
    import ml4t.diagnostic

    assert Path(identity[0]["module"]).resolve() == Path(ml4t.diagnostic.__file__).resolve()
    assert identity[0]["optional_modules"] == {
        name: importlib.util.find_spec(name) is not None
        for name in ("pyfolio", "alphalens", "empyrical", "zipline")
    }
    if name == "slippage_example.ipynb":
        wealth = next(
            plot
            for plot in plots
            if plot["layout"].get("title", {}).get("text") == "Cumulative wealth"
        )
        assert wealth["data"][0]["y"][-1] == pytest.approx(1.6540425002394212)
    if name == "single_stock_example.ipynb":
        wealth = next(
            plot
            for plot in plots
            if plot["layout"].get("title", {}).get("text") == "Cumulative wealth"
        )
        prices = pd.read_parquet(tmp_path / "history.parquet")["Close"]
        live = next(trace for trace in wealth["data"] if trace["name"] == "Live")
        assert live["y"][-1] == pytest.approx(prices.iloc[-1] / prices.iloc[0])


def test_edited_original_analytical_expression_is_rejected_instead_of_silently_translated(tmp_path):
    original = json.loads((FIXTURES / "single_stock_example.ipynb").read_text(encoding="utf-8"))
    for cell in original["cells"]:
        cell["source"] = cell["source"].replace("pct_change()", "pct_change(5)")
    edited = tmp_path / "single_stock_example.ipynb"
    edited.write_text(json.dumps(original), encoding="utf-8")
    output = tmp_path / "migrated.ipynb"
    completed = _cli(edited, output, "--history", _history(tmp_path / "history.parquet"))
    assert completed.returncode != 0
    assert "cell sources must match" in completed.stderr
    assert not output.exists()


def test_single_stock_captured_input_does_not_silently_reinterpret_aware_dates(tmp_path):
    completed = _cli(
        FIXTURES / "single_stock_example.ipynb",
        tmp_path / "migrated.ipynb",
        "--history",
        _history(tmp_path / "history.parquet", aware=True),
    )
    assert completed.returncode != 0
    assert "unique ordered naive dates" in completed.stderr


def test_wrong_pickle_is_rejected_before_conversion_or_any_unpickling(tmp_path):
    unrelated = tmp_path / "results.pickle"
    pd.DataFrame({"returns": [0.01]}).to_pickle(unrelated)
    output = tmp_path / "migrated.ipynb"
    completed = _cli(FIXTURES / "zipline_algo_example.ipynb", output, "--zipline-output", unrelated)
    assert completed.returncode != 0
    assert "exact pinned original results.pickle hash" in completed.stderr
    assert not output.exists()


def test_captured_account_fixture_cannot_be_substituted_without_provenance(tmp_path):
    fixtures = tmp_path / "inputs"
    fixtures.mkdir()
    pd.Series([0.1, -0.1], index=pd.bdate_range("2025-01-01", periods=2)).to_csv(
        fixtures / "test_returns.csv.gz"
    )
    completed = _cli(
        FIXTURES / "slippage_example.ipynb",
        tmp_path / "migrated.ipynb",
        "--fixtures-dir",
        fixtures,
    )
    assert completed.returncode != 0
    assert "fixture does not match" in completed.stderr


def test_original_zipline_captured_account_analytical_replay_preserves_reported_nav(tmp_path):
    pickle = os.environ.get("ML4T_ORIGINAL_ZIPLINE_OUTPUT")
    if pickle is None:
        if os.environ.get("ML4T_REQUIRED_PYFOLIO_NOTEBOOKS") == "all":
            pytest.fail("required pinned Zipline output missing: ML4T_ORIGINAL_ZIPLINE_OUTPUT")
        pytest.skip("optional captured original Zipline output not supplied")
    _require_execution()
    output = tmp_path / "zipline.ipynb"
    completed = _cli(
        FIXTURES / "zipline_algo_example.ipynb",
        output,
        "--zipline-output",
        pickle,
        execute=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    notebook = json.loads(output.read_text(encoding="utf-8"))
    plots = [
        item["data"]["application/vnd.plotly.v1+json"]
        for cell in notebook["cells"]
        for item in cell.get("outputs", [])
        if "application/vnd.plotly.v1+json" in item.get("data", {})
    ]
    assert len(plots) == 80
    wealth = [
        plot for plot in plots if plot["layout"].get("title", {}).get("text") == "Cumulative wealth"
    ]
    assert len(wealth) == 2
    for plot in wealth:
        live = next(trace for trace in plot["data"] if trace["name"] == "Live")
        assert live["y"][-1] == pytest.approx(1.6120420817)
