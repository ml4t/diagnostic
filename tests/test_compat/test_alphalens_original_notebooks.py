"""Preserved original journeys must execute real Diagnostic APIs without Reloaded."""

import ast
import hashlib
import importlib.util
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/alphalens_original"


def _ohlc(name, original):
    assets = None
    for cell in json.loads(original.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        text = "\n".join(line for line in text.splitlines() if not line.startswith("%"))
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Assign):
                continue
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "ticker_sector" in names and isinstance(node.value, ast.Dict):
                assets = list(ast.literal_eval(node.value))
            elif "tickers" in names and isinstance(node.value, ast.List):
                assets = ast.literal_eval(node.value)
    assert assets
    dates = pd.bdate_range("2014-12-01" if name == "overview" else "2017-01-02", periods=64)
    rng = np.random.default_rng(734)
    if name == "event_study":
        prices = np.where((np.arange(64)[:, None] + np.arange(len(assets))) % 8 < 4, 31.0, 29.0)
    else:
        prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, size=(64, len(assets))), axis=0)
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    values = pd.DataFrame(
        {
            "Open": prices.ravel(),
            "Close": (prices * (1 + rng.normal(0, 0.004, prices.shape))).ravel(),
        },
        index=index,
    )
    return values.unstack("asset"), dates, len(assets)


def test_real_long_market_frame_is_rejected_before_original_yahoo_wide_stack_can_add_a_third_level(
    tmp_path,
):
    migrate = runpy.run_path(str(ROOT / "examples/migrate_alphalens_notebooks.py"))[
        "migrate_notebook"
    ]
    original = FIXTURES / "overview.ipynb"
    wide, _, _ = _ohlc("overview", original)
    market_frame = wide.stack(future_stack=True)
    supplied = tmp_path / "long-market.parquet"
    market_frame.to_parquet(supplied)
    with pytest.raises(ValueError, match="Yahoo-style"):
        migrate(original, supplied)


@pytest.mark.parametrize("name", ["overview", "intraday_factor", "event_study"])
def test_preserved_original_calculations_execute_native_reports_with_actual_keys_and_notebook_outputs(
    name, tmp_path
):
    if os.environ.get("ML4T_REQUIRED_ORIGINAL_NOTEBOOKS"):
        import nbclient
        import nbformat
    else:
        nbformat = pytest.importorskip("nbformat")
        nbclient = pytest.importorskip("nbclient")
        pytest.importorskip("ipykernel")
    if importlib.util.find_spec("alphalens") is not None:
        if os.environ.get("ML4T_REQUIRED_ORIGINAL_NOTEBOOKS"):
            pytest.fail("Standalone qualification must not install Alphalens")
        pytest.skip("Native-only original journey profile, separate from actual reference profiles")
    import ml4t.diagnostic

    migrate = runpy.run_path(str(ROOT / "examples/migrate_alphalens_notebooks.py"))[
        "migrate_notebook"
    ]
    original = FIXTURES / f"{name}.ipynb"
    ohlc, dates, assets = _ohlc(name, original)
    input_file = tmp_path / "captured-ohlc.parquet"
    ohlc.to_parquet(input_file)
    input_digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    notebook = nbformat.reads(json.dumps(migrate(original, input_file)), as_version=4)
    capture = tmp_path / "capture.parquet"
    metadata = tmp_path / "capture.json"
    captured_name = "non_predictive_factor_data" if name == "intraday_factor" else "factor_data"
    notebook.cells.append(
        nbformat.v4.new_code_cell(f"""import importlib.util, json, pathlib, ml4t.diagnostic
assert importlib.util.find_spec('alphalens') is None
{captured_name}.to_parquet({str(capture)!r})
pathlib.Path({str(metadata)!r}).write_text(json.dumps({{'module': ml4t.diagnostic.__file__, 'pricing_rows': len(pricing)}}))
""")
    )
    notebook.cells[-1].pop("id", None)
    try:
        nbclient.NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(tmp_path)}},
        ).execute()
    finally:
        nbformat.write(notebook, tmp_path / "executed.ipynb")
    actual = pd.read_parquet(capture)
    info = json.loads(metadata.read_text())
    assert Path(info["module"]).resolve() == Path(ml4t.diagnostic.__file__).resolve()
    assert actual.index.is_unique and actual.index.names == ["date", "asset"]
    assert str(actual.index.get_level_values("date").tz) == "UTC"
    outputs = [
        output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    ]
    mime = [
        output for output in outputs if "application/vnd.plotly.v1+json" in output.get("data", {})
    ]
    reports = [
        output
        for output in outputs
        if 'class="plot-container"' in output.get("data", {}).get("text/html", "")
    ]
    if name == "overview":
        assert len(actual) == 54 * assets - 1 and len(mime) == 33 and len(reports) == 6
        assert info["pricing_rows"] == 64
        assert list(actual.columns[:3]) == ["1D", "5D", "10D"]
    elif name == "intraday_factor":
        assert len(actual) == 62 * assets and len(reports) == 2
        assert info["pricing_rows"] == 128
        assert list(actual.columns[:2]) == ["6h30m", "1D"]
        assert actual.index.get_level_values("date")[0] == dates[1].tz_localize(
            "UTC"
        ) + pd.Timedelta("9h30m")
    else:
        expected = sum((day + offset) % 8 == 4 for day in range(1, 54) for offset in range(assets))
        assert len(actual) == expected and len(reports) == 2
        assert list(actual.columns[:7]) == ["1D", "2D", "3D", "4D", "5D", "6D", "10D"]
        np.testing.assert_array_equal(actual["factor"], -29)
        np.testing.assert_array_equal(actual["factor_quantile"], 1)
    assert hashlib.sha256(input_file.read_bytes()).hexdigest() == input_digest
