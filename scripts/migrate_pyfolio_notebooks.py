"""Replay pinned Pyfolio examples with Diagnostic and explicit captured inputs.

This is an analytical replay, not Yahoo acquisition or a Zipline algorithm run.
Conversion never executes a notebook or unpickles data. Execution is opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REFERENCE_REVISION = "185eaa1657a845a5e1c8c82809248eac7ce99340"
NOTEBOOK_IDENTITIES = {
    "round_trip_tear_sheet_example.ipynb": (
        "a82b317fb3753cefc05ceca6697181f9c9a8a64cd8bcdb9a00f97b1bf55e83b0",
        "3015ba5dc8c6eeb6093d3bb9c10c97eecb116825039a2cdfc0b98d3a223f1e4e",
    ),
    "sector_mappings_example.ipynb": (
        "589414fec60595b0be4e8141428e31acf6ebf1aef16603df084e6c4e8b656aab",
        "47c0265935356a7dbb3044f4e97f78baf2fefc5718e22f08afb7f6537e8b8d58",
    ),
    "slippage_example.ipynb": (
        "fcb194be89d4efa675c7e12af65b55e722eff44097434cfcce7ca4458b47acaf",
        "d1f08f0efb9124085f4e7953a0d57051e73eadcae1ef58ac217961a72237762d",
    ),
    "single_stock_example.ipynb": (
        "2b166ebd235b061c7b00e88147f27aa2a34c5c08d65237bab4a7fee98e72cdb8",
        "1aeacaa08949470aa2dd355f49c7077277fdcd2829169342ba8b44590a193fa2",
    ),
    "zipline_algo_example.ipynb": (
        "ba68e3fd4580c6fc83ed93d0eab14f8a1978bdd4a4f42b8d7b7f713555af0a49",
        "0ebc356f7119095b40cf9e109a074200c61beb18cee1d5f8e32ef17798c1176c",
    ),
}
FIXTURE_HASHES = {
    "test_returns.csv.gz": "1ef97a992ad9a1ca21af2c615f48dd046c39c631e021aed1313ffecd507bfa11",
    "test_pos.csv.gz": "6463ec530869f231b4da9125ae005acd254aede738f324e33525deeaed95335b",
    "test_txn.csv.gz": "bb8725c641ff33955ca2afaf3e6372f952bc10451ecba86dbb7fbc3f2c101b42",
}
ZIPLINE_HASH = "7f6480c177869eb73fc7bf6cb0715924282d02b4fa507f128f182553967ba946"
ZIPLINE_CAPTURE_HASH = "d8b45de845d7d311fd8d220d36999823af5f2e2f05055a88c012a1342abc4041"
FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/pyfolio_original"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _original_cells(notebook: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "cell_type": cell["cell_type"],
            "source": "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"],
        }
        for cell in notebook["cells"]
    ]


def migrate_notebook(
    original: Path,
    *,
    fixtures: Path = FIXTURES,
    history: Path | None = None,
    zipline_output: Path | None = None,
) -> dict[str, Any]:
    """Retain pinned cell sources except recorded import, captured I/O and display edits."""
    raw = original.read_bytes()
    notebook = json.loads(raw)
    cells = _original_cells(notebook)
    fingerprint = hashlib.sha256(
        json.dumps(cells, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    identity = NOTEBOOK_IDENTITIES.get(original.name)
    if identity is None or fingerprint != identity[1]:
        raise ValueError("notebook cell sources must match a supported pinned original")
    inputs = {}
    replacements = [
        ("import pyfolio as pf", "from ml4t.diagnostic.compat import pyfolio as pf", "import"),
        ("%matplotlib inline", "", "presentation"),
    ]
    if original.name == "single_stock_example.ipynb":
        if history is None:
            raise ValueError("single-stock replay requires --history with captured Close prices")
        import pandas as pd

        prices = pd.read_parquet(history)
        if (
            not isinstance(prices.index, pd.DatetimeIndex)
            or prices.index.tz is not None
            or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing
            or prices.index.hasnans
            or "Close" not in prices
        ):
            raise ValueError("history requires unique ordered naive dates and a Close column")
        inputs["captured_history"] = {"sha256": _digest(history), "path": str(history.resolve())}
        replacements.extend(
            [
                ("import yfinance as yf", "import pandas as pd", "captured_provider_io"),
                ("fb = yf.Ticker('FB')", "", "captured_provider_io"),
                (
                    "history = fb.history('max')",
                    f"history = pd.read_parquet({str(history.resolve())!r})",
                    "captured_provider_io",
                ),
            ]
        )
    elif original.name == "zipline_algo_example.ipynb":
        if zipline_output is not None and _digest(zipline_output) != ZIPLINE_HASH:
            raise ValueError(
                "Zipline replay requires the exact pinned original results.pickle hash"
            )
        capture = fixtures / "zipline_original_account.json.gz"
        if _digest(capture) != ZIPLINE_CAPTURE_HASH:
            raise ValueError("portable captured account does not match the pinned original")
        inputs["original_zipline_output"] = {
            "sha256": ZIPLINE_HASH,
            "path": str(zipline_output.resolve()) if zipline_output is not None else None,
            "deserialized": False,
        }
        inputs["portable_zipline_account"] = {
            "sha256": ZIPLINE_CAPTURE_HASH,
            "path": str(capture.resolve()),
        }
        replacements.extend(
            [
                ("%load_ext zipline", "", "producer_presentation"),
                ("import zipline\n", "", "producer_presentation"),
                (
                    "results = pd.read_pickle('results.pickle')",
                    "import gzip\nimport json\n"
                    f"with gzip.open({str(capture.resolve())!r}, 'rt', encoding='utf-8') as stream:\n"
                    "    captured_account = json.load(stream)\n"
                    "results = pd.DataFrame(captured_account['records'])\n"
                    "results.index = pd.DatetimeIndex(pd.to_datetime(results.pop('date')))"
                    ".rename(captured_account['index_name'])",
                    "portable_original_producer_io",
                ),
                (
                    "pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date');",
                    "pf.plotting.plot_drawdown_periods(returns, top=5)"
                    ".update_xaxes(title_text='Date').show(renderer='plotly_mimetype');",
                    "presentation",
                ),
            ]
        )
    else:
        for name, expected in FIXTURE_HASHES.items():
            path = fixtures / name
            if _digest(path) != expected:
                raise ValueError(f"captured fixture does not match the pinned original: {name}")
            inputs[name] = {"sha256": expected, "path": str(path.resolve())}
            replacements.append(
                (
                    repr("../tests/test_data/" + name),
                    repr(str(path.resolve())),
                    "captured_fixture_io",
                )
            )
    edits = []
    migrated = []
    for number, cell in enumerate(cells):
        text = cell["source"]
        if cell["cell_type"] == "code":
            if text.startswith("%%zipline"):
                edits.append(
                    {
                        "cell": number,
                        "category": "producer_not_rerun",
                        "original": text,
                        "replacement": "",
                    }
                )
                text = ""
            for before, after, category in replacements:
                if before in text:
                    text = text.replace(before, after)
                    edits.append(
                        {
                            "cell": number,
                            "category": category,
                            "original": before,
                            "replacement": after,
                        }
                    )
        migrated.append(
            dict(
                cell_type=cell["cell_type"],
                source=text,
                metadata={},
                **({"outputs": [], "execution_count": None} if cell["cell_type"] == "code" else {}),
            )
        )
    license_text = (FIXTURES / "LICENSE").read_text(encoding="utf-8")
    migrated.insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "Modified Pyfolio Reloaded example, Copyright Quantopian, Inc., "
            "Apache License 2.0. Original analytical cells are retained with recorded "
            "Diagnostic imports, captured-data I/O and Plotly presentation changes. "
            "This replay does not acquire Yahoo data or rerun the Zipline algorithm. "
            "The original source URL, complete license and edits are in "
            "this notebook's diagnostic_migration metadata.",
        },
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 2,
        "cells": migrated,
        "metadata": {
            "diagnostic_migration": {
                "original_notebook": original.name,
                "reference_revision": REFERENCE_REVISION,
                "original_sha256": identity[0],
                "input_notebook_sha256": hashlib.sha256(raw).hexdigest(),
                "original_cells_sha256": fingerprint,
                "reference_url": "https://github.com/stefan-jansen/pyfolio-reloaded/blob/"
                + REFERENCE_REVISION
                + "/src/pyfolio/examples/"
                + original.name,
                "license_text": license_text,
                "inputs": inputs,
                "edits": edits,
                "scope": "Original analytical expressions retained. Plotly replaces Matplotlib; "
                "captured data replaces acquisition. Zipline producer code is not rerun.",
            }
        },
    }


def main() -> None:
    """Write a separate converted notebook, optionally executing in the current environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--zipline-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.source_notebook.resolve():
        raise FileExistsError("output must be a new file, not the original notebook")
    notebook = migrate_notebook(
        args.source_notebook,
        fixtures=args.fixtures_dir,
        history=args.history,
        zipline_output=args.zipline_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.execute:
        import sys

        import nbformat
        from nbclient import NotebookClient

        import ml4t.diagnostic

        identity = nbformat.v4.new_code_cell(
            "import importlib.util, json, sys\nfrom pathlib import Path\n"
            "from importlib.metadata import version\nimport ml4t.diagnostic\n"
            "import plotly.io as pio\npio.renderers.default = 'plotly_mimetype'\n"
            f"assert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve()\n"
            "assert Path(ml4t.diagnostic.__file__).resolve() == "
            f"Path({ml4t.diagnostic.__file__!r}).resolve()\n"
            "print(json.dumps({'diagnostic_runtime': {'module': ml4t.diagnostic.__file__, "
            "'python': sys.version, 'optional_modules': {name: importlib.util.find_spec(name) "
            "is not None for name in ('pyfolio', 'alphalens', 'empyrical', 'zipline')}, "
            "'packages': {name: version(name) for name in "
            "('ml4t-diagnostic', 'pandas', 'numpy', 'plotly')}}}))\n"
        )
        executing = nbformat.reads(json.dumps(notebook), as_version=4)
        executing.cells.insert(0, identity)
        try:
            NotebookClient(
                executing,
                timeout=180,
                kernel_name="python3",
                resources={"metadata": {"path": str(args.output.parent.resolve())}},
            ).execute()
        finally:
            nbformat.write(executing, args.output)
    else:
        args.output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote captured analytical replay: {args.output}")


if __name__ == "__main__":
    main()
