"""Convert original Alphalens notebooks using explicit captured inputs and native display.

Supply an original pinned notebook and a captured Yahoo-shaped OHLC parquet file.
The script never downloads prices or modifies the original notebook. Execution
requires optional nbclient/ipykernel; Diagnostic itself needs no legacy package.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REFERENCE_REVISION = "1bc181de4aafbe33437ed0e362ff96aa7ce20acd"
NOTEBOOK_HASHES = {
    "overview.ipynb": "f18b00e4c0cf8d1496bf8fd56942c2dda50b83edc774672931222ed40eb130ae",
    "intraday_factor.ipynb": "e62616677251e7e83364373109b7ae362c464b03b91c1e071642d87b89111edd",
    "event_study.ipynb": "18efb5472ea0be379dec1a6a19f86813b4c6bedddffd084ee5a137a3f4544d4f",
}


def _display_calls(text: str) -> tuple[str, list[dict[str, str]]]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    replacements = []
    for node in ast.parse(text).body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and ast.unparse(node.value.func.value) == "alphalens.plotting"
        ):
            replacements.append(
                (
                    offsets[node.lineno - 1] + node.col_offset,
                    offsets[node.end_lineno - 1] + node.end_col_offset,
                )
            )
    edits = []
    for start, end in reversed(replacements):
        call = text[start:end]
        replacement = "_display_migrated_figures(" + call + ")"
        text = text[:start] + replacement + text[end:]
        edits.append({"category": "presentation", "original": call, "replacement": replacement})
    return text, edits


def _provider_input(text: str, ohlc: Path) -> tuple[str, list[dict[str, str]], int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    replacements = []
    providers = 0
    for node in ast.parse(text).body:
        replacement = None
        if isinstance(node, ast.Import) and any(
            alias.name in {"yfinance", "pandas_datareader.data"} for alias in node.names
        ):
            replacement = ""
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "yf.pdr_override"
        ):
            replacement = ""
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "web.get_data_yahoo"
        ):
            replacement = f"df = pd.read_parquet({str(ohlc.resolve())!r})"
            providers += 1
        if replacement is not None:
            replacements.append(
                (
                    offsets[node.lineno - 1] + node.col_offset,
                    offsets[node.end_lineno - 1] + node.end_col_offset,
                    replacement,
                )
            )
    edits = []
    for start, end, replacement in reversed(replacements):
        original = text[start:end]
        text = text[:start] + replacement + text[end:]
        edits.append({"category": "fixture_io", "original": original, "replacement": replacement})
    return text, edits, providers


def _observed_stack_calls(text: str) -> tuple[str, list[dict[str, str]]]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    replacements = []
    for node in ast.walk(ast.parse(text)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "stack"
            and not node.args
            and not node.keywords
        ):
            frame = ast.get_source_segment(text, node.func.value)
            replacements.append(
                (
                    offsets[node.lineno - 1] + node.col_offset,
                    offsets[node.end_lineno - 1] + node.end_col_offset,
                    f"_stack_observed({frame})",
                )
            )
    edits = []
    for start, end, replacement in sorted(replacements, reverse=True):
        original = text[start:end]
        text = text[:start] + replacement + text[end:]
        edits.append(
            {
                "category": "explicit_missing_value_method",
                "original": original,
                "replacement": replacement,
            }
        )
    return text, edits


def migrate_notebook(original: Path, ohlc: Path) -> dict[str, Any]:
    """Translate allowlisted imports, fixture I/O and presentation, retaining calculations."""
    raw = original.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != NOTEBOOK_HASHES.get(original.name):
        raise ValueError("original notebook must match a supported pinned reference hash")
    prices = pd.read_parquet(ohlc)
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or not prices.index.is_unique
        or prices.index.hasnans
        or not isinstance(prices.columns, pd.MultiIndex)
        or prices.columns.nlevels != 2
        or not {"Open", "Close"}.issubset(prices.columns.get_level_values(0))
    ):
        raise ValueError(
            "OHLC requires unique date rows and Yahoo-style(field,asset) columns including Open/Close"
        )
    notebook = json.loads(raw)
    edits = []
    provider_cells = 0
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "web.get_data_yahoo(" in text:
            text, replaced, count = _provider_input(text, ohlc)
            edits.extend({"cell": number, **edit} for edit in replaced)
            provider_cells += count
        else:
            for before, after in [
                ("import alphalens", "from ml4t.diagnostic.compat import alphalens"),
                ("%matplotlib inline", ""),
                ("sns.set_style('white')", ""),
                ("sns.despine();", ""),
                ("sns.despine()", ""),
            ]:
                if before in text:
                    text = text.replace(before, after)
                    edits.append(
                        {
                            "cell": number,
                            "category": "import"
                            if before == "import alphalens"
                            else "presentation",
                            "original": before,
                            "replacement": after,
                        }
                    )
            text, displayed = _display_calls(text)
            edits.extend({"cell": number, **edit} for edit in displayed)
        text, stacked = _observed_stack_calls(text)
        edits.extend({"cell": number, **edit} for edit in stacked)
        cell.update(source=text.splitlines(keepends=True), outputs=[], execution_count=None)
    if provider_cells != 1:
        raise ValueError("expected exactly one pinned source provider cell")
    notebook["cells"].insert(
        0,
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [
                "def _stack_observed(frame):\n",
                "    observed = frame.stack()\n",
                "    return observed.dropna(how='all') if isinstance(observed, pd.DataFrame) else observed.dropna()\n",
                "def _display_migrated_figures(value):\n",
                "    for figure in value if isinstance(value, list) else [value]:\n",
                "        figure.show(renderer='plotly_mimetype')\n",
            ],
        },
    )
    notebook.setdefault("metadata", {})["diagnostic_migration"] = {
        "reference_revision": REFERENCE_REVISION,
        "original_notebook": original.name,
        "original_sha256": digest,
        "ohlc_sha256": hashlib.sha256(ohlc.read_bytes()).hexdigest(),
        "edits": edits,
        "license": "Apache-2.0; original notebooks Copyright 2018 Quantopian, Inc.",
        "license_text": (
            Path(__file__).parents[1] / "tests/fixtures/alphalens_original/LICENSE"
        ).read_text(),
        "scope": "Original analytical expressions retained with recorded import, I/O, explicit legacy stack missingness and presentation edits. Other original notebook families and remaining analytical routes require their own qualification.",
    }
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "This notebook is a modified Alphalens Reloaded example. Original analytical expressions are retained with recorded import, captured-data I/O, legacy stack missingness and Plotly presentation edits.\n",
                f"Original source: https://github.com/stefan-jansen/alphalens-reloaded/blob/{REFERENCE_REVISION}/src/alphalens/examples/{original.name}\n",
                "Original copyright 2018 Quantopian, Inc.; Apache License 2.0. The complete original license and edit record are included in this notebook's diagnostic_migration metadata.\n",
            ],
        },
    )
    return notebook


def main() -> None:
    """Write a separate migrated notebook, optionally executing with the active installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--ohlc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.source_notebook.resolve():
        raise FileExistsError("output must be a new file, not the original notebook")
    notebook = migrate_notebook(args.source_notebook, args.ohlc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.execute:
        import nbformat
        from nbclient import NotebookClient

        executing = nbformat.reads(json.dumps(notebook), as_version=4)
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
        args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote preserved-original migration: {args.output}")


if __name__ == "__main__":
    main()
