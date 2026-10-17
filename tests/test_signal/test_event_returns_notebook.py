"""A real notebook must display every event-returns panel and export the same samples."""

import asyncio
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from ml4t import diagnostic


def test_real_notebook_kernel_displays_all_five_event_returns_panels_and_owned_export(tmp_path):
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    from jupyter_client import AsyncKernelManager

    code = """
import json
from pathlib import Path
import pandas as pd
import plotly.io as pio
from IPython.display import HTML, display
from ml4t import diagnostic
from ml4t.diagnostic.signal import EventReturnsResult, analyze_event_returns
from ml4t.diagnostic.visualization.signal import generate_event_returns_report, plot_event_returns
pio.renderers.default='plotly_mimetype'
dates=pd.date_range('2025-01-01',periods=3,tz='UTC')
prices=pd.DataFrame({'A':[10.,20.,30.],'B':[100.,200.,300.]},index=dates)
events=pd.DataFrame({'date':dates.repeat(2),'asset':['A','B']*3,'quantile':[1,2]*3,'group':['small','large']*3})
result=analyze_event_returns(events,prices,input_kind='levels',before=1,after=1,demeaned=False,by_group=True)
Path('event-returns.json').write_text(result.to_json())
restored=EventReturnsResult.from_json(Path('event-returns.json').read_text())
assert restored.overall.paths.equals(result.overall.paths)
assert restored.groups.paths.equals(result.groups.paths)
for figure in plot_event_returns(result): figure.show()
display(HTML(generate_event_returns_report(result)))
print('ML4T_EVENT_NOTEBOOK:'+json.dumps({'module':diagnostic.__file__,'events':result.overall.event_counts['n_events'].sum(),'path_cells':result.overall.paths.height,'export':str(Path('event-returns.json').resolve())}))
"""
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])

    async def execute():
        manager = AsyncKernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = nbclient.NotebookClient(
            notebook,
            km=manager,
            timeout=120,
            record_timing=False,
            resources={"metadata": {"path": str(tmp_path)}},
        )
        try:
            return await client.async_execute()
        finally:
            if client.kc is not None:
                client.kc.stop_channels()
            if manager.has_kernel:
                await manager.shutdown_kernel(now=True)

    executed = asyncio.run(execute())
    path = tmp_path / "event-returns.ipynb"
    nbformat.write(executed, path)
    outputs = executed.cells[0].outputs
    figures = [
        out["data"]["application/vnd.plotly.v1+json"]
        for out in outputs
        if "application/vnd.plotly.v1+json" in out.get("data", {})
    ]
    assert [fig["layout"]["title"]["text"] for fig in figures] == [
        "All quantiles",
        "Quantile 1",
        "Quantile 2",
        "Group: large",
        "Group: small",
    ]
    assert figures[0]["data"][0]["x"] == [-1, 0, 1]
    assert figures[0]["data"][0]["y"] == [15, 20, 25]
    assert figures[0]["data"][1]["y"] == [150, 200, 250]
    assert figures[1]["data"][0]["error_y"]["array"][1] == 10
    assert figures[2]["data"][0]["error_y"]["array"][1] == 100
    assert figures[-1]["data"][0]["y"] == [15, 20, 25]
    html = "\n".join(out.get("data", {}).get("text/html", "") for out in outputs)
    assert html.count('class="plot-container"') == 5
    stream = "".join(out.get("text", "") for out in outputs if out.get("output_type") == "stream")
    metadata = json.loads(
        next(
            line.removeprefix("ML4T_EVENT_NOTEBOOK:")
            for line in stream.splitlines()
            if line.startswith("ML4T_EVENT_NOTEBOOK:")
        )
    )
    assert Path(metadata["module"]).resolve() == Path(diagnostic.__file__).resolve()
    assert metadata["events"] == 6 and metadata["path_cells"] == 18
    assert Path(metadata["export"]).parent == tmp_path
    print(
        json.dumps(
            {
                "module": metadata["module"],
                "packages": {
                    name: version(name)
                    for name in [
                        "ml4t-diagnostic",
                        "pandas",
                        "plotly",
                        "nbclient",
                        "nbformat",
                        "ipykernel",
                    ]
                },
                "mime_figures": len(figures),
                "html_panels": 5,
                "exported_events": 6,
                "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )
