"""A real notebook must retain every turnover lag and its owned observations."""

import asyncio
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from ml4t import diagnostic


def test_notebook_displays_all_six_turnover_panels_and_roundtrips_exact_dated_tables(tmp_path):
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    from jupyter_client import AsyncKernelManager

    code = """
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.io as pio
from IPython.display import HTML, display
from ml4t import diagnostic
from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
from ml4t.diagnostic.signal import prepare_signal_panel
from ml4t.diagnostic.visualization.signal import generate_signal_turnover_report, plot_signal_turnover
pio.renderers.default='plotly_mimetype'
dates=pd.bdate_range('2025-03-03',periods=14,tz='UTC')
changes=np.array([np.roll([.01,.02,-.01,-.02],i) for i in range(13)])
prices=pd.DataFrame(np.vstack([np.full(4,100.),100*np.cumprod(1+changes,axis=0)]),index=dates,columns=list('ABCD'))
factor=prices.pct_change(fill_method=None).iloc[1:-1].stack(future_stack=True).rename('factor')
forward=(prices.shift(-1)/prices-1).stack(future_stack=True).rename('1D').to_frame()
factor.index.names=forward.index.names=['date','asset']
panel=prepare_signal_panel(factor,forward,quantiles=2,max_loss=0)
result=TurnoverAnalysisResult.from_observations(panel.quantile_turnover(lags=(1,2,5)),panel.rank_autocorrelation(lags=(1,2,5)))
Path('turnover.json').write_text(result.model_dump_json())
restored=TurnoverAnalysisResult.model_validate_json(Path('turnover.json').read_text())
for name in ['turnover','autocorrelation','dated_turnover','dated_autocorrelation']:
    assert restored.get_dataframe(name).equals(result.get_dataframe(name))
for figure in plot_signal_turnover(restored): figure.show()
display(HTML(generate_signal_turnover_report(restored)))
print('ML4T_TURNOVER_NOTEBOOK:'+json.dumps({'module':diagnostic.__file__,'scored_rows':panel.data.height,'membership_rows':restored.get_dataframe('dated_turnover').height,'rank_rows':restored.get_dataframe('dated_autocorrelation').height,'rank_counts':restored.get_dataframe('autocorrelation')['n_dates'].to_list(),'export':str(Path('turnover.json').resolve())}))
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
    path = tmp_path / "turnover.ipynb"
    nbformat.write(executed, path)
    outputs = executed.cells[0].outputs
    figures = [
        out["data"]["application/vnd.plotly.v1+json"]
        for out in outputs
        if "application/vnd.plotly.v1+json" in out.get("data", {})
    ]
    assert [figure["layout"]["meta"]["lag"] for figure in figures] == [1, 2, 5, 1, 2, 5]
    assert all(figure["layout"]["meta"]["clock_source"] == "observed" for figure in figures)
    for index, lag in enumerate([1, 2, 5]):
        for trace in figures[index]["data"]:
            assert trace["y"][:lag] == [None] * lag
            assert trace["y"][lag:] == pytest.approx([1 if lag == 2 else 0.5] * (12 - lag))
        rank = figures[index + 3]["data"][0]
        assert rank["y"][:lag] == [None] * lag
        assert rank["y"][lag:] == pytest.approx([-1 if lag == 2 else 0] * (12 - lag), abs=1e-12)
        assert rank["x"][0].startswith("2025-03-04T00:00:00")
        assert rank["x"][-1].startswith("2025-03-19T00:00:00")
        assert rank["connectgaps"] is False
    html = "\n".join(out.get("data", {}).get("text/html", "") for out in outputs)
    assert html.count('class="plot-container"') == 6
    assert "All-quantile mean turnover" in html and "Dated rank observations" in html
    stream = "".join(out.get("text", "") for out in outputs if out.get("output_type") == "stream")
    metadata = json.loads(
        next(
            line.removeprefix("ML4T_TURNOVER_NOTEBOOK:")
            for line in stream.splitlines()
            if line.startswith("ML4T_TURNOVER_NOTEBOOK:")
        )
    )
    assert Path(metadata["module"]).resolve() == Path(diagnostic.__file__).resolve()
    assert [metadata[key] for key in ["scored_rows", "membership_rows", "rank_rows"]] == [
        48,
        72,
        36,
    ]
    assert metadata["rank_counts"] == [11, 10, 7]
    assert Path(metadata["export"]).parent == tmp_path
    print(
        json.dumps(
            dict(
                **metadata,
                packages={
                    name: version(name)
                    for name in [
                        "ml4t-diagnostic",
                        "pandas",
                        "polars",
                        "plotly",
                        "nbclient",
                        "ipykernel",
                    ]
                },
                mime_figures=len(figures),
                html_panels=6,
                notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    )
