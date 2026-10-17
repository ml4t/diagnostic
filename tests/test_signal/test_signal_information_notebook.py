"""A real notebook executes every complete information-report flag combination."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from ml4t import diagnostic


def test_real_information_notebook_displays_all_horizons_and_flag_combinations(tmp_path):
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    from jupyter_client import AsyncKernelManager

    recipe = Path(__file__).resolve().parents[2] / "examples/legacy_migration_comparison.py"
    code = f"""
import json,runpy
from pathlib import Path
import numpy as np
import plotly.io as pio
from IPython.display import HTML,display
from ml4t import diagnostic
from ml4t.diagnostic.signal import SignalInformationResult,analyze_signal_information,prepare_signal_panel
from ml4t.diagnostic.visualization.signal import generate_signal_information_report,plot_signal_information
pio.renderers.default='plotly_mimetype'
factor,forward,groups=runpy.run_path({str(recipe)!r})['_signal_information_inputs']()
panel=prepare_signal_panel(factor,forward,groups=groups,quantiles=3,max_loss=0)
first=factor.index.get_level_values('date')[0]
figures_count=0
for adjusted in [False,True]:
    scores=factor.loc[first].rank()
    outcomes=forward.loc[first,'1D'].copy()
    if adjusted: outcomes-=outcomes.groupby(groups.loc[first]).transform('mean')
    expected=1-6*np.sum((scores-outcomes.rank())**2)/(7*(7**2-1))
    for grouped in [False,True]:
        result=analyze_signal_information(panel,group_adjust=adjusted,by_group=grouped)
        np.testing.assert_allclose(result.get_table('ic_by_date')['ic'][0],expected,atol=1e-12,rtol=0)
        target=Path(f'information-{{int(adjusted)}}-{{int(grouped)}}.json')
        target.write_text(result.to_json())
        restored=SignalInformationResult.from_json(target.read_text())
        assert restored.get_table('prepared_data').height==224
        assert restored.get_table('ic_by_date').height==64
        assert restored.get_table('group_ic_by_date').height==(128 if grouped else 0)
        figures=plot_signal_information(restored)
        assert len(figures)==(7 if grouped else 8)
        for figure in figures: figure.show()
        display(HTML(generate_signal_information_report(restored)))
        figures_count+=len(figures)
print('ML4T_INFORMATION_NOTEBOOK:'+json.dumps({{'module':diagnostic.__file__,'figures':figures_count,'prepared_rows':224,'dates':32,'flag_combinations':4}}))
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
    nbformat.write(executed, tmp_path / "signal-information.ipynb")
    outputs = executed.cells[0].outputs
    figures = [
        out["data"]["application/vnd.plotly.v1+json"]
        for out in outputs
        if "application/vnd.plotly.v1+json" in out.get("data", {})
    ]
    assert len(figures) == 30
    html = "\n".join(out.get("data", {}).get("text/html", "") for out in outputs)
    assert html.count('class="plot-container"') == 30
    stream = "".join(out.get("text", "") for out in outputs if out.get("output_type") == "stream")
    metadata = json.loads(
        next(
            line.removeprefix("ML4T_INFORMATION_NOTEBOOK:")
            for line in stream.splitlines()
            if line.startswith("ML4T_INFORMATION_NOTEBOOK:")
        )
    )
    assert Path(metadata["module"]).resolve() == Path(diagnostic.__file__).resolve()
    assert metadata["figures"] == 30 and metadata["flag_combinations"] == 4
    assert len(list(tmp_path.glob("information-*.json"))) == 4
    print(json.dumps(metadata))
