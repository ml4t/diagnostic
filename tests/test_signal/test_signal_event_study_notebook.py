"""Execute the original event notebook's analytical sequence in an actual kernel."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from ml4t import diagnostic


def test_real_event_study_notebook_executes_threshold_one_bin_seven_horizons_and_both_signs(
    tmp_path,
):
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    pytest.importorskip("plotly")
    from jupyter_client import AsyncKernelManager

    recipe = Path(__file__).resolve().parents[2] / "examples/legacy_migration_comparison.py"
    code = f"""
import json,runpy
from pathlib import Path
import plotly.io as pio
from IPython.display import HTML,display
from ml4t import diagnostic
from ml4t.diagnostic.signal import SignalEventStudyResult,analyze_signal_event_study,prepare_signal_panel
from ml4t.diagnostic.visualization.signal import generate_signal_event_study_report,plot_signal_event_study
pio.renderers.default='plotly_mimetype'
pricing,events,forward=runpy.run_path({str(recipe)!r})['_signal_event_study_inputs']()
cases=[]
for name,direction in [('positive',1),('negative',-1)]:
    panel=prepare_signal_panel(events*direction,forward,quantiles=None,bins=1,max_loss=0)
    result=analyze_signal_event_study(panel,pricing,input_kind='levels',event_window=(5,10),rate_conversion_factors={{f'{{period}}D':1/period for period in [1,2,3,4,5,6,10]}})
    Path(name+'.json').write_text(result.to_json())
    restored=SignalEventStudyResult.from_json(Path(name+'.json').read_text())
    assert restored.prepared_events.equals(result.prepared_events)
    assert restored.event_membership.equals(result.event_membership)
    assert restored.event_returns.overall.paths.equals(result.event_returns.overall.paths)
    for figure in plot_signal_event_study(result): figure.show()
    display(HTML(generate_signal_event_study_report(result)))
    cases.append({{'direction':direction,'events':panel.data.height,'paths':result.event_returns.overall.paths.height,'factor_mean':result.factor_statistics['mean'][0],'horizons':result.metadata['return_columns']}})
print('ML4T_SIGNAL_EVENT_NOTEBOOK:'+json.dumps({{'module':diagnostic.__file__,'cases':cases}}))
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
    path = tmp_path / "signal-event-study.ipynb"
    nbformat.write(executed, path)
    outputs = executed.cells[0].outputs
    figures = [
        out["data"]["application/vnd.plotly.v1+json"]
        for out in outputs
        if "application/vnd.plotly.v1+json" in out.get("data", {})
    ]
    assert len(figures) == 10
    html = "\n".join(out.get("data", {}).get("text/html", "") for out in outputs)
    assert html.count('class="plot-container"') == 10
    assert html.count("Factor statistics") >= 2
    first_day_mean = (
        sum(
            [
                32 / 29 - 1,
                33 / 29 - 1,
                31 / 28 - 1,
                32 / 27 - 1,
                30 / 27 - 1,
                34 / 28 - 1,
                33 / 29 - 1,
                35 / 26 - 1,
                35 / 28 - 1,
            ]
        )
        / 9
    )
    for position in [3, 8]:
        assert len(figures[position]["data"]) == 7
        assert figures[position]["data"][0]["y"][0] == pytest.approx(first_day_mean, abs=1e-12)
        assert figures[position]["layout"]["showlegend"] is True
    for position in [4, 9]:
        assert all(len(trace["y"]) == 9 for trace in figures[position]["data"])
    stream = "".join(out.get("text", "") for out in outputs if out.get("output_type") == "stream")
    metadata = json.loads(
        next(
            line.removeprefix("ML4T_SIGNAL_EVENT_NOTEBOOK:")
            for line in stream.splitlines()
            if line.startswith("ML4T_SIGNAL_EVENT_NOTEBOOK:")
        )
    )
    assert Path(metadata["module"]).resolve() == Path(diagnostic.__file__).resolve()
    assert [row["events"] for row in metadata["cases"]] == [9, 9]
    assert [row["paths"] for row in metadata["cases"]] == [144, 144]
    assert metadata["cases"][0]["factor_mean"] == pytest.approx(251 / 9, abs=1e-12)
    assert metadata["cases"][1]["factor_mean"] == pytest.approx(-251 / 9, abs=1e-12)
    assert all(
        row["horizons"] == ["1D", "2D", "3D", "4D", "5D", "6D", "10D"] for row in metadata["cases"]
    )
    assert (tmp_path / "positive.json").is_file() and (tmp_path / "negative.json").is_file()
    print(
        json.dumps(
            {
                "module": metadata["module"],
                "mime_figures": 10,
                "html_panels": 10,
                "events_per_sign": 9,
                "path_cells_per_sign": 144,
                "horizons": 7,
                "factor_means": [row["factor_mean"] for row in metadata["cases"]],
            }
        )
    )
