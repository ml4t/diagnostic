"""Notebook display must retain prepared data and explain unavailable report components."""

import asyncio
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from ml4t import diagnostic


def test_real_notebook_kernel_displays_holdings_and_the_same_availability_as_html(tmp_path):
    nbclient = pytest.importorskip("nbclient", reason="Notebook execution is checked in package CI")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    from jupyter_client import AsyncKernelManager

    code = """
import json
import pandas as pd
import plotly.io as pio
from ml4t import diagnostic
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import create_portfolio_dashboard
pio.renderers.default = 'plotly_mimetype'
dates = pd.to_datetime(['2025-01-02','2025-01-03','2025-01-06'])
positions = pd.DataFrame({'A':[0.,400.,400.],'B':[0.,-300.,-300.]},index=dates)
valuations = pd.DataFrame({'equity':[1000.]*3,'cash':[1000.,900.,900.]},index=dates)
analysis = PortfolioAnalysis.from_frames(pd.Series([0.]*3,index=dates),positions=positions,valuations=valuations,complete_snapshots=True,calendar=None)
holdings = analysis.compute_holdings_analysis(valuations,complete_snapshots=True,sector_mapping={'A':'Technology'})
stress = analysis.compute_stress({'selected':(dates[0],dates[1]),'unobserved':('1990','1990')})
live = analysis.compute_live_split(live_start=dates[2])
cones = analysis.compute_wealth_cones(live_start=dates[2],horizon=2,n_samples=4,initial_capital=1000)
market = pd.DataFrame([{'date':when,'asset':asset,'price':10.,'volume':10000.} for when in dates for asset in ('A','B')])
capacity = analysis.compute_capacity(market,currency='USD',price_basis='raw',volume_basis='raw',lookback=1)
report = create_portfolio_dashboard(analysis,holdings=holdings,include_transactions=False,stress=stress,live=live,wealth_cones=cones,capacity=capacity)
report.show()
print('ML4T_NOTEBOOK:'+json.dumps({'module':diagnostic.__file__,'availability':report.availability,'holdings':report.to_dict()['holdings'],'stress_counts':report.stress.coverage['n_observations'].to_list(),'n_training':report.wealth_cones.training.height,'n_paths':report.wealth_cones.paths.height},default=str))
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
        client = nbclient.NotebookClient(notebook, km=manager, timeout=120, record_timing=False)
        try:
            return await client.async_execute()
        finally:
            if client.kc is not None:
                client.kc.stop_channels()
            if manager.has_kernel:
                await manager.shutdown_kernel(now=True)

    executed = asyncio.run(execute())
    path = tmp_path / "portfolio-report.ipynb"
    nbformat.write(executed, path)
    outputs = executed.cells[0].outputs
    figures = [
        out["data"]["application/vnd.plotly.v1+json"]
        for out in outputs
        if "application/vnd.plotly.v1+json" in out.get("data", {})
    ]
    html = "\n".join(out.get("data", {}).get("text/html", "") for out in outputs)
    stream = "".join(out.get("text", "") for out in outputs if out.get("output_type") == "stream")
    metadata = json.loads(
        next(
            line.removeprefix("ML4T_NOTEBOOK:")
            for line in stream.splitlines()
            if line.startswith("ML4T_NOTEBOOK:")
        )
    )
    assert Path(metadata["module"]).resolve() == Path(diagnostic.__file__).resolve()
    assert len(figures) == 22
    prepared = [
        fig
        for fig in figures
        if (fig["layout"].get("meta") or {}).get("analysis") == "prepared_holdings"
    ]
    assert len(prepared) == 9
    exposure = next(
        fig for fig in prepared if fig["layout"]["title"]["text"] == "Holdings Exposure"
    )
    assert next(trace["y"] for trace in exposure["data"] if trace["name"] == "Gross") == [
        0,
        0.7,
        0.7,
    ]
    assert next(trace["y"] for trace in exposure["data"] if trace["name"] == "Short (signed)") == [
        0,
        -0.3,
        -0.3,
    ]
    assert metadata["holdings"]["timeline"][-1]["cash"] == 900
    assert "Analysis availability" in html
    assert "Transaction analysis was not requested" in html
    assert "Missing labels remain unclassified" in html
    assert metadata["availability"]["normal_fit"]["status"] == "unavailable"
    assert "zero dispersion" in html
    assert "Stress windows" in html and "Live split" in html
    assert "Worst liquidation: full/recent observations" in html
    assert "unobserved" in html and "n_observations" in html
    assert metadata["stress_counts"] == [2, 0]
    assert metadata["n_training"] == 2 and metadata["n_paths"] == 8
    cone_figure = next(
        fig
        for fig in figures
        if (fig["layout"].get("meta") or {}).get("analysis") == "wealth_cones"
    )
    assert next(
        trace["y"] for trace in cone_figure["data"] if trace["name"] == "Actual live wealth"
    ) == [1000]
    distribution = next(
        fig for fig in figures if fig["layout"]["title"]["text"] == "Returns Distribution"
    )
    assert not any(trace["name"].startswith("Normal") for trace in distribution["data"])
    print(
        json.dumps(
            {
                "module": metadata["module"],
                "versions": {
                    name: version(name)
                    for name in [
                        "ml4t-diagnostic",
                        "plotly",
                        "pandas",
                        "nbclient",
                        "nbformat",
                        "ipykernel",
                        "ipython",
                    ]
                },
                "mime_figures": len(figures),
                "prepared_figures": len(prepared),
                "availability_html": True,
                "regime_tables_visible": True,
                "cone_training_observations": metadata["n_training"],
                "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "holdings": metadata["holdings"],
            },
            default=str,
        )
    )
