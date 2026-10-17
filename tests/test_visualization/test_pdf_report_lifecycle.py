"""Actual multipage PDF export must share and close the modern renderer."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_multipage_pdf_uses_one_real_renderer_and_closes_it_after_all_pages(scale, tmp_path):
    kaleido = pytest.importorskip("kaleido")
    pio = pytest.importorskip("plotly.io")
    pytest.importorskip("pypdf")
    if not callable(getattr(pio, "write_images", None)) or not callable(
        getattr(kaleido, "write_fig_sync", None)
    ):
        pytest.skip("Modern batch lifecycle requires Plotly write_images and Kaleido1")
    code = """
import hashlib,json,sys
from importlib.metadata import version
from pathlib import Path
import kaleido
import plotly.graph_objects as go
from pypdf import PdfReader
from ml4t import diagnostic
from ml4t.diagnostic.visualization import export_figures_to_pdf
instances=[]
original_init=kaleido.Kaleido.__init__
def observe_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    instances.append(self)
kaleido.Kaleido.__init__=observe_init
figures=[go.Figure(go.Bar(x=['A','B'],y=[1,-2])),
         go.Figure(go.Scatter(x=[1,2,3],y=[.85,.92,.78])),
         go.Figure(go.Histogram(x=[0.,.1,-.05,.02]))]
titles=['Holdings values','Method agreement','Observed returns']
for figure,title in zip(figures,titles):
    figure.update_layout(title=title)
before=[figure.to_json() for figure in figures]
path=export_figures_to_pdf(figures,sys.argv[1],page_size=(800,600),scale=float(sys.argv[2]))
reader=PdfReader(path)
texts=[page.extract_text() for page in reader.pages]
assert len(texts)==3 and all(title in text for title,text in zip(titles,texts))
assert [figure.to_json() for figure in figures]==before
print(json.dumps({'module':diagnostic.__file__,
    'versions':{name:version(name) for name in ['ml4t-diagnostic','plotly','kaleido','choreographer','pypdf']},
    'renderer_instances':len(instances),'renderer_returncodes':[instance.subprocess.poll() for instance in instances],
    'page_boxes':[[float(value) for value in page.mediabox] for page in reader.pages],
    'page_text':texts,'pdf_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    'original_figures_unchanged':True}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "report.pdf"), str(scale)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["renderer_instances"] == 1, result
    assert all(code is not None for code in result["renderer_returncodes"]), result
    assert result["page_boxes"] == [[0, 0, 600 * scale, 450 * scale]] * 3
    print(json.dumps(result))
