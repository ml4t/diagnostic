"""Collecting optional PDF tests must not start a rendering subprocess."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_pdf_test_collection_does_not_launch_a_python_child():
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = Path(__file__).parents[1] / "test_visualization/test_report_generation.py"
    code = """
import json,os,sys,pytest
launched=[]
def observe(event,args):
    if event=='subprocess.Popen':
        launched.append(str(args[0]))
sys.addaudithook(observe)
status=pytest.main(['--noconftest','--collect-only','-q','-c',os.devnull,'-p','no:cacheprovider','-o','addopts=',sys.argv[1]])
python_children=[path for path in launched if os.path.normcase(os.path.realpath(path))==os.path.normcase(os.path.realpath(sys.executable))]
print(json.dumps({'collection_exit_code':status,'launched_executables':launched,'python_children':python_children}))
assert status==0
assert not python_children, 'Collecting PDF tests unexpectedly starts a rendering probe'
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    print(completed.stdout)
