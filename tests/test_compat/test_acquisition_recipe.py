"""Acquisition recipes must preserve return units, missing dates and provider ownership."""

import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/acquisition_to_portfolio_report.py"
spec = importlib.util.spec_from_file_location("acquisition_recipe", EXAMPLE)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def test_provider_derived_inputs_keep_daily_rf_and_annual_yields_separate():
    returns, factors, treasury = recipe.load_captured_inputs()
    originals = (returns.copy(deep=True), factors.copy(deep=True), treasury.copy(deep=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error", getattr(pd.errors, "Pandas4Warning", FutureWarning))
        result = recipe.analyze_acquired_inputs(returns, factors, treasury)
    expected = 453.09259033203125 / 457.67266845703125
    assert result["analysis"].compute_summary_stats().total_return == pytest.approx(expected - 1)
    np.testing.assert_allclose(result["analysis"].benchmark, [-0.0099, -0.0031, 0.0017])
    excess = (
        np.array(
            [
                453.9350891113281 / 457.67266845703125 - 1,
                452.4728698730469 / 453.9350891113281 - 1,
                453.09259033203125 / 452.4728698730469 - 1,
            ]
        )
        - 0.0002
    )
    assert result["dated_rf_sharpe"] == pytest.approx(
        excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    )
    report = result["report"]
    strategy = next(
        t for t in report.figures["Returns/Cumulative wealth"].data if t.name == "Backtest"
    )
    assert strategy.y[-1] == pytest.approx(expected)
    assert report.availability["positions"]["status"] == "not_requested"
    assert report.availability["transactions"]["status"] == "not_requested"
    availability = report.availability["acquisition_inputs"]
    assert availability["treasury_units"] == "annual_percent_yield"
    assert availability["report_statistics_risk_free"].endswith("zero RF")
    assert treasury["DGS3MO"].tolist() == [5.46, 5.48, 5.48]
    assert pd.Timestamp("2024-01-05") not in treasury.index
    pd.testing.assert_series_equal(returns, originals[0])
    pd.testing.assert_frame_equal(factors, originals[1])
    pd.testing.assert_frame_equal(treasury, originals[2])


@pytest.mark.parametrize(
    "input_name,wrong_units",
    [
        ("factors", "percent_returns"),
        ("treasury", "decimal_returns"),
    ],
)
@pytest.mark.parametrize("declaration", ["keyword", "metadata"])
def test_actual_percent_and_return_inputs_cannot_be_silently_relabelled(
    input_name, wrong_units, declaration
):
    returns, factors, treasury = recipe.load_captured_inputs()
    target = factors if input_name == "factors" else treasury
    arguments = {}
    if declaration == "keyword":
        arguments["factor_units" if input_name == "factors" else "treasury_units"] = wrong_units
    else:
        target.attrs["units"] = wrong_units
    with pytest.raises(ValueError, match="Expected declared"):
        recipe.analyze_acquired_inputs(returns, factors, treasury, **arguments)


def test_benchmark_does_not_substitute_market_excess_for_total_return():
    returns, factors, treasury = recipe.load_captured_inputs()
    with pytest.raises(ValueError, match="separate Mkt-RF and RF"):
        recipe.analyze_acquired_inputs(returns, factors.drop(columns="RF"), treasury)


def test_dated_rf_recipe_matches_actual_published_empyrical_on_the_shared_sample():
    if importlib.util.find_spec("empyrical") is None:
        if os.environ.get("ML4T_REQUIRED_ACQUISITION_REFERENCE"):
            pytest.fail("Required published Empyrical reference is absent")
        pytest.skip("Published Empyrical is an optional qualification reference")
    import empyrical

    returns, factors, treasury = recipe.load_captured_inputs()
    result = recipe.analyze_acquired_inputs(returns, factors, treasury)
    expected = empyrical.sharpe_ratio(
        returns, risk_free=factors["RF"].reindex(returns.index), annualization=252
    )
    assert result["dated_rf_sharpe"] == pytest.approx(expected, rel=1e-12)


def test_captured_cli_writes_standalone_report_without_optional_producer(tmp_path):
    target = tmp_path / "report.html"
    completed = subprocess.run(
        [sys.executable, "-I", str(EXAMPLE), "--output", str(target)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert "captured provider-derived CSVs; no network" in completed.stdout
    assert "plotly" in target.read_text().lower()
    payload = completed.stdout.split("\nWrote acquired-input", 1)[0]
    assert json.loads(payload)["return_observations"] == 3


@pytest.mark.parametrize(
    "symbol,start,end",
    [
        (" ", "2024-01-02", "2024-01-06"),
        ("SPY", "2024-01-06", "2024-01-02"),
        ("SPY", "not-a-date", "2024-01-06"),
    ],
)
def test_invalid_fetch_request_is_rejected_before_optional_import_or_cache_write(
    symbol, start, end, tmp_path
):
    cache = tmp_path / "absent-cache"
    with pytest.raises(ValueError):
        recipe.fetch_acquired_inputs(symbol, start, end, cache)
    assert not cache.exists()


def test_invalid_cli_scale_does_not_create_output(tmp_path):
    output = tmp_path / "absent.html"
    completed = subprocess.run(
        [sys.executable, "-I", str(EXAMPLE), "--periods-per-year", "0", "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "must be positive" in completed.stderr
    assert not output.exists()


def test_fetch_recipe_uses_actual_data_parsers_caches_and_explicit_fred_requests(
    tmp_path, monkeypatch
):
    captures = os.environ.get("ML4T_ACQUISITION_CAPTURE_DIRECTORY")
    if not captures:
        if os.environ.get("ML4T_REQUIRED_ACQUISITION_RECIPE"):
            pytest.fail("Required provider-derived recipe captures are absent")
        pytest.skip("Captured Data provider responses are required for optional producer replay")
    if importlib.util.find_spec("ml4t.data") is None:
        if os.environ.get("ML4T_REQUIRED_ACQUISITION_RECIPE"):
            pytest.fail("Required published Data producer is absent")
        pytest.skip("The Data producer is an optional example dependency")
    import httpx
    from ml4t.data.core.exceptions import NetworkError
    from ml4t.data.providers import FamaFrenchProvider, yahoo

    capture_directory = Path(captures)
    capture = json.loads((capture_directory / "yahoo-adjustment-provider.json").read_text())
    prices = capture["records"]["default"]
    adjusted = pd.DataFrame(
        prices["data"],
        index=pd.DatetimeIndex(prices["index"]),
        columns=pd.MultiIndex.from_tuples(prices["columns"]),
    )
    calls = []

    def download(symbol, **kwargs):
        assert symbol == "SPY"
        assert pd.Timestamp(kwargs["start"]) == pd.Timestamp("2024-01-02")
        assert pd.Timestamp(kwargs["end"]) == pd.Timestamp("2024-01-06")
        assert kwargs["auto_adjust"] and kwargs["keepna"] and not kwargs["actions"]
        calls.append("yahoo")
        return adjusted.copy(deep=True)

    def get(_client, url, **kwargs):
        if str(url).startswith(FamaFrenchProvider.BASE_URL + "/"):
            body = (capture_directory / str(url).rsplit("/", 1)[-1]).read_bytes()
        else:
            assert str(url) == "https://fred.stlouisfed.org/graph/fredgraph.csv"
            assert kwargs["params"] == {"id": "DGS3MO", "cosd": "2024-01-02", "coed": "2024-01-06"}
            body = (capture_directory / "fred-dgs3mo.csv").read_bytes()
        calls.append(str(url))
        return httpx.Response(200, content=body, request=httpx.Request("GET", url))

    monkeypatch.setenv("FRED_API_KEY", "")
    monkeypatch.setattr(yahoo.yf, "download", download)
    monkeypatch.setattr(httpx.Client, "get", get)
    cache = tmp_path / "request-cache"
    acquired = recipe.fetch_acquired_inputs("SPY", "2024-01-02", "2024-01-06", cache)
    assert len(calls) == 4
    for observed, expected in zip(acquired, recipe.load_captured_inputs(), strict=True):
        if isinstance(observed, pd.Series):
            pd.testing.assert_series_equal(observed, expected, check_freq=False)
        else:
            pd.testing.assert_frame_equal(observed, expected, check_freq=False)
    hit = recipe.fetch_acquired_inputs("SPY", "2024-01-02", "2024-01-06", cache)
    assert len(calls) == 5  # FRED is explicit, not part of the legacy CSV caches.
    result = recipe.analyze_acquired_inputs(*hit)
    assert result["analysis"].compute_summary_stats().total_return == pytest.approx(
        453.09259033203125 / 457.67266845703125 - 1
    )
    for path in cache.glob("*.csv"):
        os.utime(path, (0, 0))
    recipe.fetch_acquired_inputs("SPY", "2024-01-02", "2024-01-06", cache)
    assert len(calls) == 9

    def offline(*args, **kwargs):
        raise httpx.ConnectError(
            "offline transport", request=httpx.Request("GET", "https://fred.stlouisfed.org")
        )

    monkeypatch.setattr(httpx.Client, "get", offline)
    monkeypatch.setattr(yahoo.yf, "download", offline)
    with pytest.raises(NetworkError, match="offline transport"):
        recipe.fetch_acquired_inputs("SPY", "2024-01-02", "2024-01-06", cache)
    # Offline replay is the bundled/caller-supplied input route, not hidden FRED caching.
    recipe.analyze_acquired_inputs(*recipe.load_captured_inputs())
