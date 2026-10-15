"""Report builders must retain one declared scale at every sample length."""

import json
import re
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.visualization.backtest import (
    BacktestTearsheet,
    export_workspaces,
    generate_backtest_tearsheet,
)


def report_inputs(n):
    returns = np.random.default_rng(15).normal(0.001, 0.01, n)
    dates = [datetime(2025, 1, 2) + timedelta(days=i) for i in range(n)]
    equity = pl.DataFrame({"timestamp": dates, "equity": 100 * np.cumprod(1 + returns)})
    return returns, equity


def chart_payloads(html, section):
    start = html.index(f'data-section="{section}"')
    tail = html[start:]
    end = tail.find('data-section="', 1)
    if end >= 0:
        tail = tail[:end]
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', tail)
    assert match, section
    decoder = json.JSONDecoder()
    data, offset = decoder.raw_decode(tail[match.end() :])
    layout, _ = decoder.raw_decode(tail[match.end() + offset :].lstrip()[1:].lstrip())
    return data, layout


@pytest.mark.parametrize("n", [80, 160])
def test_short_daily_report_does_not_reinterpret_sample_length_as_monthly(n):
    returns, equity = report_inputs(n)
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    metrics = {"sharpe_ratio": expected_sharpe}
    generate_backtest_tearsheet(
        returns=returns, equity_curve=equity, metrics=metrics, template="risk_manager"
    )
    assert metrics["sharpe_ratio"] == expected_sharpe
    assert metrics["annual_volatility"] == pytest.approx(returns.std(ddof=1) * np.sqrt(252))


@pytest.mark.parametrize("scale", [12, 252, 365])
@pytest.mark.parametrize("route", ["function", "builder", "export"])
def test_declared_scale_reaches_relative_metrics_and_actual_chart_data(scale, route):
    returns, equity = report_inputs(300)
    benchmark = returns * 0.5 + 0.0001
    metrics = {}
    if route == "function":
        html = generate_backtest_tearsheet(
            returns=returns,
            equity_curve=equity,
            metrics=metrics,
            benchmark_returns=benchmark,
            periods_per_year=scale,
        )
    elif route == "builder":
        builder = BacktestTearsheet(periods_per_year=scale)
        builder.add_returns(returns).add_equity_curve(equity).add_benchmark(benchmark)
        html = builder.generate()
        metrics = builder.metrics
    else:
        html = export_workspaces(
            ("overview", "performance", "validation", "trading"),
            returns=returns,
            equity_curve=equity,
            metrics=metrics,
            benchmark_returns=benchmark,
            periods_per_year=scale,
        )
    assert metrics["annual_volatility"] == pytest.approx(returns.std(ddof=1) * np.sqrt(scale))
    assert metrics["sharpe_ratio"] == pytest.approx(
        returns.mean() / returns.std(ddof=1) * np.sqrt(scale)
    )
    assert metrics["annual_return"] == pytest.approx(np.prod(1 + returns) ** (scale / 300) - 1)
    assert metrics["alpha"] == pytest.approx(-0.0002 * scale, abs=1e-12)
    assert metrics["tracking_error"] == pytest.approx(
        (returns - benchmark).std(ddof=1) * np.sqrt(scale)
    )
    sr_period = returns.mean() / returns.std(ddof=1)
    expected_ci_se = np.sqrt((1 + sr_period**2 / 2) / len(returns)) * np.sqrt(scale)
    assert metrics["sharpe_upper_95"] == pytest.approx(metrics["sharpe"] + 1.96 * expected_ci_se)
    centered = returns - returns.mean()
    m2 = np.mean(centered**2)
    skew = np.mean(centered**3) / m2**1.5
    kurt = np.mean(centered**4) / m2**2
    expected_se = np.sqrt(
        (1 - skew * sr_period + (kurt - 1) * sr_period**2 / 4) / len(returns)
    ) * np.sqrt(scale)
    assert metrics["sharpe_se"] == pytest.approx(expected_se)
    assert f'name="ml4t-periods-per-year" content="{scale}"' in html

    data, _ = chart_payloads(html, "rolling_metrics")
    window = returns[-252:]
    assert data[1]["y"][-1] == pytest.approx(window.mean() / window.std(ddof=1) * np.sqrt(scale))
    assert data[2]["y"][-1] == pytest.approx(window.std(ddof=1) * np.sqrt(scale))
    assert len(data[2]["y"]) == 300
    np.testing.assert_array_equal(
        np.isnan(np.asarray(data[2]["y"], dtype=float)), np.arange(300) < 251
    )

    cost_data, _ = chart_payloads(html, "cost_sensitivity")
    assert cost_data[0]["y"][0] == pytest.approx(
        returns.mean() / returns.std(ddof=1) * np.sqrt(scale)
    )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, 12.5])
def test_invalid_report_scale_does_not_mutate_metrics_or_publish_output(scale, tmp_path):
    returns, equity = report_inputs(80)
    metrics = {"sharpe_ratio": 1.5}
    output = tmp_path / "report.html"
    output.write_text("previous report")
    with pytest.raises(ValueError, match="periods_per_year"):
        generate_backtest_tearsheet(
            returns=returns,
            equity_curve=equity,
            metrics=metrics,
            periods_per_year=scale,
            output_path=output,
        )
    assert metrics == {"sharpe_ratio": 1.5}
    assert output.read_text() == "previous report"


@pytest.mark.parametrize("scale", [None, 12, 365])
def test_factor_table_scales_intercept_and_its_standard_error_identically(scale):
    from ml4t.diagnostic.evaluation.factor import FactorData

    factor, equity = report_inputs(80)
    noise = np.random.default_rng(19).normal(0, 0.002, 80)
    returns = 0.0003 + 0.75 * factor + noise
    factor_data = FactorData.from_dataframe(
        equity.select("timestamp").with_columns(pl.Series("f", factor))
    )
    metrics = {}
    kwargs = {} if scale is None else {"periods_per_year": scale}
    html = generate_backtest_tearsheet(
        returns=returns, equity_curve=equity, metrics=metrics, factor_data=factor_data, **kwargs
    )
    design = np.column_stack([np.ones(80), factor])
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ returns
    scores = design * (returns - design @ beta)[:, None]
    sandwich = scores.T @ scores
    lags = max(1, int(4 * (80 / 100) ** (2 / 9)))
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        sandwich += (1 - lag / (lags + 1)) * (cross + cross.T)
    expected_se = np.sqrt((inverse @ sandwich @ inverse)[0, 0])
    alpha_row = re.search(r"Alpha \(annlzd\)</em></td>(.*?)</tr>", html).group(1)
    values = re.findall(r'<td class="num[^\"]*">([^<]+)</td>', alpha_row)
    resolved = 252 if scale is None else scale
    assert float(values[0]) == pytest.approx(beta[0] * resolved, abs=0.000051)
    assert float(values[1]) == pytest.approx(expected_se * resolved, abs=0.000051)


@pytest.mark.parametrize("route", ["function", "builder", "export", "integration"])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_profile_report_inherits_or_overrides_scale_without_changing_profile(route):
    from ml4t.backtest import BacktestResult

    from ml4t.diagnostic.integration import analyze_backtest_result, generate_tearsheet_from_result

    _, equity = report_inputs(300)
    result = BacktestResult(trades=[], equity_curve=equity.rows(), fills=[], metrics={})
    profile = analyze_backtest_result(result, calendar="NYSE")
    original_scale = profile.periods_per_year
    original_stats = profile.performance["metrics"].copy()
    returns = profile.daily_returns.to_numpy()
    if route == "function":
        html = generate_backtest_tearsheet(profile=profile, periods_per_year=365)
    elif route == "builder":
        html = BacktestTearsheet(periods_per_year=365).add_profile(profile).generate()
    elif route == "export":
        html = export_workspaces("full_report", profile=profile, periods_per_year=365)
    else:
        html = generate_tearsheet_from_result(result, calendar="NYSE", periods_per_year=365)
    data, _ = chart_payloads(html, "rolling_metrics")
    expected_vol = returns[-252:].std(ddof=1) * np.sqrt(365)
    assert data[2]["y"][-1] == pytest.approx(expected_vol)
    assert profile.periods_per_year == original_scale
    assert profile.performance["metrics"] == original_stats
    inherited = generate_backtest_tearsheet(profile=profile)
    inherited_data, _ = chart_payloads(inherited, "rolling_metrics")
    assert inherited_data[2]["y"][-1] == pytest.approx(
        returns[-252:].std(ddof=1) * np.sqrt(original_scale)
    )
