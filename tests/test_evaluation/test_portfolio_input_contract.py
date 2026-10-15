"""Public regressions for portfolio sample identity and input ownership."""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis import (
    alpha_beta,
    information_ratio,
    up_down_capture,
)
from ml4t.diagnostic.visualization.portfolio import (
    plot_cumulative_returns,
    plot_monthly_returns_heatmap,
)


def test_labeled_returns_keep_real_months_in_tables_and_rendered_axes():
    returns = pd.Series([0.1, -0.1], index=pd.to_datetime(["2025-01-31", "2025-02-03"]))
    analysis = PortfolioAnalysis(returns)
    monthly = analysis.compute_monthly_returns()
    assert monthly["year"].to_list() == [2025, 2025]
    assert monthly["month"].to_list() == [1, 2]
    np.testing.assert_allclose(monthly["monthly_return"], [0.1, -0.1])
    figure = plot_cumulative_returns(analysis)
    assert list(figure.data[0].x) == list(returns.index.to_pydatetime())
    heatmap = plot_monthly_returns_heatmap(analysis)
    assert list(heatmap.data[0].y) == [2025]
    np.testing.assert_allclose(np.asarray(heatmap.data[0].z)[0, :2], [0.1, -0.1])


def test_timezone_and_nanosecond_input_dates_survive_without_day_conversion():
    index = pd.DatetimeIndex(["2024-12-31 23:59:59.000000001", "2025-01-01"]).tz_localize(
        "America/New_York"
    )
    analysis = PortfolioAnalysis(pd.Series([0.01, 0.02], index=index))
    assert analysis.dates.dtype == pl.Datetime("ns", "America/New_York")
    np.testing.assert_array_equal(analysis.dates.cast(pl.Int64).to_numpy(), index.asi8)
    assert analysis.compute_annual_returns()["year"].to_list() == [2024, 2025]


@pytest.mark.parametrize("as_polars", [False, True])
def test_explicit_matching_dates_retain_labeled_sample(as_polars):
    index = pd.date_range("2025-01-31", periods=2)
    dates = pl.from_pandas(index) if as_polars else index
    analysis = PortfolioAnalysis(pd.Series([0.1, -0.1], index=index), dates=dates)
    assert analysis.dates.to_list() == list(index.to_pydatetime())


def test_explicit_dates_cannot_relabel_a_dated_return_series():
    returns = pd.Series([0.1, -0.1], index=pd.date_range("2025-01-31", periods=2))
    with pytest.raises(ValueError, match="dates.*(match|agree)"):
        PortfolioAnalysis(returns, dates=pd.date_range("2024-01-31", periods=2))


def test_undated_arrays_do_not_invent_calendar_tables_or_reports(tmp_path):
    analysis = PortfolioAnalysis([0.1, -0.1, 0.02])
    assert analysis.dates.dtype == pl.Int64
    assert analysis.dates.to_list() == [0, 1, 2]
    assert analysis.compute_summary_stats().total_return == pytest.approx(0.0098)
    figure = plot_cumulative_returns(analysis)
    assert list(figure.data[0].x) == [0, 1, 2]
    assert figure.layout.xaxis.title.text == "Observation"
    for method in (analysis.compute_monthly_returns, analysis.compute_annual_returns):
        with pytest.raises(ValueError, match="dates"):
            method()
    output = tmp_path / "report.html"
    output.write_text("existing report")
    with pytest.raises(ValueError, match="dates"):
        analysis.create_tear_sheet().save_html(output)
    assert output.read_text() == "existing report"


def test_undated_backtest_report_marks_calendar_panels_unavailable_instead_of_inventing_dates():
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    html = generate_backtest_tearsheet(
        returns=np.random.default_rng(116).normal(0.001, 0.01, 80), template="full"
    )
    assert 'data-availability="missing-dates"' in html
    assert "Calendar returns unavailable" in html
    assert "2000-01" not in html


@pytest.mark.parametrize("relative_count", [79, 81])
def test_raw_report_rejects_unequal_relative_sample_before_metrics_or_file_changes(
    tmp_path, relative_count
):
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    output = tmp_path / "existing.html"
    output.write_text("existing report")
    metrics = {"producer_stat": 1.0}
    with pytest.raises(ValueError, match="benchmark"):
        generate_backtest_tearsheet(
            returns=np.random.default_rng(116).normal(0.001, 0.01, 80),
            benchmark_returns=np.random.default_rng(117).normal(0.001, 0.01, relative_count),
            metrics=metrics,
            output_path=output,
        )
    assert metrics == {"producer_stat": 1.0}
    assert output.read_text() == "existing report"


def test_reference_permutation_and_extra_dates_keep_one_paired_sample():
    benchmark = pd.Series(
        [0.01, -0.02, 0.03, 0, -0.01, 0.02], index=pd.bdate_range("2025-01-02", periods=6)
    )
    extra = pd.Series([0.5], index=pd.to_datetime(["2024-12-31"]))
    reordered = pd.concat([extra, benchmark.iloc[[2, 5, 0, 4, 1, 3]]])
    expected = PortfolioAnalysis(2 * benchmark, benchmark=benchmark)
    actual = PortfolioAnalysis(2 * benchmark, benchmark=reordered)
    assert actual.compute_summary_stats().beta == pytest.approx(2)
    np.testing.assert_array_equal(actual.benchmark, benchmark.to_numpy())
    assert actual.compute_summary_stats().to_dict() == expected.compute_summary_stats().to_dict()
    rolling = actual.compute_rolling_metrics(windows=[3], metrics=["beta"])
    np.testing.assert_allclose(rolling.beta[3].drop_nulls(), [2, 2, 2, 2])
    assert rolling.dates.to_list() == list(benchmark.index.to_pydatetime())


@pytest.mark.parametrize("problem", ["missing", "duplicate", "undated_strategy", "timezone"])
def test_reference_labels_cannot_be_silently_dropped(problem):
    index = pd.date_range("2025-01-01", periods=3)
    returns = pd.Series([0.02, -0.04, 0.06], index=index)
    benchmark = pd.Series([0.01, -0.02, 0.03], index=index)
    if problem == "missing":
        benchmark = benchmark.drop(index[1])
    elif problem == "duplicate":
        benchmark = pd.concat([benchmark, benchmark.iloc[:1]])
    elif problem == "undated_strategy":
        returns = returns.to_numpy()
    else:
        benchmark.index = benchmark.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="benchmark"):
        PortfolioAnalysis(returns, benchmark=benchmark)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"returns": pd.DataFrame({"a": [0.01, 0.02], "b": [0.03, 0.04]})}, "returns"),
        ({"returns": pd.Series([0.01], index=[0]).reindex([0, 1])}, "returns"),
        ({"returns": pl.Series([0.01, None])}, "returns"),
        ({"returns": [0.01, np.inf]}, "returns"),
        ({"returns": [0.01, -1.01]}, "returns"),
        ({"returns": [True, False]}, "returns"),
        ({"returns": ["0.01", "0.02"]}, "returns"),
        ({"returns": []}, "returns"),
        ({"returns": [np.nan, np.nan]}, "returns"),
        ({"dates": pd.date_range("2025-01-01", periods=2)}, "dates"),
        ({"dates": [date(2025, 1, 1)] * 3}, "dates"),
        ({"dates": [date(2025, 1, 3), date(2025, 1, 1), date(2025, 1, 2)]}, "dates"),
        ({"dates": [date(2025, 1, 1), None, date(2025, 1, 3)]}, "dates"),
        ({"dates": [1, 2, 3]}, "dates"),
        pytest.param({"benchmark": [0.01]}, "benchmark", id="short-relative-series"),
        pytest.param({"benchmark": [0.01, np.nan, 0.03]}, "benchmark", id="missing-relative-value"),
        ({"periods_per_year": 0}, "periods_per_year"),
        ({"periods_per_year": -1}, "periods_per_year"),
        ({"periods_per_year": 12.5}, "periods_per_year"),
        ({"periods_per_year": True}, "periods_per_year"),
        ({"risk_free": -1}, "risk_free"),
        ({"risk_free": np.inf}, "risk_free"),
    ],
)
def test_invalid_pipeline_inputs_fail_at_constructor_boundary(kwargs, field):
    args = {"returns": np.array([0.01, -0.02, 0.03]), **kwargs}
    with pytest.raises((ValueError, TypeError), match=field):
        PortfolioAnalysis(**args)


@pytest.mark.parametrize("kwargs", [{"windows": [0]}, {"windows": [1.5]}, {"metrics": ["typo"]}])
def test_invalid_rolling_requests_do_not_produce_empty_or_misleading_results(kwargs):
    analysis = PortfolioAnalysis([0.1, -0.1, 0.02])
    with pytest.raises((ValueError, TypeError), match="windows|metrics"):
        analysis.compute_rolling_metrics(**kwargs)


@pytest.mark.parametrize("function", [alpha_beta, information_ratio, up_down_capture])
def test_relative_metric_functions_reject_unequal_arrays_instead_of_prefix_truncation(function):
    with pytest.raises(ValueError, match="(benchmark|length)"):
        function([0.1, -0.1, 0.02], [0.01, -0.01])


@pytest.mark.parametrize("container", [np.array, pd.Series, pl.Series])
def test_caller_mutation_does_not_change_owned_input_or_cached_results(container):
    returns = container([0.1, -0.05, 0.02])
    benchmark = container([0.03, -0.02, 0.01])
    analysis = PortfolioAnalysis(returns, benchmark=benchmark)
    analysis.compute_summary_stats()
    returns[0] = 0.5
    benchmark[0] = 0.5
    np.testing.assert_allclose(analysis.returns, [0.1, -0.05, 0.02])
    np.testing.assert_allclose(analysis.benchmark, [0.03, -0.02, 0.01])
    assert analysis.compute_summary_stats(force_recompute=True).total_return == pytest.approx(
        0.0659
    )
    exposed = analysis.returns
    exposed[0] = 0.7
    exposed_benchmark = analysis.benchmark
    exposed_benchmark[0] = 0.7
    np.testing.assert_allclose(analysis.returns, [0.1, -0.05, 0.02])
    np.testing.assert_allclose(analysis.benchmark, [0.03, -0.02, 0.01])


def test_mutating_returned_results_cannot_poison_summary_rolling_or_drawdown_caches():
    analysis = PortfolioAnalysis([0.1, -0.05, 0.02])
    metrics = analysis.compute_summary_stats()
    metrics.total_return = 999
    assert analysis.compute_summary_stats().total_return == pytest.approx(0.0659)
    rolling = analysis.compute_rolling_metrics(windows=[2], metrics=["returns"])
    rolling.returns[2][2] = 999
    rolling.windows.append(999)
    rolling.dates[0] = 999
    original = analysis.compute_rolling_metrics(windows=[2], metrics=["returns"])
    assert original.windows == [2]
    assert original.dates[0] == 0
    assert original.returns[2][2] == pytest.approx(-0.031)
    drawdown = analysis.compute_drawdown_analysis()
    drawdown.top_drawdowns[0].depth = -999
    drawdown.underwater_curve[1] = -999
    assert analysis.compute_drawdown_analysis().top_drawdowns[0].depth == pytest.approx(-0.05)
    assert analysis.compute_drawdown_analysis().underwater_curve[1] == pytest.approx(-0.05)


@pytest.mark.parametrize("returns", [[0.0], [0.0, 0.0, 0.0], [0.01]])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_finite_cash_and_one_observation_inputs_have_defined_availability(returns):
    with np.errstate(all="raise"):
        summary = PortfolioAnalysis(returns).compute_summary_stats()
    assert summary.total_return == pytest.approx(returns[0] if len(returns) == 1 else 0)
    assert np.isnan(summary.sharpe_ratio)
    assert np.isnan(summary.skewness)


def test_dates_are_owned_snapshots_before_and_after_caching():
    index = pd.date_range("2025-01-31", periods=3)
    dates = pl.from_pandas(index)
    analysis = PortfolioAnalysis([0.1, -0.05, 0.02], dates=dates)
    rolling = analysis.compute_rolling_metrics(windows=[2])
    dates[0] = pd.Timestamp("2024-01-01")
    exposed = analysis.dates
    exposed[0] = pd.Timestamp("2024-02-01")
    assert analysis.compute_monthly_returns()["year"].to_list() == [2025, 2025]
    assert rolling.dates.to_list() == list(index.to_pydatetime())


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_constant_and_single_observation_risk_windows_are_unavailable_without_division_errors():
    analysis = PortfolioAnalysis([0.0, 0.0, 0.0], benchmark=[0.0, 0.0, 0.0])
    rolling = analysis.compute_rolling_metrics(
        windows=[1, 2], metrics=["sharpe", "volatility", "beta"]
    )
    assert rolling.sharpe[1].null_count() == 3
    assert rolling.volatility[1].null_count() == 3
    assert rolling.sharpe[2].null_count() == 3
    assert rolling.beta[2].null_count() == 3
