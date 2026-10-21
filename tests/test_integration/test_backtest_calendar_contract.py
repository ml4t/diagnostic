"""The real engine bridge must distinguish an omitted calendar from explicit None."""

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

pytest.importorskip("ml4t.backtest")
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import (
    BacktestReportMetadata,
    analyze_backtest_result,
    compute_metrics_from_result,
    generate_tearsheet_from_result,
    portfolio_analysis_from_result,
)
from ml4t.diagnostic.integration.backtest_profile import BacktestProfile


class BuyOnce(Strategy):
    def __init__(self):
        self.ordered = False

    def on_data(self, timestamp, data, context, broker):
        if not self.ordered:
            broker.submit_order("A", 2)
            self.ordered = True


@pytest.fixture
def overnight_result():
    prices = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 7, 6, 23, 30, tzinfo=UTC),
                datetime(2025, 7, 7, 0, 30, tzinfo=UTC),
                datetime(2025, 7, 7, 1, 30, tzinfo=UTC),
            ],
            "asset": ["A"] * 3,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0, 100.0, 110.0]),
            "volume": [10000.0] * 3,
        }
    )
    return Engine(
        DataFeed(prices_df=prices),
        BuyOnce(),
        BacktestConfig(
            initial_cash=1000,
            calendar="CME_Equity",
            timezone="America/Chicago",
        ),
    ).run()


def test_explicit_none_disables_engine_calendar_without_changing_omitted_metadata(overnight_result):
    omitted = analyze_backtest_result(overnight_result)
    assert omitted.resolved_calendar == "CME_Equity"
    disabled = analyze_backtest_result(overnight_result, calendar=None)
    assert disabled.resolved_calendar is None
    assert BacktestProfile(overnight_result, calendar=None).resolved_calendar is None
    assert disabled.daily_returns.to_list() == pytest.approx([0.0, 0.0, 0.02])
    assert disabled.daily_returns.equals(overnight_result.to_returns_series().alias("daily_return"))
    assert overnight_result.config.resolved_calendar == "CME_Equity"


def test_row_mode_uses_observation_keys_and_the_explicit_return_scale_in_every_wrapper(
    overnight_result,
):
    scale = 365 * 24
    profile = analyze_backtest_result(overnight_result, calendar=None, periods_per_year=scale)
    analysis = portfolio_analysis_from_result(
        overnight_result, calendar=None, periods_per_year=scale
    )
    np.testing.assert_allclose(analysis.returns, [0.0, 0.0, 0.02])
    assert analysis.dates.equals(overnight_result.to_equity_dataframe()["timestamp"].rename("date"))
    expected_volatility = np.std([0.0, 0.0, 0.02], ddof=1) * np.sqrt(scale)
    assert analysis.compute_summary_stats().annual_volatility == pytest.approx(expected_volatility)
    assert profile.performance["metrics"]["annualized_volatility"] == pytest.approx(
        expected_volatility
    )
    computed = compute_metrics_from_result(overnight_result, calendar=None, periods_per_year=scale)
    assert computed["annualized_volatility"] == pytest.approx(expected_volatility)


def test_row_mode_reaches_html_with_resolved_metadata_and_no_hidden_calendar_fallback(
    overnight_result, tmp_path
):
    pytest.importorskip("plotly")
    destination = tmp_path / "overnight-observations.html"
    html = generate_tearsheet_from_result(
        overnight_result,
        calendar=None,
        periods_per_year=365 * 24,
        include_statistical=False,
        output_path=destination,
    )
    assert destination.read_text() == html
    assert "calendar CME_Equity" not in html


def test_observation_metrics_require_a_scale_and_invalid_calls_never_publish(
    overnight_result, tmp_path
):
    profile = analyze_backtest_result(overnight_result, calendar=None)
    with pytest.raises(ValueError, match="periods_per_year is required"):
        profile.performance
    with pytest.raises(ValueError, match="periods_per_year is required"):
        portfolio_analysis_from_result(overnight_result, calendar=None)
    output = tmp_path / "unpublished.html"
    with pytest.raises(ValueError, match="periods_per_year is required"):
        generate_tearsheet_from_result(overnight_result, calendar=None, output_path=output)
    assert not output.exists()
    with pytest.raises(ValueError, match="metadata calendar conflicts"):
        generate_tearsheet_from_result(
            overnight_result,
            calendar=None,
            periods_per_year=8760,
            report_metadata=BacktestReportMetadata(calendar="CME_Equity"),
            output_path=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("scale", [0, -1, True, float("nan"), 12.5])
def test_invalid_observation_scales_are_rejected_before_any_lazy_cache(scale, overnight_result):
    with pytest.raises(ValueError, match="positive integer"):
        analyze_backtest_result(overnight_result, calendar=None, periods_per_year=scale)
