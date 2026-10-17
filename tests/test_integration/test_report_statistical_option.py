"""A disabled report option must not silently render statistical inference."""

from datetime import UTC, datetime
from html.parser import HTMLParser

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import generate_tearsheet_from_result

STATISTICAL_SECTIONS = {
    "validity_card",
    "confidence_intervals",
    "min_trl",
    "sharpe_bootstrap",
    "credibility_box",
}


@pytest.fixture
def result():
    class BuyOnce(Strategy):
        def __init__(self):
            self.ordered = False

        def on_data(self, timestamp, data, context, broker):
            if not self.ordered:
                broker.submit_order("A", 4)
                self.ordered = True

    marks = [100.0, 100.0, 110.0, 105.0, 115.0, 108.0, 120.0, 117.0]
    bars = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 2, d, 15, tzinfo=UTC) for d in [1, 2, 5, 6, 7, 8, 9, 12]],
            "asset": ["A"] * 8,
            **dict.fromkeys(["open", "high", "low", "close"], marks),
            "volume": [10000.0] * 8,
        }
    )
    return Engine(DataFeed(prices_df=bars), BuyOnce(), BacktestConfig(initial_cash=1000)).run()


def _sections(rendered):
    class Sections(HTMLParser):
        def __init__(self):
            super().__init__()
            self.names = set()

        def handle_starttag(self, tag, attrs):
            for name, value in attrs:
                if name == "data-section":
                    self.names.add(value)

    parser = Sections()
    parser.feed(rendered)
    return parser.names


@pytest.mark.parametrize("template", ["quant_trader", "hedge_fund", "risk_manager", "full"])
def test_actual_bridge_disables_inference_without_changing_descriptive_report_or_next_default(
    result, tmp_path, template
):
    trades, fills = result.to_trades_dataframe().clone(), result.to_fills_dataframe().clone()
    equity = list(result.equity_curve)
    enabled = generate_tearsheet_from_result(result, template=template, include_statistical=True)
    disabled_path = tmp_path / "disabled.html"
    disabled = generate_tearsheet_from_result(
        result, template=template, include_statistical=False, output_path=disabled_path
    )
    assert disabled_path.read_text(encoding="utf-8") == disabled
    assert not (_sections(disabled) & STATISTICAL_SECTIONS)
    assert "executive_summary" in _sections(disabled)
    assert "overview_snapshot" in _sections(disabled)
    if template in ("hedge_fund", "risk_manager", "full"):
        assert "drawdown_anatomy" in _sections(disabled)
    default = generate_tearsheet_from_result(result, template=template)
    assert _sections(default) == _sections(enabled)
    assert _sections(disabled) == _sections(enabled) - STATISTICAL_SECTIONS
    if template in ("risk_manager", "full"):
        assert {"confidence_intervals", "min_trl"} <= _sections(enabled)
        assert "Deflated Sharpe" not in disabled
    assert result.equity_curve == equity
    assert result.to_trades_dataframe().equals(trades)
    assert result.to_fills_dataframe().equals(fills)


@pytest.mark.parametrize("template", ["quant_trader", "hedge_fund", "risk_manager", "full"])
def test_direct_renderer_disabled_option_skips_actual_inference_enrichment(result, template):
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    returns = result.to_daily_returns().to_numpy()
    disabled_metrics = {}
    disabled = generate_backtest_tearsheet(
        returns=returns, metrics=disabled_metrics, template=template, include_statistical=False
    )
    assert not (_sections(disabled) & STATISTICAL_SECTIONS)
    assert "annual_volatility" in disabled_metrics
    for key in ("dsr_probability", "min_trl", "sharpe_lower_95", "sharpe_se"):
        assert key not in disabled_metrics
    enabled_metrics = {}
    generate_backtest_tearsheet(returns=returns, metrics=enabled_metrics, template=template)
    assert "dsr_probability" in enabled_metrics
    assert "sharpe_lower_95" in enabled_metrics
    assert disabled_metrics["sharpe_ratio"] == enabled_metrics["sharpe_ratio"]


@pytest.mark.parametrize("template", ["risk_manager", "full"])
def test_disabled_report_omits_supplied_inference_and_keeps_next_preset_unchanged(result, template):
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    returns = result.to_daily_returns().to_numpy()
    metrics = {"dsr_probability": 0.99, "min_trl": 12.0}
    enabled = generate_backtest_tearsheet(
        returns=returns, metrics=metrics.copy(), template=template
    )
    disabled = generate_backtest_tearsheet(
        returns=returns, metrics=metrics, template=template, include_statistical=False
    )
    default = generate_backtest_tearsheet(
        returns=returns, metrics=metrics.copy(), template=template
    )
    assert "Deflated Sharpe Ratio" in enabled and "Deflated Sharpe Ratio" in default
    assert "Deflated Sharpe Ratio" not in disabled
    assert "MinTRL" not in disabled
    assert metrics["dsr_probability"] == 0.99 and metrics["min_trl"] == 12.0
    assert _sections(enabled) == _sections(default)
