"""A disabled report option must not silently render statistical inference."""

import re
from datetime import UTC, datetime

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
    return set(re.findall(r'data-section="([^"]+)"', rendered))


@pytest.mark.parametrize("template", ["quant_trader", "hedge_fund", "risk_manager", "full"])
def test_actual_bridge_disables_inference_without_changing_descriptive_report_or_next_default(
    result, tmp_path, template
):
    trades, fills = result.to_trades_dataframe().clone(), result.to_fills_dataframe().clone()
    equity = list(result.equity_curve)
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
    enabled = generate_tearsheet_from_result(result, template=template, include_statistical=True)
    default = generate_tearsheet_from_result(result, template=template)
    assert _sections(default) == _sections(enabled)
    assert _sections(disabled) == _sections(enabled) - STATISTICAL_SECTIONS
    if template in ("risk_manager", "full"):
        assert {"confidence_intervals", "min_trl"} <= _sections(enabled)
        assert "Deflated Sharpe" not in disabled
    assert result.equity_curve == equity
    assert result.to_trades_dataframe().equals(trades)
    assert result.to_fills_dataframe().equals(fills)
