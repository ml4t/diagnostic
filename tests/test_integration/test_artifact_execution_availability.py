"""Saved target weights are not evidence that executions or account marks occurred."""

from datetime import UTC, date, datetime

import polars as pl
import pytest

pytest.importorskip("ml4t.backtest")

from ml4t.diagnostic.integration import profile_from_run_artifacts


def test_weights_only_artifacts_do_not_fabricate_fill_prices_quantities_or_account_cash(tmp_path):
    # These are valid files emitted by a legacy target-weight run export. Only
    # returns and intended weights were saved, not an execution/account ledger.
    pl.DataFrame(schema={"symbol": pl.String, "pnl": pl.Float64}).write_parquet(
        tmp_path / "trades.parquet"
    )
    pl.DataFrame(
        {"timestamp": [date(2025, 7, 7), date(2025, 7, 8)], "daily_return": [0.01, -0.02]}
    ).write_parquet(tmp_path / "daily_returns.parquet")
    pl.DataFrame(
        {
            "timestamp": [datetime(2025, 7, 7, tzinfo=UTC)],
            "symbol": ["ES"],
            "weight": [0.2],
        }
    ).write_parquet(tmp_path / "weights.parquet")
    profile = profile_from_run_artifacts(tmp_path)
    assert profile.has_signals
    assert profile.fills_df.is_empty(), "target weights cannot determine fills at price 1"
    assert profile.portfolio_state_df.is_empty(), "futures cash cannot be equity*(1-gross_weight)"
    assert profile.availability.families["activity"].status.value == "unavailable"
    assert profile.availability.families["occupancy"].status.value == "unavailable"
    assert profile.data_sources["fills"] == "unavailable (no execution artifact)"
    assert profile.data_sources["portfolio_state"] == "unavailable (no account-state artifact)"
