"""Familiar capacity calls share native formulas and preserve actual ledger keys."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat.pyfolio import capacity


def _inputs():
    dates = pd.date_range("2025-01-13", periods=8, freq="B", tz="UTC")
    positions = pd.DataFrame(
        {"A": [10000.0] * 8, "B": [-10000.0] * 8, "cash": [100000.0] * 8}, index=dates
    )
    market = pd.DataFrame(
        {"A": [10.0, 10000.0] * 8, "B": [10.0, 10000.0] * 8},
        index=pd.MultiIndex.from_product([dates, ["price", "volume"]]),
    )
    fills = pd.DataFrame(
        {"symbol": ["A", "B", "A"], "amount": [100.0, -100.0, -50.0], "price": [10.0, 10.0, 10.0]},
        index=dates[[1, 1, 3]],
    )
    returns = pd.Series(
        [0.01, -0.02, 0.03, 0.0, 0.01, 0.02, -0.03, 0.01], index=dates, name="returns"
    )
    return returns, positions, fills, market


def test_two_assets_at_one_date_retain_distinct_transaction_keys_without_mutation():
    _, _, fills, market = _inputs()
    before = fills.copy(deep=True)
    result = capacity.daily_txns_with_bar_data(fills, market)
    assert result["symbol"].dropna().tolist() == ["A", "B", "A"]
    assert result["amount"].dropna().tolist() == [100.0, 100.0, 50.0]
    assert result["volume"].dropna().tolist() == [10000.0] * 3
    assert len(result) == 4
    pd.testing.assert_frame_equal(fills, before)
    worst = capacity.get_low_liquidity_transactions(fills, market)
    assert worst["max_pct_bar_consumed"].tolist() == [1.0, 1.0]


def test_fixed_capital_legacy_liquidation_and_trimmed_window_are_explicit():
    _, positions, _, market = _inputs()
    result = capacity.days_to_liquidate_positions(
        positions, market, max_bar_consumption=0.1, capital_base=100000.0, mean_volume_window=2
    )
    assert result.index.equals(positions.index[2:])
    np.testing.assert_allclose(result, 1.0)
    doubled_book = positions * 2
    pd.testing.assert_frame_equal(
        result,
        capacity.days_to_liquidate_positions(
            doubled_book,
            market,
            max_bar_consumption=0.1,
            capital_base=100000.0,
            mean_volume_window=2,
        ),
    )


def test_known_zero_volume_exposure_is_unbounded_not_missing_or_free():
    _, positions, _, market = _inputs()
    market.loc[market.index.get_level_values(1) == "volume", :] = 0.0
    result = capacity.days_to_liquidate_positions(positions, market, mean_volume_window=1)
    assert np.isposinf(result.to_numpy()).all()


def test_closing_nav_legacy_impact_is_distinct_from_native_beginning_nav():
    returns, _, fills, market = _inputs()
    daily = capacity.daily_txns_with_bar_data(fills, market)
    adjusted = capacity.apply_slippage_penalty(returns, daily, 100000.0, 100000.0)
    assert adjusted.iloc[1] == pytest.approx(-0.02 - 0.02 / (100000 * 1.01 * 0.98))
    assert adjusted.iloc[3] == pytest.approx(0.0 - 0.00125 / (100000 * 1.01 * 0.98 * 1.03))
    assert adjusted.iloc[0] == returns.iloc[0]
    assert adjusted.name is None


def test_unknown_active_volume_is_not_a_zero_slippage_penalty():
    returns, _, fills, market = _inputs()
    market.loc[(returns.index[1], "volume"), "A"] = np.nan
    result = capacity.apply_slippage_penalty(
        returns, capacity.daily_txns_with_bar_data(fills, market), 100000.0, 100000.0
    )
    assert np.isnan(result.iloc[1])
    assert result.iloc[0] == returns.iloc[0]


def test_recent_windows_preserve_actual_worst_observation_fields_and_cutoffs():
    _, positions, fills, market = _inputs()
    result = capacity.get_max_days_to_liquidate_by_ticker(
        positions, market, mean_volume_window=1, last_n_days=2
    )
    assert (result["date"] >= positions.index.max() - pd.Timedelta(days=2)).all()
    assert result.loc["A", "pos_alloc_pct"] == 10.0
    trades = capacity.get_low_liquidity_transactions(fills, market, last_n_days=1)
    assert trades.index.tolist() == ["A"]
    assert trades.iloc[0]["date"] == fills.index[-1]


def test_real_published_reference_matches_defined_daily_and_liquidation_workflows():
    reference = pytest.importorskip("pyfolio.capacity")
    returns, positions, fills, market = _inputs()
    for window in (1, 2, 5, 20):
        pd.testing.assert_frame_equal(
            capacity.days_to_liquidate_positions(positions, market, 0.1, 100000.0, window),
            reference.days_to_liquidate_positions(positions, market, 0.1, 100000.0, window),
        )
    one_asset = fills[fills["symbol"] == "A"].copy()
    daily = capacity.daily_txns_with_bar_data(one_asset, market)
    expected = reference.daily_txns_with_bar_data(one_asset.copy(), market)
    pd.testing.assert_frame_equal(daily, expected, check_dtype=False, check_freq=False)
    pd.testing.assert_series_equal(
        capacity.apply_slippage_penalty(returns, daily, 200000.0, 100000.0),
        reference.apply_slippage_penalty(returns, expected, 200000.0, 100000.0),
    )


def test_capital_sweep_preserves_legacy_capital_grid_and_explicit_metric_scale():
    pytest.importorskip("plotly")
    from ml4t.diagnostic.compat.pyfolio import plotting

    returns, _, fills, market = _inputs()
    figure = plotting.plot_capacity_sweep(
        returns,
        fills,
        market,
        100000.0,
        min_pv=100000,
        max_pv=400000,
        step_size=100000,
        periods_per_year=12,
    )
    assert list(figure.data[0].x) == [0.1, 0.2, 0.3]
    for ratio, capital in zip(figure.data[0].y, [100000.0, 200000.0, 300000.0], strict=True):
        sample = returns.to_numpy().copy()
        nav = capital * np.cumprod(1 + sample)
        scale = capital / 100000
        sample[1] -= 0.02 * scale**3 / nav[1]
        sample[3] -= 0.00125 * scale**3 / nav[3]
        assert ratio == pytest.approx(sample.mean() / sample.std(ddof=1) * np.sqrt(12))
    assert figure.layout.meta["nav_timing"] == "compounded_closing"


def test_real_reference_capacity_summary_options_preserve_dates_and_percent_units():
    reference = pytest.importorskip("pyfolio.capacity")
    _, positions, fills, market = _inputs()
    for day, value in zip(positions.index, np.arange(10000.0, 18000.0, 1000.0), strict=True):
        market.loc[(day, "volume"), :] = value
    for recent in (None, 1, 3):
        pd.testing.assert_frame_equal(
            capacity.get_max_days_to_liquidate_by_ticker(
                positions, market, mean_volume_window=2, last_n_days=recent
            ),
            reference.get_max_days_to_liquidate_by_ticker(
                positions, market, mean_volume_window=2, last_n_days=recent
            ),
            check_like=True,
        )
        single = fills[fills["symbol"] == "A"].copy()
        pd.testing.assert_frame_equal(
            capacity.get_low_liquidity_transactions(single, market, last_n_days=recent),
            reference.get_low_liquidity_transactions(single.copy(), market, last_n_days=recent),
            check_freq=False,
        )
