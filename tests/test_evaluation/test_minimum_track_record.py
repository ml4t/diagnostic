"""Regression tests for Minimum Track Record Length calculations."""

import numpy as np

from ml4t.diagnostic.evaluation.stats import compute_min_trl


def test_min_trl_includes_finite_sample_offset_for_strategy_14() -> None:
    """Match the seeded Chapter 16 strategy-selection example."""
    rng = np.random.default_rng(123)
    n_strategies = 30
    n_days = 504
    true_sharpes = rng.permutation([0.8] * 5 + [0.0] * (n_strategies - 5))

    strategy_returns: dict[str, np.ndarray] = {}
    observed_sharpes: dict[str, float] = {}
    for index, true_sharpe in enumerate(true_sharpes, start=1):
        daily_volatility = 0.15 / np.sqrt(252)
        daily_mean = true_sharpe * 0.15 / 252
        returns = rng.normal(daily_mean, daily_volatility, n_days)
        name = f"Strategy_{index}"
        strategy_returns[name] = returns
        observed_sharpes[name] = returns.mean() / returns.std(ddof=1) * np.sqrt(252)

    best_name = max(observed_sharpes, key=observed_sharpes.__getitem__)
    result = compute_min_trl(
        returns=strategy_returns[best_name],
        target_sharpe=0.5 / np.sqrt(252),
        frequency="daily",
    )

    assert best_name == "Strategy_14"
    assert result.min_trl == 213.0
