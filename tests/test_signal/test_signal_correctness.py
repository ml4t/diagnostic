"""High-quality correctness tests for signal analysis functions.

These tests verify mathematical properties of signal quality metrics,
not just that they "work" or "return something".

Key properties tested:
1. IC = Spearman correlation between factor and forward returns
2. Quantile returns: spread = top - bottom, monotonicity = Spearman(q, r)
3. Turnover = 1 - overlap / max(sizes), range [0, 1]
4. Autocorrelation = Spearman(factor_t, factor_{t+lag})
5. Half-life: lag where AC drops to 50% of lag-1 value
"""

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from ml4t.diagnostic.signal.quantile import (
    compute_monotonicity,
    compute_quantile_returns,
    compute_spread,
)
from ml4t.diagnostic.signal.signal_ic import compute_ic_series, compute_ic_summary
from ml4t.diagnostic.signal.turnover import (
    compute_autocorrelation,
    compute_turnover,
    estimate_half_life,
)


class TestICCorrectness:
    """Tests verifying Information Coefficient mathematical correctness."""

    def test_ic_equals_spearman_correlation(self):
        """Verify IC = Spearman(factor, forward_return) for each date.

        This is the fundamental definition of IC in quantitative finance.
        """
        np.random.seed(42)
        n_dates = 10
        n_assets = 50

        # Create data where factor predicts returns
        dates = [f"2024-01-{i + 1:02d}" for i in range(n_dates)]
        data_rows = []

        expected_ics = []
        for date in dates:
            factors = np.random.randn(n_assets)
            # Returns have positive correlation with factors
            returns = factors * 0.5 + np.random.randn(n_assets) * 0.3

            # Compute expected IC manually
            expected_ic, _ = spearmanr(factors, returns)
            expected_ics.append(expected_ic)

            for i in range(n_assets):
                data_rows.append(
                    {
                        "date": date,
                        "asset": f"asset_{i}",
                        "factor": factors[i],
                        "1D_fwd_return": returns[i],
                    }
                )

        df = pl.DataFrame(data_rows)

        # Compute IC using the library
        computed_dates, computed_ics = compute_ic_series(
            df, period=1, method="spearman", min_obs=10
        )

        # Verify each IC matches manual calculation
        assert len(computed_ics) == n_dates
        for i, (expected, computed) in enumerate(zip(expected_ics, computed_ics)):
            assert abs(computed - expected) < 1e-10, (
                f"Date {i}: IC mismatch. Expected {expected:.6f}, got {computed:.6f}"
            )

    def test_perfect_positive_correlation_ic_is_one(self):
        """When factor perfectly predicts returns (monotonically), IC = 1.0."""
        n_assets = 30

        # Factor values
        factors = np.arange(n_assets, dtype=float)
        # Returns are monotonically increasing function of factor
        returns = factors * 2 + 10  # Perfect linear relationship

        df = pl.DataFrame(
            {
                "date": ["2024-01-01"] * n_assets,
                "asset": [f"asset_{i}" for i in range(n_assets)],
                "factor": factors.tolist(),
                "1D_fwd_return": returns.tolist(),
            }
        )

        dates, ics = compute_ic_series(df, period=1, min_obs=5)

        assert len(ics) == 1
        assert abs(ics[0] - 1.0) < 1e-10, f"Perfect correlation should give IC=1.0, got {ics[0]}"

    def test_reversed_factor_flips_ic_sign(self):
        """Negating the factor should negate the IC."""
        np.random.seed(42)
        n_assets = 50

        factors = np.random.randn(n_assets)
        returns = factors * 0.5 + np.random.randn(n_assets) * 0.3

        # Original
        df_original = pl.DataFrame(
            {
                "date": ["2024-01-01"] * n_assets,
                "asset": [f"asset_{i}" for i in range(n_assets)],
                "factor": factors.tolist(),
                "1D_fwd_return": returns.tolist(),
            }
        )

        # Negated factor
        df_negated = pl.DataFrame(
            {
                "date": ["2024-01-01"] * n_assets,
                "asset": [f"asset_{i}" for i in range(n_assets)],
                "factor": (-factors).tolist(),
                "1D_fwd_return": returns.tolist(),
            }
        )

        _, ic_original = compute_ic_series(df_original, period=1, min_obs=5)
        _, ic_negated = compute_ic_series(df_negated, period=1, min_obs=5)

        assert abs(ic_original[0] + ic_negated[0]) < 1e-10, (
            f"Negating factor should negate IC: {ic_original[0]} vs {ic_negated[0]}"
        )

    def test_t_stat_formula_exact(self):
        """Verify t-stat = mean / (std / sqrt(n))."""
        ic_series = [0.05, 0.08, 0.03, 0.07, 0.04, 0.06, 0.09, 0.05, 0.07, 0.06]

        result = compute_ic_summary(ic_series)

        # Calculate expected t-stat manually
        arr = np.array(ic_series)
        n = len(arr)
        mean_ic = np.mean(arr)
        std_ic = np.std(arr, ddof=1)
        expected_t = mean_ic / (std_ic / np.sqrt(n))

        assert abs(result["t_stat"] - expected_t) < 1e-10, (
            f"t-stat mismatch: expected {expected_t:.6f}, got {result['t_stat']:.6f}"
        )

    def test_pct_positive_calculation(self):
        """Verify pct_positive = fraction of IC > 0."""
        ic_series = [0.05, -0.02, 0.03, -0.01, 0.06, 0.04, -0.03, 0.02]
        # Positive: 0.05, 0.03, 0.06, 0.04, 0.02 = 5 out of 8

        result = compute_ic_summary(ic_series)

        expected_pct = 5 / 8
        assert abs(result["pct_positive"] - expected_pct) < 1e-10


class TestQuantileReturnsCorrectness:
    """Tests verifying quantile return analysis correctness."""

    def test_spread_equals_top_minus_bottom(self):
        """Verify spread = mean(top quantile) - mean(bottom quantile)."""
        # Create data with known returns per quantile
        data_rows = []
        n_per_quantile = 20

        # Quantile 1 (bottom): mean return = -0.05
        for i in range(n_per_quantile):
            data_rows.append(
                {
                    "quantile": 1,
                    "asset": f"asset_1_{i}",
                    "5D_fwd_return": -0.05 + np.random.randn() * 0.01,
                }
            )

        # Quantile 5 (top): mean return = 0.10
        for i in range(n_per_quantile):
            data_rows.append(
                {
                    "quantile": 5,
                    "asset": f"asset_5_{i}",
                    "5D_fwd_return": 0.10 + np.random.randn() * 0.01,
                }
            )

        # Middle quantiles
        for q in [2, 3, 4]:
            for i in range(n_per_quantile):
                data_rows.append(
                    {
                        "quantile": q,
                        "asset": f"asset_{q}_{i}",
                        "5D_fwd_return": 0.02 * q + np.random.randn() * 0.01,
                    }
                )

        df = pl.DataFrame(data_rows)

        result = compute_spread(df, period=5, n_quantiles=5)

        # Calculate expected spread manually
        top_returns = df.filter(pl.col("quantile") == 5)["5D_fwd_return"].to_numpy()
        bottom_returns = df.filter(pl.col("quantile") == 1)["5D_fwd_return"].to_numpy()
        expected_spread = np.mean(top_returns) - np.mean(bottom_returns)

        assert abs(result["spread"] - expected_spread) < 1e-10, (
            f"Spread mismatch: expected {expected_spread:.6f}, got {result['spread']:.6f}"
        )

    def test_monotonicity_perfect_increasing(self):
        """Perfect monotonic increase across quantiles → monotonicity = 1.0."""
        # Quantile 1 has lowest return, quantile 5 has highest
        quantile_returns = {
            1: 0.01,
            2: 0.02,
            3: 0.03,
            4: 0.04,
            5: 0.05,
        }

        mono = compute_monotonicity(quantile_returns)

        assert abs(mono - 1.0) < 1e-10, f"Perfect increase should give mono=1.0, got {mono}"

    def test_monotonicity_perfect_decreasing(self):
        """Perfect monotonic decrease across quantiles → monotonicity = -1.0."""
        quantile_returns = {
            1: 0.05,
            2: 0.04,
            3: 0.03,
            4: 0.02,
            5: 0.01,
        }

        mono = compute_monotonicity(quantile_returns)

        assert abs(mono - (-1.0)) < 1e-10, f"Perfect decrease should give mono=-1.0, got {mono}"

    def test_monotonicity_equals_spearman(self):
        """Monotonicity = Spearman(quantile, return)."""
        np.random.seed(42)
        # Random returns per quantile
        quantile_returns = {
            1: 0.02 + np.random.randn() * 0.01,
            2: 0.03 + np.random.randn() * 0.01,
            3: 0.025 + np.random.randn() * 0.01,  # Not monotonic
            4: 0.04 + np.random.randn() * 0.01,
            5: 0.035 + np.random.randn() * 0.01,
        }

        mono = compute_monotonicity(quantile_returns)

        # Calculate expected Spearman manually
        quantiles = list(range(1, 6))
        returns = [quantile_returns[q] for q in quantiles]
        expected_mono, _ = spearmanr(quantiles, returns)

        assert abs(mono - expected_mono) < 1e-10, (
            f"Monotonicity should equal Spearman: expected {expected_mono:.6f}, got {mono:.6f}"
        )

    def test_quantile_returns_mean_by_quantile(self):
        """Verify quantile returns are mean of returns in each quantile."""
        data_rows = []

        # Known returns for each quantile
        expected_means = {1: -0.03, 2: 0.01, 3: 0.04}
        n_per_q = 10

        for q, target_mean in expected_means.items():
            returns = target_mean + np.random.randn(n_per_q) * 0.001  # Small noise
            for i, ret in enumerate(returns):
                data_rows.append(
                    {
                        "quantile": q,
                        "asset": f"asset_{q}_{i}",
                        "1D_fwd_return": ret,
                    }
                )

        df = pl.DataFrame(data_rows)

        result = compute_quantile_returns(df, period=1, n_quantiles=3)

        for q in expected_means:
            assert abs(result[q] - expected_means[q]) < 0.002, (
                f"Quantile {q}: expected ~{expected_means[q]:.4f}, got {result[q]:.4f}"
            )


class TestTurnoverCorrectness:
    """Tests verifying turnover calculation correctness."""

    def test_turnover_formula_manual_verification(self):
        """Verify turnover = 1 - overlap / max(|assets_t|, |assets_t+1|)."""
        # Date 1, Q1: assets A, B, C (3 assets)
        # Date 2, Q1: assets A, D, E (3 assets)
        # Overlap = 1 (only A), max = 3
        # Turnover = 1 - 1/3 = 0.667

        df = pl.DataFrame(
            [
                {"date": "2024-01-01", "asset": "A", "quantile": 1},
                {"date": "2024-01-01", "asset": "B", "quantile": 1},
                {"date": "2024-01-01", "asset": "C", "quantile": 1},
                {"date": "2024-01-02", "asset": "A", "quantile": 1},
                {"date": "2024-01-02", "asset": "D", "quantile": 1},
                {"date": "2024-01-02", "asset": "E", "quantile": 1},
            ]
        )

        turnover = compute_turnover(df, n_quantiles=1)

        expected = 1 - 1 / 3  # = 0.6667
        assert abs(turnover - expected) < 1e-10, (
            f"Turnover formula mismatch: expected {expected:.4f}, got {turnover:.4f}"
        )

    def test_zero_turnover_identical_assignments(self):
        """When all assets stay in same quantile, turnover = 0."""
        assets = ["A", "B", "C", "D", "E"]
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]

        data_rows = []
        for date in dates:
            for asset in assets:
                data_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "quantile": 1,  # All in quantile 1
                    }
                )

        df = pl.DataFrame(data_rows)

        turnover = compute_turnover(df, n_quantiles=1)

        assert abs(turnover - 0.0) < 1e-10, (
            f"Identical assignments should give turnover=0, got {turnover}"
        )

    def test_full_turnover_complete_change(self):
        """When all assets change quantile, turnover = 1."""
        # Date 1, Q1: assets A, B, C
        # Date 2, Q1: assets D, E, F (no overlap)

        df = pl.DataFrame(
            [
                {"date": "2024-01-01", "asset": "A", "quantile": 1},
                {"date": "2024-01-01", "asset": "B", "quantile": 1},
                {"date": "2024-01-01", "asset": "C", "quantile": 1},
                {"date": "2024-01-02", "asset": "D", "quantile": 1},
                {"date": "2024-01-02", "asset": "E", "quantile": 1},
                {"date": "2024-01-02", "asset": "F", "quantile": 1},
            ]
        )

        turnover = compute_turnover(df, n_quantiles=1)

        assert abs(turnover - 1.0) < 1e-10, (
            f"Complete change should give turnover=1, got {turnover}"
        )

    def test_turnover_bounded_zero_one(self):
        """Turnover should always be in [0, 1]."""
        np.random.seed(42)

        for _ in range(10):
            # Random data
            n_dates = 5
            n_assets = 20
            n_quantiles = 5

            data_rows = []
            for d in range(n_dates):
                for a in range(n_assets):
                    data_rows.append(
                        {
                            "date": f"2024-01-{d + 1:02d}",
                            "asset": f"asset_{a}",
                            "quantile": np.random.randint(1, n_quantiles + 1),
                        }
                    )

            df = pl.DataFrame(data_rows)
            turnover = compute_turnover(df, n_quantiles=n_quantiles)

            assert 0.0 <= turnover <= 1.0, f"Turnover {turnover} out of bounds [0, 1]"


class TestAutocorrelationCorrectness:
    """Tests verifying autocorrelation calculation correctness."""

    def test_autocorrelation_equals_spearman_lag(self):
        """Verify AC(lag) = Spearman(factor_t, factor_{t+lag})."""
        np.random.seed(42)
        n_dates = 20
        n_assets = 30

        # Create persistent factor (AR(1) process)
        rho = 0.8
        factors_by_date = {}
        prev_factors = np.random.randn(n_assets)

        data_rows = []
        for d in range(n_dates):
            date = f"2024-01-{d + 1:02d}"
            if d == 0:
                factors = prev_factors
            else:
                factors = rho * prev_factors + np.random.randn(n_assets) * np.sqrt(1 - rho**2)

            factors_by_date[date] = factors.copy()
            prev_factors = factors

            for a in range(n_assets):
                data_rows.append(
                    {
                        "date": date,
                        "asset": f"asset_{a}",
                        "factor": factors[a],
                    }
                )

        df = pl.DataFrame(data_rows)

        # Compute autocorrelation at lag 1
        computed_ac = compute_autocorrelation(df, lags=[1], min_obs=10)

        # Calculate expected AC manually (average Spearman across date pairs)
        dates = sorted(factors_by_date.keys())
        expected_corrs = []
        for i in range(len(dates) - 1):
            rho_manual, _ = spearmanr(factors_by_date[dates[i]], factors_by_date[dates[i + 1]])
            expected_corrs.append(rho_manual)

        expected_ac = np.mean(expected_corrs)

        assert abs(computed_ac[0] - expected_ac) < 1e-10, (
            f"AC(1) mismatch: expected {expected_ac:.6f}, got {computed_ac[0]:.6f}"
        )

    def test_constant_factor_ac_is_one(self):
        """When factor doesn't change, autocorrelation = 1.0 at all lags."""
        n_dates = 10
        n_assets = 20

        # Same factor values every date
        constant_factors = np.arange(n_assets, dtype=float)

        data_rows = []
        for d in range(n_dates):
            for a in range(n_assets):
                data_rows.append(
                    {
                        "date": f"2024-01-{d + 1:02d}",
                        "asset": f"asset_{a}",
                        "factor": constant_factors[a],
                    }
                )

        df = pl.DataFrame(data_rows)

        ac = compute_autocorrelation(df, lags=[1, 2, 3], min_obs=10)

        for i, lag in enumerate([1, 2, 3]):
            assert abs(ac[i] - 1.0) < 1e-10, (
                f"Constant factor should give AC=1.0 at lag {lag}, got {ac[i]}"
            )


class TestHalfLifeCorrectness:
    """Tests verifying half-life calculation correctness."""

    def test_half_life_definition(self):
        """Half-life = lag where AC drops to 50% of lag-1 value."""
        # Autocorrelations: [0.8, 0.64, 0.512, 0.4096, 0.328, ...]
        # 50% of 0.8 = 0.4
        # Half-life is between lag 3 (0.512) and lag 4 (0.4096)

        ac = [0.8, 0.64, 0.512, 0.4096, 0.328]
        half_life = estimate_half_life(ac)

        # 50% threshold = 0.4
        # At lag 4, AC = 0.4096 > 0.4
        # At lag 5, AC = 0.328 < 0.4
        # Linear interpolation: 4 + (0.4096 - 0.4) / (0.4096 - 0.328) = 4 + 0.118

        assert half_life is not None
        assert 4 < half_life < 5, f"Half-life should be between 4 and 5, got {half_life}"

    def test_half_life_immediate_decay(self):
        """If AC drops below 50% at lag 2, half-life < 2."""
        ac = [0.6, 0.2, 0.1]  # 50% of 0.6 = 0.3, already below at lag 2
        half_life = estimate_half_life(ac)

        assert half_life is not None
        assert half_life < 2, f"Should decay by lag 2, got half_life={half_life}"

    def test_half_life_no_decay(self):
        """If AC never drops to 50%, return None."""
        ac = [0.8, 0.7, 0.65, 0.6]  # Never drops to 0.4
        half_life = estimate_half_life(ac)

        assert half_life is None, f"Should return None when no decay, got {half_life}"


class TestEdgeCasesSignal:
    """Edge cases for signal analysis functions."""

    def test_ic_insufficient_data(self):
        """IC summary with < 2 samples returns NaN."""
        result = compute_ic_summary([0.05])  # Single value

        assert np.isnan(result["t_stat"])
        assert np.isnan(result["p_value"])

    def test_quantile_returns_missing_quantile(self):
        """Missing quantile in data returns NaN for that quantile."""
        df = pl.DataFrame(
            [
                {"quantile": 1, "asset": "A", "1D_fwd_return": 0.01},
                {"quantile": 3, "asset": "B", "1D_fwd_return": 0.03},
                # Quantile 2 is missing
            ]
        )

        result = compute_quantile_returns(df, period=1, n_quantiles=3)

        assert np.isnan(result[2]), "Missing quantile should return NaN"
        assert not np.isnan(result[1])
        assert not np.isnan(result[3])

    def test_turnover_single_date(self):
        """Single date returns NaN (need 2 dates for turnover)."""
        df = pl.DataFrame(
            [
                {"date": "2024-01-01", "asset": "A", "quantile": 1},
                {"date": "2024-01-01", "asset": "B", "quantile": 1},
            ]
        )

        turnover = compute_turnover(df, n_quantiles=1)

        assert np.isnan(turnover), "Single date should give NaN turnover"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
