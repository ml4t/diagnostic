"""Tests for SignalResult bridge methods (to_ic_result, to_quantile_result, to_tear_sheet).

Verifies the conversion from frozen dataclass SignalResult to Pydantic viz models.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.signal import SignalResult, analyze_signal

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def predictive_data():
    """Factor data with positive IC (higher factor -> higher returns)."""
    np.random.seed(42)

    n_assets = 50
    n_dates = 60
    base_date = date(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_dates)]
    assets = [f"A{i:03d}" for i in range(n_assets)]

    base_quality = np.random.randn(n_assets)
    factor_rows = []
    price_rows = []
    prices = np.full(n_assets, 100.0)

    for d in dates:
        factors = base_quality + np.random.randn(n_assets) * 0.3
        returns = base_quality * 0.002 + np.random.randn(n_assets) * 0.01
        prices = prices * (1 + returns)

        for i, asset in enumerate(assets):
            factor_rows.append({"date": d, "asset": asset, "factor": factors[i]})
            price_rows.append({"date": d, "asset": asset, "price": prices[i]})

    return pl.DataFrame(factor_rows), pl.DataFrame(price_rows)


@pytest.fixture
def signal_result(predictive_data):
    """Full SignalResult from analyze_signal()."""
    factor_df, prices_df = predictive_data
    return analyze_signal(factor_df, prices_df, periods=(1, 5, 21), quantiles=5)


@pytest.fixture
def minimal_result():
    """Minimal SignalResult without bridge fields (backward compat test)."""
    return SignalResult(
        ic={"1D": 0.05, "5D": 0.08},
        ic_std={"1D": 0.02, "5D": 0.03},
        ic_t_stat={"1D": 2.5, "5D": 2.7},
        ic_p_value={"1D": 0.01, "5D": 0.007},
        periods=(1, 5),
        quantiles=5,
    )


# =============================================================================
# Test backward compatibility
# =============================================================================


class TestBackwardCompat:
    """SignalResult without new fields still works."""

    def test_construct_without_new_fields(self, minimal_result):
        """New fields default to empty dicts."""
        assert minimal_result.ic_dates == {}
        assert minimal_result.quantile_returns_std == {}
        assert minimal_result.count_by_quantile == {}
        assert minimal_result.spread_std == {}

    def test_summary_without_new_fields(self, minimal_result):
        """summary() works without new fields."""
        s = minimal_result.summary()
        assert "IC=" in s

    def test_to_dict_includes_new_fields(self, minimal_result):
        """to_dict() includes the new fields as empty dicts."""
        d = minimal_result.to_dict()
        assert "ic_dates" in d
        assert "quantile_returns_std" in d
        assert "count_by_quantile" in d
        assert "spread_std" in d

    def test_json_roundtrip(self, signal_result, tmp_path):
        """JSON export/import preserves new fields."""
        path = str(tmp_path / "result.json")
        signal_result.to_json(path)
        loaded = SignalResult.from_json(path)

        assert loaded.ic == signal_result.ic
        assert loaded.ic_dates == signal_result.ic_dates
        assert loaded.count_by_quantile == signal_result.count_by_quantile
        assert loaded.spread_std.keys() == signal_result.spread_std.keys()
        # Check quantile_returns_std keys are int after roundtrip
        for pk in loaded.quantile_returns_std:
            for k in loaded.quantile_returns_std[pk]:
                assert isinstance(k, int)


# =============================================================================
# Test data capture in analyze_signal
# =============================================================================


class TestDataCapture:
    """analyze_signal() populates the new fields."""

    def test_ic_dates_populated(self, signal_result):
        """ic_dates has entries for each period."""
        for p in (1, 5, 21):
            key = f"{p}D"
            assert key in signal_result.ic_dates
            assert len(signal_result.ic_dates[key]) > 0
            assert len(signal_result.ic_dates[key]) == len(signal_result.ic_series[key])

    def test_ic_dates_are_strings(self, signal_result):
        """ic_dates values are ISO date strings."""
        for dates in signal_result.ic_dates.values():
            for d in dates:
                assert isinstance(d, str)

    def test_quantile_returns_std_populated(self, signal_result):
        """quantile_returns_std has entries for each period and quantile."""
        for p in (1, 5, 21):
            key = f"{p}D"
            assert key in signal_result.quantile_returns_std
            q_std = signal_result.quantile_returns_std[key]
            assert len(q_std) == signal_result.quantiles
            for q, val in q_std.items():
                assert isinstance(q, int)
                assert val >= 0

    def test_count_by_quantile_populated(self, signal_result):
        """count_by_quantile has an entry per quantile."""
        assert len(signal_result.count_by_quantile) == signal_result.quantiles
        total = sum(signal_result.count_by_quantile.values())
        assert total > 0

    def test_spread_std_populated(self, signal_result):
        """spread_std has entries for each period."""
        for p in (1, 5, 21):
            key = f"{p}D"
            assert key in signal_result.spread_std
            assert signal_result.spread_std[key] >= 0


# =============================================================================
# Test to_ic_result()
# =============================================================================


class TestToICResult:
    """Bridge from SignalResult to SignalICResult."""

    def test_all_periods(self, signal_result):
        """to_ic_result() without period returns all periods."""
        ic_result = signal_result.to_ic_result()
        assert set(ic_result.ic_mean.keys()) == {"1D", "5D", "21D"}

    def test_single_period(self, signal_result):
        """to_ic_result(period=21) returns single period."""
        ic_result = signal_result.to_ic_result(period=21)
        assert list(ic_result.ic_mean.keys()) == ["21D"]

    def test_single_period_string(self, signal_result):
        """to_ic_result(period="5D") works with string."""
        ic_result = signal_result.to_ic_result(period="5D")
        assert list(ic_result.ic_mean.keys()) == ["5D"]

    def test_field_mapping(self, signal_result):
        """Fields are correctly mapped from SignalResult to SignalICResult."""
        ic_result = signal_result.to_ic_result()
        for pk in ("1D", "5D", "21D"):
            assert ic_result.ic_mean[pk] == signal_result.ic[pk]
            assert ic_result.ic_std[pk] == signal_result.ic_std[pk]
            assert ic_result.ic_t_stat[pk] == signal_result.ic_t_stat[pk]
            assert ic_result.ic_p_value[pk] == signal_result.ic_p_value[pk]

    def test_dates_aligned(self, signal_result):
        """IC series aligned to common date intersection."""
        ic_result = signal_result.to_ic_result()
        n_dates = len(ic_result.dates)
        for pk in ic_result.ic_by_date:
            assert len(ic_result.ic_by_date[pk]) == n_dates

    def test_hac_ras_default_none(self, signal_result):
        """HAC/RAS fields default to None (not computed in analyze_signal)."""
        ic_result = signal_result.to_ic_result()
        assert ic_result.ic_t_stat_hac is None
        assert ic_result.ras_adjusted_ic is None

    def test_invalid_period_raises(self, signal_result):
        """to_ic_result() with invalid period raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            signal_result.to_ic_result(period=999)

    def test_no_dates_raises(self, minimal_result):
        """to_ic_result() without ic_dates raises ValueError."""
        with pytest.raises(ValueError, match="ic_dates not available"):
            minimal_result.to_ic_result()

    def test_get_dataframe(self, signal_result):
        """Resulting SignalICResult produces valid DataFrames."""
        ic_result = signal_result.to_ic_result()
        df = ic_result.get_dataframe("ic_by_date")
        assert "date" in df.columns
        summary_df = ic_result.get_dataframe("summary")
        assert "ic_mean" in summary_df.columns


# =============================================================================
# Test to_quantile_result()
# =============================================================================


class TestToQuantileResult:
    """Bridge from SignalResult to QuantileAnalysisResult."""

    def test_basic_structure(self, signal_result):
        """to_quantile_result() produces valid QuantileAnalysisResult."""
        qr = signal_result.to_quantile_result()
        assert qr.n_quantiles == 5
        assert qr.quantile_labels == ["Q1", "Q2", "Q3", "Q4", "Q5"]
        assert qr.periods == ["1D", "5D", "21D"]

    def test_mean_returns_relabeled(self, signal_result):
        """Int quantile keys converted to Q1..QN string labels."""
        qr = signal_result.to_quantile_result()
        for pk in qr.periods:
            assert set(qr.mean_returns[pk].keys()) == {"Q1", "Q2", "Q3", "Q4", "Q5"}

    def test_mean_returns_values_match(self, signal_result):
        """Mean return values match original SignalResult."""
        qr = signal_result.to_quantile_result()
        for pk in qr.periods:
            for q_int, val in signal_result.quantile_returns[pk].items():
                assert qr.mean_returns[pk][f"Q{q_int}"] == val

    def test_std_returns_populated(self, signal_result):
        """std_returns are populated from quantile_returns_std."""
        qr = signal_result.to_quantile_result()
        for pk in qr.periods:
            for q_label in qr.quantile_labels:
                assert isinstance(qr.std_returns[pk][q_label], float)

    def test_count_by_quantile(self, signal_result):
        """count_by_quantile uses string labels."""
        qr = signal_result.to_quantile_result()
        assert set(qr.count_by_quantile.keys()) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
        assert all(v > 0 for v in qr.count_by_quantile.values())

    def test_spread_metrics(self, signal_result):
        """Spread mean/std/t/p are populated."""
        qr = signal_result.to_quantile_result()
        for pk in qr.periods:
            assert pk in qr.spread_mean
            assert pk in qr.spread_std
            assert pk in qr.spread_t_stat
            assert pk in qr.spread_p_value

    def test_spread_ci(self, signal_result):
        """Confidence intervals computed from spread ± 1.96*se."""
        qr = signal_result.to_quantile_result()
        for pk in qr.periods:
            spread = qr.spread_mean[pk]
            se = qr.spread_std[pk]
            assert abs(qr.spread_ci_lower[pk] - (spread - 1.96 * se)) < 1e-10
            assert abs(qr.spread_ci_upper[pk] - (spread + 1.96 * se)) < 1e-10

    def test_monotonicity(self, signal_result):
        """Monotonicity derived from rank_correlation threshold."""
        qr = signal_result.to_quantile_result()
        for pk in qr.periods:
            rho = qr.rank_correlation[pk]
            assert qr.is_monotonic[pk] == (abs(rho) > 0.8)
            if rho > 0.8:
                assert qr.monotonicity_direction[pk] == "increasing"
            elif rho < -0.8:
                assert qr.monotonicity_direction[pk] == "decreasing"
            else:
                assert qr.monotonicity_direction[pk] == "none"

    def test_get_dataframe(self, signal_result):
        """Resulting QuantileAnalysisResult produces valid DataFrames."""
        qr = signal_result.to_quantile_result()
        df = qr.get_dataframe("mean_returns")
        assert "quantile" in df.columns
        assert "mean_return" in df.columns

    def test_without_std_data(self):
        """to_quantile_result() works when quantile_returns_std is empty."""
        result = SignalResult(
            ic={"1D": 0.05},
            ic_std={"1D": 0.02},
            ic_t_stat={"1D": 2.5},
            ic_p_value={"1D": 0.01},
            quantile_returns={"1D": {1: 0.01, 2: 0.02, 3: 0.03}},
            spread={"1D": 0.02},
            spread_t_stat={"1D": 2.0},
            spread_p_value={"1D": 0.05},
            monotonicity={"1D": 0.9},
            periods=(1,),
            quantiles=3,
        )
        qr = result.to_quantile_result()
        # std_returns should be 0.0 when not available
        assert qr.std_returns["1D"]["Q1"] == 0.0


# =============================================================================
# Test to_tear_sheet()
# =============================================================================


class TestToTearSheet:
    """Bridge from SignalResult to SignalTearSheet."""

    def test_basic_structure(self, signal_result):
        """to_tear_sheet() produces valid SignalTearSheet."""
        ts = signal_result.to_tear_sheet("my_signal")
        assert ts.signal_name == "my_signal"
        assert ts.n_assets == signal_result.n_assets
        assert ts.n_dates == signal_result.n_dates
        assert ts.date_range == signal_result.date_range

    def test_ic_analysis_present(self, signal_result):
        """Tear sheet includes IC analysis."""
        ts = signal_result.to_tear_sheet()
        assert ts.ic_analysis is not None
        assert set(ts.ic_analysis.ic_mean.keys()) == {"1D", "5D", "21D"}

    def test_quantile_analysis_present(self, signal_result):
        """Tear sheet includes quantile analysis."""
        ts = signal_result.to_tear_sheet()
        assert ts.quantile_analysis is not None
        assert ts.quantile_analysis.n_quantiles == 5

    def test_summary(self, signal_result):
        """Tear sheet summary includes both IC and quantile sections."""
        ts = signal_result.to_tear_sheet()
        s = ts.summary()
        assert "IC Analysis" in s
        assert "Quantile Analysis" in s

    def test_default_signal_name(self, signal_result):
        """Default signal name is 'signal'."""
        ts = signal_result.to_tear_sheet()
        assert ts.signal_name == "signal"


# =============================================================================
# Test viz roundtrip (analyze_signal -> bridge -> plot function)
# =============================================================================


class TestVizRoundtrip:
    """End-to-end: analyze_signal() -> bridge -> viz function doesn't error."""

    def test_ic_ts_roundtrip(self, signal_result):
        """analyze_signal -> to_ic_result -> plot_ic_ts produces a figure."""
        from ml4t.diagnostic.visualization.signal.ic_plots import plot_ic_ts

        ic_result = signal_result.to_ic_result()
        fig = plot_ic_ts(ic_result)
        assert fig is not None
        assert len(fig.data) > 0

    def test_quantile_bar_roundtrip(self, signal_result):
        """analyze_signal -> to_quantile_result -> plot_quantile_returns_bar produces a figure."""
        from ml4t.diagnostic.visualization.signal.quantile_plots import (
            plot_quantile_returns_bar,
        )

        qr = signal_result.to_quantile_result()
        fig = plot_quantile_returns_bar(qr)
        assert fig is not None
        assert len(fig.data) > 0

    def test_ic_single_period_roundtrip(self, signal_result):
        """Single-period IC result works with plot_ic_ts."""
        from ml4t.diagnostic.visualization.signal.ic_plots import plot_ic_ts

        ic_result = signal_result.to_ic_result(period=21)
        fig = plot_ic_ts(ic_result, period="21D")
        assert fig is not None
