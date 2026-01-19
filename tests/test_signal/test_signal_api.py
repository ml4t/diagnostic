"""Tests for the new signal analysis API.

Tests the clean functional API at ml4t.diagnostic.signal.
"""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_data():
    """Create sample factor and price data."""
    np.random.seed(42)
    n_dates = 50
    n_assets = 100

    dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 3, 20), eager=True)[:n_dates]

    # Factor data
    factor_data = []
    for d in dates:
        for a in range(n_assets):
            factor_data.append({"date": d, "asset": f"A{a}", "factor": np.random.randn()})
    factor_df = pl.DataFrame(factor_data)

    # Price data (random walk)
    price_data = []
    prices = np.random.randn(n_assets).cumsum() + 100
    for d in dates:
        prices = prices * (1 + np.random.randn(n_assets) * 0.01)
        for a, p in enumerate(prices):
            price_data.append({"date": d, "asset": f"A{a}", "price": p})
    prices_df = pl.DataFrame(price_data)

    return factor_df, prices_df


class TestAnalyzeSignal:
    """Tests for analyze_signal function."""

    def test_basic_usage(self, sample_data):
        """Test basic analyze_signal call."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df)

        assert result.n_assets == 100
        assert result.n_dates == 50
        assert result.periods == (1, 5, 21)
        assert result.quantiles == 5

    def test_custom_periods(self, sample_data):
        """Test with custom periods."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, periods=(1, 5))

        assert result.periods == (1, 5)
        assert "1D" in result.ic
        assert "5D" in result.ic
        assert "21D" not in result.ic

    def test_custom_quantiles(self, sample_data):
        """Test with custom quantiles."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, quantiles=10, periods=(1,))

        assert result.quantiles == 10
        # Should have 10 quantile returns
        assert len(result.quantile_returns["1D"]) == 10

    def test_ic_values(self, sample_data):
        """Test IC values are computed."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, periods=(1,))

        # IC should be a float (may be close to 0 for random data)
        assert isinstance(result.ic["1D"], float)
        assert isinstance(result.ic_std["1D"], float)
        assert isinstance(result.ic_t_stat["1D"], float)
        assert isinstance(result.ic_p_value["1D"], float)

    def test_spread_values(self, sample_data):
        """Test spread values are computed."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, periods=(1,))

        assert isinstance(result.spread["1D"], float)
        assert isinstance(result.spread_t_stat["1D"], float)
        assert isinstance(result.spread_p_value["1D"], float)

    def test_turnover_disabled(self, sample_data):
        """Test turnover can be disabled."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, compute_turnover_flag=False, periods=(1,))

        assert result.turnover is None
        assert result.autocorrelation is None
        assert result.half_life is None

    def test_turnover_enabled(self, sample_data):
        """Test turnover when enabled."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, compute_turnover_flag=True, periods=(1,))

        assert result.turnover is not None
        assert "1D" in result.turnover
        assert result.autocorrelation is not None
        assert len(result.autocorrelation) == 10  # default lags

    def test_no_filter(self, sample_data):
        """Test with outlier filtering disabled."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, filter_zscore=None, periods=(1,))

        # Should still work
        assert result.n_assets == 100


class TestSignalResult:
    """Tests for SignalResult dataclass."""

    def test_summary(self, sample_data):
        """Test summary method."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, periods=(1,))

        summary = result.summary()
        assert "Signal Analysis" in summary
        assert "IC Summary" in summary
        assert "Spread" in summary

    def test_to_dict(self, sample_data):
        """Test to_dict method."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, periods=(1,))

        d = result.to_dict()
        assert "ic" in d
        assert "spread" in d
        assert "n_assets" in d

    def test_to_json(self, sample_data, tmp_path):
        """Test to_json method."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, periods=(1,))

        # To string
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "ic" in json_str

        # To file
        path = tmp_path / "result.json"
        result.to_json(str(path))
        assert path.exists()

    def test_from_json(self, sample_data, tmp_path):
        """Test from_json method."""
        from ml4t.diagnostic.signal import SignalResult, analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, periods=(1,))

        path = tmp_path / "result.json"
        result.to_json(str(path))

        loaded = SignalResult.from_json(str(path))
        assert loaded.n_assets == result.n_assets
        assert loaded.n_dates == result.n_dates
        assert loaded.periods == result.periods

    def test_frozen(self, sample_data):
        """Test that result is frozen (immutable)."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_df, prices_df = sample_data
        result = analyze_signal(factor_df, prices_df, periods=(1,))

        with pytest.raises(AttributeError):  # FrozenInstanceError
            result.n_assets = 999


class TestPrepareData:
    """Tests for prepare_data function."""

    def test_basic(self, sample_data):
        """Test basic prepare_data call."""
        from ml4t.diagnostic.signal import prepare_data

        factor_df, prices_df = sample_data
        data = prepare_data(factor_df, prices_df, periods=(1, 5))

        assert "factor" in data.columns
        assert "quantile" in data.columns
        assert "1D_fwd_return" in data.columns
        assert "5D_fwd_return" in data.columns

    def test_quantile_assignment(self, sample_data):
        """Test quantile column is assigned correctly."""
        from ml4t.diagnostic.signal import prepare_data

        factor_df, prices_df = sample_data
        data = prepare_data(factor_df, prices_df, quantiles=5, periods=(1,))

        # Check quantiles are 1-5
        unique_q = data.select("quantile").unique().sort("quantile").to_series().to_list()
        assert unique_q == [1, 2, 3, 4, 5]


class TestBuildingBlocks:
    """Tests for individual functions."""

    def test_compute_ic_series(self, sample_data):
        """Test compute_ic_series function."""
        from ml4t.diagnostic.signal import compute_ic_series, prepare_data

        factor_df, prices_df = sample_data
        data = prepare_data(factor_df, prices_df, periods=(1,))

        dates, ic_vals = compute_ic_series(data, period=1)

        assert len(dates) == len(ic_vals)
        assert len(dates) > 0
        assert all(isinstance(ic, float) for ic in ic_vals)

    def test_compute_quantile_returns(self, sample_data):
        """Test compute_quantile_returns function."""
        from ml4t.diagnostic.signal import compute_quantile_returns, prepare_data

        factor_df, prices_df = sample_data
        data = prepare_data(factor_df, prices_df, periods=(1,), quantiles=5)

        q_returns = compute_quantile_returns(data, period=1, n_quantiles=5)

        assert len(q_returns) == 5
        assert 1 in q_returns
        assert 5 in q_returns

    def test_compute_turnover(self, sample_data):
        """Test compute_turnover function."""
        from ml4t.diagnostic.signal import compute_turnover, prepare_data

        factor_df, prices_df = sample_data
        data = prepare_data(factor_df, prices_df, periods=(1,), quantiles=5)

        turnover = compute_turnover(data, n_quantiles=5)

        assert isinstance(turnover, float)
        assert 0 <= turnover <= 1 or np.isnan(turnover)

    def test_filter_outliers(self, sample_data):
        """Test filter_outliers function."""
        from ml4t.diagnostic.signal import filter_outliers
        from ml4t.diagnostic.signal._utils import ensure_polars

        factor_df, _ = sample_data
        factor_df = ensure_polars(factor_df)

        # Add some extreme outliers
        factor_df = factor_df.with_columns(
            pl.when(pl.col("asset") == "A0")
            .then(pl.lit(100.0))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )

        filtered = filter_outliers(factor_df, z_threshold=3.0)

        # Should have fewer rows (outliers removed)
        assert filtered.height < factor_df.height


class TestPandasCompatibility:
    """Tests for pandas DataFrame input."""

    def test_pandas_input(self, sample_data):
        """Test that pandas DataFrames work."""
        from ml4t.diagnostic.signal import analyze_signal

        factor_pl, prices_pl = sample_data

        # Convert to pandas
        factor_pd = factor_pl.to_pandas()
        prices_pd = prices_pl.to_pandas()

        result = analyze_signal(factor_pd, prices_pd, periods=(1,))

        assert result.n_assets == 100
        assert result.n_dates == 50


class TestSerializationEdgeCases:
    """Tests for SignalResult serialization edge cases."""

    def test_to_json_nan_values(self, sample_data, tmp_path):
        """Test that NaN values are converted to null in JSON."""
        import json

        from ml4t.diagnostic.signal import SignalResult

        # Create result with NaN values
        result = SignalResult(
            ic={"1D": float("nan")},
            ic_std={"1D": float("nan")},
            ic_t_stat={"1D": float("nan")},
            ic_p_value={"1D": float("nan")},
            n_assets=10,
            n_dates=5,
            periods=(1,),
        )

        json_str = result.to_json()

        # NaN should become null in JSON
        assert "null" in json_str
        assert "NaN" not in json_str

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["ic"]["1D"] is None

    def test_to_json_nested_dicts(self, sample_data, tmp_path):
        """Test that nested dicts (quantile_returns) serialize correctly."""
        import json

        from ml4t.diagnostic.signal import SignalResult

        result = SignalResult(
            ic={"1D": 0.05},
            ic_std={"1D": 0.02},
            ic_t_stat={"1D": 2.5},
            ic_p_value={"1D": 0.01},
            quantile_returns={"1D": {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05}},
            n_assets=10,
            n_dates=5,
            periods=(1,),
        )

        json_str = result.to_json()
        data = json.loads(json_str)

        # Nested dict keys become strings in JSON
        assert "1" in data["quantile_returns"]["1D"]
        assert data["quantile_returns"]["1D"]["1"] == 0.01

    def test_from_json_null_to_nan(self, tmp_path):
        """Test that JSON null values become NaN when loaded."""
        import json

        from ml4t.diagnostic.signal import SignalResult

        # Write JSON with null values
        json_data = {
            "ic": {"1D": None},
            "ic_std": {"1D": None},
            "ic_t_stat": {"1D": None},
            "ic_p_value": {"1D": None},
            "ic_ir": {},
            "ic_positive_pct": {},
            "ic_series": {},
            "quantile_returns": {},
            "spread": {},
            "spread_t_stat": {},
            "spread_p_value": {},
            "monotonicity": {},
            "turnover": None,
            "autocorrelation": None,
            "half_life": None,
            "n_assets": 10,
            "n_dates": 5,
            "date_range": ["2024-01-01", "2024-01-05"],
            "periods": [1],
            "quantiles": 5,
        }

        path = tmp_path / "result.json"
        with open(path, "w") as f:
            json.dump(json_data, f)

        loaded = SignalResult.from_json(str(path))

        # null should remain None (not NaN for dict values)
        assert loaded.n_assets == 10
        assert loaded.periods == (1,)

    def test_from_json_quantile_key_conversion(self, tmp_path):
        """Test that quantile keys are converted back to int."""
        import json

        from ml4t.diagnostic.signal import SignalResult

        json_data = {
            "ic": {"1D": 0.05},
            "ic_std": {"1D": 0.02},
            "ic_t_stat": {"1D": 2.5},
            "ic_p_value": {"1D": 0.01},
            "ic_ir": {},
            "ic_positive_pct": {},
            "ic_series": {},
            "quantile_returns": {"1D": {"1": 0.01, "2": 0.02, "3": 0.03}},
            "spread": {},
            "spread_t_stat": {},
            "spread_p_value": {},
            "monotonicity": {},
            "turnover": None,
            "autocorrelation": None,
            "half_life": None,
            "n_assets": 10,
            "n_dates": 5,
            "date_range": ["2024-01-01", "2024-01-05"],
            "periods": [1],
            "quantiles": 3,
        }

        path = tmp_path / "result.json"
        with open(path, "w") as f:
            json.dump(json_data, f)

        loaded = SignalResult.from_json(str(path))

        # String keys should become int
        assert 1 in loaded.quantile_returns["1D"]
        assert isinstance(list(loaded.quantile_returns["1D"].keys())[0], int)

    def test_json_round_trip(self, sample_data, tmp_path):
        """Test complete JSON round-trip preserves data."""
        from ml4t.diagnostic.signal import SignalResult, analyze_signal

        factor_df, prices_df = sample_data
        original = analyze_signal(factor_df, prices_df, periods=(1,))

        path = tmp_path / "result.json"
        original.to_json(str(path))
        loaded = SignalResult.from_json(str(path))

        # Key fields should match
        assert loaded.n_assets == original.n_assets
        assert loaded.n_dates == original.n_dates
        assert loaded.periods == original.periods
        assert loaded.quantiles == original.quantiles
        # IC values should be close (allowing for NaN->null->None conversions)
        for period in original.ic:
            if not np.isnan(original.ic[period]):
                assert loaded.ic[period] is not None


class TestSummaryEdgeCases:
    """Tests for SignalResult.summary() edge cases."""

    def test_summary_with_nan_values(self):
        """Test summary handles NaN values gracefully."""
        from ml4t.diagnostic.signal import SignalResult

        result = SignalResult(
            ic={"1D": float("nan")},
            ic_std={"1D": float("nan")},
            ic_t_stat={"1D": float("nan")},
            ic_p_value={"1D": float("nan")},
            n_assets=10,
            n_dates=5,
            periods=(1,),
        )

        summary = result.summary()

        # Should not raise, should include "nan" in output
        assert isinstance(summary, str)
        assert "Signal Analysis" in summary

    def test_summary_no_turnover(self):
        """Test summary when turnover is None."""
        from ml4t.diagnostic.signal import SignalResult

        result = SignalResult(
            ic={"1D": 0.05},
            ic_std={"1D": 0.02},
            ic_t_stat={"1D": 2.5},
            ic_p_value={"1D": 0.01},
            turnover=None,
            half_life=None,
            n_assets=10,
            n_dates=5,
            periods=(1,),
        )

        summary = result.summary()

        # Should not include turnover section
        assert "Turnover:" not in summary
        assert "Half-life:" not in summary

    def test_summary_with_turnover(self):
        """Test summary includes turnover when present."""
        from ml4t.diagnostic.signal import SignalResult

        result = SignalResult(
            ic={"1D": 0.05},
            ic_std={"1D": 0.02},
            ic_t_stat={"1D": 2.5},
            ic_p_value={"1D": 0.01},
            turnover={"1D": 0.15},
            half_life=3.5,
            n_assets=10,
            n_dates=5,
            periods=(1,),
        )

        summary = result.summary()

        # Should include turnover section
        assert "Turnover:" in summary
        assert "Half-life:" in summary

    def test_summary_significance_asterisk(self):
        """Test that significant ICs get asterisk."""
        from ml4t.diagnostic.signal import SignalResult

        result = SignalResult(
            ic={"1D": 0.05},
            ic_std={"1D": 0.02},
            ic_t_stat={"1D": 5.0},
            ic_p_value={"1D": 0.001},  # p < 0.05, significant
            n_assets=10,
            n_dates=5,
            periods=(1,),
        )

        summary = result.summary()

        # Significant IC should have asterisk
        assert "*" in summary
