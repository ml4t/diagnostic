"""Data correctness tests for factor visualization functions.

Unlike test_factor_plots.py which checks "does it render?", these tests
verify that the plotted data values actually match the input data.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.attribution import compute_return_attribution
from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.risk import compute_risk_attribution
from ml4t.diagnostic.evaluation.factor.rolling_model import compute_rolling_exposures
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model
from ml4t.diagnostic.visualization.factor import (
    plot_factor_betas_bar,
    plot_factor_correlation_heatmap,
    plot_return_attribution_waterfall,
    plot_risk_attribution_pie,
    plot_rolling_betas,
)


@pytest.fixture
def factor_setup() -> tuple[np.ndarray, FactorData]:
    np.random.seed(42)
    T = 500
    mkt = np.random.normal(0.0004, 0.01, T)
    smb = np.random.normal(0.0001, 0.005, T)
    hml = np.random.normal(0.0001, 0.005, T)
    eps = np.random.normal(0, 0.003, T)
    returns = 0.0002 + 1.0 * mkt + 0.3 * smb - 0.1 * hml + eps

    dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
    fd = FactorData.from_dataframe(
        pl.DataFrame({"timestamp": dates, "Mkt-RF": mkt, "SMB": smb, "HML": hml})
    )
    return returns, fd


class TestBetaBarDataCorrectness:
    def test_plotted_betas_match_model(self, factor_setup: tuple) -> None:
        """Bar chart x-values should equal model betas."""
        returns, fd = factor_setup
        model = compute_factor_model(returns, fd)
        fig = plot_factor_betas_bar(model)

        # The first trace should be a Bar with factor betas
        bar = fig.data[0]
        plotted_names = list(bar.y)
        plotted_values = list(bar.x)

        for i, f in enumerate(model.factor_names):
            assert f in plotted_names
            idx = plotted_names.index(f)
            assert abs(plotted_values[idx] - model.betas[f]) < 1e-10, (
                f"Plotted beta for {f} = {plotted_values[idx]}, expected {model.betas[f]}"
            )

    def test_error_bars_match_cis(self, factor_setup: tuple) -> None:
        """Error bars should match CI widths from model."""
        returns, fd = factor_setup
        model = compute_factor_model(returns, fd)
        fig = plot_factor_betas_bar(model)

        bar = fig.data[0]
        plotted_names = list(bar.y)
        error_plus = list(bar.error_x.array)
        error_minus = list(bar.error_x.arrayminus)

        for f in model.factor_names:
            idx = plotted_names.index(f)
            expected_plus = model.beta_cis[f][1] - model.betas[f]
            expected_minus = model.betas[f] - model.beta_cis[f][0]
            assert abs(error_plus[idx] - expected_plus) < 1e-10
            assert abs(error_minus[idx] - expected_minus) < 1e-10


class TestRollingBetasDataCorrectness:
    def test_plotted_traces_match_rolling(self, factor_setup: tuple) -> None:
        """Each trace y-values should match rolling betas from model."""
        returns, fd = factor_setup
        rolling = compute_rolling_exposures(returns, fd, window=63)
        fig = plot_rolling_betas(rolling)

        # Find scatter traces by name (factor names)
        for f in fd.factor_names:
            matching = [t for t in fig.data if t.name == f]
            assert len(matching) >= 1, f"No trace named '{f}'"
            trace = matching[0]
            expected = rolling.rolling_betas[f]
            # Compare valid (non-NaN) values
            valid = np.isfinite(expected)
            plotted = np.array(trace.y)
            np.testing.assert_allclose(
                plotted[valid],
                expected[valid],
                atol=1e-10,
                err_msg=f"Rolling betas for {f} don't match",
            )


class TestWaterfallDataCorrectness:
    def test_waterfall_values_match_attribution(self, factor_setup: tuple) -> None:
        """Waterfall bar values should match cumulative attribution."""
        returns, fd = factor_setup
        attr = compute_return_attribution(returns, fd, window=63)
        fig = plot_return_attribution_waterfall(attr)

        # Waterfall trace
        wf = fig.data[0]
        labels = list(wf.x)
        values = list(wf.y)

        # Check each factor's cumulative contribution
        for f in fd.factor_names:
            assert f in labels
            idx = labels.index(f)
            expected = attr.cumulative_factor[f][-1]
            assert abs(values[idx] - expected) < 1e-10, (
                f"Waterfall {f}: plotted={values[idx]}, expected={expected}"
            )

        # Check alpha
        alpha_idx = labels.index("Alpha")
        assert abs(values[alpha_idx] - attr.cumulative_alpha[-1]) < 1e-10

        # Check total
        total_idx = labels.index("Total")
        assert abs(values[total_idx] - attr.cumulative_total[-1]) < 1e-10


class TestRiskPieDataCorrectness:
    def test_pie_values_match_risk(self, factor_setup: tuple) -> None:
        """Pie chart values should match variance contributions."""
        returns, fd = factor_setup
        risk = compute_risk_attribution(returns, fd)
        fig = plot_risk_attribution_pie(risk)

        pie = fig.data[0]
        labels = list(pie.labels)
        values = list(pie.values)

        # Factor contributions (abs values used for pie)
        for f in fd.factor_names:
            assert f in labels
            idx = labels.index(f)
            expected = abs(risk.factor_contributions[f])
            assert abs(values[idx] - expected) < 1e-10, (
                f"Pie {f}: plotted={values[idx]}, expected={expected}"
            )

        # Idiosyncratic
        assert "Idiosyncratic" in labels
        idx = labels.index("Idiosyncratic")
        assert abs(values[idx] - risk.idiosyncratic_variance) < 1e-10


class TestCorrelationHeatmapDataCorrectness:
    def test_heatmap_values_match_correlation(self, factor_setup: tuple) -> None:
        """Heatmap z-values should match factor correlation matrix."""
        _, fd = factor_setup
        fig = plot_factor_correlation_heatmap(fd)

        heatmap = fig.data[0]
        z = np.array(heatmap.z)

        # Compute expected correlation
        X = fd.get_factor_array()
        expected_corr = np.corrcoef(X, rowvar=False)

        np.testing.assert_allclose(z, expected_corr, atol=1e-10)
