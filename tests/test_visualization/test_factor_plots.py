"""Tests for factor analysis visualization functions."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import plotly.graph_objects as go
import pytest

from ml4t.diagnostic.evaluation.factor.analysis import FactorAnalysis
from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.rolling_model import compute_rolling_exposures
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model
from ml4t.diagnostic.evaluation.factor.attribution import compute_return_attribution
from ml4t.diagnostic.evaluation.factor.risk import compute_risk_attribution
from ml4t.diagnostic.visualization.factor import (
    plot_factor_betas_bar,
    plot_factor_correlation_heatmap,
    plot_residual_diagnostics,
    plot_return_attribution_area,
    plot_return_attribution_waterfall,
    plot_risk_attribution_bar,
    plot_risk_attribution_pie,
    plot_rolling_betas,
    plot_vif_bar,
)


@pytest.fixture
def factor_data() -> FactorData:
    np.random.seed(42)
    T = 500
    dates = pl.date_range(
        date(2018, 1, 1), date(2019, 12, 31), eager=True
    )[:T]
    return FactorData.from_dataframe(pl.DataFrame({
        "timestamp": dates,
        "Mkt-RF": np.random.normal(0.0004, 0.01, T),
        "SMB": np.random.normal(0.0001, 0.005, T),
        "HML": np.random.normal(0.0001, 0.005, T),
    }))


@pytest.fixture
def returns(factor_data: FactorData) -> np.ndarray:
    np.random.seed(42)
    T = factor_data.n_periods
    X = factor_data.get_factor_array()
    eps = np.random.normal(0, 0.003, T)
    return 0.0002 + X @ np.array([1.0, 0.3, -0.1]) + eps


@pytest.fixture
def model_result(returns: np.ndarray, factor_data: FactorData):
    return compute_factor_model(returns, factor_data)


@pytest.fixture
def rolling_result(returns: np.ndarray, factor_data: FactorData):
    return compute_rolling_exposures(returns, factor_data, window=63, compute_vif=True)


@pytest.fixture
def attribution_result(returns: np.ndarray, factor_data: FactorData):
    return compute_return_attribution(returns, factor_data, window=63)


@pytest.fixture
def risk_result(returns: np.ndarray, factor_data: FactorData):
    return compute_risk_attribution(returns, factor_data)


class TestExposurePlots:
    def test_factor_betas_bar(self, model_result) -> None:
        fig = plot_factor_betas_bar(model_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_factor_betas_bar_dark_theme(self, model_result) -> None:
        fig = plot_factor_betas_bar(model_result, theme="dark")
        assert isinstance(fig, go.Figure)

    def test_factor_betas_bar_custom_size(self, model_result) -> None:
        fig = plot_factor_betas_bar(model_result, height=400, width=800)
        assert fig.layout.height == 400
        assert fig.layout.width == 800

    def test_rolling_betas(self, rolling_result) -> None:
        fig = plot_rolling_betas(rolling_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_rolling_betas_no_r2(self, rolling_result) -> None:
        fig = plot_rolling_betas(rolling_result, show_r_squared=False)
        assert isinstance(fig, go.Figure)

    def test_rolling_betas_themes(self, rolling_result) -> None:
        for theme in ["default", "dark", "print", "presentation"]:
            fig = plot_rolling_betas(rolling_result, theme=theme)
            assert isinstance(fig, go.Figure)


class TestAttributionPlots:
    def test_waterfall(self, attribution_result) -> None:
        fig = plot_return_attribution_waterfall(attribution_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_waterfall_dark(self, attribution_result) -> None:
        fig = plot_return_attribution_waterfall(attribution_result, theme="dark")
        assert isinstance(fig, go.Figure)

    def test_area(self, attribution_result) -> None:
        fig = plot_return_attribution_area(attribution_result)
        assert isinstance(fig, go.Figure)
        # Should have traces for each factor + alpha + residual + total
        assert len(fig.data) >= 4

    def test_area_themes(self, attribution_result) -> None:
        for theme in ["default", "dark", "print", "presentation"]:
            fig = plot_return_attribution_area(attribution_result, theme=theme)
            assert isinstance(fig, go.Figure)


class TestRiskPlots:
    def test_pie(self, risk_result) -> None:
        fig = plot_risk_attribution_pie(risk_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_pie_dark(self, risk_result) -> None:
        fig = plot_risk_attribution_pie(risk_result, theme="dark")
        assert isinstance(fig, go.Figure)

    def test_mctr_bar(self, risk_result) -> None:
        fig = plot_risk_attribution_bar(risk_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestDiagnosticPlots:
    def test_residual_diagnostics(self, model_result) -> None:
        fig = plot_residual_diagnostics(model_result)
        assert isinstance(fig, go.Figure)
        # Should have 4 subplots worth of traces
        assert len(fig.data) >= 4

    def test_correlation_heatmap(self, factor_data) -> None:
        fig = plot_factor_correlation_heatmap(factor_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_vif_bar_from_rolling(self, rolling_result) -> None:
        fig = plot_vif_bar(rolling_result)
        assert isinstance(fig, go.Figure)

    def test_vif_bar_from_factor_data(self, model_result, factor_data) -> None:
        fig = plot_vif_bar(model_result, factor_data=factor_data)
        assert isinstance(fig, go.Figure)

    def test_vif_bar_no_data_raises(self, model_result) -> None:
        with pytest.raises(ValueError, match="VIF not available"):
            plot_vif_bar(model_result)


class TestGenerateReport:
    def test_generate_report(
        self, returns: np.ndarray, factor_data: FactorData
    ) -> None:
        fa = FactorAnalysis(returns, factor_data)
        report = fa.generate_report()
        assert isinstance(report, dict)
        assert "factor_betas" in report
        assert "rolling_betas" in report
        assert "attribution_waterfall" in report
        assert "risk_decomposition" in report
        assert "residual_diagnostics" in report
        assert "factor_correlation" in report
        for fig in report.values():
            assert isinstance(fig, go.Figure)
