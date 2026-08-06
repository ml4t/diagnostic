"""Tests for signal analysis plots."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml4t.diagnostic.results.signal_results import SignalICResult, TurnoverAnalysisResult
from ml4t.diagnostic.visualization.signal import ic_plots
from ml4t.diagnostic.visualization.signal.turnover_plots import (
    plot_autocorrelation,
    plot_top_bottom_turnover,
)


@pytest.fixture
def ic_result() -> SignalICResult:
    """Create an IC result with enough observations for normality tests."""
    values = [0.01, 0.03, -0.02, 0.04, -0.01, 0.02, 0.0, 0.05]
    return SignalICResult(
        ic_by_date={"1D": values},
        dates=[f"2020-01-{day:02d}" for day in range(1, 9)],
        ic_mean={"1D": 0.015},
        ic_std={"1D": 0.025},
        ic_t_stat={"1D": 1.7},
        ic_p_value={"1D": 0.1},
        ic_positive_pct={"1D": 62.5},
        ic_ir={"1D": 0.6},
    )


@pytest.fixture
def turnover_result() -> TurnoverAnalysisResult:
    """Create a turnover result with one horizon."""
    return TurnoverAnalysisResult(
        quantile_turnover={"1D": {"Q1": 0.15, "Q5": 0.17}},
        mean_turnover={"1D": 0.16},
        top_quantile_turnover={"1D": 0.17},
        bottom_quantile_turnover={"1D": 0.15},
        autocorrelation={"1D": [0.8, 0.65, 0.53, 0.43, 0.35]},
        autocorrelation_lags=[1, 2, 3, 4, 5],
        mean_autocorrelation={"1D": 0.552},
        half_life={"1D": 3.0},
    )


def test_qq_plot_does_not_infer_normality_from_unavailable_test(
    ic_result: SignalICResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable normality test does not produce a verdict."""
    monkeypatch.setattr(ic_plots, "_shapiro", lambda values: (0.0, float("nan")))
    monkeypatch.setattr(
        ic_plots,
        "_jarque_bera",
        lambda values: SimpleNamespace(pvalue=0.9),
    )

    figure = ic_plots.plot_ic_qq(ic_result, period="1D")
    annotation_text = " ".join(annotation.text for annotation in figure.layout.annotations)

    assert "Shapiro-Wilk p: N/A" in annotation_text
    assert "Jarque-Bera p: 0.9000" in annotation_text
    assert "Normality: N/A" in annotation_text
    assert "✓ Normal" not in annotation_text
    assert "✗ Non-normal" not in annotation_text


def test_autocorrelation_distinguishes_zero_and_missing_half_life(
    turnover_result: TurnoverAnalysisResult,
) -> None:
    """Zero is a valid half-life while None is unavailable."""
    missing = turnover_result.model_copy(deep=True)
    missing.half_life["1D"] = None
    missing_figure = plot_autocorrelation(missing, period="1D")
    missing_text = next(
        annotation.text
        for annotation in missing_figure.layout.annotations
        if "Half-life" in annotation.text
    )

    zero = turnover_result.model_copy(deep=True)
    zero.half_life["1D"] = 0.0
    zero_figure = plot_autocorrelation(zero, period="1D")
    zero_text = next(
        annotation.text
        for annotation in zero_figure.layout.annotations
        if "Half-life" in annotation.text
    )

    assert "Mean AC (Lag 1-5):" in missing_text
    assert "Half-life: N/A" in missing_text
    assert "N/A periods" not in missing_text
    assert "Half-life: 0.0 periods" in zero_text


def test_turnover_plot_omits_unit_for_unavailable_half_life(
    turnover_result: TurnoverAnalysisResult,
) -> None:
    """Unavailable half-life values do not retain a numeric unit."""
    result = turnover_result.model_copy(deep=True)
    result.half_life["1D"] = float("nan")

    figure = plot_top_bottom_turnover(result)
    annotation_text = " ".join(annotation.text for annotation in figure.layout.annotations)

    assert "1D: N/A<br>" in annotation_text
    assert "N/A periods" not in annotation_text
